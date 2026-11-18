#!/usr/bin/env python3
"""CARLA driver's-POV demo: steering from the EMG Steering Host, pedals from the keyboard.

Controls
    UP arrow     increase throttle (hold)
    DOWN arrow   decrease throttle (hold)
    SPACE        brake (also zeroes the throttle)
    ESC / close  quit

Steering comes from the angle streamed over the network by the EMG Steering Host
(same protocol as steering_client.py: newline-delimited JSON with seq / angle_deg / angle_norm).

    python carla_emg_drive.py                       # TCP, listen on 0.0.0.0:5005
    python carla_emg_drive.py --protocol udp
    python carla_emg_drive.py --steer-source deg --max-angle 90
"""
from __future__ import annotations

import argparse
import json
import random
import socket
import threading
import time

import carla
import numpy as np
import pygame

CARLA_HOST = "localhost"
CARLA_PORT = 2000
IMG_W, IMG_H = 1280, 720

THROTTLE_RATE = 0.8     # throttle change per second while UP / DOWN is held
STALE_AGE_S = 0.5       # host says EMG hasn't updated this long -> steer straight
NO_DATA_S = 1.0         # no network message for this long -> steer straight
IDLE_TIMEOUT_S = 5.0    # TCP: drop the connection after this much silence

# Approximate driver's eye position relative to the vehicle origin (metres).
DRIVER_EYE = carla.Transform(carla.Location(x=0.1, y=-0.35, z=1.25))


class SteeringReceiver(threading.Thread):
    """Background thread that receives steering messages and keeps the latest one."""

    def __init__(self, protocol: str, bind: str, port: int):
        super().__init__(daemon=True)
        self.protocol, self.bind, self.port = protocol, bind, port
        self._lock = threading.Lock()
        self._msg = None
        self._rx_time = 0.0
        self.connected = False
        self._stop_evt = threading.Event()

    def stop(self):
        self._stop_evt.set()

    def latest(self):
        """Return (message dict or None, seconds since it arrived)."""
        with self._lock:
            return self._msg, time.monotonic() - self._rx_time

    def _handle(self, line: bytes):
        try:
            msg = json.loads(line)
            float(msg["angle_deg"])
        except (ValueError, KeyError, TypeError):
            return
        with self._lock:
            self._msg, self._rx_time = msg, time.monotonic()

    def run(self):
        try:
            (self._run_tcp if self.protocol == "tcp" else self._run_udp)()
        except OSError as exc:
            print(f"[network] error: {exc}")

    def _run_tcp(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
            srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            srv.bind((self.bind, self.port))
            srv.listen(1)
            srv.settimeout(0.5)
            print(f"[network] listening on TCP {self.bind}:{self.port}")
            while not self._stop_evt.is_set():
                try:
                    conn, peer = srv.accept()
                except socket.timeout:
                    continue
                print(f"[network] connected: {peer[0]}:{peer[1]}")
                self.connected = True
                with conn:
                    conn.settimeout(1.0)
                    buf, last_rx = b"", time.monotonic()
                    while not self._stop_evt.is_set():
                        try:
                            chunk = conn.recv(4096)
                        except socket.timeout:
                            if time.monotonic() - last_rx > IDLE_TIMEOUT_S:
                                print("[network] no data, closing connection")
                                break
                            continue
                        except OSError:
                            break
                        if not chunk:
                            print("[network] sender disconnected")
                            break
                        last_rx = time.monotonic()
                        buf += chunk
                        *lines, buf = buf.split(b"\n")
                        for line in lines:
                            if line.strip():
                                self._handle(line)
                self.connected = False

    def _run_udp(self):
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.bind((self.bind, self.port))
            sock.settimeout(0.5)
            print(f"[network] listening on UDP {self.bind}:{self.port}")
            while not self._stop_evt.is_set():
                try:
                    data, _ = sock.recvfrom(65535)
                except socket.timeout:
                    continue
                except ConnectionResetError:  # Windows ICMP noise
                    continue
                for line in data.split(b"\n"):
                    if line.strip():
                        self._handle(line)


def steer_from_message(msg, age, args) -> tuple[float, str]:
    """Convert the latest network message into a CARLA steer value in [-1, 1]."""
    if msg is None or age > NO_DATA_S:
        return 0.0, "no data"
    if float(msg.get("age_s", 0)) > STALE_AGE_S:
        return 0.0, "EMG stale"
    if args.steer_source == "norm" and "angle_norm" in msg:
        value = float(msg["angle_norm"])
    else:
        value = float(msg["angle_deg"]) / args.max_angle
    return max(-1.0, min(1.0, value * args.steer_gain)), "ok"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--carla-host", default=CARLA_HOST)
    ap.add_argument("--carla-port", type=int, default=CARLA_PORT)
    ap.add_argument("--protocol", choices=("tcp", "udp"), default="tcp", help="must match the host (default tcp)")
    ap.add_argument("--bind", default="0.0.0.0", help="local address to listen on")
    ap.add_argument("--port", type=int, default=5005, help="steering port to listen on (default 5005)")
    ap.add_argument("--steer-source", choices=("norm", "deg"), default="norm",
                    help="use angle_norm (-1..1) directly, or angle_deg / --max-angle (default norm)")
    ap.add_argument("--max-angle", type=float, default=90.0,
                    help="angle in degrees that maps to full steering lock when --steer-source deg")
    ap.add_argument("--steer-gain", type=float, default=1.0, help="multiplier applied to the steering value")
    args = ap.parse_args()

    receiver = SteeringReceiver(args.protocol, args.bind, args.port)
    receiver.start()

    client = carla.Client(args.carla_host, args.carla_port)
    client.set_timeout(10.0)
    world = client.get_world()
    blueprints = world.get_blueprint_library()

    vehicle = None
    camera = None
    latest = {"image": None}

    try:
        # --- Spawn the vehicle at a random free spawn point ---------------
        vehicle_bp = blueprints.find("vehicle.tesla.model3")
        spawn_points = world.get_map().get_spawn_points()
        random.shuffle(spawn_points)
        for sp in spawn_points:
            vehicle = world.try_spawn_actor(vehicle_bp, sp)
            if vehicle is not None:
                break
        if vehicle is None:
            raise RuntimeError("Could not find a free spawn point.")
        print(f"Spawned {vehicle.type_id} at {vehicle.get_location()}")

        # --- Camera at the driver's seat -----------------------------------
        cam_bp = blueprints.find("sensor.camera.rgb")
        cam_bp.set_attribute("image_size_x", str(IMG_W))
        cam_bp.set_attribute("image_size_y", str(IMG_H))
        cam_bp.set_attribute("fov", "90")
        camera = world.spawn_actor(cam_bp, DRIVER_EYE, attach_to=vehicle)
        camera.listen(lambda img: latest.update(image=img))

        # --- Pygame window ---------------------------------------------------
        pygame.init()
        display = pygame.display.set_mode((IMG_W, IMG_H))
        pygame.display.set_caption("CARLA - Driver's POV")
        font = pygame.font.SysFont("monospace", 18)
        clock = pygame.time.Clock()

        throttle = 0.0
        running = True
        while running:
            dt = clock.tick(30) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False

            # --- Pedals from the keyboard ------------------------------------
            keys = pygame.key.get_pressed()
            braking = bool(keys[pygame.K_SPACE])
            if keys[pygame.K_UP]:
                throttle += THROTTLE_RATE * dt
            if keys[pygame.K_DOWN]:
                throttle -= THROTTLE_RATE * dt
            throttle = max(0.0, min(1.0, throttle))
            if braking:
                throttle = 0.0

            # --- Steering from the network -----------------------------------
            msg, age = receiver.latest()
            steer, status = steer_from_message(msg, age, args)

            vehicle.apply_control(carla.VehicleControl(
                throttle=throttle, brake=1.0 if braking else 0.0, steer=steer))

            # Move the simulator's own spectator window to the driver's seat too
            world.get_spectator().set_transform(camera.get_transform())

            # --- Draw ----------------------------------------------------------
            image = latest["image"]
            if image is not None:
                frame = np.frombuffer(image.raw_data, dtype=np.uint8)
                frame = frame.reshape((image.height, image.width, 4))[:, :, :3]
                frame = frame[:, :, ::-1]  # BGRA -> RGB
                display.blit(pygame.surfarray.make_surface(frame.swapaxes(0, 1)), (0, 0))

            speed = vehicle.get_velocity()
            kmh = 3.6 * (speed.x ** 2 + speed.y ** 2 + speed.z ** 2) ** 0.5
            net = "connected" if receiver.connected or args.protocol == "udp" else "waiting for host"
            hud = (f"{kmh:5.1f} km/h   throttle {throttle:.2f}   brake {'ON ' if braking else 'off'}   "
                   f"steer {steer:+.2f}   [{net}; {status}]")
            text = font.render(hud, True, (255, 255, 255), (0, 0, 0))
            display.blit(text, (10, 10))
            pygame.display.flip()

        print("Done driving.")

    finally:
        receiver.stop()
        if camera is not None:
            camera.stop()
            camera.destroy()
        if vehicle is not None:
            vehicle.destroy()
            print("Vehicle destroyed.")
        pygame.quit()


if __name__ == "__main__":
    main()
