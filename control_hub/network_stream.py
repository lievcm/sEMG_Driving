"""Network streaming thread: sends the steering angle to another computer over TCP or UDP.

The remote computer runs ``client/steering_client.py``, which listens on a port; this thread
connects (TCP) or sends datagrams (UDP) to it. Every ``1 / send_hz`` seconds the latest value
from LiveState is sent, so the EMG thread is never blocked by the network.

Wire format: one JSON object per line (newline-delimited JSON; one line per UDP datagram)::

    {"v":1,"seq":42,"t":1767000000.123,"angle_deg":-12.3,"angle_norm":-0.027,
     "arm_left":-0.05,"arm_right":0.0,"age_s":0.004}

``angle_deg`` is the steering angle computed from the EMG (full +-lock_to_lock_deg/2 range, not
the clamped wheel target). ``age_s`` is the time since the EMG thread last published; if it grows
the EMG is not streaming (the angle is then 0).

TCP: if the link drops after a successful connection the thread keeps retrying every
``reconnect_interval_s``. If the *first* connection fails the thread stops and reports it.
UDP: fire-and-forget; send errors are shown in the status but never stop the stream.
"""
from __future__ import annotations

import json
import select
import socket
import threading
import time
import traceback

from PyQt6.QtCore import QThread, pyqtSignal

from config import NetworkConfig
from sources import high_res_timer
from steering import LiveState, Snapshot

PROTOCOL_VERSION = 1


def encode_message(seq: int, snap: Snapshot, age_s: float, now: float | None = None) -> bytes:
    """One newline-terminated JSON message."""
    msg = {
        "v": PROTOCOL_VERSION,
        "seq": seq,
        "t": round(time.time() if now is None else now, 4),     # sender wall-clock, Unix seconds
        "angle_deg": round(snap.angle_deg, 4),
        "angle_norm": round(snap.angle_norm, 6),
        "arm_left": round(snap.arm_left, 6),
        "arm_right": round(snap.arm_right, 6),
        "age_s": round(age_s, 4),
    }
    return (json.dumps(msg, separators=(",", ":")) + "\n").encode("ascii")


def _peer_closed(sock: socket.socket) -> bool:
    """True if the remote end closed the TCP connection (the receiver never sends anything)."""
    readable, _, _ = select.select([sock], [], [], 0)
    if not readable:
        return False
    try:
        return sock.recv(4096) == b""
    except (BlockingIOError, socket.timeout):
        return False


class NetworkStreamer(QThread):
    status = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, cfg: NetworkConfig, live: LiveState, parent=None):
        super().__init__(parent)
        self._cfg = cfg                       # replaced atomically by update_config()
        self._live = live
        self._stop_evt = threading.Event()
        self._seq = 0

    def update_config(self, cfg: NetworkConfig) -> None:
        """send_hz applies immediately; protocol / host / port need a reconnect."""
        self._cfg = cfg

    def stop(self) -> None:
        self._stop_evt.set()

    # ------------------------------------------------------------------ thread
    def run(self) -> None:
        cfg = self._cfg                       # destination is fixed for this connection
        tcp = cfg.protocol == "tcp"
        label = f"{cfg.host}:{cfg.port} ({cfg.protocol.upper()})"
        sock = None
        try:
            self.status.emit(f"Connecting to {label}…")
            try:
                sock, addr = self._open(cfg)
            except OSError as exc:
                raise RuntimeError(
                    f"Could not reach {label}: {exc}\n\nCheck that steering_client.py is running on "
                    f"the remote computer, that the address and port are correct, and that the "
                    f"port is allowed through its firewall.") from exc
            self.status.emit(f"Streaming to {label}")
            with high_res_timer():
                while not self._stop_evt.is_set():
                    try:
                        self._stream(sock, addr, tcp, label)
                    except OSError as exc:    # TCP link lost -> reconnect
                        _close(sock)
                        sock = None
                        sock, addr = self._reconnect(cfg, label, exc)
                        if sock is None:      # stopped while reconnecting
                            break
                        self.status.emit(f"Streaming to {label}")
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            _close(sock)
            self.status.emit("Disconnected")

    def _open(self, cfg: NetworkConfig):
        """Returns (socket, address). TCP: connected socket. UDP: unconnected socket + resolved address."""
        if cfg.protocol == "tcp":
            sock = socket.create_connection((cfg.host, cfg.port), timeout=cfg.connect_timeout_s)
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)   # no batching: lowest latency
            sock.settimeout(1.0)                                         # send timeout
            return sock, None
        family, _, _, _, addr = socket.getaddrinfo(cfg.host, cfg.port, type=socket.SOCK_DGRAM)[0]
        return socket.socket(family, socket.SOCK_DGRAM), addr

    def _reconnect(self, cfg: NetworkConfig, label: str, reason: Exception):
        """Retry until connected; returns (None, None) if the thread was stopped meanwhile."""
        while True:
            self.status.emit(f"Connection lost ({reason}) — retrying {label}…")
            if self._stop_evt.wait(cfg.reconnect_interval_s):
                return None, None
            try:
                return self._open(cfg)
            except OSError as exc:
                reason = exc

    def _stream(self, sock: socket.socket, addr, tcp: bool, label: str) -> None:
        """Send at ``send_hz`` until stopped. Raises OSError if the TCP link is lost."""
        next_t = time.perf_counter()
        send_failing = False
        while not self._stop_evt.is_set():
            _, age = self._live.get_target()
            data = encode_message(self._seq, self._live.snapshot(), age)
            self._seq += 1
            if tcp:
                if _peer_closed(sock):
                    raise ConnectionResetError("remote computer closed the connection")
                sock.sendall(data)
            else:
                try:
                    sock.sendto(data, addr)
                    if send_failing:
                        send_failing = False
                        self.status.emit(f"Streaming to {label}")
                except OSError as exc:
                    if not send_failing:
                        send_failing = True
                        self.status.emit(f"Streaming to {label} — send failing: {exc}")

            period = 1.0 / self._cfg.send_hz
            next_t += period
            delay = next_t - time.perf_counter()
            if delay < -period:               # fell far behind (e.g. PC slept): don't burst
                next_t = time.perf_counter()
            elif delay > 0:
                self._stop_evt.wait(delay)


def _close(sock) -> None:
    if sock is not None:
        try:
            sock.close()
        except OSError:
            pass
