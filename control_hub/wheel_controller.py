"""Logitech G29/G920 control thread (closed-loop constant-force method, via SDL2).

On connect: a short, gentle force ramp learns which way positive force turns the wheel
(this differs between drivers/platforms).

Every cycle: read the wheel angle, clamp the EMG target, move the commanded angle toward it
(speed- and acceleration-limited so the heavy wheel never overshoots), and set one
constant-force effect from a PD controller that steers the wheel toward that target.
The target comes from LiveState (written by the EMG thread); if it goes stale the wheel
is centred. If the wheel ends up far from where it should be (grabbed, wrong range
setting, ...) the motor is released and the thread stops.

Works on Windows (Logitech G HUB driver), Linux (in-kernel hid-logitech) and macOS.
"""
from __future__ import annotations

import ctypes
import math
import threading
import time
import traceback

import sdl2
from PyQt6.QtCore import QThread, pyqtSignal

from config import WheelConfig
from sources import high_res_timer
from steering import LiveState

LOGITECH_VID = 0x046D
WHEEL_PIDS = {0xC24F, 0xC260,   # G29 (PS3 mode / PS4 mode)
              0xC261, 0xC262}   # G920 (Xbox mode / native)
WHEEL_NAMES = ("G29", "G920")
FULL_SCALE = 32767              # max magnitude of SDL axis values and force levels


class SafetyStop(RuntimeError):
    pass


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def pd_level(err_deg: float, err_vel: float, kp: float, kd: float, max_level: float) -> float:
    """PD control law: force toward the target, damped, clamped to +-max_level."""
    return _clamp(kp * err_deg + kd * err_vel, -max_level, max_level)


def step_toward(current: float, vel: float, target: float, max_vel: float, max_accel: float,
                dt: float) -> tuple[float, float]:
    """Move the commanded angle toward `target` with speed <= max_vel and, if max_accel > 0,
    acceleration <= max_accel, braking in time to stop at the target without overshoot.
    Returns the new (angle, velocity)."""
    if dt <= 0:
        return current, vel
    rem = target - current
    if max_accel <= 0:                                   # plain slew limit
        step = _clamp(rem, -max_vel * dt, max_vel * dt)
        return current + step, step / dt
    # Fastest speed from which we can still stop at the target, braking at max_accel in
    # steps of dt (the discrete-time version of v = sqrt(2 a d)).
    ad = max_accel * dt
    v_stop = -ad / 2.0 + math.sqrt(ad * ad / 4.0 + 2.0 * max_accel * abs(rem))
    want = _clamp(math.copysign(v_stop, rem), -max_vel, max_vel)
    vel += _clamp(want - vel, -max_accel * dt, max_accel * dt)
    nxt = current + vel * dt
    if (target - nxt) * rem <= 0:                        # would reach/pass it: arrive
        return target, 0.0
    return nxt, vel


def _sdl_error() -> str:
    return sdl2.SDL_GetError().decode("utf-8", "replace")


class SdlWheel:
    """SDL joystick + haptic handles and one running constant-force effect."""

    _SUBSYSTEMS = sdl2.SDL_INIT_JOYSTICK | sdl2.SDL_INIT_HAPTIC

    def __init__(self, device_index: int, axis: int):
        self.joy = self.haptic = None
        self.effect_id = -1
        self.axis = axis
        self.name = "?"

        # Hints must be set before SDL starts the joystick subsystem.
        sdl2.SDL_SetHint(b"SDL_JOYSTICK_ALLOW_BACKGROUND_EVENTS", b"1")
        sdl2.SDL_SetHint(b"SDL_JOYSTICK_HIDAPI", b"0")     # use the OS driver path, which
        sdl2.SDL_SetHint(b"SDL_JOYSTICK_RAWINPUT", b"0")   # is where force feedback lives
        if sdl2.SDL_InitSubSystem(self._SUBSYSTEMS) != 0:
            raise RuntimeError(f"SDL_Init failed: {_sdl_error()}")
        self._sdl_up = True
        try:
            self._open(device_index)
        except Exception:
            self.close()
            raise

    def _open(self, device_index: int) -> None:
        index, self.name = self._pick_device(device_index)
        self.joy = sdl2.SDL_JoystickOpen(index)
        if not self.joy:
            raise RuntimeError(f"Could not open the wheel: {_sdl_error()}")
        if sdl2.SDL_JoystickIsHaptic(self.joy) != 1:
            raise RuntimeError(
                "SDL can't use force feedback on this device. On Windows, check that "
                "Logitech G HUB is installed. On Linux, check write access to the "
                "wheel's /dev/input/event* node.")
        self.haptic = sdl2.SDL_HapticOpenFromJoystick(self.joy)
        if not self.haptic:
            raise RuntimeError(f"Could not open force feedback: {_sdl_error()}")

        features = sdl2.SDL_HapticQuery(self.haptic)
        if not features & sdl2.SDL_HAPTIC_CONSTANT:
            raise RuntimeError("This device doesn't support constant-force effects.")
        if features & sdl2.SDL_HAPTIC_GAIN:
            sdl2.SDL_HapticSetGain(self.haptic, 100)
        if features & sdl2.SDL_HAPTIC_AUTOCENTER:
            sdl2.SDL_HapticSetAutocenter(self.haptic, 0)   # don't fight the centring spring

        self._effect = sdl2.SDL_HapticEffect()
        ctypes.memset(ctypes.byref(self._effect), 0, ctypes.sizeof(self._effect))
        self._effect.type = sdl2.SDL_HAPTIC_CONSTANT
        const = self._effect.constant
        const.type = sdl2.SDL_HAPTIC_CONSTANT
        v = sdl2.SDL_version()
        sdl2.SDL_GetVersion(ctypes.byref(v))
        if (v.major, v.minor, v.patch) >= (2, 26, 0):
            # SDL picks the steering axis itself (more reliable across drivers).
            const.direction.type = getattr(sdl2, "SDL_HAPTIC_STEERING_AXIS", 3)
        else:
            const.direction.type = sdl2.SDL_HAPTIC_CARTESIAN
            const.direction.dir[0] = 1
        const.length = sdl2.SDL_HAPTIC_INFINITY
        const.level = 0

        self.effect_id = sdl2.SDL_HapticNewEffect(self.haptic, ctypes.byref(self._effect))
        if self.effect_id < 0:
            raise RuntimeError(f"Could not create the force effect: {_sdl_error()}")
        if sdl2.SDL_HapticRunEffect(self.haptic, self.effect_id, 1) != 0:
            raise RuntimeError(f"Could not start the force effect: {_sdl_error()}")

    @staticmethod
    def _pick_device(device_index: int) -> tuple[int, str]:
        devices = []
        for i in range(sdl2.SDL_NumJoysticks()):
            name = (sdl2.SDL_JoystickNameForIndex(i) or b"?").decode("utf-8", "replace")
            devices.append((i, name, sdl2.SDL_JoystickGetDeviceVendor(i),
                            sdl2.SDL_JoystickGetDeviceProduct(i)))
        if device_index >= 0:
            for i, name, _, _ in devices:
                if i == device_index:
                    return i, name
            raise RuntimeError(f"No game controller at device index {device_index}")
        for i, name, vid, pid in devices:
            if (vid == LOGITECH_VID and pid in WHEEL_PIDS) or any(n in name.upper() for n in WHEEL_NAMES):
                return i, name
        listing = "\n".join(f"  [{i}] {n}  (VID {v:04x}, PID {p:04x})" for i, n, v, p in devices)
        raise RuntimeError("No Logitech G29/G920 found. Detected devices:\n"
                           + (listing or "  (no game controllers found)")
                           + "\nSet 'Device index' in the wheel settings to choose one.")

    def position(self) -> float:
        """Wheel position, -1 (full left) .. +1 (full right)."""
        sdl2.SDL_JoystickUpdate()
        return sdl2.SDL_JoystickGetAxis(self.joy, self.axis) / FULL_SCALE

    def push(self, level: float) -> None:
        """Raw constant-force level, -32767..32767 (sign as the driver defines it)."""
        self._effect.constant.level = int(round(_clamp(level, -FULL_SCALE, FULL_SCALE)))
        sdl2.SDL_HapticUpdateEffect(self.haptic, self.effect_id, ctypes.byref(self._effect))

    def close(self) -> None:
        if self.haptic:
            if self.effect_id >= 0:
                try:
                    self.push(0)
                finally:
                    sdl2.SDL_HapticStopEffect(self.haptic, self.effect_id)
                    sdl2.SDL_HapticDestroyEffect(self.haptic, self.effect_id)
                    self.effect_id = -1
            sdl2.SDL_HapticClose(self.haptic)
            self.haptic = None
        if self.joy:
            sdl2.SDL_JoystickClose(self.joy)
            self.joy = None
        if getattr(self, "_sdl_up", False):
            sdl2.SDL_QuitSubSystem(self._SUBSYSTEMS)
            self._sdl_up = False


class WheelController(QThread):
    status = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, cfg: WheelConfig, live: LiveState, parent=None):
        super().__init__(parent)
        self._cfg = cfg                        # replaced atomically by update_config()
        self._live = live
        self._stop_evt = threading.Event()
        self._direction = 1                    # -1 if positive force turns toward negative angles

    def update_config(self, cfg: WheelConfig) -> None:
        """Gains, force cap, limits and range apply live. device_index / axis need a reconnect."""
        self._cfg = cfg

    def stop(self) -> None:
        self._stop_evt.set()

    def _open_wheel(self, cfg: WheelConfig):
        return SdlWheel(cfg.device_index, cfg.axis)

    # ------------------------------------------------------------------ thread
    def run(self) -> None:
        wheel = None
        try:
            self.status.emit("Connecting…")
            with high_res_timer():
                wheel = self._open_wheel(self._cfg)
                self.status.emit("Checking motor direction…")
                if not self._detect_direction(wheel):
                    return                            # stopped while calibrating
                self.status.emit(f"Connected ({getattr(wheel, 'name', 'wheel')})")
                self._control_loop(wheel)
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if wheel is not None:
                try:
                    wheel.close()                     # zero the force and free the motor
                except Exception:
                    traceback.print_exc()
            self._live.set_wheel_deg(None)
            self.status.emit("Disconnected")

    def _detect_direction(self, wheel) -> bool:
        """Ramp up a small force until the wheel moves 3 deg, to learn which way +force turns it."""
        cfg = self._cfg
        half = cfg.wheel_range_deg / 2.0
        max_level = cfg.max_force_pct / 100.0 * FULL_SCALE
        ramp_s = 1.5                                  # time to reach the force cap
        for applied in (+1, -1):                      # other way if already against a stop
            start = wheel.position() * half
            t0 = time.perf_counter()
            while time.perf_counter() - t0 < ramp_s + 0.5:
                if self._stop_evt.is_set():
                    wheel.push(0)
                    return False
                frac = min((time.perf_counter() - t0) / ramp_s, 1.0)
                wheel.push(applied * max_level * frac)
                moved = wheel.position() * half - start
                if abs(moved) >= 3.0:
                    wheel.push(0)
                    self._direction = applied * (1 if moved > 0 else -1)
                    time.sleep(0.3)                   # let it settle
                    return True
                time.sleep(0.01)
            wheel.push(0)
            time.sleep(0.3)
        raise RuntimeError("The wheel didn't move when force was applied. Is something holding "
                           "it? Try a higher 'Max motor force'.")

    def _control_loop(self, wheel) -> None:
        cfg = self._cfg
        pos = wheel.position() * cfg.wheel_range_deg / 2.0
        current, current_vel = pos, 0.0  # commanded angle starts where the wheel is: no snap
        last_pos, vel = pos, 0.0
        last = next_t = time.perf_counter()
        n = 0
        while not self._stop_evt.is_set():
            cfg = self._cfg
            now = time.perf_counter()
            dt = min(now - last, 0.05)
            last = now

            pos = wheel.position() * cfg.wheel_range_deg / 2.0
            if dt > 0:
                vel += 0.3 * ((pos - last_pos) / dt - vel)     # low-pass filtered deg/s
            last_pos = pos

            target, age = self._live.get_target()
            if cfg.target_timeout_s > 0 and age > cfg.target_timeout_s:
                target = 0.0                                  # EMG stalled/stopped -> centre
            target = _clamp(target, -cfg.max_target_deg, cfg.max_target_deg)

            current, current_vel = step_toward(current, current_vel, target, cfg.max_slew_deg_per_s,
                                               cfg.max_accel_deg_per_s2, dt)

            err = current - pos
            if cfg.safety_max_error_deg > 0 and abs(err) > cfg.safety_max_error_deg:
                raise SafetyStop(f"wheel is {err:+.0f}° away from where it should be "
                                 "(was it grabbed, or is the wheel range wrong?); motor released")

            max_level = cfg.max_force_pct / 100.0 * FULL_SCALE
            wheel.push(self._direction * pd_level(err, current_vel - vel, cfg.kp, cfg.kd, max_level))

            n += 1
            if n % max(1, int(cfg.loop_hz / 30)) == 0:        # ~30 Hz readback for the GUI
                self._live.set_wheel_deg(pos)

            next_t += 1.0 / cfg.loop_hz
            delay = next_t - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            else:
                next_t = time.perf_counter()                  # fell behind: don't accumulate backlog
