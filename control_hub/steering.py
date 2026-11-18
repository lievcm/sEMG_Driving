"""Band values -> normalized steering angle -> degrees, plus the state shared between threads."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from config import ARMS, CHANNEL_KEYS, MUSCLES, UserConfig


def _clip1(x: float) -> float:
    return -1.0 if x < -1.0 else 1.0 if x > 1.0 else x


@dataclass
class SteeringResult:
    arm_left: float
    arm_right: float
    angle_norm: float
    angle_deg: float
    norm_power: dict = field(default_factory=dict)   # key -> clip(n * p, -1, 1)


def compute_steering(values: dict, user: UserConfig, lock_to_lock_deg: float) -> SteeringResult:
    """angle_arm  = clip( sum(w * clip(n * p)), -1, 1 );  angle = (angle_L + angle_R) / 2."""
    norm_power, arm = {}, {}
    for a in ARMS:
        total = 0.0
        for m in MUSCLES:
            key = f"{a}_{m}"
            ch = user.channel(key)
            p = _clip1(ch.n * values.get(key, 0.0))
            norm_power[key] = p
            total += ch.w * p
        arm[a] = _clip1(total)
    angle = (arm["L"] + arm["R"]) / 2.0
    return SteeringResult(arm["L"], arm["R"], angle, angle * lock_to_lock_deg / 2.0, norm_power)


@dataclass
class Snapshot:
    angle_deg: float = 0.0
    angle_norm: float = 0.0
    arm_left: float = 0.0
    arm_right: float = 0.0
    wheel_deg: float | None = None
    norm_power: dict = field(default_factory=lambda: dict.fromkeys(CHANNEL_KEYS, 0.0))
    raw_value: dict = field(default_factory=lambda: dict.fromkeys(CHANNEL_KEYS, 0.0))


class LiveState:
    """Written by the EMG thread, read by the wheel thread (target) and the GUI (snapshot)."""

    _N = 65536

    def __init__(self):
        self._lock = threading.Lock()
        self._snap = Snapshot()
        self._target_deg = 0.0
        self._target_time = time.perf_counter()
        self._ht = np.zeros(self._N)
        self._hv = np.zeros(self._N)
        self._hn = 0

    def publish(self, res: SteeringResult, raw: dict) -> None:
        now = time.perf_counter()
        with self._lock:
            s = self._snap
            s.angle_deg, s.angle_norm = res.angle_deg, res.angle_norm
            s.arm_left, s.arm_right = res.arm_left, res.arm_right
            s.norm_power = dict(res.norm_power)
            s.raw_value = dict(raw)
            self._target_deg, self._target_time = res.angle_deg, now
            i = self._hn % self._N
            self._ht[i], self._hv[i] = now, res.angle_deg
            self._hn += 1

    def get_target(self) -> tuple[float, float]:
        """(target degrees, age in seconds)."""
        with self._lock:
            return self._target_deg, time.perf_counter() - self._target_time

    def set_wheel_deg(self, deg: float | None) -> None:
        with self._lock:
            self._snap.wheel_deg = deg

    def clear(self) -> None:
        """EMG stopped: zero everything so the wheel returns to centre."""
        with self._lock:
            wheel = self._snap.wheel_deg
            self._snap = Snapshot(wheel_deg=wheel)
            self._target_deg, self._target_time = 0.0, time.perf_counter()

    def snapshot(self) -> Snapshot:
        with self._lock:
            s = self._snap
            return Snapshot(s.angle_deg, s.angle_norm, s.arm_left, s.arm_right, s.wheel_deg,
                            dict(s.norm_power), dict(s.raw_value))

    def history(self, window_s: float) -> tuple[np.ndarray, np.ndarray]:
        """(timestamps [perf_counter s], angle_deg) for the last ``window_s`` seconds."""
        with self._lock:
            if self._hn == 0:
                return np.empty(0), np.empty(0)
            if self._hn <= self._N:
                t, v = self._ht[:self._hn].copy(), self._hv[:self._hn].copy()
            else:
                e = self._hn % self._N
                t = np.concatenate((self._ht[e:], self._ht[:e]))
                v = np.concatenate((self._hv[e:], self._hv[:e]))
        keep = t >= time.perf_counter() - window_s
        return t[keep], v[keep]
