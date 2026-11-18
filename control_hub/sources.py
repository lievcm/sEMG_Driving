"""EMG sample sources. Both expose: open() -> {channel_key: fs_hz}, poll() -> {key: samples}|None, close()."""
from __future__ import annotations

import contextlib
import ctypes
import sys
import time

import numpy as np

from config import CHANNEL_KEYS, DelsysConfig, resolve_path


@contextlib.contextmanager
def high_res_timer(ms: int = 1):
    """1 ms Windows timer resolution so time.sleep() is accurate (no-op elsewhere)."""
    winmm = None
    if sys.platform == "win32":
        try:
            winmm = ctypes.WinDLL("winmm")
            winmm.timeBeginPeriod(ms)
        except OSError:
            winmm = None
    try:
        yield
    finally:
        if winmm is not None:
            winmm.timeEndPeriod(ms)


class DelsysSource:
    def __init__(self, cfg: DelsysConfig):
        self.cfg = cfg
        self.client = None
        self._guid_key: dict = {}
        self._errors = 0

    def open(self) -> dict:
        from delsys_client import DelsysClient
        c = self.cfg
        self.client = DelsysClient(c.key, c.license, resolve_path(c.api_dll_dir), c.pythonnet_runtime)
        self.client.connect()
        self.client.scan_and_select({int(p) for p in c.sensor_map.values()})
        self.client.configure(c.configure_timeout_s, c.emg_channel_type)
        self.client.start()
        pair_to_key = {int(p): k for k, p in c.sensor_map.items()}
        rates = {}
        for guid, info in self.client.channels.items():
            key = pair_to_key[info["pair"]]
            self._guid_key[guid] = key
            rates[key] = c.sample_rate_override_hz or info["rate"]
        return rates

    def poll(self):
        try:
            data = self.client.poll_data()
            self._errors = 0
        except Exception:                      # tolerate transient poll errors
            self._errors += 1
            if self._errors > 100:
                raise
            return None
        if not data:
            return None
        return {self._guid_key[g]: v for g, v in data.items() if g in self._guid_key}

    def close(self) -> None:
        if self.client is not None:
            self.client.close()


class SimulatedSource:
    """Gaussian noise whose amplitude is slowly modulated per channel (for testing without hardware)."""
    FS = 1259.259

    def open(self) -> dict:
        self._last = time.perf_counter()
        self._n = 0
        self._rng = np.random.default_rng(0)
        return {k: self.FS for k in CHANNEL_KEYS}

    def poll(self):
        now = time.perf_counter()
        n = int((now - self._last) * self.FS)
        if n < 1:
            return None
        self._last += n / self.FS
        t = (self._n + np.arange(n)) / self.FS
        self._n += n
        out = {}
        for i, k in enumerate(CHANNEL_KEYS):
            env = 0.5 * (1 + np.sin(2 * np.pi * (0.15 + 0.05 * i) * t + i))
            out[k] = self._rng.standard_normal(n) * (2e-4 * env + 5e-6)
        return out

    def close(self) -> None:
        pass


def make_source(cfg: DelsysConfig):
    return SimulatedSource() if cfg.source == "simulated" else DelsysSource(cfg)
