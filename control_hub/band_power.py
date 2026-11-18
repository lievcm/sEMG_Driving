"""Streaming (causal) version of ``rolling_band_power``.

Differences from the offline reference:
  * samples arrive in chunks; a window is evaluated every ``step_s`` as soon as its last
    sample has arrived (windows line up exactly with the offline version);
  * the smoothing moving average is *trailing* (uses only past windows) instead of
    ``center=True``, and its length is given in seconds;
  * all windows that completed within one chunk are processed in one batched ``rfft``.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from config import METRICS

_EMPTY = np.empty(0)


class BandPowerProcessor:
    def __init__(self, fs: float, window_s: float, step_s: float, f_low: float, f_high: float,
                 avg_window_s: float, metric: str = "power"):
        if metric not in METRICS:
            raise ValueError("unknown metric")
        self.fs = float(fs)
        self.metric = metric
        self.win_n = int(round(window_s * self.fs))
        self.step_n = max(1, int(round(step_s * self.fs)))
        if self.win_n < 8:
            raise ValueError(f"window_s={window_s}s is only {self.win_n} samples at {self.fs:.1f} Hz")
        if f_high >= self.fs / 2:
            raise ValueError(f"f_high={f_high} Hz must be below Nyquist ({self.fs / 2:.1f} Hz)")

        self._hann = np.hanning(self.win_n)
        freqs = np.fft.rfftfreq(self.win_n, d=1.0 / self.fs)
        self._band = (freqs >= f_low) & (freqs <= f_high)
        if not self._band.any():
            raise ValueError("No FFT bins fall inside [f_low, f_high]; lengthen window_s")
        self._band_freqs = freqs[self._band]
        self._offsets = np.arange(self.win_n)

        n_avg = max(1, int(round(avg_window_s / (self.step_n / self.fs))))
        self._ring: deque = deque(maxlen=n_avg)

        self._data = _EMPTY      # unprocessed samples (+ overlap needed by the next window)
        self._base = 0           # absolute index of self._data[0]
        self._next_end = self.win_n   # absolute (exclusive) end index of the next window

    def push(self, chunk) -> np.ndarray:
        """Feed new samples; returns the smoothed band values of every window completed."""
        chunk = np.asarray(chunk, dtype=np.float64)
        if chunk.size == 0:
            return _EMPTY
        data = np.concatenate((self._data, chunk)) if self._data.size else chunk
        total_end = self._base + data.size
        if self._next_end > total_end:
            self._data = data
            return _EMPTY

        ends = np.arange(self._next_end, total_end + 1, self.step_n)
        idx = (ends - self.win_n - self._base)[:, None] + self._offsets
        segs = data[idx]
        segs = (segs - segs.mean(axis=1, keepdims=True)) * self._hann   # remove DC, window
        spec = np.abs(np.fft.rfft(segs, axis=1)) / self.win_n
        sb = spec[:, self._band]

        if self.metric == "power":
            raw = np.sum(sb ** 2, axis=1)
        elif self.metric == "mean_amp":
            raw = sb.mean(axis=1)
        else:  # peak_freq
            raw = self._band_freqs[np.argmax(sb, axis=1)]

        # keep only what the next window still needs
        self._next_end = int(ends[-1]) + self.step_n
        drop = min(self._next_end - self.win_n - self._base, data.size)
        self._base += drop
        self._data = data[drop:]

        out = np.empty(raw.size)
        ring = self._ring
        for i, v in enumerate(raw):          # trailing moving average
            ring.append(v)
            out[i] = sum(ring) / len(ring)
        return out
