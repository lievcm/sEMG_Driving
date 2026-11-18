"""EMG acquisition thread: poll source -> band power per channel -> steering angle -> LiveState."""
from __future__ import annotations

import threading
import time
import traceback

from PyQt6.QtCore import QThread, pyqtSignal

from band_power import BandPowerProcessor
from config import CHANNEL_KEYS, AppConfig, BandPowerConfig, UserConfig
from sources import high_res_timer, make_source
from steering import LiveState, compute_steering


def _make_processors(band: BandPowerConfig, rates: dict) -> dict:
    return {k: BandPowerProcessor(rates[k], band.window_s, band.step_s, band.f_low, band.f_high,
                                  band.avg_window_s, band.metric) for k in CHANNEL_KEYS}


class EmgWorker(QThread):
    status = pyqtSignal(str)
    failed = pyqtSignal(str)

    def __init__(self, app_cfg: AppConfig, user_cfg: UserConfig, live: LiveState, parent=None):
        super().__init__(parent)
        self._cfg = (app_cfg, user_cfg)       # replaced atomically by update_configs()
        self._live = live
        self._stop_evt = threading.Event()

    def update_configs(self, app_cfg: AppConfig, user_cfg: UserConfig) -> None:
        """Weights / steering scale apply immediately; band-power changes rebuild the filters."""
        self._cfg = (app_cfg, user_cfg)

    def stop(self) -> None:
        self._stop_evt.set()

    def run(self) -> None:
        source = None
        try:
            self.status.emit("Connecting…")
            source = make_source(self._cfg[0].delsys)
            rates = source.open()
            missing = [k for k in CHANNEL_KEYS if k not in rates]
            if missing:
                raise RuntimeError(f"No EMG data channel for: {', '.join(missing)}")
            with high_res_timer():
                self._stream(source, rates)
        except Exception as exc:
            traceback.print_exc()
            self.failed.emit(f"{type(exc).__name__}: {exc}")
        finally:
            if source is not None:
                try:
                    source.close()
                except Exception:
                    traceback.print_exc()
            self._live.clear()
            self.status.emit("Disconnected")

    def _stream(self, source, rates: dict) -> None:
        app, user = self._cfg
        band = app.band_power
        procs = _make_processors(band, rates)
        latest = dict.fromkeys(CHANNEL_KEYS, 0.0)       # last smoothed band value per channel
        self.status.emit("Streaming")

        while not self._stop_evt.is_set():
            app, user = self._cfg
            if app.band_power != band:
                try:
                    procs = _make_processors(app.band_power, rates)
                    band = app.band_power
                except ValueError as exc:
                    band = app.band_power               # don't retry every loop
                    self.status.emit(f"Streaming (band-power change rejected: {exc})")

            data = source.poll()
            if not data:
                time.sleep(app.delsys.poll_interval_ms / 1000.0)
                continue

            updated = False
            for key, chunk in data.items():
                out = procs[key].push(chunk)
                if out.size:
                    latest[key] = float(out[-1])
                    updated = True
            if updated:
                res = compute_steering(latest, user, app.steering.lock_to_lock_deg)
                self._live.publish(res, latest)
