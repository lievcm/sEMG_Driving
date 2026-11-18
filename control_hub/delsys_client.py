"""Minimal Delsys AeroPy client (trimmed from the reference client).

Call sequence (unchanged from the working reference):
    ValidateBase(key, license) -> ScanSensors().Result -> GetScannedSensorsFound()
    -> SelectSensor(i) -> Configure(False, False) -> Start(False)
    -> loop: CheckDataQueue() / PollData() -> Stop() -> ResetPipeline()

The .NET runtime is loaded lazily so the rest of the application (GUI, simulated source)
works on machines without the Delsys DLL.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np

_AeroPy = None


def load_aeropy(dll_dir: str, runtime: str = "coreclr"):
    """Import the AeroPy class. PYTHONNET_RUNTIME must be set before ``clr`` is imported."""
    global _AeroPy
    if _AeroPy is not None:
        return _AeroPy
    path = Path(dll_dir)
    if not path.exists():
        raise FileNotFoundError(f"DelsysAPI folder not found: {path}\n"
                                "Set 'DelsysAPI folder' in the application settings.")
    os.environ["PYTHONNET_RUNTIME"] = runtime
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    import clr  # noqa: pythonnet
    clr.AddReference("DelsysAPI")
    clr.AddReference("System.Collections")
    from Aero import AeroPy  # noqa
    _AeroPy = AeroPy
    return AeroPy


class DelsysClient:
    def __init__(self, key: str, license: str, dll_dir: str, runtime: str = "coreclr"):
        self.base = load_aeropy(dll_dir, runtime)()
        self.key, self.license = key, license
        self.selected = []        # selected sensor objects
        self.channels = {}        # channel guid (str) -> {"pair": int, "rate": float}
        self._streaming = False
        self._armed = False

    # ------------------------------------------------------------------ setup
    def connect(self) -> None:
        self.base.ValidateBase(self.key, self.license)
        state = self.base.GetPipelineState()
        if state not in ("Off", "Connected"):
            raise RuntimeError(f"Unexpected pipeline state after ValidateBase: {state}")

    def scan_and_select(self, wanted_pairs: set) -> None:
        last_exc = None
        for _ in range(2):                              # reference pattern: one retry
            try:
                self.base.ScanSensors().Result
                break
            except Exception as exc:                    # noqa
                last_exc = exc
                time.sleep(1)
        sensors = list(self.base.GetScannedSensorsFound())
        if not sensors:
            raise RuntimeError("No sensors found (powered on, paired and in range?)"
                               + (f" Scan error: {last_exc}" if last_exc else ""))
        self.selected = []
        for i, sensor in enumerate(sensors):
            if int(sensor.PairNumber) in wanted_pairs:
                self.base.SelectSensor(i)
                self.selected.append(sensor)
        missing = wanted_pairs - {int(s.PairNumber) for s in self.selected}
        if missing:
            found = sorted(int(s.PairNumber) for s in sensors)
            raise RuntimeError(f"Sensor pair numbers {sorted(missing)} not found (found: {found})")

    def configure(self, timeout_s: float, emg_type: str = "EMG") -> None:
        state = self.base.GetPipelineState()
        if state != "Armed":
            if state != "Connected":
                raise RuntimeError(f"Cannot configure from pipeline state '{state}'")
            self.base.Configure(False, False)
            deadline = time.monotonic() + timeout_s
            while not (self.base.IsPipelineConfigured() and self.base.GetPipelineState() == "Armed"):
                if time.monotonic() > deadline:
                    raise TimeoutError(f"Pipeline not Armed after {timeout_s}s "
                                       f"(state: {self.base.GetPipelineState()})")
                time.sleep(0.1)
        self._armed = True
        self._parse_channels(emg_type)

    def _parse_channels(self, emg_type: str) -> None:
        """One EMG channel per selected sensor."""
        self.channels = {}
        for sensor in self.selected:
            for ch in sensor.TrignoChannels:
                if ch.IsEnabled and emg_type in str(ch.Type):
                    self.channels[str(ch.Id)] = {"pair": int(sensor.PairNumber),
                                                 "rate": float(ch.SampleRate)}
                    break
            else:
                raise RuntimeError(f"Sensor pair {sensor.PairNumber} has no enabled "
                                   f"'{emg_type}' channel")

    # ------------------------------------------------------------------ streaming
    def start(self) -> None:
        state = self.base.GetPipelineState()
        if state != "Armed":
            raise RuntimeError(f"Cannot start from pipeline state '{state}'")
        self.base.Start(False)
        self._streaming = True
        deadline = time.monotonic() + 3.0
        while self.base.GetPipelineState() != "Running":
            if time.monotonic() > deadline:
                raise RuntimeError(f"Pipeline did not reach 'Running' "
                                   f"(state: {self.base.GetPipelineState()})")
            time.sleep(0.05)

    def poll_data(self):
        """{guid: np.ndarray} of newly available samples, or None if nothing is ready."""
        if not self.base.CheckDataQueue():
            return None
        return {str(guid): np.array(list(values), dtype=np.float64)
                for guid, values in self.base.PollData().items()}

    def close(self) -> None:
        if self._streaming:
            try:
                self.base.Stop()
            finally:
                self._streaming = False
        if self._armed:
            try:
                self.base.ResetPipeline()
            finally:
                self._armed = False
