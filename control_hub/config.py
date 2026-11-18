"""Application + user configuration (dataclasses with JSON load/save/validate).

Application config : everything that describes the setup (Delsys connection, band-power
                     parameters, steering scale, G29 settings, GUI).
User config        : the 12 per-user values - for each of the 6 channels a
                     normalization factor ``n`` and a weight ``w``.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent
CONFIG_DIR = APP_ROOT / "config"
DEFAULT_APP_CONFIG_PATH = CONFIG_DIR / "app_config.json"
DEFAULT_USER_CONFIG_PATH = CONFIG_DIR / "user_config.json"

ARMS = ("L", "R")
MUSCLES = ("biceps", "triceps", "brachioradialis")
CHANNEL_KEYS = tuple(f"{a}_{m}" for a in ARMS for m in MUSCLES)  # L_biceps, ... R_brachioradialis
METRICS = ("power", "mean_amp", "peak_freq")
SOURCES = ("delsys", "simulated")
PROTOCOLS = ("tcp", "udp")


def resolve_path(p: str) -> str:
    """Relative paths are looked up next to the application first."""
    path = Path(p)
    if path.is_absolute():
        return str(path)
    candidate = APP_ROOT / path
    return str(candidate) if candidate.exists() else p


# --------------------------------------------------------------------------- application
@dataclass
class DelsysConfig:
    source: str = "delsys"                      # "delsys" or "simulated" (no hardware needed)
    key: str = ""                               # Delsys API key
    license: str = ""                           # Delsys API license
    api_dll_dir: str = "resources/DelsysAPI/resources"   # folder containing DelsysAPI.dll
    pythonnet_runtime: str = "coreclr"          # "coreclr" or "netfx"
    # Sensor pair number (1-16) used for each muscle channel
    sensor_map: dict = field(default_factory=lambda: {
        "L_biceps": 1, "L_triceps": 2, "L_brachioradialis": 3,
        "R_biceps": 4, "R_triceps": 5, "R_brachioradialis": 6,
    })
    emg_channel_type: str = "EMG"               # substring matched against channel type
    sample_rate_override_hz: float = 0.0        # 0 = use the rate reported by the sensor
    configure_timeout_s: float = 10.0
    poll_interval_ms: float = 1.0               # sleep between empty polls


@dataclass
class BandPowerConfig:
    window_s: float = 0.2        # FFT window length
    step_s: float = 0.01         # time between windows (output rate = 1/step_s)
    f_low: float = 20.0
    f_high: float = 450.0
    avg_window_s: float = 0.1    # trailing moving-average length (seconds)
    metric: str = "power"        # power | mean_amp | peak_freq


@dataclass
class SteeringConfig:
    lock_to_lock_deg: float = 900.0   # +-1 normalized maps to +-(lock_to_lock_deg / 2)


@dataclass
class WheelConfig:
    device_index: int = -1                 # SDL game-controller index; -1 = auto-detect G29/G920
    axis: int = 0                          # steering axis number
    wheel_range_deg: float = 900.0         # must match the range set in G HUB / the driver
    max_target_deg: float = 90.0           # clamp on the commanded angle (+-)
    max_slew_deg_per_s: float = 180.0
    max_accel_deg_per_s2: float = 360.0    # eases starts/stops so the wheel doesn't overshoot (0 = off)
    max_force_pct: float = 30.0            # motor force cap, % of the wheel's maximum
    kp: float = 400.0                      # force units (of 32767) per degree of position error
    kd: float = 200.0                      # force units per deg/s of velocity error
    safety_max_error_deg: float = 90.0     # wheel this far off target -> release motor (0 = off)
    loop_hz: float = 100.0
    target_timeout_s: float = 0.5          # no new EMG target for this long -> centre wheel (0 = off)


@dataclass
class NetworkConfig:
    """Streaming of the steering angle to another computer (see network_stream.py)."""
    protocol: str = "tcp"                  # "tcp" (reliable) or "udp" (lowest latency, may drop packets)
    host: str = "127.0.0.1"                # IP address / host name of the receiving computer
    port: int = 5005
    send_hz: float = 100.0                 # messages per second
    connect_timeout_s: float = 3.0         # TCP connect timeout
    reconnect_interval_s: float = 2.0      # TCP: wait this long between reconnect attempts


@dataclass
class GuiConfig:
    history_s: float = 10.0
    refresh_hz: float = 30.0


@dataclass
class AppConfig:
    delsys: DelsysConfig = field(default_factory=DelsysConfig)
    band_power: BandPowerConfig = field(default_factory=BandPowerConfig)
    steering: SteeringConfig = field(default_factory=SteeringConfig)
    wheel: WheelConfig = field(default_factory=WheelConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    gui: GuiConfig = field(default_factory=GuiConfig)

    def validate(self) -> None:
        d, b, s, w, g = self.delsys, self.band_power, self.steering, self.wheel, self.gui
        n = self.network
        if d.source not in SOURCES:
            raise ValueError(f"EMG source must be one of {SOURCES}")
        if set(d.sensor_map) != set(CHANNEL_KEYS):
            raise ValueError(f"sensor_map must have exactly the keys {CHANNEL_KEYS}")
        pairs = list(d.sensor_map.values())
        if d.source == "delsys" and (len(set(pairs)) != 6 or not all(1 <= int(p) <= 16 for p in pairs)):
            raise ValueError("Sensor pair numbers must be unique and between 1 and 16")
        if b.metric not in METRICS:
            raise ValueError(f"metric must be one of {METRICS}")
        if b.window_s <= 0 or b.step_s <= 0 or b.avg_window_s < 0:
            raise ValueError("window_s and step_s must be > 0, avg_window_s >= 0")
        if not (0 <= b.f_low < b.f_high):
            raise ValueError("Need 0 <= f_low < f_high")
        if s.lock_to_lock_deg <= 0:
            raise ValueError("lock_to_lock_deg must be > 0")
        if w.wheel_range_deg <= 0 or w.loop_hz <= 0 or w.max_slew_deg_per_s <= 0:
            raise ValueError("wheel_range_deg, loop_hz and max_slew_deg_per_s must be > 0")
        if not (0 <= w.max_target_deg <= w.wheel_range_deg / 2):
            raise ValueError("max_target_deg must be between 0 and wheel_range_deg / 2")
        if w.device_index < -1 or w.axis < 0:
            raise ValueError("device_index must be >= -1 and axis >= 0")
        if not 0 < w.max_force_pct <= 100:
            raise ValueError("max_force_pct must be between 0 (exclusive) and 100")
        if w.kp < 0 or w.kd < 0 or w.safety_max_error_deg < 0 or w.max_accel_deg_per_s2 < 0:
            raise ValueError("kp, kd, max_accel_deg_per_s2 and safety_max_error_deg must be >= 0")
        if n.protocol not in PROTOCOLS:
            raise ValueError(f"Network protocol must be one of {PROTOCOLS}")
        if not n.host.strip():
            raise ValueError("Network host (IP address or name of the remote computer) is empty")
        if not 1 <= n.port <= 65535:
            raise ValueError("Network port must be between 1 and 65535")
        if not 0 < n.send_hz <= 1000:
            raise ValueError("Network send rate must be > 0 and <= 1000 Hz")
        if n.connect_timeout_s <= 0 or n.reconnect_interval_s <= 0:
            raise ValueError("Network connect timeout and reconnect interval must be > 0")
        if g.history_s <= 0 or g.refresh_hz <= 0:
            raise ValueError("history_s and refresh_hz must be > 0")


# --------------------------------------------------------------------------- user
@dataclass
class ChannelWeights:
    n: float = 1.0    # normalization factor: normalized power = clip(n * band_value, -1, 1)
    w: float = 0.0    # weight (signed: negative = steer left)


def _arm(bn, bw, tn, tw, rn, rw) -> "ArmWeights":
    return ArmWeights(ChannelWeights(bn, bw), ChannelWeights(tn, tw), ChannelWeights(rn, rw))


@dataclass
class ArmWeights:
    biceps: ChannelWeights = field(default_factory=ChannelWeights)
    triceps: ChannelWeights = field(default_factory=ChannelWeights)
    brachioradialis: ChannelWeights = field(default_factory=ChannelWeights)


@dataclass
class UserConfig:
    # Example defaults: sum(|w|) per arm = 1. The n values are placeholders and must be
    # calibrated per user/sensor (n ~ 1 / band value at a strong contraction).
    left: ArmWeights = field(default_factory=lambda: _arm(1e8, -0.5, 1e8, 0.4, 1e8, -0.1))
    right: ArmWeights = field(default_factory=lambda: _arm(1e8, 0.5, 1e8, -0.4, 1e8, 0.1))

    def channel(self, key: str) -> ChannelWeights:
        arm, muscle = key.split("_", 1)
        return getattr(self.left if arm == "L" else self.right, muscle)

    def validate(self) -> None:
        for key in CHANNEL_KEYS:
            ch = self.channel(key)
            if not (math.isfinite(ch.n) and math.isfinite(ch.w)):
                raise ValueError(f"{key}: n and w must be finite numbers")


# --------------------------------------------------------------------------- JSON io
def _from_dict(cls, data):
    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object for {cls.__name__}")
    obj = cls()
    for f in fields(cls):
        if f.name not in data:
            continue                        # missing keys keep their defaults
        cur, val = getattr(obj, f.name), data[f.name]
        if is_dataclass(cur):
            val = _from_dict(type(cur), val)
        elif isinstance(cur, dict):
            val = {**cur, **val}
        elif isinstance(cur, bool):
            val = bool(val)
        elif isinstance(cur, float):
            val = float(val)
        elif isinstance(cur, int):
            val = int(val)
        elif isinstance(cur, str):
            val = str(val)
        setattr(obj, f.name, val)
    return obj


def _save(cfg, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _load(cls, path):
    cfg = _from_dict(cls, json.loads(Path(path).read_text(encoding="utf-8")))
    cfg.validate()
    return cfg


def load_app_config(path) -> AppConfig:
    return _load(AppConfig, path)


def save_app_config(cfg: AppConfig, path) -> None:
    cfg.validate()
    _save(cfg, path)


def load_user_config(path) -> UserConfig:
    return _load(UserConfig, path)


def save_user_config(cfg: UserConfig, path) -> None:
    cfg.validate()
    _save(cfg, path)
