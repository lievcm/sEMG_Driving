"""Run with:  python -m pytest tests   (or: python tests/test_core.py)"""
import math
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from band_power import BandPowerProcessor
from config import (AppConfig, UserConfig, load_app_config, load_user_config, save_app_config,
                    save_user_config)
from steering import LiveState, compute_steering


def reference_band_values(x, fs, window_s, step_s, f_low, f_high, metric="power"):
    """The reference rolling_band_power without pandas and without the smoothing step."""
    win_n, step_n = int(round(window_s * fs)), max(1, int(round(step_s * fs)))
    hann = np.hanning(win_n)
    freqs = np.fft.rfftfreq(win_n, d=1.0 / fs)
    band = (freqs >= f_low) & (freqs <= f_high)
    out = []
    for s in range(0, len(x) - win_n + 1, step_n):
        seg = (x[s:s + win_n] - x[s:s + win_n].mean()) * hann
        spec = np.abs(np.fft.rfft(seg)) / win_n
        out.append(np.sum(spec[band] ** 2) if metric == "power" else
                   spec[band].mean() if metric == "mean_amp" else freqs[band][np.argmax(spec[band])])
    return np.array(out)


def _signal(n=6000, fs=1259.259, seed=1):
    return np.random.default_rng(seed).standard_normal(n) * 1e-4, fs


def test_matches_reference_for_every_metric_and_chunking():
    x, fs = _signal()
    for metric in ("power", "mean_amp", "peak_freq"):
        ref = reference_band_values(x, fs, 0.2, 0.01, 20, 450, metric)
        for chunking in ("one", "random", "tiny"):
            p = BandPowerProcessor(fs, 0.2, 0.01, 20, 450, avg_window_s=0.0, metric=metric)
            rng = np.random.default_rng(3)
            outs, i = [], 0
            while i < len(x):
                step = {"one": len(x), "tiny": 1, "random": int(rng.integers(1, 90))}[chunking]
                outs.append(p.push(x[i:i + step]))
                i += step
            got = np.concatenate(outs)
            assert got.shape == ref.shape, (metric, chunking, got.shape, ref.shape)
            assert np.allclose(got, ref, rtol=1e-9, atol=0), (metric, chunking)


def test_trailing_average_is_causal_and_correct():
    x, fs = _signal()
    ref = reference_band_values(x, fs, 0.2, 0.01, 20, 450)
    p = BandPowerProcessor(fs, 0.2, 0.01, 20, 450, avg_window_s=0.05)   # 5 windows
    got = np.concatenate([p.push(x[i:i + 37]) for i in range(0, len(x), 37)])
    expect = np.array([ref[max(0, i - 4):i + 1].mean() for i in range(len(ref))])
    assert np.allclose(got, expect, rtol=1e-9)


def test_steering_math_and_clamps():
    u = UserConfig()
    zero = compute_steering({}, u, 900)
    assert zero.angle_deg == 0
    # huge powers: every normalized power clamps to 1; arm angles clamp to +-1
    big = compute_steering({k: 1e9 for k in ("L_biceps", "L_triceps", "L_brachioradialis",
                                              "R_biceps", "R_triceps", "R_brachioradialis")}, u, 900)
    assert all(v == 1.0 for v in big.norm_power.values())
    assert abs(big.angle_norm) <= 1.0
    # right biceps only -> right (positive) steer
    r = compute_steering({"R_biceps": 1.0 / u.right.biceps.n}, u, 900)
    assert np.isclose(r.arm_right, u.right.biceps.w) and r.angle_deg > 0
    assert np.isclose(r.angle_deg, r.arm_right / 2 * 450)
    # weights summing above 1 still clamp the arm at +-1
    u.left.biceps.w, u.left.triceps.w = -1.0, -1.0
    c = compute_steering({"L_biceps": 1e9, "L_triceps": 1e9}, u, 900)
    assert c.arm_left == -1.0


def test_config_roundtrip(tmp_path=Path("/tmp/emg_cfg_test")):
    a, u = AppConfig(), UserConfig()
    a.band_power.metric, a.wheel.max_target_deg, u.left.biceps.n = "mean_amp", 123.0, 3.5e6
    save_app_config(a, tmp_path / "a.json")
    save_user_config(u, tmp_path / "u.json")
    a2, u2 = load_app_config(tmp_path / "a.json"), load_user_config(tmp_path / "u.json")
    assert a2 == a and u2 == u
    (tmp_path / "partial.json").write_text('{"wheel": {"loop_hz": 50}}')       # missing keys -> defaults
    assert load_app_config(tmp_path / "partial.json").wheel.loop_hz == 50


class _SimWheel:
    """Stands in for SdlWheel: a wheel with inertia and friction, integrated in real time."""

    J, FC, B, NM_MAX, RANGE = 0.03, 0.1, 0.01, 2.1, 900.0   # kg m^2, Nm, Nm/(rad/s), Nm, deg

    def __init__(self, start_deg=0.0, polarity=1):
        self.name, self.polarity = "SimWheel", polarity
        self.th, self.w, self.level = math.radians(start_deg), 0.0, 0.0
        self.frozen, self.closed, self.levels = False, False, []
        self._t = time.perf_counter()

    def _advance(self):
        now = time.perf_counter()
        steps, self._t = int((now - self._t) / 0.0005), self._t + int((now - self._t) / 0.0005) * 0.0005
        tau = self.polarity * self.level / 32767 * self.NM_MAX
        for _ in range(steps):
            if self.frozen:
                self.w = 0.0
                return
            if abs(self.w) < 1e-3 and abs(tau) <= self.FC * 1.3:     # stiction
                self.w = 0.0
                continue
            f = self.FC * math.copysign(1, self.w if self.w else tau)
            self.w += (tau - self.B * self.w - f) / self.J * 0.0005
            self.th += self.w * 0.0005

    @property
    def deg(self):
        return math.degrees(self.th)

    def position(self):
        self._advance()
        return max(-1.0, min(1.0, self.deg / (self.RANGE / 2)))

    def push(self, level):
        self._advance()
        self.level = level
        self.levels.append(level)

    def close(self):
        self.push(0)
        self.closed = True


def _run_wheel(sim, publish_deg, seconds, cfg=None, silent_after=None, during=None):
    """Run WheelController against `sim`, publishing an EMG target every 10 ms."""
    from PyQt6.QtCore import QCoreApplication, Qt
    from steering import SteeringResult
    from wheel_controller import WheelController
    QCoreApplication.instance() or QCoreApplication([])
    live, errors, trace = LiveState(), [], []
    w = WheelController(cfg or AppConfig().wheel, live, None)
    w._open_wheel = lambda c: sim
    w.failed.connect(errors.append, Qt.ConnectionType.DirectConnection)
    w.start()
    t0 = time.perf_counter()
    while (t := time.perf_counter() - t0) < seconds and not w.isFinished():
        if silent_after is None or t < silent_after:
            live.publish(SteeringResult(0, 0, 0, publish_deg, {}), {})
        if during:
            during(t)
        trace.append((t, sim.deg))
        time.sleep(0.01)
    w.stop()
    w.wait(3000)
    return w, errors, trace, live


def test_pd_level_clamps():
    from wheel_controller import pd_level
    assert pd_level(10, 0, 400, 40, 1e9) == 4000
    assert pd_level(-1000, 0, 400, 40, 5000) == -5000
    assert pd_level(0, -10, 400, 40, 1e9) == -400


def test_step_toward_limits_speed_and_accel_without_overshoot():
    from wheel_controller import step_toward
    x, v, xs, vs = 0.0, 0.0, [], []
    for _ in range(300):                                     # 3 s at 100 Hz (accel 720)
        x, v = step_toward(x, v, 90.0, 180.0, 720.0, 0.01)
        xs.append(x); vs.append(v)
    assert x == 90.0 and v == 0.0
    assert max(xs) <= 90.0 and max(vs) <= 180.0 + 1e-9
    dv = np.abs(np.diff(vs))
    arrive = vs.index(0.0) - 1                               # step where it lands on the target
    assert np.delete(dv, arrive).max() <= 720.0 * 0.01 + 1e-9
    assert dv[arrive] <= 2 * 720.0 * 0.01 + 1e-9, "arrives at low speed"
    x, v = step_toward(0.0, 0.0, 90.0, 180.0, 0.0, 0.01)    # accel 0 -> plain slew limit
    assert np.isclose(x, 1.8) and np.isclose(v, 180.0)


def test_wheel_tracks_clamped_target_with_slew_and_cleanup():
    sim = _SimWheel()
    cfg = AppConfig().wheel                      # max 90 deg, slew 180 deg/s, force cap 30 %
    w, errors, trace, live = _run_wheel(sim, 400.0, 3.5, cfg)   # EMG asks 400 -> clamp 90
    assert not errors, errors
    final = trace[-1][1]
    assert 84 < final < 93, f"wheel should settle near 90 deg, got {final:.1f}"
    assert max(d for _, d in trace) < 94, "no real overshoot past the clamp"
    # slew-limited: 90 deg takes >= 0.5 s after calibration, so no sudden jump
    jumps = np.abs(np.diff([d for _, d in trace]))
    assert jumps.max() < 6, f"wheel moved {jumps.max():.1f} deg in one 10 ms step"
    assert max(abs(l) for l in sim.levels) <= 0.30 * 32767 + 1, "force cap respected"
    assert sim.closed and sim.level == 0, "motor released on stop"
    assert live.snapshot().wheel_deg is None


def test_direction_detection_handles_reversed_motor():
    sim = _SimWheel(polarity=-1)                 # driver defines force the other way round
    _, errors, trace, _ = _run_wheel(sim, -60.0, 3.0)
    assert not errors, errors
    assert -64 < trace[-1][1] < -55, trace[-1][1]


def test_wheel_centres_when_target_goes_stale():
    sim = _SimWheel(start_deg=20.0)
    _, errors, trace, _ = _run_wheel(sim, 60.0, 4.0, silent_after=2.0)
    assert not errors, errors
    assert max(d for _, d in trace) > 50, "should first follow the EMG target"
    assert abs(trace[-1][1]) < 4, f"should return to centre after timeout, got {trace[-1][1]:.1f}"


def test_safety_stop_releases_motor_when_wheel_is_held():
    sim = _SimWheel()
    cfg = AppConfig().wheel
    cfg.safety_max_error_deg = 30.0

    def grab(t):
        if t > 1.0:
            sim.frozen = True                    # someone holds the wheel still

    w, errors, _, _ = _run_wheel(sim, 90.0, 4.0, cfg, during=grab)
    assert errors and "SafetyStop" in errors[0], errors
    assert w.isFinished() and sim.closed and sim.level == 0


# --------------------------------------------------------------------------- network streaming
def _net_streamer(protocol, port, send_hz=200.0, **kw):
    import json  # noqa: F401
    from PyQt6.QtCore import QCoreApplication, Qt
    from config import NetworkConfig
    from network_stream import NetworkStreamer
    from steering import SteeringResult
    QCoreApplication.instance() or QCoreApplication([])
    live, errors, statuses = LiveState(), [], []
    live.publish(SteeringResult(0.1, -0.3, -0.1, -45.0, {}), {})
    cfg = NetworkConfig(protocol=protocol, host="127.0.0.1", port=port, send_hz=send_hz,
                        connect_timeout_s=1.0, reconnect_interval_s=0.2, **kw)
    s = NetworkStreamer(cfg, live, None)
    s.failed.connect(errors.append, Qt.ConnectionType.DirectConnection)
    s.status.connect(statuses.append, Qt.ConnectionType.DirectConnection)
    return s, live, errors, statuses


def _read_lines(conn, n, timeout=3.0):
    import json
    conn.settimeout(timeout)
    buf, out = b"", []
    while len(out) < n:
        buf += conn.recv(4096)
        *lines, buf = buf.split(b"\n")
        out += [json.loads(l) for l in lines if l]
    return out


def test_network_config_validation_and_defaults(tmp_path=Path("/tmp/emg_cfg_test")):
    import json
    a = AppConfig()
    assert a.network.protocol == "tcp" and a.network.port == 5005
    for field_, bad in (("protocol", "sctp"), ("host", "  "), ("port", 0), ("port", 70000),
                        ("send_hz", 0.0), ("send_hz", 5000.0), ("connect_timeout_s", 0.0)):
        b = AppConfig()
        setattr(b.network, field_, bad)
        try:
            b.validate()
        except ValueError:
            continue
        raise AssertionError(f"network.{field_}={bad!r} should be rejected")
    a.network.host, a.network.protocol = "192.168.1.50", "udp"
    save_app_config(a, tmp_path / "net.json")
    assert load_app_config(tmp_path / "net.json") == a
    old = json.loads((tmp_path / "net.json").read_text())
    del old["network"]                                    # configs from before this feature still load
    (tmp_path / "old.json").write_text(json.dumps(old))
    assert load_app_config(tmp_path / "old.json").network.host == "127.0.0.1"


def test_encode_message_fields():
    import json
    from network_stream import encode_message
    from steering import Snapshot
    raw = encode_message(7, Snapshot(angle_deg=-12.34567, angle_norm=-0.0274, arm_left=0.1, arm_right=-0.2), 0.0123)
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    m = json.loads(raw)
    assert m["v"] == 1 and m["seq"] == 7 and m["angle_deg"] == -12.3457 and m["age_s"] == 0.0123
    assert m["arm_left"] == 0.1 and m["arm_right"] == -0.2 and abs(m["t"] - time.time()) < 5


def test_tcp_stream_to_listener_and_clean_stop():
    import socket
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    s, live, errors, statuses = _net_streamer("tcp", srv.getsockname()[1])
    s.start()
    conn, _ = srv.accept()
    msgs = _read_lines(conn, 20)
    assert [m["seq"] for m in msgs] == list(range(msgs[0]["seq"], msgs[0]["seq"] + 20)), "ordered, no gaps"
    assert msgs[0]["angle_deg"] == -45.0 and msgs[0]["arm_left"] == 0.1
    live.publish(__import__("steering").SteeringResult(0, 0, 0.5, 225.0, {}), {})   # new value is picked up
    assert any(m["angle_deg"] == 225.0 for m in _read_lines(conn, 10))
    s.stop()
    assert s.wait(3000) and not errors
    assert statuses[0].startswith("Connecting") and any(x.startswith("Streaming") for x in statuses)
    assert statuses[-1] == "Disconnected"
    conn.settimeout(2.0)
    while conn.recv(4096):                              # drain; streamer must then close (EOF)
        pass
    conn.close(); srv.close()


def test_tcp_reconnects_after_receiver_restarts():
    import socket
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    s, _, errors, statuses = _net_streamer("tcp", srv.getsockname()[1])
    s.start()
    c1, _ = srv.accept()
    _read_lines(c1, 5)
    c1.close()                                           # receiver drops the connection
    srv.settimeout(5.0)
    c2, _ = srv.accept()                                 # streamer must come back by itself
    assert len(_read_lines(c2, 5)) == 5
    s.stop()
    assert s.wait(3000) and not errors
    assert any("retrying" in x for x in statuses)
    c2.close(); srv.close()


def test_tcp_first_connection_failure_is_reported():
    import socket
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()                                        # nothing listens on this port now
    s, _, errors, statuses = _net_streamer("tcp", port)
    s.start()
    assert s.wait(5000), "thread should stop after a failed first connect"
    assert errors and "Could not reach" in errors[0] and "steering_client.py" in errors[0]
    assert statuses[-1] == "Disconnected"


def test_udp_stream_one_message_per_datagram():
    import json
    import socket
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(3.0)
    s, _, errors, statuses = _net_streamer("udp", rx.getsockname()[1])
    s.start()
    got = [json.loads(rx.recvfrom(4096)[0]) for _ in range(10)]
    s.stop()
    assert s.wait(3000) and not errors
    assert got[0]["angle_deg"] == -45.0 and got[-1]["seq"] > got[0]["seq"]
    rx.close()


def test_udp_does_not_need_a_listener():
    import socket
    probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    probe.bind(("127.0.0.1", 0))
    port = probe.getsockname()[1]
    probe.close()                                        # nobody listening: UDP just keeps sending
    s, _, errors, _ = _net_streamer("udp", port)
    s.start()
    time.sleep(0.5)
    assert s.isRunning() and not errors
    s.stop()
    assert s.wait(3000)


def test_client_end_to_end_prints_angle():
    """Real streamer -> real steering_client.py process (both protocols)."""
    import socket
    import subprocess
    client = Path(__file__).resolve().parent.parent / "client" / "steering_client.py"
    for protocol in ("tcp", "udp"):
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
        probe.close()
        proc = subprocess.Popen([sys.executable, "-u", str(client), "--protocol", protocol, "--port", str(port),
                                 "--bind", "127.0.0.1", "--print-hz", "0"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        try:
            time.sleep(0.8)                              # let it start listening
            s, _, errors, _ = _net_streamer(protocol, port, send_hz=50.0)
            s.start()
            lines = [proc.stdout.readline() for _ in range(4)]       # "Listening", [Connected], data...
            time.sleep(0.2)
            s.stop()
            s.wait(3000)
            assert not errors, errors
        finally:
            proc.terminate()
            rest = proc.communicate(timeout=5)[0]
        text = "".join(lines) + rest
        assert "angle   -45.00°" in text and "#" in text, (protocol, text)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok ", name)
