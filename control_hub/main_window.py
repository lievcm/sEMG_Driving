"""Main window: live steering angle + EMG readouts. All configuration lives in menus/dialogs."""
from __future__ import annotations

import time

import pyqtgraph as pg
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QAction, QFont
from PyQt6.QtWidgets import (QGridLayout, QGroupBox, QLabel, QMainWindow, QMessageBox, QProgressBar,
                             QVBoxLayout, QWidget)

from config import ARMS, MUSCLES, AppConfig, UserConfig
from emg_worker import EmgWorker
from network_stream import NetworkStreamer
from settings_dialogs import AppSettingsDialog, UserSettingsDialog
from steering import LiveState
from wheel_controller import WheelController


class MainWindow(QMainWindow):
    def __init__(self, app_cfg: AppConfig, user_cfg: UserConfig, app_path, user_path):
        super().__init__()
        self.setWindowTitle("EMG Steering Host")
        self.app_cfg, self.user_cfg = app_cfg, user_cfg
        self.app_path, self.user_path = app_path, user_path
        self.live = LiveState()
        self.emg: EmgWorker | None = None
        self.wheel: WheelController | None = None
        self.net: NetworkStreamer | None = None

        self._build_ui()
        self._build_menus()
        self._update_actions()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(int(1000 / app_cfg.gui.refresh_hz))

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        central = QWidget()
        lay = QVBoxLayout(central)

        self.angle_label = QLabel("+0.0°")
        font = QFont()
        font.setPointSize(44)
        font.setBold(True)
        self.angle_label.setFont(font)
        self.angle_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sub_label = QLabel()
        self.sub_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(self.angle_label)
        lay.addWidget(self.sub_label)

        self.plot = pg.PlotWidget()
        self.plot.setMenuEnabled(False)
        self.plot.setMouseEnabled(False, False)
        self.plot.showGrid(x=True, y=True, alpha=0.3)
        self.plot.setLabel("left", "Steering", units="deg")
        self.plot.setLabel("bottom", "Time", units="s")
        self.plot.addLine(y=0, pen=pg.mkPen("gray", width=1))
        self.curve = self.plot.plot(pen=pg.mkPen("c", width=2))
        self.plot.setMinimumHeight(200)
        self._apply_plot_ranges()
        lay.addWidget(self.plot, 1)

        box = QGroupBox("EMG readouts (normalized band value, raw band value)")
        grid = QGridLayout(box)
        self.bars, self.vals = {}, {}
        for c, arm in enumerate(ARMS):
            grid.addWidget(QLabel(f"<b>{'Left' if arm == 'L' else 'Right'} arm</b>"), 0, c * 3, 1, 3)
            for r, m in enumerate(MUSCLES):
                key = f"{arm}_{m}"
                bar = QProgressBar()
                bar.setRange(0, 1000)
                bar.setTextVisible(False)
                val = QLabel("0")
                val.setMinimumWidth(60)
                self.bars[key], self.vals[key] = bar, val
                grid.addWidget(QLabel(m.capitalize()), r + 1, c * 3)
                grid.addWidget(bar, r + 1, c * 3 + 1)
                grid.addWidget(val, r + 1, c * 3 + 2)
        lay.addWidget(box)
        self.setCentralWidget(central)

        self.emg_status = QLabel("EMG: disconnected")
        self.wheel_status = QLabel("Wheel: disconnected")
        self.net_status = QLabel("Network: disconnected")
        self.statusBar().addWidget(self.emg_status)
        self.statusBar().addPermanentWidget(self.net_status)
        self.statusBar().addPermanentWidget(self.wheel_status)
        self.resize(820, 700)

    def _apply_plot_ranges(self):
        half = self.app_cfg.steering.lock_to_lock_deg / 2
        self.plot.setYRange(-half, half, padding=0.05)
        self.plot.setXRange(-self.app_cfg.gui.history_s, 0, padding=0)

    def _build_menus(self):
        mb = self.menuBar()
        m = mb.addMenu("&Settings")
        m.addAction("Application settings…", self.open_app_settings)
        m.addAction("User settings (weights)…", self.open_user_settings)

        m = mb.addMenu("&EMG")
        self.act_emg_on = m.addAction("Connect EMG", self.connect_emg)
        self.act_emg_off = m.addAction("Disconnect EMG", self.disconnect_emg)

        m = mb.addMenu("&Wheel")
        self.act_wheel_on = m.addAction("Connect G29 / G920", self.connect_wheel)
        self.act_wheel_off = m.addAction("Disconnect (disable) wheel", self.disconnect_wheel)

        m = mb.addMenu("&Network")
        self.act_net_on = m.addAction("Connect to remote computer", self.connect_network)
        self.act_net_off = m.addAction("Disconnect", self.disconnect_network)
        m.addSeparator()
        m.addAction("Network settings…", self.open_app_settings)

        m = mb.addMenu("&File")
        m.addAction("Quit", self.close)

    def _update_actions(self):
        self.act_emg_on.setEnabled(self.emg is None)
        self.act_emg_off.setEnabled(self.emg is not None)
        self.act_wheel_on.setEnabled(self.wheel is None)
        self.act_wheel_off.setEnabled(self.wheel is not None)
        self.act_net_on.setEnabled(self.net is None)
        self.act_net_off.setEnabled(self.net is not None)

    # ------------------------------------------------------------------ refresh
    def _refresh(self):
        s = self.live.snapshot()
        self.angle_label.setText(f"{s.angle_deg:+.1f}°")
        wheel = "—" if s.wheel_deg is None else f"{s.wheel_deg:+.1f}°"
        self.sub_label.setText(f"normalized {s.angle_norm:+.3f}   |   left arm {s.arm_left:+.3f}   "
                               f"right arm {s.arm_right:+.3f}   |   wheel position {wheel}")
        for key, bar in self.bars.items():
            bar.setValue(int(abs(s.norm_power[key]) * 1000))
            self.vals[key].setText(f"{s.raw_value[key]:.3g}")
        t, v = self.live.history(self.app_cfg.gui.history_s)
        if t.size:
            self.curve.setData(t - time.perf_counter(), v)

    # ------------------------------------------------------------------ settings
    def open_app_settings(self):
        AppSettingsDialog(self, self.app_cfg, self.app_path, self.apply_app_config).exec()

    def open_user_settings(self):
        UserSettingsDialog(self, self.user_cfg, self.user_path, self.apply_user_config).exec()

    def apply_app_config(self, cfg: AppConfig, path=None):
        old, self.app_cfg = self.app_cfg, cfg
        if path:
            self.app_path = path
        if self.emg:
            self.emg.update_configs(self.app_cfg, self.user_cfg)
        if self.wheel:
            self.wheel.update_config(cfg.wheel)
        if self.net:
            self.net.update_config(cfg.network)
        self._timer.setInterval(int(1000 / cfg.gui.refresh_hz))
        self._apply_plot_ranges()
        notes = []
        if self.emg and old.delsys != cfg.delsys:
            notes.append("reconnect EMG for Delsys changes")
        keys = ("device_index", "axis")
        if self.wheel and any(getattr(old.wheel, k) != getattr(cfg.wheel, k) for k in keys):
            notes.append("reconnect wheel for device/axis changes")
        keys = ("protocol", "host", "port", "connect_timeout_s", "reconnect_interval_s")
        if self.net and any(getattr(old.network, k) != getattr(cfg.network, k) for k in keys):
            notes.append("reconnect network for protocol/host/port changes")
        self.statusBar().showMessage("Applied" + (f" — {'; '.join(notes)}" if notes else ""), 5000)

    def apply_user_config(self, cfg: UserConfig, path=None):
        self.user_cfg = cfg
        if path:
            self.user_path = path
        if self.emg:
            self.emg.update_configs(self.app_cfg, self.user_cfg)
        self.statusBar().showMessage("Applied", 3000)

    # ------------------------------------------------------------------ EMG / wheel
    def connect_emg(self):
        if self.emg:
            return
        w = EmgWorker(self.app_cfg, self.user_cfg, self.live, self)
        w.status.connect(lambda s: self.emg_status.setText(f"EMG: {s}"))
        w.failed.connect(lambda e: QMessageBox.warning(self, "EMG", e))
        w.finished.connect(self._emg_finished)
        self.emg = w
        w.start()
        self._update_actions()

    def disconnect_emg(self):
        if self.emg:
            self.emg.stop()

    def _emg_finished(self):
        w, self.emg = self.emg, None
        if w:
            w.deleteLater()
        self._update_actions()

    def connect_wheel(self):
        if self.wheel:
            return
        w = WheelController(self.app_cfg.wheel, self.live, self)
        w.status.connect(lambda s: self.wheel_status.setText(f"Wheel: {s}"))
        w.failed.connect(lambda e: QMessageBox.warning(self, "Wheel", e))
        w.finished.connect(self._wheel_finished)
        self.wheel = w
        w.start()
        self._update_actions()

    def disconnect_wheel(self):
        if self.wheel:
            self.wheel.stop()

    def _wheel_finished(self):
        w, self.wheel = self.wheel, None
        if w:
            w.deleteLater()
        self._update_actions()

    def connect_network(self):
        if self.net:
            return
        w = NetworkStreamer(self.app_cfg.network, self.live, self)
        w.status.connect(lambda s: self.net_status.setText(f"Network: {s}"))
        w.failed.connect(lambda e: QMessageBox.warning(self, "Network", e))
        w.finished.connect(self._net_finished)
        self.net = w
        w.start()
        self._update_actions()

    def disconnect_network(self):
        if self.net:
            self.net.stop()

    def _net_finished(self):
        w, self.net = self.net, None
        if w:
            w.deleteLater()
        self._update_actions()

    def closeEvent(self, event):
        workers = [w for w in (self.emg, self.wheel, self.net) if w]
        for w in workers:
            w.stop()
        for w in workers:
            w.wait(5000)
        event.accept()
