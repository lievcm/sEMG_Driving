"""Application / user settings dialogs. Each configuration section is its own group box."""
from __future__ import annotations

import copy
from pathlib import Path

from PyQt6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDoubleSpinBox, QFileDialog, QFormLayout,
                             QGridLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                             QPlainTextEdit, QPushButton, QScrollArea, QSpinBox, QVBoxLayout, QWidget)

from config import (ARMS, CHANNEL_KEYS, METRICS, MUSCLES, PROTOCOLS, SOURCES, AppConfig, UserConfig,
                    load_app_config, load_user_config, save_app_config, save_user_config)


# --------------------------------------------------------------------------- small widgets
class _PathEdit(QWidget):
    def __init__(self, directory: bool = False, filt: str = "All files (*)"):
        super().__init__()
        self._dir, self._filt = directory, filt
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self.edit = QLineEdit()
        btn = QPushButton("…")
        btn.setFixedWidth(30)
        btn.clicked.connect(self._browse)
        lay.addWidget(self.edit)
        lay.addWidget(btn)

    def _browse(self):
        if self._dir:
            p = QFileDialog.getExistingDirectory(self, "Select folder", self.edit.text())
        else:
            p, _ = QFileDialog.getOpenFileName(self, "Select file", self.edit.text(), self._filt)
        if p:
            self.edit.setText(p)


class _SciEdit(QLineEdit):
    """Float entry that accepts scientific notation (e.g. 2e7)."""

    def value(self) -> float:
        try:
            return float(self.text())
        except ValueError:
            raise ValueError(f"'{self.text()}' is not a valid number")

    def setValue(self, v: float) -> None:
        self.setText(f"{v:.6g}")


def _dspin(lo, hi, decimals=3, suffix="", step=None):
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    if suffix:
        w.setSuffix(suffix)
    if step:
        w.setSingleStep(step)
    return w


def _ispin(lo, hi, suffix=""):
    w = QSpinBox()
    w.setRange(lo, hi)
    if suffix:
        w.setSuffix(suffix)
    return w


def _combo(items):
    w = QComboBox()
    w.addItems(list(items))
    return w


def _text(height=54):
    w = QPlainTextEdit()
    w.setFixedHeight(height)
    return w


def _get(w):
    if isinstance(w, QComboBox):
        return w.currentText()
    if isinstance(w, QCheckBox):
        return w.isChecked()
    if isinstance(w, QPlainTextEdit):
        return w.toPlainText().strip()
    if isinstance(w, (QSpinBox, QDoubleSpinBox)):
        return w.value()
    if isinstance(w, _PathEdit):
        return w.edit.text().strip()
    if isinstance(w, _SciEdit):
        return w.value()
    return w.text().strip()


def _set(w, v):
    if isinstance(w, QComboBox):
        w.setCurrentText(str(v))
    elif isinstance(w, QCheckBox):
        w.setChecked(bool(v))
    elif isinstance(w, QPlainTextEdit):
        w.setPlainText(str(v))
    elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
        w.setValue(v)
    elif isinstance(w, _PathEdit):
        w.edit.setText(str(v))
    elif isinstance(w, _SciEdit):
        w.setValue(v)
    else:
        w.setText(str(v))


# --------------------------------------------------------------------------- base dialog
class _ConfigDialog(QDialog):
    """Load… / Save As… / Save / Apply / Close around a form built by the subclass."""

    FILTER = "JSON files (*.json)"

    def __init__(self, parent, title, cfg, path, loader, saver, on_apply):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(620, 680)
        self._cfg, self._path = cfg, Path(path)
        self._loader, self._saver, self._on_apply = loader, saver, on_apply

        inner = QWidget()
        form_lay = QVBoxLayout(inner)
        self._build(form_lay)
        form_lay.addStretch(1)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(inner)

        self._path_label = QLabel()
        buttons = QHBoxLayout()
        for text, slot in (("Load…", self._load), ("Save As…", self._save_as)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)
        buttons.addStretch(1)
        for text, slot in (("Save", self._save), ("Apply", self._apply), ("Close", self.close)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            buttons.addWidget(b)

        lay = QVBoxLayout(self)
        lay.addWidget(scroll)
        lay.addWidget(self._path_label)
        lay.addLayout(buttons)
        self._write(cfg)
        self._refresh_path()

    # subclass hooks
    def _build(self, layout): raise NotImplementedError
    def _write(self, cfg): raise NotImplementedError
    def _read(self): raise NotImplementedError

    def _refresh_path(self):
        self._path_label.setText(f"File: {self._path}")

    def _guard(self, fn):
        try:
            fn()
        except Exception as exc:
            QMessageBox.warning(self, "Configuration", str(exc))

    def _apply(self):
        def go():
            self._cfg = self._read()
            self._on_apply(self._cfg, None)
        self._guard(go)

    def _save(self):
        def go():
            cfg = self._read()
            self._saver(cfg, self._path)
            self._cfg = cfg
            self._on_apply(cfg, self._path)
        self._guard(go)

    def _save_as(self):
        p, _ = QFileDialog.getSaveFileName(self, "Save configuration", str(self._path), self.FILTER)
        if p:
            self._path = Path(p)
            self._refresh_path()
            self._save()

    def _load(self):
        p, _ = QFileDialog.getOpenFileName(self, "Load configuration", str(self._path), self.FILTER)
        if not p:
            return

        def go():
            cfg = self._loader(p)
            self._path = Path(p)
            self._cfg = cfg
            self._write(cfg)               # fills the form; press Apply/Save to use it
            self._refresh_path()
        self._guard(go)


# --------------------------------------------------------------------------- application
class AppSettingsDialog(_ConfigDialog):
    def __init__(self, parent, cfg: AppConfig, path, on_apply):
        super().__init__(parent, "Application settings", cfg, path,
                         load_app_config, save_app_config, on_apply)

    def _build(self, layout):
        sections = [
            ("Delsys connection", "delsys", [
                ("source", "EMG source", lambda: _combo(SOURCES)),
                ("key", "API key", _text),
                ("license", "API license", _text),
                ("api_dll_dir", "DelsysAPI folder", lambda: _PathEdit(directory=True)),
                ("pythonnet_runtime", ".NET runtime", lambda: _combo(("coreclr", "netfx"))),
                ("emg_channel_type", "EMG channel type contains", QLineEdit),
                ("sample_rate_override_hz", "Sample rate override (0 = sensor)",
                 lambda: _dspin(0, 20000, 3, " Hz")),
                ("configure_timeout_s", "Configure timeout", lambda: _dspin(1, 120, 1, " s")),
                ("poll_interval_ms", "Idle poll interval", lambda: _dspin(0.1, 100, 1, " ms")),
            ]),
            ("Band power", "band_power", [
                ("metric", "Metric", lambda: _combo(METRICS)),
                ("window_s", "FFT window", lambda: _dspin(0.01, 5, 3, " s", 0.01)),
                ("step_s", "Window step", lambda: _dspin(0.001, 1, 3, " s", 0.005)),
                ("f_low", "Band low", lambda: _dspin(0, 5000, 1, " Hz")),
                ("f_high", "Band high", lambda: _dspin(1, 5000, 1, " Hz")),
                ("avg_window_s", "Trailing average length", lambda: _dspin(0, 5, 3, " s", 0.01)),
            ]),
            ("Steering", "steering", [
                ("lock_to_lock_deg", "Max lock-to-lock angle", lambda: _dspin(1, 5000, 1, "°")),
            ]),
            ("Logitech G29 / G920", "wheel", [
                ("device_index", "Device index (-1 = auto-detect)", lambda: _ispin(-1, 15)),
                ("axis", "Steering axis", lambda: _ispin(0, 7)),
                ("wheel_range_deg", "Wheel range (match G HUB)", lambda: _dspin(40, 1080, 0, "°")),
                ("max_target_deg", "Max target angle (±)", lambda: _dspin(0, 540, 1, "°")),
                ("max_slew_deg_per_s", "Max slew rate", lambda: _dspin(1, 5000, 0, " °/s")),
                ("max_accel_deg_per_s2", "Max acceleration (0 = off)", lambda: _dspin(0, 20000, 0, " °/s²")),
                ("max_force_pct", "Max motor force", lambda: _dspin(1, 100, 0, " %")),
                ("kp", "Position gain (Kp)", lambda: _dspin(0, 5000, 1, " /°", 10)),
                ("kd", "Damping gain (Kd)", lambda: _dspin(0, 1000, 1, " /(°/s)", 5)),
                ("safety_max_error_deg", "Release motor if off target by (0 = off)",
                 lambda: _dspin(0, 540, 0, "°")),
                ("loop_hz", "Control loop rate", lambda: _dspin(10, 500, 0, " Hz")),
                ("target_timeout_s", "Centre wheel if EMG silent for (0 = off)",
                 lambda: _dspin(0, 10, 2, " s")),
            ]),
            ("Network streaming (to another computer)", "network", [
                ("protocol", "Protocol", lambda: _combo(PROTOCOLS)),
                ("host", "Remote computer (IP address / name)", QLineEdit),
                ("port", "Port", lambda: _ispin(1, 65535)),
                ("send_hz", "Send rate", lambda: _dspin(1, 1000, 0, " Hz")),
                ("connect_timeout_s", "TCP connect timeout", lambda: _dspin(0.5, 60, 1, " s")),
                ("reconnect_interval_s", "TCP reconnect interval", lambda: _dspin(0.5, 60, 1, " s")),
            ]),
            ("Display", "gui", [
                ("history_s", "Plot history", lambda: _dspin(1, 120, 1, " s")),
                ("refresh_hz", "Refresh rate", lambda: _dspin(1, 120, 0, " Hz")),
            ]),
        ]
        self._fields = []        # (section attr, field name, widget)
        self._map_widgets = {}
        for title, attr, rows in sections:
            if attr == "band_power":                       # sensor map goes before band power
                layout.addWidget(self._build_sensor_map())
            box = QGroupBox(title)
            form = QFormLayout(box)
            for name, label, factory in rows:
                w = factory()
                form.addRow(label, w)
                self._fields.append((attr, name, w))
            layout.addWidget(box)

    def _build_sensor_map(self):
        box = QGroupBox("Sensor → muscle mapping (Delsys pair number 1–16)")
        grid = QGridLayout(box)
        for c, arm in enumerate(("Left arm", "Right arm")):
            grid.addWidget(QLabel(f"<b>{arm}</b>"), 0, c * 2, 1, 2)
        for c, a in enumerate(ARMS):
            for r, m in enumerate(MUSCLES):
                key = f"{a}_{m}"
                w = _ispin(1, 16)
                self._map_widgets[key] = w
                grid.addWidget(QLabel(m.capitalize()), r + 1, c * 2)
                grid.addWidget(w, r + 1, c * 2 + 1)
        return box

    def _write(self, cfg: AppConfig):
        for attr, name, w in self._fields:
            _set(w, getattr(getattr(cfg, attr), name))
        for key, w in self._map_widgets.items():
            w.setValue(int(cfg.delsys.sensor_map[key]))

    def _read(self) -> AppConfig:
        cfg = copy.deepcopy(self._cfg)
        for attr, name, w in self._fields:
            setattr(getattr(cfg, attr), name, _get(w))
        cfg.delsys.sensor_map = {k: w.value() for k, w in self._map_widgets.items()}
        cfg.validate()
        return cfg


# --------------------------------------------------------------------------- user
class UserSettingsDialog(_ConfigDialog):
    def __init__(self, parent, cfg: UserConfig, path, on_apply):
        super().__init__(parent, "User settings (weights)", cfg, path,
                         load_user_config, save_user_config, on_apply)

    def _build(self, layout):
        self._n, self._w, self._sum_labels = {}, {}, {}
        for arm, title in (("L", "Left arm"), ("R", "Right arm")):
            box = QGroupBox(title)
            grid = QGridLayout(box)
            for c, h in enumerate(("Muscle", "n  (normalization)", "w  (weight, −1…1)")):
                grid.addWidget(QLabel(f"<b>{h}</b>"), 0, c)
            for r, m in enumerate(MUSCLES):
                key = f"{arm}_{m}"
                n, w = _SciEdit(), _dspin(-1, 1, 3, "", 0.05)
                w.valueChanged.connect(self._update_sums)
                self._n[key], self._w[key] = n, w
                grid.addWidget(QLabel(m.capitalize()), r + 1, 0)
                grid.addWidget(n, r + 1, 1)
                grid.addWidget(w, r + 1, 2)
            lbl = QLabel()
            self._sum_labels[arm] = lbl
            grid.addWidget(lbl, 4, 0, 1, 3)
            layout.addWidget(box)
        note = QLabel("normalized power = clip(n · band value, −1, 1);  arm angle = clip(Σ w · normalized "
                      "power, −1, 1);  steering = mean of both arms.  Negative = left.")
        note.setWordWrap(True)
        layout.addWidget(note)

    def _update_sums(self):
        for arm, lbl in self._sum_labels.items():
            s = sum(abs(self._w[f"{arm}_{m}"].value()) for m in MUSCLES)
            color = "red" if s > 1.0 + 1e-9 else "gray"
            lbl.setText(f"<span style='color:{color}'>Σ|w| = {s:.3f} "
                        f"{'(arm output will saturate at ±1)' if s > 1 else ''}</span>")

    def _write(self, cfg: UserConfig):
        for key in CHANNEL_KEYS:
            ch = cfg.channel(key)
            self._n[key].setValue(ch.n)
            self._w[key].setValue(ch.w)
        self._update_sums()

    def _read(self) -> UserConfig:
        cfg = copy.deepcopy(self._cfg)
        for key in CHANNEL_KEYS:
            ch = cfg.channel(key)
            ch.n, ch.w = self._n[key].value(), self._w[key].value()
        cfg.validate()
        return cfg
