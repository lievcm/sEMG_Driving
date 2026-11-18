"""EMG -> steering host. Run:  python main.py"""
import os
import sys

os.environ.setdefault("PYQTGRAPH_QT_LIB", "PyQt6")

import pyqtgraph as pg
from PyQt6.QtWidgets import QApplication, QMessageBox

from config import (DEFAULT_APP_CONFIG_PATH, DEFAULT_USER_CONFIG_PATH, AppConfig, UserConfig,
                    load_app_config, load_user_config)
from main_window import MainWindow


def _load(loader, default_factory, path, label, problems):
    if not path.exists():
        return default_factory()
    try:
        return loader(path)
    except Exception as exc:
        problems.append(f"{label} config ({path.name}) could not be loaded: {exc}\nUsing defaults.")
        return default_factory()


def main() -> int:
    app = QApplication(sys.argv)
    pg.setConfigOptions(antialias=False)
    problems: list[str] = []
    app_cfg = _load(load_app_config, AppConfig, DEFAULT_APP_CONFIG_PATH, "Application", problems)
    user_cfg = _load(load_user_config, UserConfig, DEFAULT_USER_CONFIG_PATH, "User", problems)
    win = MainWindow(app_cfg, user_cfg, DEFAULT_APP_CONFIG_PATH, DEFAULT_USER_CONFIG_PATH)
    win.show()
    if problems:
        QMessageBox.warning(win, "Configuration", "\n\n".join(problems))
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
