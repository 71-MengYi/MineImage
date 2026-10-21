"""应用入口：``python -m app``。"""

from __future__ import annotations

import logging
import os
import sys
import traceback
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from app import APP_DISPLAY_NAME, ORGANIZATION, __version__
from app.config import AppSettings, default_config_path, resource_root
from app.ui.main_window import MainWindow
from app.ui.theme import apply_theme

LOG = logging.getLogger("mineimage")


def _configure_logging() -> Path:
    log_path = default_config_path().with_name("MineImage.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return log_path


def _install_excepthook(app: QApplication) -> None:
    def hook(exc_type, exc_value, exc_tb):  # noqa: ANN001
        detail = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        LOG.error("未捕获异常:\n%s", detail)
        QMessageBox.critical(
            None,
            "MineImage 发生错误",
            f"{exc_type.__name__}: {exc_value}\n\n详细信息已写入日志。",
        )

    sys.excepthook = hook


def _windows_app_id() -> None:
    if sys.platform != "win32":
        return
    try:
        import ctypes

        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID("MineImage.ImageLab.1")
    except Exception:  # noqa: BLE001
        pass


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv
    _windows_app_id()
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )
    app = QApplication(argv)
    app.setApplicationName("MineImage")
    app.setOrganizationName(ORGANIZATION)
    app.setApplicationVersion(__version__)
    app.setApplicationDisplayName(APP_DISPLAY_NAME)
    app.setStyle("Fusion")

    icon_path = resource_root() / "assets" / "icon.svg"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    log_path = _configure_logging()
    _install_excepthook(app)
    LOG.info("MineImage %s 启动，Python %s，设置 %s", __version__, sys.version, default_config_path())

    apply_theme(app)
    settings = AppSettings()
    window = MainWindow(settings)
    LOG.info(
        "算子库：%d 个算子，%d 个加载问题",
        len(window.registry),
        len(window.registry.issues),
    )

    target = None
    args = list(argv[1:])
    if "--open" in args:
        index = args.index("--open")
        if index + 1 < len(args):
            target = args[index + 1]
    elif args and not args[0].startswith("-"):
        target = args[0]
    if target:
        path = Path(target)
        if path.suffix.lower() == ".mi":
            window.open_project_path(path)
        elif path.exists():
            window.open_image_path(path)

    window.show()
    LOG.info("日志文件：%s", log_path)
    return app.exec()
