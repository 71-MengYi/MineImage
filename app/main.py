"""应用入口：``python -m app``。"""

from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QIcon
from PySide6.QtWidgets import QApplication, QMessageBox

from app import APP_DISPLAY_NAME, ORGANIZATION, __version__
from app.config import AppSettings, default_config_path, default_log_db_path, resource_root
from app.core.logstore import LogStore, SQLiteLogHandler
from app.ui.main_window import MainWindow
from app.ui.theme import apply_theme

LOG = logging.getLogger("mineimage")


def _configure_logging() -> tuple[Path, SQLiteLogHandler | None]:
    """配置文件日志与 SQLite 应用日志，返回（日志文件, SQLite 处理器）。"""

    log_path = default_config_path().with_name("MineImage.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handlers: list[logging.Handler] = []
    store_handler: SQLiteLogHandler | None = None
    try:
        store = LogStore(default_log_db_path())
        store_handler = SQLiteLogHandler(store, level=logging.INFO)
        handlers.append(store_handler)
    except Exception as exc:  # noqa: BLE001 - 日志库初始化失败不能阻止应用启动
        print(f"MineImage: 无法初始化 SQLite 日志库：{exc}", file=sys.stderr)
        store_handler = None
    handlers.extend(
        [
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(),
        ]
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=handlers,
    )
    return log_path, store_handler


def _install_excepthook(app: QApplication, store_handler: SQLiteLogHandler | None) -> None:
    def hook(exc_type, exc_value, exc_tb):  # noqa: ANN001
        LOG.error(
            "未捕获异常：%s: %s", exc_type.__name__, exc_value,
            exc_info=(exc_type, exc_value, exc_tb),
        )
        entry = store_handler.last_entry if store_handler is not None else None
        log_hint = f"\n\n日志 ID：{entry.entry_id}" if entry is not None else ""
        QMessageBox.critical(
            None,
            "MineImage 发生错误",
            f"{exc_type.__name__}: {exc_value}\n\n"
            f"详细信息已写入日志（帮助 → 日志）。{log_hint}",
        )

    sys.excepthook = hook

    def thread_hook(args: threading.ExceptHookArgs) -> None:
        if args.exc_value is None:
            return
        LOG.error(
            "后台线程未捕获异常（%s）：%s: %s",
            args.thread.name if args.thread is not None else "?",
            args.exc_type.__name__,
            args.exc_value,
            exc_info=(args.exc_type, args.exc_value, args.exc_traceback),
        )

    threading.excepthook = thread_hook


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

    log_path, log_handler = _configure_logging()
    _install_excepthook(app, log_handler)
    LOG.info("MineImage %s 启动，Python %s，设置 %s", __version__, sys.version, default_config_path())

    apply_theme(app)
    settings = AppSettings()
    window = MainWindow(
        settings,
        log_store=log_handler.store if log_handler is not None else None,
    )
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
    if log_handler is not None:
        LOG.info("应用日志数据库：%s", log_handler.store.path)
    code = app.exec()
    LOG.info("MineImage 退出，返回码 %s", code)
    return code
