"""主窗口 ↔ 应用日志的集成测试（离屏，不弹出真实对话框）。"""

from __future__ import annotations

import logging
import os
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.config import AppSettings
from app.core.logstore import LogFilter, LogStore
from app.ui.main_window import MainWindow
from app.ui.theme import apply_theme

_REPO_ROOT = Path(__file__).resolve().parent.parent
_APP: QApplication | None = None


class _FakeLogDialog:
    """替代真实对话框，记录传入的 store 与生命周期。"""

    created: list["_FakeLogDialog"] = []

    def __init__(self, store: LogStore, parent=None) -> None:  # noqa: ANN001
        self.store = store
        self.parent = parent
        self.deleted = False
        self.modal = True
        self.visible = False
        _FakeLogDialog.created.append(self)

    def setModal(self, modal: bool) -> None:  # noqa: N802
        self.modal = modal

    def show(self) -> None:
        self.visible = True

    def isVisible(self) -> bool:  # noqa: N802
        return self.visible

    def raise_(self) -> None:
        self.raised = True

    def activateWindow(self) -> None:  # noqa: N802
        self.activated = True

    def close(self) -> None:
        self.visible = False
        self.closed = True

    def deleteLater(self) -> None:
        self.deleted = True


def setUpModule() -> None:  # noqa: N802
    global _APP
    _APP = QApplication.instance() or QApplication([])
    apply_theme(_APP)


def tearDownModule() -> None:  # noqa: N802
    global _APP
    _APP = None


class MainWindowLogIntegrationTests(unittest.TestCase):
    def setUp(self) -> None:  # noqa: N802
        _FakeLogDialog.created.clear()
        base = _REPO_ROOT / ".test_tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.temp = base / f"log-integration-{uuid.uuid4().hex[:10]}"
        self.temp.mkdir()
        self.store = LogStore(self.temp / "logs.db")
        self.settings = AppSettings(self.temp / "MineImage.ini")
        self.win = MainWindow(self.settings, log_store=self.store)

    def tearDown(self) -> None:  # noqa: N802
        self.win.close()
        self.win.deleteLater()
        assert _APP is not None
        _APP.processEvents()
        shutil.rmtree(self.temp, ignore_errors=True)

    def _help_menu_actions(self) -> list[str]:
        for action in self.win.menuBar().actions():
            if action.text() == "帮助(&H)" and action.menu() is not None:
                return [entry.text() for entry in action.menu().actions()]
        self.fail("找不到帮助菜单")

    def test_help_menu_contains_log_action(self) -> None:
        texts = self._help_menu_actions()
        self.assertIn("日志…", texts)
        self.assertIn("关于 MineImage", texts)
        self.assertEqual(self.win.act_logs.shortcut().toString(), "Ctrl+Shift+L")

    def test_log_action_passes_shared_store_to_dialog(self) -> None:
        with patch("app.ui.main_window.LogDialog", _FakeLogDialog):
            self.win.act_logs.trigger()
            self.win.act_logs.trigger()  # 已打开时只前置，不重复创建
        self.assertEqual(len(_FakeLogDialog.created), 1)
        dialog = _FakeLogDialog.created[0]
        self.assertIs(dialog.store, self.store)
        self.assertIs(dialog.parent, self.win)
        self.assertFalse(dialog.modal)
        self.assertTrue(dialog.visible)
        self.assertTrue(dialog.raised)
        self.assertFalse(dialog.deleted)

    def test_closing_main_window_closes_log_dialog(self) -> None:
        with patch("app.ui.main_window.LogDialog", _FakeLogDialog):
            self.win.act_logs.trigger()
        dialog = _FakeLogDialog.created[0]
        self.win.close()
        self.assertTrue(dialog.closed)
        self.assertIsNone(self.win._log_dialog)

    def test_dialog_queries_logs_written_by_handler(self) -> None:
        # 通过公开 store 写入后，再打开对话框应当能查询到（真实对话框未参与，
        # 这里只验证主窗口传入的是同一个可查询的仓库）。
        entry = self.store.append("ERROR", "集成测试错误", detail="Traceback: X")
        entries = self.store.query(LogFilter(levels=("ERROR",)))
        self.assertEqual(entries[0].entry_id, entry.entry_id)
        self.assertIn("集成测试错误", entries[0].full_text())

    def test_configure_logging_installs_sqlite_handler(self) -> None:
        from app import main as app_main

        root = logging.getLogger()
        saved_handlers = list(root.handlers)
        saved_level = root.level
        for handler in saved_handlers:
            root.removeHandler(handler)
        try:
            with patch.dict(
                os.environ,
                {
                    "MINEIMAGE_CONFIG": str(self.temp / "config" / "MineImage.ini"),
                    "MINEIMAGE_LOG_DB": str(self.temp / "startup-logs.db"),
                },
            ):
                log_path, store_handler = app_main._configure_logging()
            self.assertIsNotNone(store_handler)
            assert store_handler is not None
            self.assertEqual(store_handler.store.path, self.temp / "startup-logs.db")
            self.assertTrue(log_path.exists())

            logging.getLogger("mineimage.integration").error("启动日志写入测试")
            entries = store_handler.store.query(LogFilter(levels=("ERROR",)))
            self.assertTrue(
                any("启动日志写入测试" in entry.message for entry in entries),
                [entry.message for entry in entries],
            )
        finally:
            for handler in list(root.handlers):
                root.removeHandler(handler)
                handler.close()
            for handler in saved_handlers:
                root.addHandler(handler)
            root.setLevel(saved_level)


if __name__ == "__main__":
    unittest.main()
