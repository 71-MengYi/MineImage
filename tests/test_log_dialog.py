"""应用日志查看器对话框测试。

全部使用 offscreen 平台，不执行模态事件循环；模态确认框通过 patch 绕过。
临时目录放在仓库内 .test_tmp 下（与 tests/test_widgets.py 保持一致），
避免 Windows 沙箱下系统临时目录的 ACL 问题。
"""

from __future__ import annotations

import csv
import io
import os
import shutil
import sqlite3
import unittest
import uuid
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QDate, Qt
from PySide6.QtGui import QBrush
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QHeaderView,
    QMessageBox,
)

from app.core.logstore import LOG_LEVELS, LogStore
from app.ui.log_dialog import LogDialog, LogTableModel
from app.ui.theme import apply_theme

_APP: QApplication | None = None
_REPO_ROOT = Path(__file__).resolve().parent.parent

_TRACEBACK = (
    "Traceback (most recent call last):\n"
    '  File "project.py", line 301, in load\n'
    "    raise ValueError('工程文件损坏')\n"
    "ValueError: 工程文件损坏"
)


def _workspace_tempdir() -> Path:
    """在仓库工作区内创建临时目录（不要用 tempfile.mkdtemp）。"""

    base = _REPO_ROOT / ".test_tmp"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"logdialog-{uuid.uuid4().hex[:10]}"
    path.mkdir()
    return path


def _cleanup_tempdir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def setUpModule() -> None:
    global _APP
    _APP = QApplication.instance() or QApplication([])
    apply_theme(_APP)


def tearDownModule() -> None:
    global _APP
    _APP = None


def _app() -> QApplication:
    assert _APP is not None
    return _APP


class LogDialogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = _workspace_tempdir()
        self.addCleanup(_cleanup_tempdir, self.tmpdir)
        self.store = LogStore(self.tmpdir / "logs.db")

        # 旧记录：日期筛选应把它排除在外。
        self.old_entry = self.store.append(
            "INFO",
            "旧日志：项目已归档",
            logger="mineimage.ui",
            source="legacy.py:1",
            created_at=datetime(2020, 1, 1, 8, 0, 0),
        )
        self.info_entry = self.store.append(
            "INFO",
            "应用启动完成",
            logger="mineimage.ui",
            source="main.py:12",
            thread="MainThread",
        )
        self.warning_entry = self.store.append(
            "WARNING",
            "磁盘空间不足",
            logger="mineimage.io",
            source="io.py:88",
        )
        self.error_entry = self.store.append(
            "ERROR",
            "预览渲染失败",
            logger="mineimage.preview",
            source="preview.py:45",
        )
        self.detail_error = self.store.append(
            "ERROR",
            "读取工程失败",
            logger="mineimage.project",
            detail=_TRACEBACK,
            source="project.py:301",
        )

        self.dialog = LogDialog(self.store)
        self.dialog.show()
        _app().processEvents()

    def tearDown(self) -> None:
        try:
            self.dialog.close()
            self.dialog.deleteLater()
            _app().processEvents()
        finally:
            self.store.close()

    # ------------------------------------------------------------ 基础
    def test_table_shape_and_newest_first(self) -> None:
        model = self.dialog.model
        # 默认按最近 7 天筛选，2020 年的旧记录被排除。
        self.assertEqual(model.rowCount(), 4)
        self.assertEqual(model.columnCount(), 6)
        self.assertEqual(len(LogTableModel.COLUMNS), 6)

        headers = [
            model.headerData(column, Qt.Orientation.Horizontal)
            for column in range(model.columnCount())
        ]
        self.assertEqual(headers, list(LogTableModel.COLUMNS))
        self.assertEqual(headers[0], "日志 ID")
        self.assertEqual(headers[4], "消息")

        # LogStore 按 seq 倒序返回：最后写入的带 detail 的 ERROR 在最前。
        first = model.entry_at(0)
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first.entry_id, self.detail_error.entry_id)
        self.assertEqual(first.message, "读取工程失败")
        last = model.entry_at(3)
        self.assertIsNotNone(last)
        assert last is not None
        self.assertEqual(last.entry_id, self.info_entry.entry_id)
        self.assertIsNone(model.entry_at(-1))
        self.assertIsNone(model.entry_at(99))

        view = self.dialog.table_view
        self.assertFalse(view.showGrid())
        self.assertTrue(view.alternatingRowColors())
        self.assertFalse(view.isSortingEnabled())
        self.assertEqual(
            view.selectionBehavior(),
            QAbstractItemView.SelectionBehavior.SelectRows,
        )
        self.assertEqual(
            view.selectionMode(),
            QAbstractItemView.SelectionMode.ExtendedSelection,
        )
        self.assertEqual(
            view.editTriggers(),
            QAbstractItemView.EditTrigger.NoEditTriggers,
        )
        self.assertEqual(
            view.horizontalHeader().sectionResizeMode(4),
            QHeaderView.ResizeMode.Stretch,
        )

        # 等级列颜色：ERROR 红、WARNING 橙、INFO 默认。
        error_brush = model.data(
            model.index(0, LogTableModel.LEVEL_COLUMN),
            Qt.ItemDataRole.ForegroundRole,
        )
        self.assertIsInstance(error_brush, QBrush)
        self.assertEqual(error_brush.color().name(), "#b91c1c")

        warning_row = model.row_for_entry_id(self.warning_entry.entry_id)
        self.assertGreaterEqual(warning_row, 0)
        warning_brush = model.data(
            model.index(warning_row, LogTableModel.LEVEL_COLUMN),
            Qt.ItemDataRole.ForegroundRole,
        )
        self.assertIsInstance(warning_brush, QBrush)
        self.assertEqual(warning_brush.color().name(), "#b45309")

        info_row = model.row_for_entry_id(self.info_entry.entry_id)
        self.assertIsNone(
            model.data(
                model.index(info_row, LogTableModel.LEVEL_COLUMN),
                Qt.ItemDataRole.ForegroundRole,
            )
        )
        self.assertIsNone(
            model.data(
                model.index(0, 4), Qt.ItemDataRole.ForegroundRole
            )
        )

        # 消息列 ToolTip 展示完整消息。
        tooltip = model.data(
            model.index(0, 4), Qt.ItemDataRole.ToolTipRole
        )
        self.assertEqual(tooltip, "读取工程失败")

    def test_header_hint_and_status(self) -> None:
        hint = self.dialog.hint_label.text()
        self.assertIn("日志保存在", hint)
        self.assertIn(str(self.store.path), hint)
        self.assertIn("共 5 条", hint)
        self.assertIn("数据库", hint)
        self.assertIn("KB", hint)

        status = self.dialog.status_label.text()
        self.assertIn("共 4 条符合条件的日志", status)
        self.assertIn("数据库", status)
        self.assertIn("最后刷新", status)
        self.assertEqual(self.dialog.windowTitle(), "应用日志")
        self.assertGreaterEqual(self.dialog.minimumWidth(), 860)
        self.assertGreaterEqual(self.dialog.minimumHeight(), 560)

    # ------------------------------------------------------------ 筛选
    def test_level_checkboxes_filter_rows(self) -> None:
        checks = self.dialog.level_checks
        self.assertEqual(set(checks), set(LOG_LEVELS))
        expected_labels = {
            "DEBUG": "调试",
            "INFO": "信息",
            "WARNING": "警告",
            "ERROR": "错误",
            "CRITICAL": "严重",
        }
        for level, check in checks.items():
            self.assertTrue(check.isChecked())
            self.assertEqual(check.text(), expected_labels[level])
            self.assertEqual(check.toolTip(), level)

        checks["INFO"].setChecked(False)
        _app().processEvents()
        self.assertEqual(self.dialog.model.rowCount(), 3)
        levels = {
            self.dialog.model.entry_at(row).level
            for row in range(self.dialog.model.rowCount())
        }
        self.assertNotIn("INFO", levels)
        self.assertEqual(levels, {"WARNING", "ERROR"})

        checks["INFO"].setChecked(True)
        _app().processEvents()
        self.assertEqual(self.dialog.model.rowCount(), 4)

    def test_date_filter_and_disabled_editors(self) -> None:
        dialog = self.dialog
        self.assertTrue(dialog.date_check.isChecked())
        self.assertTrue(dialog.start_date.isEnabled())
        self.assertTrue(dialog.end_date.isEnabled())
        self.assertEqual(dialog.start_date.displayFormat(), "yyyy-MM-dd")
        self.assertEqual(dialog.end_date.displayFormat(), "yyyy-MM-dd")
        self.assertTrue(dialog.start_date.calendarPopup())
        self.assertTrue(dialog.end_date.calendarPopup())
        self.assertEqual(
            dialog.start_date.date(), QDate.currentDate().addDays(-6)
        )
        self.assertEqual(dialog.end_date.date(), QDate.currentDate())
        self.assertEqual(dialog.model.rowCount(), 4)

        dialog.date_check.setChecked(False)
        _app().processEvents()
        self.assertFalse(dialog.start_date.isEnabled())
        self.assertFalse(dialog.end_date.isEnabled())
        self.assertEqual(dialog.model.rowCount(), 5)
        all_ids = {
            dialog.model.entry_at(row).entry_id
            for row in range(dialog.model.rowCount())
        }
        self.assertIn(self.old_entry.entry_id, all_ids)

        dialog.date_check.setChecked(True)
        _app().processEvents()
        self.assertTrue(dialog.start_date.isEnabled())
        self.assertEqual(dialog.model.rowCount(), 4)
        visible_ids = {
            dialog.model.entry_at(row).entry_id
            for row in range(dialog.model.rowCount())
        }
        self.assertNotIn(self.old_entry.entry_id, visible_ids)

    def test_logger_combo_filter(self) -> None:
        combo = self.dialog.logger_combo
        self.assertEqual(combo.itemText(0), "全部模块")
        self.assertEqual(combo.itemData(0), "")
        self.assertEqual(combo.count(), 5)  # 全部模块 + 4 个模块

        index = combo.findData("mineimage.project")
        self.assertGreater(index, 0)
        combo.setCurrentIndex(index)
        _app().processEvents()
        self.assertEqual(self.dialog.model.rowCount(), 1)
        entry = self.dialog.model.entry_at(0)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.entry_id, self.detail_error.entry_id)
        self.assertEqual(entry.logger, "mineimage.project")

        index = combo.findData("mineimage.io")
        combo.setCurrentIndex(index)
        _app().processEvents()
        self.assertEqual(self.dialog.model.rowCount(), 1)
        self.assertEqual(
            self.dialog.model.entry_at(0).entry_id,
            self.warning_entry.entry_id,
        )

        combo.setCurrentIndex(0)
        _app().processEvents()
        self.assertEqual(self.dialog.model.rowCount(), 4)

    def test_keyword_debounce_searches_message_and_detail(self) -> None:
        edit = self.dialog.keyword_edit
        self.assertEqual(
            edit.placeholderText(), "搜索消息 / 详细信息 / 日志 ID"
        )
        self.assertTrue(edit.isClearButtonEnabled())

        edit.setText("磁盘空间不足")
        # 防抖 250ms，等待定时器触发刷新。
        QTest.qWait(320)
        self.assertEqual(self.dialog.model.rowCount(), 1)
        entry = self.dialog.model.entry_at(0)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.entry_id, self.warning_entry.entry_id)

        edit.setText("Traceback")
        QTest.qWait(320)
        self.assertEqual(self.dialog.model.rowCount(), 1)
        entry = self.dialog.model.entry_at(0)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.entry_id, self.detail_error.entry_id)

        # entry_id 也是可搜索字段。
        edit.setText("ERR-")
        QTest.qWait(320)
        self.assertEqual(self.dialog.model.rowCount(), 2)

        edit.clear()
        QTest.qWait(320)
        self.assertEqual(self.dialog.model.rowCount(), 4)

    def test_all_levels_unchecked_is_empty_state(self) -> None:
        for check in self.dialog.level_checks.values():
            check.setChecked(False)
        _app().processEvents()

        self.assertEqual(self.dialog.model.rowCount(), 0)
        status = self.dialog.status_label.text()
        self.assertIn("没有符合条件的日志", status)
        self.assertIn(
            "选择一条日志查看完整信息", self.dialog.detail_edit.toPlainText()
        )
        # 顶部摘要仍显示数据库内总条数。
        self.assertIn("共 5 条", self.dialog.hint_label.text())

        self.dialog.level_checks["ERROR"].setChecked(True)
        _app().processEvents()
        self.assertEqual(self.dialog.model.rowCount(), 2)

    # ------------------------------------------------------------ 详情/复制
    def test_selection_detail_and_copy(self) -> None:
        dialog = self.dialog
        view = dialog.table_view
        model = dialog.model
        clipboard = _app().clipboard()
        clipboard.clear()

        dialog.table_view.selectRow(0)
        dialog.table_view.setCurrentIndex(model.index(0, 0))
        _app().processEvents()

        detail = dialog.detail_edit.toPlainText()
        self.assertIn(self.detail_error.entry_id, detail)
        self.assertIn("日志 ID", detail)
        self.assertIn("Traceback", detail)
        self.assertIn("ValueError", detail)
        self.assertTrue(dialog.detail_edit.isReadOnly())

        # 详细信息区可直接用 Ctrl+A / Ctrl+C 选中复制。
        clipboard.clear()
        dialog.detail_edit.setFocus()
        _app().processEvents()
        QTest.keyClick(
            dialog.detail_edit,
            Qt.Key.Key_A,
            Qt.KeyboardModifier.ControlModifier,
        )
        QTest.keyClick(
            dialog.detail_edit,
            Qt.Key.Key_C,
            Qt.KeyboardModifier.ControlModifier,
        )
        _app().processEvents()
        self.assertIn(self.detail_error.entry_id, clipboard.text())
        self.assertIn("Traceback", clipboard.text())

        # 复制当前单元格（消息列）。
        view.setCurrentIndex(model.index(0, 4))
        copied_cell = view.copy_current_cell()
        self.assertEqual(copied_cell, self.detail_error.message)
        self.assertEqual(clipboard.text(), self.detail_error.message)

        # Ctrl+C 复制选中行（整行选择 → 6 列制表符分隔）。
        view.selectRow(0)
        _app().processEvents()
        QTest.keyClick(
            view, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier
        )
        _app().processEvents()
        row_text = clipboard.text()
        self.assertIn(self.detail_error.entry_id, row_text)
        self.assertIn(self.detail_error.message, row_text)
        self.assertEqual(len(row_text.splitlines()[0].split("\t")), 6)

        # 右键菜单包含要求的动作，且选中行时可用。
        menu = view.build_context_menu()
        try:
            texts = [action.text() for action in menu.actions()]
            for expected in ("复制单元格", "复制整行", "复制日志 ID", "复制详细信息"):
                self.assertIn(expected, texts)
            for action in menu.actions():
                self.assertTrue(action.isEnabled())
            menu.actions()[2].trigger()  # 复制日志 ID
            self.assertEqual(clipboard.text(), self.detail_error.entry_id)
            menu.actions()[3].trigger()  # 复制详细信息
            self.assertIn("Traceback", clipboard.text())
        finally:
            menu.deleteLater()

        # 双击一行：完整日志写入剪贴板，并在状态栏临时提示。
        clipboard.clear()
        view.doubleClicked.emit(model.index(0, 0))
        _app().processEvents()
        self.assertIn("日志 ID：", clipboard.text())
        self.assertIn(self.detail_error.entry_id, clipboard.text())
        flash = dialog.status_label.text()
        self.assertIn("已复制日志", flash)
        self.assertIn(self.detail_error.entry_id, flash)
        dialog._restore_status()
        self.assertIn("共 4 条符合条件的日志", dialog.status_label.text())

        # 无选中时复制动作全部禁用。
        view.selectionModel().clearSelection()
        view.selectionModel().clearCurrentIndex()
        _app().processEvents()
        self.assertIn(
            "选择一条日志查看完整信息", dialog.detail_edit.toPlainText()
        )
        menu = view.build_context_menu()
        try:
            for action in menu.actions():
                self.assertFalse(action.isEnabled())
        finally:
            menu.deleteLater()

    # ------------------------------------------------------------ 刷新
    def test_refresh_preserves_selected_entry_id(self) -> None:
        dialog = self.dialog
        view = dialog.table_view
        model = dialog.model
        selected_row = model.row_for_entry_id(self.warning_entry.entry_id)
        self.assertGreaterEqual(selected_row, 0)
        view.selectRow(selected_row)
        view.setCurrentIndex(model.index(selected_row, 0))
        _app().processEvents()

        # 外部继续写日志后刷新：新记录出现在第一行，选中行跟随 entry_id 移动。
        new_entry = self.store.append(
            "INFO", "刷新期间新增日志", logger="mineimage.ui"
        )
        dialog.refresh()
        _app().processEvents()

        self.assertEqual(model.row_for_entry_id(new_entry.entry_id), 0)
        current = view.currentIndex()
        self.assertTrue(current.isValid())
        entry = model.entry_at(current.row())
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.entry_id, self.warning_entry.entry_id)

    def test_auto_refresh_timer_follows_checkbox(self) -> None:
        dialog = self.dialog
        self.assertTrue(dialog.auto_refresh_check.isChecked())
        self.assertTrue(dialog._auto_timer.isActive())

        dialog.auto_refresh_check.setChecked(False)
        self.assertFalse(dialog._auto_timer.isActive())
        dialog.auto_refresh_check.setChecked(True)
        self.assertTrue(dialog._auto_timer.isActive())

    def test_query_error_shows_status_and_clears_table(self) -> None:
        with patch.object(
            self.store,
            "query",
            side_effect=sqlite3.OperationalError("database is locked"),
        ):
            self.dialog.refresh()
        _app().processEvents()

        self.assertEqual(self.dialog.model.rowCount(), 0)
        status = self.dialog.status_label.text()
        self.assertTrue(status.startswith("读取日志失败："), status)
        self.assertIn("数据库错误", status)
        self.assertIn(
            "选择一条日志查看完整信息",
            self.dialog.detail_edit.toPlainText(),
        )

    # ------------------------------------------------------------ 清空/导出
    def test_clear_logs_requires_confirmation_and_refreshes(self) -> None:
        with patch(
            "app.ui.log_dialog.QMessageBox.question",
            return_value=QMessageBox.StandardButton.No,
        ) as question:
            self.dialog._on_clear_logs()
        question.assert_called_once()
        self.assertEqual(self.store.count(), 5)
        self.assertEqual(self.dialog.model.rowCount(), 4)

        with patch(
            "app.ui.log_dialog.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            self.dialog._on_clear_logs()
        _app().processEvents()

        self.assertEqual(self.store.count(), 0)
        self.assertEqual(self.dialog.model.rowCount(), 0)
        self.assertIn(
            "没有符合条件的日志", self.dialog.status_label.text()
        )
        # 清空后模块下拉只剩“全部模块”。
        self.assertEqual(self.dialog.logger_combo.count(), 1)
        self.assertEqual(self.dialog.logger_combo.itemText(0), "全部模块")

    def test_export_uses_current_filter_and_reports_count(self) -> None:
        target = self.tmpdir / "export-test.csv"
        with patch(
            "app.ui.log_dialog.QFileDialog.getSaveFileName",
            return_value=(str(target), "CSV 文件 (*.csv)"),
        ) as save_dialog, patch(
            "app.ui.log_dialog.QMessageBox.information"
        ) as information:
            self.dialog._on_export()

        save_dialog.assert_called_once()
        suggested = Path(save_dialog.call_args.args[2]).name
        self.assertTrue(suggested.startswith("MineImage-logs-"), suggested)
        self.assertTrue(suggested.endswith(".csv"), suggested)
        information.assert_called_once()
        message = information.call_args.args[2]
        self.assertIn("已导出 4 条日志", message)

        self.assertTrue(target.exists())
        text = target.read_text(encoding="utf-8-sig")
        self.assertIn("磁盘空间不足", text)
        self.assertIn("读取工程失败", text)
        self.assertNotIn("旧日志：项目已归档", text)
        rows = list(csv.reader(io.StringIO(text)))
        self.assertEqual(len(rows), 5)  # 表头 + 4 行（detail 中含换行）
        self.assertEqual(rows[0][0], "日志ID")
        self.assertEqual(rows[0][1], "时间")

        # 全部等级取消勾选时没有可导出的记录，不应弹出保存对话框。
        for check in self.dialog.level_checks.values():
            check.setChecked(False)
        _app().processEvents()
        with patch(
            "app.ui.log_dialog.QFileDialog.getSaveFileName"
        ) as save_dialog, patch(
            "app.ui.log_dialog.QMessageBox.information"
        ) as information:
            self.dialog._on_export()
        save_dialog.assert_not_called()
        information.assert_called_once()


if __name__ == "__main__":
    unittest.main()
