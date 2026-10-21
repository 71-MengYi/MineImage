"""应用日志查看器对话框（「帮助 → 日志」）。

设计要点：

* 日志仓库由调用方持有，本对话框只读查询；仅「清空日志」「导出 CSV」
  会调用 LogStore.clear / LogStore.export_csv；
* 所有查询都在 UI 线程内同步完成。LogStore 每次调用自建 SQLite 连接，
  数据量很小，不阻塞交互；任何 sqlite3.Error / OSError 都降级为状态栏
  提示，不向调用方抛异常；
* 表格支持整行选择，Ctrl+C / 右键菜单可复制单元格、整行、日志 ID
  与完整详细信息；双击一行会复制整条日志全文；
* 自动刷新（每 3 秒）与关键字防抖（250 毫秒）使用以对话框为父对象的
  QTimer，窗口销毁后定时器随之销毁，不会回调已删除的控件。
"""

from __future__ import annotations

import sqlite3
from collections.abc import Callable, Sequence
from datetime import date, datetime
from pathlib import Path

from PySide6.QtCore import (
    QAbstractTableModel,
    QDate,
    QItemSelection,
    QModelIndex,
    QPoint,
    QRect,
    QSize,
    Qt,
    QTimer,
)
from PySide6.QtGui import (
    QBrush,
    QColor,
    QContextMenuEvent,
    QFont,
    QFontDatabase,
    QKeyEvent,
    QKeySequence,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLayout,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from app.core.logstore import LOG_LEVELS, LogEntry, LogFilter, LogStore

_LEVEL_LABELS: dict[str, str] = {
    "DEBUG": "调试",
    "INFO": "信息",
    "WARNING": "警告",
    "ERROR": "错误",
    "CRITICAL": "严重",
}

_EMPTY_DETAIL = "选择一条日志查看完整信息（可直接选中文字复制）"
_ERROR_FOREGROUND = "#b91c1c"
_WARNING_FOREGROUND = "#b45309"

_KEYWORD_DEBOUNCE_MS = 250
_AUTO_REFRESH_MS = 3000
_STATUS_FLASH_MS = 2000

#: 表格列宽（0 表示自适应剩余宽度，即消息列）。
_COLUMN_WIDTHS: tuple[int, ...] = (265, 165, 95, 190, 0, 170)


def _format_size(size_bytes: int) -> str:
    """把字节数格式化为 X KB（契约要求统一使用 KB）。"""

    return f"{max(0, int(size_bytes)) / 1024:.1f} KB"


def _describe_error(exc: BaseException) -> str:
    """把数据库 / 文件异常转成简短中文原因，避免只暴露英文原文。"""

    message = str(exc).strip()
    if isinstance(exc, sqlite3.Error):
        return f"数据库错误（{message}）" if message else "数据库错误"
    if isinstance(exc, OSError):
        reason = str(exc.strerror or message).strip()
        return f"文件访问失败（{reason}）" if reason else "文件访问失败"
    return message or "未知错误"


def _monospace_font() -> QFont:
    """详细信息区使用的等宽字体：Consolas → Cascadia Mono → 系统等宽。"""

    available = set(QFontDatabase.families())
    for family in ("Consolas", "Cascadia Mono", "Microsoft YaHei UI"):
        if family in available:
            font = QFont(family)
            font.setStyleHint(QFont.StyleHint.Monospace)
            font.setPointSizeF(9.5)
            return font
    font = QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont)
    font.setPointSizeF(9.5)
    return font


def _level_foreground(level: str) -> QColor | None:
    if level in ("ERROR", "CRITICAL"):
        return QColor(_ERROR_FOREGROUND)
    if level == "WARNING":
        return QColor(_WARNING_FOREGROUND)
    return None


class _FlowLayout(QLayout):
    """轻量换行布局：子控件按行排列，宽度不够时自动换行。

    参考 Qt 官方 Flow Layout 示例实现 heightForWidth，这样父级 QGroupBox
    的高度会随可用宽度变化，窗口变窄时筛选控件不会被挤爆。
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        hspacing: int = 10,
        vspacing: int = 8,
    ) -> None:
        super().__init__(parent)
        self._items: list = []
        self._hspacing = max(0, int(hspacing))
        self._vspacing = max(0, int(vspacing))
        self.setContentsMargins(0, 0, 0, 0)

    # ------------------------------------------------------------ QLayout
    def addItem(self, item) -> None:  # noqa: N802 - Qt 接口
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index: int):  # noqa: N802 - Qt 接口
        if 0 <= index < len(self._items):
            return self._items[index]
        return None

    def takeAt(self, index: int):  # noqa: N802 - Qt 接口
        if 0 <= index < len(self._items):
            return self._items.pop(index)
        return None

    def expandingDirections(self) -> Qt.Orientation:  # noqa: N802 - Qt 接口
        return Qt.Orientation(0)

    def hasHeightForWidth(self) -> bool:  # noqa: N802 - Qt 接口
        return True

    def heightForWidth(self, width: int) -> int:  # noqa: N802 - Qt 接口
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect: QRect) -> None:  # noqa: N802 - Qt 接口
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt 接口
        return self.minimumSize()

    def minimumSize(self) -> QSize:  # noqa: N802 - Qt 接口
        size = QSize()
        for item in self._items:
            widget = item.widget()
            if widget is not None and widget.isHidden():
                continue
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        size += QSize(
            margins.left() + margins.right(),
            margins.top() + margins.bottom(),
        )
        return size

    # ------------------------------------------------------------ 布局
    def _do_layout(self, rect: QRect, *, test_only: bool) -> int:
        margins = self.contentsMargins()
        effective = rect.adjusted(
            margins.left(),
            margins.top(),
            -margins.right(),
            -margins.bottom(),
        )
        x = effective.x()
        y = effective.y()
        line_height = 0
        for item in self._items:
            widget = item.widget()
            if widget is not None and widget.isHidden():
                continue
            hint = item.sizeHint()
            next_x = x + hint.width() + self._hspacing
            if next_x - self._hspacing > effective.right() and line_height > 0:
                x = effective.x()
                y += line_height + self._vspacing
                next_x = x + hint.width() + self._hspacing
                line_height = 0
            if not test_only:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x = next_x
            line_height = max(line_height, hint.height())
        return y + line_height - rect.y() + margins.bottom()


class LogTableModel(QAbstractTableModel):
    """日志表格模型：列顺序为日志 ID / 时间 / 等级 / 模块 / 消息 / 来源。"""

    COLUMNS: tuple[str, ...] = ("日志 ID", "时间", "等级", "模块", "消息", "来源")
    ATTRS: tuple[str, ...] = (
        "entry_id",
        "short_time",
        "level",
        "logger",
        "message",
        "source",
    )
    LEVEL_COLUMN = 2

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._entries: list[LogEntry] = []

    # ------------------------------------------------------------ 数据
    def set_entries(self, entries: Sequence[LogEntry]) -> None:
        self.beginResetModel()
        self._entries = list(entries)
        self.endResetModel()

    def entries(self) -> list[LogEntry]:
        return list(self._entries)

    def entry_at(self, row: int) -> LogEntry | None:
        if 0 <= row < len(self._entries):
            return self._entries[row]
        return None

    def row_for_entry_id(self, entry_id: str) -> int:
        for row, entry in enumerate(self._entries):
            if entry.entry_id == entry_id:
                return row
        return -1

    # ------------------------------------------------------------ Qt 接口
    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self._entries)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:  # noqa: N802
        if parent.isValid():
            return 0
        return len(self.COLUMNS)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        entry = self.entry_at(index.row())
        if entry is None:
            return None
        column = index.column()
        if role == Qt.ItemDataRole.DisplayRole:
            if 0 <= column < len(self.ATTRS):
                return str(getattr(entry, self.ATTRS[column]) or "")
            return None
        if role == Qt.ItemDataRole.ForegroundRole:
            if column == self.LEVEL_COLUMN:
                color = _level_foreground(entry.level)
                if color is not None:
                    return QBrush(color)
            return None
        if role == Qt.ItemDataRole.ToolTipRole:
            if column == 0:
                return entry.entry_id
            if column == 1:
                return entry.created_at
            if column == 2:
                return f"{entry.level}（{_LEVEL_LABELS.get(entry.level, entry.level)}）"
            if column == 3:
                return entry.logger
            if column == 4:
                return entry.message
            if column == 5:
                return entry.source or "-"
            return None
        if role == Qt.ItemDataRole.UserRole:
            return entry
        return None

    def headerData(  # noqa: N802 - Qt 接口
        self,
        section: int,
        orientation: Qt.Orientation,
        role: int = Qt.ItemDataRole.DisplayRole,
    ):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal:
            if 0 <= section < len(self.COLUMNS):
                return self.COLUMNS[section]
            return None
        return str(section + 1)

    def flags(self, index: QModelIndex) -> Qt.ItemFlags:
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable


class CopyableTableView(QTableView):
    """支持 Ctrl+C 与右键复制菜单的日志表格。

    文本复制策略：选中区域按行、列顺序拼接为制表符分隔的文本；整行选择
    时得到完整的 6 列；复制完整详细信息通过 detail_provider 回调获取。
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.detail_provider: Callable[[], str] | None = None
        self.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setSortingEnabled(False)
        self.setShowGrid(False)
        self.setAlternatingRowColors(True)
        self.setWordWrap(False)
        self.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.setHorizontalScrollMode(
            QAbstractItemView.ScrollMode.ScrollPerPixel
        )
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerItem)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.DefaultContextMenu)

    # ------------------------------------------------------------ 文本
    @staticmethod
    def _index_text(index: QModelIndex) -> str:
        if not index.isValid():
            return ""
        value = index.data(Qt.ItemDataRole.DisplayRole)
        return "" if value is None else str(value)

    def current_row(self) -> int:
        index = self.currentIndex()
        if index.isValid():
            return index.row()
        selection = self.selectionModel()
        rows = selection.selectedRows() if selection is not None else []
        return rows[0].row() if rows else -1

    def selection_text(self) -> str:
        selection = self.selectionModel()
        if selection is None:
            return ""
        indexes = sorted(
            selection.selectedIndexes(),
            key=lambda item: (item.row(), item.column()),
        )
        if not indexes:
            return ""
        rows: dict[int, list[tuple[int, str]]] = {}
        for index in indexes:
            rows.setdefault(index.row(), []).append(
                (index.column(), self._index_text(index))
            )
        lines: list[str] = []
        for row in sorted(rows):
            cells = [text for _column, text in sorted(rows[row])]
            lines.append("\t".join(cells))
        return "\n".join(lines)

    def current_row_text(self) -> str:
        row = self.current_row()
        model = self.model()
        if model is None or row < 0:
            return ""
        cells = [
            self._index_text(model.index(row, column))
            for column in range(model.columnCount())
        ]
        return "\t".join(cells)

    def detail_text(self) -> str:
        provider = self.detail_provider
        if provider is None:
            return ""
        try:
            return str(provider() or "")
        except RuntimeError:
            # 控件（或对话框）已销毁，静默忽略迟到的复制请求。
            return ""

    # ------------------------------------------------------------ 复制
    @staticmethod
    def _set_clipboard(text: str) -> None:
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(text)

    def copy_selection(self) -> str:
        text = self.selection_text()
        if text:
            self._set_clipboard(text)
        return text

    def copy_current_cell(self) -> str:
        text = self._index_text(self.currentIndex())
        if text:
            self._set_clipboard(text)
        return text

    def copy_current_row(self) -> str:
        """复制当前行（整行、制表符分隔），返回复制的文本。"""

        text = self.current_row_text()
        if text:
            self._set_clipboard(text)
        return text

    def copy_current_entry_id(self) -> str:
        model = self.model()
        row = self.current_row()
        if model is None or row < 0:
            return ""
        text = self._index_text(model.index(row, 0))
        if text:
            self._set_clipboard(text)
        return text

    def copy_detail(self) -> str:
        text = self.detail_text()
        if text:
            self._set_clipboard(text)
        return text

    # ------------------------------------------------------------ 菜单
    def build_context_menu(self) -> QMenu:
        """构造右键菜单；无选中内容时对应动作禁用。"""

        menu = QMenu(self)
        selection = self.selectionModel()
        has_selection = bool(
            selection is not None and selection.selectedIndexes()
        )
        row = self.current_row()

        self.copy_cell_action = menu.addAction("复制单元格")
        self.copy_cell_action.setEnabled(has_selection)
        self.copy_cell_action.triggered.connect(self.copy_current_cell)

        self.copy_row_action = menu.addAction("复制整行")
        self.copy_row_action.setEnabled(row >= 0)
        self.copy_row_action.triggered.connect(self.copy_current_row)

        self.copy_id_action = menu.addAction("复制日志 ID")
        self.copy_id_action.setEnabled(row >= 0)
        self.copy_id_action.triggered.connect(self.copy_current_entry_id)

        self.copy_detail_action = menu.addAction("复制详细信息")
        self.copy_detail_action.setEnabled(bool(self.detail_text()))
        self.copy_detail_action.triggered.connect(self.copy_detail)
        return menu

    def contextMenuEvent(self, event: QContextMenuEvent) -> None:  # noqa: N802
        menu = self.build_context_menu()
        try:
            menu.exec(event.globalPos())
        finally:
            menu.deleteLater()

    # ------------------------------------------------------------ 键盘
    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802
        if event.matches(QKeySequence.StandardKey.Copy):
            if self.copy_selection():
                event.accept()
                return
        super().keyPressEvent(event)


class LogDialog(QDialog):
    """应用日志查看器；日志仓库由调用方持有，对话框只负责展示与导出。"""

    def __init__(self, store: LogStore, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.store = store
        self._last_count = 0
        self._last_size = 0
        self._last_refresh = ""
        self._status_snapshot = ""

        self.setWindowTitle("应用日志")
        self.setMinimumSize(860, 560)
        self.resize(1120, 720)
        self.setSizeGripEnabled(True)

        # 定时器都以对话框为父对象；对话框销毁时自动销毁，避免命中已删除控件。
        self._status_timer = QTimer(self)
        self._status_timer.setSingleShot(True)
        self._status_timer.setInterval(_STATUS_FLASH_MS)
        self._status_timer.timeout.connect(self._restore_status)

        self._keyword_timer = QTimer(self)
        self._keyword_timer.setSingleShot(True)
        self._keyword_timer.setInterval(_KEYWORD_DEBOUNCE_MS)
        self._keyword_timer.timeout.connect(self.refresh)

        self._auto_timer = QTimer(self)
        self._auto_timer.setInterval(_AUTO_REFRESH_MS)
        self._auto_timer.timeout.connect(self._auto_refresh)

        self._build_ui()
        self.refresh()
        if self.auto_refresh_check.isChecked():
            self._auto_timer.start()

    # ================================================================== UI
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 12)
        layout.setSpacing(10)

        layout.addLayout(self._build_title_row())
        layout.addWidget(self._build_filter_panel())
        layout.addWidget(self._build_table(), 3)
        layout.addLayout(self._build_detail_section(), 1)

        self.status_label = QLabel(self)
        self.status_label.setObjectName("hintLabel")
        self.status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        layout.addWidget(self.status_label)

        self._refresh_shortcut = QShortcut(QKeySequence("F5"), self)
        self._refresh_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self._refresh_shortcut.activated.connect(self.refresh)

    def _build_title_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)

        title = QLabel("应用日志", self)
        title.setObjectName("sectionTitle")
        row.addWidget(title, 0)

        self.hint_label = QLabel(self)
        self.hint_label.setObjectName("hintLabel")
        self.hint_label.setWordWrap(True)
        self.hint_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.hint_label.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        row.addWidget(self.hint_label, 1)

        self.refresh_button = QPushButton("刷新", self)
        self.refresh_button.setObjectName("subtleButton")
        self.refresh_button.setToolTip("刷新日志列表（F5）")
        self.refresh_button.clicked.connect(self.refresh)
        row.addWidget(self.refresh_button, 0)

        self.export_button = QPushButton("导出 CSV…", self)
        self.export_button.setToolTip("把当前筛选条件下的全部日志导出为 CSV")
        self.export_button.clicked.connect(self._on_export)
        row.addWidget(self.export_button, 0)

        self.clear_button = QPushButton("清空日志", self)
        self.clear_button.setObjectName("dangerButton")
        self.clear_button.setToolTip("清空数据库中的全部日志（需要确认）")
        self.clear_button.clicked.connect(self._on_clear_logs)
        row.addWidget(self.clear_button, 0)

        self.close_button = QPushButton("关闭", self)
        self.close_button.setObjectName("subtleButton")
        self.close_button.clicked.connect(self.close)
        row.addWidget(self.close_button, 0)
        return row

    def _build_filter_panel(self) -> QGroupBox:
        box = QGroupBox("筛选", self)
        flow = _FlowLayout(box, hspacing=12, vspacing=8)
        flow.setContentsMargins(4, 4, 4, 4)

        # 等级 --------------------------------------------------------
        level_group = QWidget(box)
        level_layout = QHBoxLayout(level_group)
        level_layout.setContentsMargins(0, 0, 0, 0)
        level_layout.setSpacing(6)
        level_layout.addWidget(QLabel("等级：", level_group))
        self.level_checks: dict[str, QCheckBox] = {}
        for level in LOG_LEVELS:
            check = QCheckBox(_LEVEL_LABELS[level], level_group)
            check.setToolTip(level)
            check.setChecked(True)
            check.toggled.connect(self.refresh)
            self.level_checks[level] = check
            level_layout.addWidget(check)
        flow.addWidget(level_group)

        # 时间 --------------------------------------------------------
        date_group = QWidget(box)
        date_layout = QHBoxLayout(date_group)
        date_layout.setContentsMargins(0, 0, 0, 0)
        date_layout.setSpacing(6)
        date_layout.addWidget(QLabel("时间：", date_group))
        self.date_check = QCheckBox("按日期筛选", date_group)
        self.date_check.setToolTip("按整天筛选日志（含起止两天）")
        self.date_check.setChecked(True)
        date_layout.addWidget(self.date_check)
        self.start_date = QDateEdit(date_group)
        self.start_date.setCalendarPopup(True)
        self.start_date.setDisplayFormat("yyyy-MM-dd")
        self.start_date.setMinimumWidth(112)
        self.start_date.setDate(QDate.currentDate().addDays(-6))
        self.start_date.setToolTip("起始日期（含）")
        date_layout.addWidget(self.start_date)
        date_layout.addWidget(QLabel("至", date_group))
        self.end_date = QDateEdit(date_group)
        self.end_date.setCalendarPopup(True)
        self.end_date.setDisplayFormat("yyyy-MM-dd")
        self.end_date.setMinimumWidth(112)
        self.end_date.setDate(QDate.currentDate())
        self.end_date.setToolTip("结束日期（含）")
        date_layout.addWidget(self.end_date)
        self.date_check.toggled.connect(self._on_date_toggled)
        self.start_date.dateChanged.connect(lambda _value: self.refresh())
        self.end_date.dateChanged.connect(lambda _value: self.refresh())
        flow.addWidget(date_group)

        # 模块 --------------------------------------------------------
        logger_group = QWidget(box)
        logger_layout = QHBoxLayout(logger_group)
        logger_layout.setContentsMargins(0, 0, 0, 0)
        logger_layout.setSpacing(6)
        logger_layout.addWidget(QLabel("模块：", logger_group))
        self.logger_combo = QComboBox(logger_group)
        self.logger_combo.setMinimumWidth(150)
        self.logger_combo.addItem("全部模块", "")
        self.logger_combo.setToolTip("按日志模块（logger 名称）筛选")
        self.logger_combo.currentIndexChanged.connect(lambda _index: self.refresh())
        logger_layout.addWidget(self.logger_combo)
        flow.addWidget(logger_group)

        # 关键字 ------------------------------------------------------
        keyword_group = QWidget(box)
        keyword_layout = QHBoxLayout(keyword_group)
        keyword_layout.setContentsMargins(0, 0, 0, 0)
        keyword_layout.setSpacing(6)
        keyword_layout.addWidget(QLabel("关键字：", keyword_group))
        self.keyword_edit = QLineEdit(keyword_group)
        self.keyword_edit.setPlaceholderText("搜索消息 / 详细信息 / 日志 ID")
        self.keyword_edit.setClearButtonEnabled(True)
        self.keyword_edit.setMinimumWidth(240)
        self.keyword_edit.textChanged.connect(self._on_keyword_changed)
        keyword_layout.addWidget(self.keyword_edit)
        flow.addWidget(keyword_group)

        # 自动刷新 ----------------------------------------------------
        self.auto_refresh_check = QCheckBox("自动刷新", box)
        self.auto_refresh_check.setChecked(True)
        self.auto_refresh_check.setToolTip("每 3 秒刷新一次日志列表")
        self.auto_refresh_check.toggled.connect(self._on_auto_refresh_toggled)
        flow.addWidget(self.auto_refresh_check)
        return box

    def _build_table(self) -> CopyableTableView:
        self.model = LogTableModel(self)
        self.table_view = CopyableTableView(self)
        self.table_view.setModel(self.model)
        self.table_view.detail_provider = self._current_detail_text
        self.table_view.setMinimumHeight(240)

        header = self.table_view.horizontalHeader()
        header.setStretchLastSection(False)
        header.setMinimumSectionSize(60)
        for column, width in enumerate(_COLUMN_WIDTHS):
            header.setSectionResizeMode(
                column, QHeaderView.ResizeMode.Interactive
            )
            if width:
                self.table_view.setColumnWidth(column, width)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)

        self.table_view.doubleClicked.connect(self._on_row_double_clicked)
        selection = self.table_view.selectionModel()
        selection.currentRowChanged.connect(self._on_current_row_changed)
        selection.selectionChanged.connect(self._on_selection_changed)
        return self.table_view

    def _build_detail_section(self) -> QVBoxLayout:
        section = QVBoxLayout()
        section.setContentsMargins(0, 0, 0, 0)
        section.setSpacing(4)

        label = QLabel("详细信息", self)
        section.addWidget(label)

        self.detail_edit = QPlainTextEdit(self)
        self.detail_edit.setReadOnly(True)
        self.detail_edit.setPlainText(_EMPTY_DETAIL)
        self.detail_edit.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        self.detail_edit.setMinimumHeight(180)
        self.detail_edit.setFont(_monospace_font())
        self.detail_edit.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        section.addWidget(self.detail_edit, 1)
        return section

    # ============================================================== 查询
    def _selected_levels(self) -> tuple[str, ...]:
        return tuple(
            level for level in LOG_LEVELS if self.level_checks[level].isChecked()
        )

    def build_filter(self, limit: int | None = None) -> LogFilter:
        """按当前筛选控件构造 LogFilter；limit=None 表示不限制条数。"""

        levels = self._selected_levels()
        logger_name = str(self.logger_combo.currentData() or "")
        keyword = self.keyword_edit.text()
        if self.date_check.isChecked():
            start: date = self.start_date.date().toPython()
            end: date = self.end_date.date().toPython()
            return LogFilter.by_dates(
                start,
                end,
                levels=levels,
                keyword=keyword,
                logger=logger_name,
                limit=limit,
            )
        return LogFilter(
            levels=levels,
            keyword=keyword,
            logger=logger_name,
            limit=limit,
        )

    def refresh(self) -> None:
        """按当前筛选条件重新查询并刷新表格、详情与状态栏。"""

        self._last_refresh = datetime.now().strftime("%H:%M:%S")
        self._stop_status_flash()
        self._refresh_loggers()
        self._refresh_database_stats()

        # 全部等级都不勾选是合法状态：结果为空，但不能把空 levels 传给
        # LogStore（空元组表示“不限制等级”，会查出全部日志）。
        if not self._selected_levels():
            self.model.set_entries([])
            self._show_detail_text(_EMPTY_DETAIL)
            self._update_status(0)
            return

        selected_entry_id = self._selected_entry_id()
        scroll_value = self.table_view.verticalScrollBar().value()
        try:
            entries = self.store.query(self.build_filter(limit=None))
        except (sqlite3.Error, OSError) as exc:
            self.model.set_entries([])
            self._show_detail_text(_EMPTY_DETAIL)
            self._last_count = 0
            self.status_label.setText(f"读取日志失败：{_describe_error(exc)}")
            return

        self.model.set_entries(entries)
        self._restore_selection(selected_entry_id, scroll_value)
        self._update_status(len(entries))

    def _refresh_loggers(self) -> None:
        current = str(self.logger_combo.currentData() or "")
        try:
            loggers = self.store.loggers()
        except (sqlite3.Error, OSError):
            return
        previous = self.logger_combo.blockSignals(True)
        try:
            self.logger_combo.clear()
            self.logger_combo.addItem("全部模块", "")
            for name in loggers:
                if name:
                    self.logger_combo.addItem(name, name)
            index = self.logger_combo.findData(current) if current else 0
            self.logger_combo.setCurrentIndex(index if index >= 0 else 0)
        finally:
            self.logger_combo.blockSignals(previous)

    def _refresh_database_stats(self) -> None:
        try:
            stats = self.store.stats()
            total = int(stats.get("total", 0) or 0)
            size = int(stats.get("size_bytes", 0) or 0)
        except (sqlite3.Error, OSError):
            total = self.model.rowCount()
            size = 0
        self._last_size = size
        self.hint_label.setText(
            f"日志保存在 {self.store.path} · 共 {total} 条 · "
            f"数据库 {_format_size(size)}"
        )
        self.hint_label.setToolTip(str(self.store.path))

    def _update_status(self, count: int) -> None:
        self._last_count = max(0, int(count))
        prefix = (
            "没有符合条件的日志"
            if self._last_count == 0
            else f"共 {self._last_count} 条符合条件的日志"
        )
        self.status_label.setText(
            f"{prefix} · 数据库 {_format_size(self._last_size)} · "
            f"最后刷新 {self._last_refresh}"
        )

    # ============================================================== 选中
    def _selected_entry_id(self) -> str:
        index = self.table_view.currentIndex()
        if not index.isValid():
            selection = self.table_view.selectionModel()
            rows = selection.selectedRows() if selection is not None else []
            if not rows:
                return ""
            index = rows[0]
        entry = self.model.entry_at(index.row())
        return entry.entry_id if entry is not None else ""

    def _restore_selection(self, entry_id: str, scroll_value: int) -> None:
        row = self.model.row_for_entry_id(entry_id) if entry_id else -1
        if row >= 0:
            self.table_view.selectRow(row)
            self.table_view.setCurrentIndex(self.model.index(row, 0))
        else:
            selection = self.table_view.selectionModel()
            if selection is not None:
                selection.clearSelection()
                selection.setCurrentIndex(
                    QModelIndex(), selection.SelectionFlag.NoUpdate
                )
        self._sync_detail()
        bar = self.table_view.verticalScrollBar()
        if scroll_value > 0:
            bar.setValue(min(scroll_value, bar.maximum()))

    def _sync_detail(self) -> None:
        row = self.table_view.current_row()
        entry = self.model.entry_at(row) if row >= 0 else None
        text = entry.full_text() if entry is not None else _EMPTY_DETAIL
        self._show_detail_text(text)

    def _show_detail_text(self, text: str) -> None:
        # 内容没变时不重设文本，避免自动刷新打断用户在详细信息里的选择。
        if self.detail_edit.toPlainText() != text:
            self.detail_edit.setPlainText(text)

    def _current_detail_text(self) -> str:
        row = self.table_view.current_row()
        entry = self.model.entry_at(row) if row >= 0 else None
        return entry.full_text() if entry is not None else ""

    def _on_current_row_changed(
        self, current: QModelIndex, previous: QModelIndex
    ) -> None:
        self._sync_detail()

    def _on_selection_changed(
        self, selected: QItemSelection, deselected: QItemSelection
    ) -> None:
        self._sync_detail()

    # ============================================================== 交互
    def _on_date_toggled(self, checked: bool) -> None:
        self.start_date.setEnabled(checked)
        self.end_date.setEnabled(checked)
        self.refresh()

    def _on_keyword_changed(self, _text: str) -> None:
        self._keyword_timer.start()

    def _on_auto_refresh_toggled(self, checked: bool) -> None:
        if checked:
            self._auto_timer.start()
        else:
            self._auto_timer.stop()

    def _auto_refresh(self) -> None:
        if not self.isVisible():
            return
        self.refresh()

    def _on_row_double_clicked(self, index: QModelIndex) -> None:
        entry = self.model.entry_at(index.row())
        if entry is None:
            return
        clipboard = QApplication.clipboard()
        if clipboard is not None:
            clipboard.setText(entry.full_text())
        self._flash_status(f"已复制日志 {entry.entry_id}")

    # ============================================================== 状态
    def _flash_status(self, message: str) -> None:
        self._status_timer.stop()
        self._status_snapshot = self.status_label.text()
        self.status_label.setText(message)
        self._status_timer.start()

    def _stop_status_flash(self) -> None:
        self._status_timer.stop()
        self._status_snapshot = ""

    def _restore_status(self) -> None:
        if not self._status_snapshot:
            return
        self.status_label.setText(self._status_snapshot)
        self._status_snapshot = ""

    # ============================================================== 按钮
    def _on_clear_logs(self) -> None:
        answer = QMessageBox.question(
            self,
            "清空日志",
            "确定要清空全部应用日志吗？此操作无法撤销。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self.store.clear()
        except (sqlite3.Error, OSError) as exc:
            self.status_label.setText(f"清空日志失败：{_describe_error(exc)}")
            return
        self.refresh()

    def _default_export_path(self) -> str:
        filename = f"MineImage-logs-{datetime.now():%Y%m%d-%H%M%S}.csv"
        return str(Path.home() / filename)

    def _on_export(self) -> None:
        if not self._selected_levels():
            QMessageBox.information(
                self,
                "导出 CSV",
                "当前筛选条件下没有日志，无需导出。",
            )
            return
        flt = self.build_filter(limit=None)
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出 CSV",
            self._default_export_path(),
            "CSV 文件 (*.csv);;所有文件 (*.*)",
        )
        if not path:
            return
        target = Path(path)
        if target.suffix.lower() != ".csv":
            target = target.with_name(target.name + ".csv")
        try:
            count = self.store.export_csv(target, flt)
        except (sqlite3.Error, OSError) as exc:
            QMessageBox.warning(
                self, "导出 CSV", f"导出失败：{_describe_error(exc)}"
            )
            return
        QMessageBox.information(
            self,
            "导出 CSV",
            f"已导出 {count} 条日志到：\n{target}",
        )

    # ============================================================== 生命周期
    def showEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        super().showEvent(event)
        if self.auto_refresh_check.isChecked() and not self._auto_timer.isActive():
            self._auto_timer.start()

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        self._auto_timer.stop()
        self._keyword_timer.stop()
        self._status_timer.stop()
        super().closeEvent(event)
