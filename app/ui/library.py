"""左侧算子库：行式一级分类手风琴、多级折叠树、拖拽与说明浮层。

* 一级分类是整行可折叠标题（左侧箭头 + 分类名 + 右侧数量），默认全部折叠；
* 点击某个一级分类就在该行下方原地展开，支持同时展开多个分类，展开后下方
  分类标题整体下移；二级、三级分类仍是折叠分组，可逐级展开 / 折叠；
* 搜索跨所有分类过滤，命中项自动展开并在列表中定位；
* 将算子叶子拖到画布时使用 :data:`app.core.constants.MIME_OPERATOR`，
  payload 为 ``{"id": op_id}`` 的 UTF-8 JSON；
* 单击算子叶子在库底部弹出说明浮层（名称 / 分类 / 描述 / 参数 / 端口），
  点击浮层外部或按 Esc 关闭。
"""

from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import QEvent, QMimeData, QPoint, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QDrag,
    QFont,
    QFontMetrics,
    QKeySequence,
    QPainter,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QScrollArea,
    QStackedWidget,
    QStyle,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.core.constants import MIME_OPERATOR
from app.operators.loader import OperatorEntry, OperatorRegistry
from app.operators.spec import ParamSpec

_TYPE_LABELS = {
    "int": "整数",
    "float": "浮点",
    "bool": "布尔",
    "enum": "枚举",
    "string": "文本",
    "color": "颜色",
    "path": "路径",
    "points": "点集",
    "roi": "矩形",
    "line": "直线",
}


def _format_default(param: ParamSpec) -> str:
    value = param.default_value()
    if value is None:
        return "—"
    if param.type == "bool":
        return "是" if value else "否"
    if param.type == "enum":
        return param.choice_label(value)
    if param.type == "color":
        return str(value).upper()
    if param.type == "points":
        count = len(value) if isinstance(value, (list, tuple)) else 0
        return f"{count} 个点" if count else "空"
    if param.type in ("roi", "line"):
        if value in (None, [], ()):
            return "未设置"
        return ", ".join(f"{float(item):g}" for item in value)
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _format_range(param: ParamSpec) -> str:
    parts: list[str] = []
    if param.type == "enum":
        parts.append(" / ".join(label for _value, label in param.choices))
    else:
        if param.min is not None and param.max is not None:
            parts.append(f"{param.min:g} ~ {param.max:g}")
        elif param.min is not None:
            parts.append(f"≥ {param.min:g}")
        elif param.max is not None:
            parts.append(f"≤ {param.max:g}")
    if param.odd:
        parts.append("奇数")
    if param.type == "points" and param.count:
        parts.append(f"需要 {param.count} 个点")
    if param.interactive:
        parts.append("可交互拾取")
    if param.unit:
        parts.append(f"单位：{param.unit}")
    return " · ".join(parts) if parts else "—"


def _port_text(ports, label: str) -> str:
    if not ports:
        return f"{label}：无"
    items = []
    for port in ports:
        name = f"{port.label}（{port.key}）"
        detail = port.type_text()
        optional = " · 可选" if port.optional else ""
        items.append(f"{name} · {detail}{optional}")
    return f"{label}：" + "；".join(items)


def _is_group_operator(operator: object) -> bool:
    """判断算子是否为算子组（GroupOperator）。

    优先使用 app.operators.group.is_group_operator；该模块异常或尚未就绪时
    按 group_definition 属性兜底，避免说明浮层因导入问题崩溃。
    """

    try:
        from app.operators.group import is_group_operator as checker
    except Exception:  # noqa: BLE001 - 延迟导入，帧模块可能尚未就绪
        checker = None
    if checker is not None:
        try:
            return bool(checker(operator))
        except Exception:  # noqa: BLE001 - 判定失败时退回属性检查
            pass
    return getattr(operator, "group_definition", None) is not None


class _OperatorTree(QTreeWidget):
    """算子库主树：一级分类为整行折叠标题，叶子可拖拽。

    * 第一列：分类 / 分组 / 算子名；第二列：分类或分组的算子数量；
    * 非叶子条目单击整行即可原地展开 / 折叠（箭头仍由 Qt 原生处理）；
    * 只有算子叶子允许拖拽，payload 为 ``MIME_OPERATOR``。
    """

    def __init__(self, library: "OperatorLibrary", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._library = library
        self.setObjectName("operatorTree")
        self.setHeaderHidden(True)
        self.setColumnCount(2)
        self.setRootIsDecorated(True)
        self.setIndentation(18)
        self.setAnimated(True)
        self.setUniformRowHeights(False)
        self.setExpandsOnDoubleClick(False)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.setColumnWidth(1, 52)
        self.setMouseTracking(True)
        self._hover_item: QTreeWidgetItem | None = None
        self.itemClicked.connect(self._on_item_clicked)
        self.itemActivated.connect(self._on_item_activated)

    # ------------------------------------------------------------------ 绘制
    def _row_background(self, item: QTreeWidgetItem | None) -> QColor | None:
        """整行底色：一级分类按展开状态，其余按选中 / 悬停。"""

        if item is None:
            return None
        if item.parent() is None:
            return QColor("#eaf2ff") if item.isExpanded() else QColor("#f8fafc")
        if item.isSelected():
            return QColor("#dbeafe")
        if item is self._hover_item:
            return QColor("#eef4ff")
        return None

    def drawRow(  # noqa: N802 - Qt 命名
        self,
        painter: QPainter,
        option: QStyleOptionViewItem,
        index,
    ) -> None:
        """整行铺一个圆角底色，并屏蔽 Qt 自带的逐列高亮。

        Qt 默认会把「分支缩进区」和「每个单元格」分别画成带圆角的选中块，
        选中算子行时左侧会多出一个空白方框、右侧多出箭头所在单元格的外框；
        这里改为先行绘制整行底色，再让基类只画文字、图标与展开三角。
        """

        item = self.itemFromIndex(index)
        background = self._row_background(item)
        if background is not None:
            row = QRectF(
                0.0,
                float(option.rect.top()) + 1.0,
                float(self.viewport().width()),
                float(option.rect.height()) - 2.0,
            )
            painter.save()
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(background)
            painter.drawRoundedRect(row, 6.0, 6.0)
            painter.restore()

        plain = QStyleOptionViewItem(option)
        plain.state = plain.state & ~(
            QStyle.StateFlag.State_Selected
            | QStyle.StateFlag.State_MouseOver
            | QStyle.StateFlag.State_HasFocus
        )
        super().drawRow(painter, plain, index)

        # 展开的一级分类：左侧强调条。
        if item is not None and item.parent() is None and item.isExpanded():
            painter.save()
            painter.fillRect(
                QRectF(
                    0.0,
                    float(option.rect.top() + 4),
                    3.0,
                    float(max(0, option.rect.height() - 8)),
                ),
                QColor("#2563eb"),
            )
            painter.restore()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        item = self.itemAt(event.position().toPoint())
        if item is not self._hover_item:
            self._hover_item = item
            self.viewport().update()
        super().mouseMoveEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if self._hover_item is not None:
            self._hover_item = None
            self.viewport().update()
        super().leaveEvent(event)

    def startDrag(self, supported_actions) -> None:  # noqa: N802 - Qt 命名
        item = self.currentItem()
        if item is None:
            return
        op_id = item.data(0, Qt.ItemDataRole.UserRole)
        if not op_id:
            return
        self._library.begin_operator_drag(self, str(op_id))

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        op_id = item.data(0, Qt.ItemDataRole.UserRole)
        if op_id:
            self._library.show_operator_info(str(op_id))
            self._library.note_leaf_clicked(item)
            return
        # 一级分类 / 二级分组：单击整行原地展开或折叠（多开，不关闭其它分类）。
        item.setExpanded(not item.isExpanded())

    def _on_item_activated(self, item: QTreeWidgetItem, _column: int) -> None:
        op_id = item.data(0, Qt.ItemDataRole.UserRole)
        if op_id:
            self._library.show_operator_info(str(op_id))


class OperatorLibrary(QWidget):
    """算子库面板。"""

    refresh_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(220)
        self._registry: OperatorRegistry | None = None
        self._category_items: dict[str, QTreeWidgetItem] = {}
        self._entry_items: dict[str, QTreeWidgetItem] = {}
        self._current_category: str | None = None
        self._match_count = 0
        self._filter_text = ""
        self._info_filter_installed = False
        self._build_ui()
        self.tree_bar.setVisible(False)
        self._content_stack.setCurrentWidget(self.empty_page)
        self._update_tree_label()
        self._update_footer()

    # ------------------------------------------------------------------ 构建
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        header = QHBoxLayout()
        header.setSpacing(6)
        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText("搜索算子（名称 / 描述 / 关键字）")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setToolTip("跨所有分类搜索算子；命中项会自动展开并在列表中定位。")
        self.search_edit.textChanged.connect(self._on_search_changed)
        # 刷新按钮与工具栏一致：只显示图标，不显示文字。
        self.refresh_button = QToolButton(self)
        self.refresh_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_BrowserReload)
        )
        self.refresh_button.setIconSize(QSize(16, 16))
        self.refresh_button.setAccessibleName("刷新算子库")
        self.refresh_button.setToolTip("重新扫描算子目录（F5）")
        self.refresh_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_button.clicked.connect(self._on_refresh_clicked)
        header.addWidget(self.search_edit, 1)
        header.addWidget(self.refresh_button)
        layout.addLayout(header)

        # 一级分类 + 多级分组：单棵树内整行折叠，支持同时展开多个分类。
        self.tree_bar = QWidget(self)
        tree_bar_layout = QHBoxLayout(self.tree_bar)
        tree_bar_layout.setContentsMargins(2, 0, 2, 0)
        tree_bar_layout.setSpacing(4)
        self.category_label = QLabel("尚未加载算子", self.tree_bar)
        self.category_label.setObjectName("hintLabel")
        self.expand_all_button = QToolButton(self.tree_bar)
        self.expand_all_button.setObjectName("treeAction")
        self.expand_all_button.setText("展开全部")
        self.expand_all_button.setToolTip("展开所有分类和分组")
        self.expand_all_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.expand_all_button.clicked.connect(self._on_expand_all)
        self.collapse_all_button = QToolButton(self.tree_bar)
        self.collapse_all_button.setObjectName("treeAction")
        self.collapse_all_button.setText("折叠全部")
        self.collapse_all_button.setToolTip("折叠所有分类和分组")
        self.collapse_all_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.collapse_all_button.clicked.connect(self._on_collapse_all)
        tree_bar_layout.addWidget(self.category_label, 1)
        tree_bar_layout.addWidget(self.expand_all_button)
        tree_bar_layout.addWidget(self.collapse_all_button)
        layout.addWidget(self.tree_bar)

        self._content_stack = QStackedWidget(self)
        self.tree = _OperatorTree(self)
        self.tree.itemExpanded.connect(self._on_tree_item_expanded)
        self.tree.itemCollapsed.connect(self._on_tree_item_collapsed)

        self.empty_page = QWidget(self)
        empty_layout = QVBoxLayout(self.empty_page)
        empty_layout.setContentsMargins(16, 24, 16, 16)
        empty_layout.addStretch(1)
        empty_title = QLabel("尚未加载算子", self.empty_page)
        empty_title.setObjectName("sectionTitle")
        empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_hint = QLabel(
            "请检查「设置 → 算子目录」，\n然后点击右上角的刷新按钮重新扫描。",
            self.empty_page,
        )
        empty_hint.setObjectName("hintLabel")
        empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_hint.setWordWrap(True)
        empty_layout.addWidget(empty_title)
        empty_layout.addWidget(empty_hint)
        empty_layout.addStretch(2)

        self.filter_empty_page = QWidget(self)
        filter_empty_layout = QVBoxLayout(self.filter_empty_page)
        filter_empty_layout.setContentsMargins(16, 24, 16, 16)
        filter_empty_layout.addStretch(1)
        filter_empty_hint = QLabel(
            "未找到匹配的算子\n试试其它关键字。",
            self.filter_empty_page,
        )
        filter_empty_hint.setObjectName("hintLabel")
        filter_empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        filter_empty_hint.setWordWrap(True)
        filter_empty_layout.addWidget(filter_empty_hint)
        filter_empty_layout.addStretch(2)

        self._content_stack.addWidget(self.tree)
        self._content_stack.addWidget(self.empty_page)
        self._content_stack.addWidget(self.filter_empty_page)
        layout.addWidget(self._content_stack, 1)

        footer = QHBoxLayout()
        footer.setSpacing(8)
        self.count_label = QLabel(self)
        self.count_label.setObjectName("hintLabel")
        self.error_label = QLabel(self)
        self.error_label.setObjectName("hintLabel")
        self.error_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        footer.addWidget(self.count_label, 1)
        footer.addWidget(self.error_label)
        layout.addLayout(footer)

        self._build_info_panel()

        shortcut = QShortcut(QKeySequence("F5"), self)
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(self._on_refresh_clicked)

    def _build_info_panel(self) -> None:
        self.info_panel = QFrame(self)
        self.info_panel.setObjectName("libraryInfoPanel")
        self.info_panel.setVisible(False)
        self.info_panel.setMinimumHeight(160)
        panel_layout = QVBoxLayout(self.info_panel)
        panel_layout.setContentsMargins(12, 10, 12, 10)
        panel_layout.setSpacing(6)

        title_row = QHBoxLayout()
        title_row.setSpacing(8)
        self.info_name = QLabel("", self.info_panel)
        self.info_name.setObjectName("sectionTitle")
        self.info_name.setWordWrap(True)
        self.info_meta = QLabel("", self.info_panel)
        self.info_meta.setObjectName("hintLabel")
        self.info_meta.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        close_button = QToolButton(self.info_panel)
        close_button.setText("✕")
        close_button.setToolTip("关闭说明（Esc）")
        close_button.clicked.connect(self._hide_info_panel)
        title_row.addWidget(self.info_name, 0)
        title_row.addWidget(self.info_meta, 1)
        title_row.addWidget(close_button, 0)
        panel_layout.addLayout(title_row)

        self.info_desc = QLabel("", self.info_panel)
        self.info_desc.setWordWrap(True)
        self.info_desc.setObjectName("hintLabel")
        panel_layout.addWidget(self.info_desc)

        scroll = QScrollArea(self.info_panel)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        content = QWidget(scroll)
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(6)

        self.info_params_title = QLabel("参数", content)
        font = QFont(self.font())
        font.setBold(True)
        self.info_params_title.setFont(font)
        self.info_table = QTableWidget(0, 3, content)
        self.info_table.setHorizontalHeaderLabels(["参数", "类型", "默认值 / 范围"])
        self.info_table.verticalHeader().setVisible(False)
        header = self.info_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.info_table.setColumnWidth(0, 92)
        self.info_table.setColumnWidth(1, 52)
        self.info_table.setMinimumWidth(260)
        self.info_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.info_table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self.info_table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self.info_table.setShowGrid(False)
        self.info_table.setWordWrap(False)
        self.info_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.info_no_params = QLabel("该算子没有可配置参数。", content)
        self.info_no_params.setObjectName("hintLabel")
        self.info_ports = QLabel("", content)
        self.info_ports.setWordWrap(True)
        self.info_ports.setObjectName("hintLabel")
        content_layout.addWidget(self.info_params_title)
        content_layout.addWidget(self.info_table)
        content_layout.addWidget(self.info_no_params)
        content_layout.addWidget(self.info_ports)
        content_layout.addStretch(1)
        scroll.setWidget(content)
        panel_layout.addWidget(scroll, 1)
        self.info_panel.hide()

    # ------------------------------------------------------------------ 数据
    def set_registry(self, registry: OperatorRegistry | None) -> None:
        """重建一级分类树（保留已展开分类与搜索状态）。"""
        previous_expanded = set(self._expanded_category_names())
        previous_current = self._current_category
        self._registry = registry
        self._hide_info_panel()
        self.setUpdatesEnabled(False)
        try:
            self.tree.clear()
            self._category_items.clear()
            self._entry_items.clear()
            self._current_category = None
            entries = registry.entries if registry is not None else {}
            if not entries or registry is None:
                self.tree_bar.setVisible(False)
                self._content_stack.setCurrentWidget(self.empty_page)
                self._update_footer()
                return
            self._build_tree(self.tree, registry.tabs())
            self.tree_bar.setVisible(True)
            self._content_stack.setCurrentWidget(self.tree)
            if not self._filter_text:
                expanded = previous_expanded & set(self._category_items)
                for name in self._category_items:
                    if name in expanded:
                        self._category_items[name].setExpanded(True)
                if previous_current in expanded:
                    self._current_category = previous_current
                elif expanded:
                    self._current_category = next(
                        name for name in self._category_items if name in expanded
                    )
            self._refresh_category_styles()
            self._update_tree_label()
            self._apply_filter()
        finally:
            self.setUpdatesEnabled(True)

    # ------------------------------------------------------------------ 分类
    def category_names(self) -> list[str]:
        """按展示顺序返回全部一级分类名。"""
        return list(self._category_items.keys())

    def category_items(self) -> list[QTreeWidgetItem]:
        """返回全部一级分类条目（与 :meth:`category_names` 同序）。"""
        return list(self._category_items.values())

    def category_buttons(self) -> list[QTreeWidgetItem]:
        """兼容旧接口：返回一级分类条目（当前实现没有独立按钮）。"""
        return self.category_items()

    def current_category(self) -> str | None:
        return self._current_category

    def category_tree(self, name: str) -> QTreeWidget | None:
        if name not in self._category_items:
            return None
        return self.tree

    @staticmethod
    def _top_category(item: QTreeWidgetItem) -> QTreeWidgetItem:
        current = item
        while current.parent() is not None:
            current = current.parent()
        return current

    def _expanded_category_names(self) -> list[str]:
        return [
            name
            for name, item in self._category_items.items()
            if item.isExpanded()
        ]

    def select_category(self, name: str) -> bool:
        """展开指定一级分类并滚动到它；分类不存在时返回 False。"""
        item = self._category_items.get(name)
        if item is None:
            return False
        self._hide_info_panel()
        item.setExpanded(True)
        self.tree.setCurrentItem(item)
        self.tree.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtTop)
        self._current_category = name
        self._refresh_category_styles()
        self._update_tree_label()
        return True

    def toggle_category(self, name: str) -> bool:
        """原地展开 / 折叠一个一级分类；其它已展开分类保持不变（多开）。"""
        item = self._category_items.get(name)
        if item is None:
            return False
        item.setExpanded(not item.isExpanded())
        if item.isExpanded():
            self.tree.setCurrentItem(item)
            self.tree.scrollToItem(item, QAbstractItemView.ScrollHint.EnsureVisible)
        return True

    def note_leaf_clicked(self, item: QTreeWidgetItem) -> None:
        """点击算子叶子时把其所属一级分类记为当前分类。"""

        top = self._top_category(item)
        self._current_category = top.text(0)
        self._update_tree_label()

    def _on_tree_item_expanded(self, item: QTreeWidgetItem) -> None:
        if item.parent() is None:
            self._current_category = item.text(0)
            # 展开一级分类时把其下所有分组也展开，让算子列表直接可见；
            # 用户仍可单独折叠任意二级 / 三级分组。
            self.tree.blockSignals(True)
            try:
                self._expand_descendants(item)
            finally:
                self.tree.blockSignals(False)
        self._refresh_category_styles()
        self._update_tree_label()

    @classmethod
    def _expand_descendants(cls, item: QTreeWidgetItem) -> None:
        for index in range(item.childCount()):
            child = item.child(index)
            if child.childCount():
                child.setExpanded(True)
                cls._expand_descendants(child)

    def _on_tree_item_collapsed(self, item: QTreeWidgetItem) -> None:
        if item.parent() is None and self._current_category == item.text(0):
            expanded = self._expanded_category_names()
            self._current_category = expanded[-1] if expanded else None
        self._refresh_category_styles()
        self._update_tree_label()

    def _refresh_category_styles(self) -> None:
        """一级分类行：展开时浅蓝底 + 强调文字，收起时浅灰底。"""

        for name, item in self._category_items.items():
            active = item.isExpanded()
            background = QColor("#eaf2ff") if active else QColor("#f8fafc")
            title_color = QColor("#1e3a8a") if active else QColor("#334155")
            count_color = QColor("#2563eb") if active else QColor("#64748b")
            for column in (0, 1):
                item.setBackground(column, QBrush(background))
            item.setForeground(0, QBrush(title_color))
            item.setForeground(1, QBrush(count_color))

    def _update_tree_label(self) -> None:
        if not self._category_items:
            self.category_label.setText("尚未加载算子")
            self.expand_all_button.setEnabled(False)
            self.collapse_all_button.setEnabled(False)
            return
        self.expand_all_button.setEnabled(True)
        self.collapse_all_button.setEnabled(True)
        expanded = self._expanded_category_names()
        total = sum(int(item.text(1) or 0) for item in self._category_items.values())
        self.category_label.setToolTip(
            f"{len(self._category_items)} 个一级分类，共 {total} 个算子；点击分类原地展开，可同时展开多个。"
        )
        if self._filter_text:
            self.category_label.setText(f"匹配 {self._match_count} 个")
        elif not expanded:
            self.category_label.setText("全部折叠")
        elif len(expanded) == len(self._category_items):
            self.category_label.setText("全部展开")
        elif len(expanded) == 1:
            # 数量已在分类行右侧显示，窄栏下只放分类名，避免被截断。
            self.category_label.setText(expanded[0])
        else:
            self.category_label.setText(f"已展开 {len(expanded)} 个")

    def _on_expand_all(self) -> None:
        if not self._category_items:
            return
        self.tree.blockSignals(True)
        self.tree.expandAll()
        self.tree.blockSignals(False)
        self._current_category = None
        self._refresh_category_styles()
        self._update_tree_label()

    def _on_collapse_all(self) -> None:
        if not self._category_items:
            return
        self.tree.blockSignals(True)
        self.tree.collapseAll()
        self.tree.blockSignals(False)
        self.tree.clearSelection()
        self._current_category = None
        self._refresh_category_styles()
        self._update_tree_label()

    def _build_tree(
        self,
        tree: QTreeWidget,
        tabs: dict[str, list[OperatorEntry]],
    ) -> None:
        tree.clear()
        # registry.tabs() 已按 CATEGORY_ORDER 排序（「用户自定义算子」固定第一），
        # 这里直接按给定顺序建树，避免二次排序破坏分类契约。
        for name, entries in tabs.items():
            if not entries:
                continue
            root: dict[str, Any] = {"groups": {}, "entries": []}
            for entry in entries:
                node = root
                for part in entry.subgroups:
                    node = node["groups"].setdefault(part, {"groups": {}, "entries": []})
                node["entries"].append(entry)

            category = QTreeWidgetItem([name, ""])
            category.setData(0, Qt.ItemDataRole.UserRole, None)
            category.setData(0, Qt.ItemDataRole.UserRole + 2, "category")
            category.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            category_font = category.font(0)
            category_font.setBold(True)
            category_font.setPointSizeF(max(10.0, category_font.pointSizeF()))
            category.setFont(0, category_font)
            category.setData(
                1,
                Qt.ItemDataRole.TextAlignmentRole,
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            )
            category.setSizeHint(0, QSize(0, 36))
            category.setToolTip(0, f"{name} · 单击展开 / 折叠")
            tree.addTopLevelItem(category)
            total = self._populate_tree(category, root, ())
            category.setText(1, str(total))
            category.setToolTip(1, f"{total} 个算子")
            self._category_items[name] = category
        tree.collapseAll()

    def _populate_tree(
        self,
        parent: QTreeWidgetItem,
        node: dict[str, Any],
        path: tuple[str, ...],
    ) -> int:
        total = 0
        for group_name, child in node["groups"].items():
            group = QTreeWidgetItem([group_name, ""])
            group_path = path + (group_name,)
            group.setData(0, Qt.ItemDataRole.UserRole, None)
            group.setData(0, Qt.ItemDataRole.UserRole + 2, "group")
            group.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
            group_font = group.font(0)
            group_font.setBold(True)
            group.setFont(0, group_font)
            group.setForeground(0, QBrush(QColor("#475569")))
            group.setData(
                1,
                Qt.ItemDataRole.TextAlignmentRole,
                int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            )
            group.setForeground(1, QBrush(QColor("#94a3b8")))
            group.setSizeHint(0, QSize(0, 28))
            group.setToolTip(0, "分类：" + " / ".join(group_path))
            parent.addChild(group)
            count = self._populate_tree(group, child, group_path)
            group.setText(1, str(count))
            group.setToolTip(1, f"{count} 个算子")
            total += count

        for entry in node["entries"]:
            item = QTreeWidgetItem([entry.name, ""])
            category_text = " / ".join(entry.spec.category) or "未分类"
            haystack = " ".join(
                [
                    entry.name,
                    entry.spec.id,
                    entry.spec.description or "",
                    category_text,
                    " ".join(str(keyword) for keyword in entry.spec.keywords),
                ]
            ).lower()
            item.setData(0, Qt.ItemDataRole.UserRole, entry.id)
            item.setData(0, Qt.ItemDataRole.UserRole + 1, haystack)
            tooltip_lines = [entry.name, f"分类：{category_text}"]
            if _is_group_operator(entry.operator):
                tooltip_lines.append("类型：算子组")
            tooltip_lines.append(entry.spec.description or "暂无描述")
            item.setToolTip(0, "\n".join(tooltip_lines))
            item.setFlags(
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsDragEnabled
            )
            item.setSizeHint(0, QSize(0, 26))
            parent.addChild(item)
            self._entry_items[entry.id] = item
            total += 1
        return total

    # ------------------------------------------------------------------ 过滤
    def _on_search_changed(self, _text: str) -> None:
        self._hide_info_panel()
        self._apply_filter()

    def _apply_filter(self) -> None:
        was_filtering = bool(self._filter_text)
        self._filter_text = self.search_edit.text().strip().lower()
        if not self._category_items:
            self._content_stack.setCurrentWidget(self.empty_page)
            self._update_footer()
            return
        if not self._filter_text:
            if was_filtering:
                self._filter_tree(self.tree, "")
                self.tree.blockSignals(True)
                self.tree.collapseAll()
                self.tree.blockSignals(False)
                self.tree.clearSelection()
                self._current_category = None
            self._content_stack.setCurrentWidget(self.tree)
            self._refresh_category_styles()
            self._update_tree_label()
            self._update_footer()
            return

        matched = self._filter_tree(self.tree, self._filter_text)
        self._match_count = matched
        if matched == 0:
            self._content_stack.setCurrentWidget(self.filter_empty_page)
            self._current_category = None
            self.tree.clearSelection()
            self._update_tree_label()
            self._update_footer(matched)
            return

        self._content_stack.setCurrentWidget(self.tree)
        self.tree.blockSignals(True)
        self.tree.expandAll()
        self.tree.blockSignals(False)
        first = next(
            (item for item in self._category_items.values() if not item.isHidden()),
            None,
        )
        if first is not None:
            self._current_category = first.text(0)
            self.tree.setCurrentItem(first)
            self.tree.scrollToItem(first, QAbstractItemView.ScrollHint.PositionAtTop)
        self._refresh_category_styles()
        self._update_tree_label()
        self._update_footer(matched)

    def _filter_tree(self, tree: QTreeWidget, text: str) -> int:
        matched = 0

        def walk(item: QTreeWidgetItem) -> bool:
            nonlocal matched
            op_id = item.data(0, Qt.ItemDataRole.UserRole)
            if op_id:
                haystack = str(
                    item.data(0, Qt.ItemDataRole.UserRole + 1) or item.text(0).lower()
                )
                hit = not text or text in haystack
                item.setHidden(not hit)
                if hit:
                    matched += 1
                return hit
            any_hit = False
            for child_index in range(item.childCount()):
                if walk(item.child(child_index)):
                    any_hit = True
            item.setHidden(not any_hit)
            return any_hit

        for index in range(tree.topLevelItemCount()):
            walk(tree.topLevelItem(index))
        return matched

    def _on_refresh_clicked(self) -> None:
        self._hide_info_panel()
        self.refresh_requested.emit()

    def _update_footer(self, matches: int | None = None) -> None:
        total = len(self._registry) if self._registry is not None else 0
        if self._filter_text and matches is not None:
            self.count_label.setText(f"共 {total} 个算子 · 匹配 {matches} 个")
        else:
            self.count_label.setText(f"共 {total} 个算子")
        issues = self._registry.issues if self._registry is not None else []
        errors = [issue for issue in issues if issue.level == "error"]
        self.error_label.setText(f"加载错误 {len(errors)} 个")
        self.error_label.setProperty("hasError", bool(errors))
        self.error_label.style().unpolish(self.error_label)
        self.error_label.style().polish(self.error_label)
        if issues:
            self.error_label.setToolTip("\n".join(issue.display() for issue in issues))
        else:
            self.error_label.setToolTip("算子目录加载正常")

    # ------------------------------------------------------------------ 说明浮层
    def show_operator_info(self, op_id: str) -> bool:
        """显示指定算子的说明浮层；返回是否找到该算子。"""
        if self._registry is None:
            return False
        entry = self._registry.entry(op_id)
        if entry is None:
            return False
        self._populate_info(entry)
        self.info_panel.show()
        self.info_panel.raise_()
        self._position_info_panel()
        app = QApplication.instance()
        if app is not None and not self._info_filter_installed:
            app.installEventFilter(self)
            self._info_filter_installed = True
        return True

    def info_panel_visible(self) -> bool:
        return self.info_panel.isVisible()

    def _populate_info(self, entry: OperatorEntry) -> None:
        spec = entry.spec
        group_operator = _is_group_operator(entry.operator)
        category = " / ".join(spec.category) or "未分类"
        source = "内置" if entry.builtin else "用户"
        meta = f"{category} · {source}"
        if group_operator:
            # 算子组与普通算子在库中的操作不同，说明浮层需要明确标识。
            meta = f"{meta} · 算子组"
        self.info_name.setText(entry.name)
        self.info_meta.setText(meta)
        tooltip = f"算子 id：{entry.id}\n文件：{entry.path}"
        if group_operator:
            tooltip = (
                "算子组节点：由算子组保存生成，拖入画布后展开为算子组与内部节点。\n"
                + tooltip
            )
        self.info_meta.setToolTip(tooltip)
        description = spec.description
        if not description:
            description = (
                "由算子组保存生成的自定义算子；拖入画布后会展开为算子组与内部节点。"
                if group_operator
                else "暂无描述"
            )
        if spec.notes:
            description = f"{description}\n{spec.notes}"
        self.info_desc.setText(description)
        keywords = "、".join(str(keyword) for keyword in spec.keywords)
        self.info_desc.setToolTip(f"关键字：{keywords}" if keywords else "")

        params = list(spec.params)
        self.info_table.setRowCount(len(params))
        for row, param in enumerate(params):
            default_text = _format_default(param)
            range_text = _format_range(param)
            detail = default_text
            if range_text and range_text not in ("—", "-") and param.type != "enum":
                detail = f"{default_text}（{range_text.replace(' ', '')}）"
            values = [
                param.label,
                _TYPE_LABELS.get(param.type, param.type),
                detail,
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                if column == 0 and param.tooltip:
                    item.setToolTip(param.tooltip)
                if column == 2:
                    item.setToolTip(f"默认值：{default_text}\n范围 / 说明：{range_text or '无'}")
                self.info_table.setItem(row, column, item)
        if params:
            self.info_table.setVisible(True)
            self.info_no_params.setVisible(False)
            height = min(168, 28 * (len(params) + 1) + 4)
            self.info_table.setMinimumHeight(height)
            self.info_table.setMaximumHeight(height)
        else:
            # 算子组的端口由算子组定义决定，没有可配置参数；不要显示空参数表。
            self.info_table.setVisible(False)
            self.info_no_params.setText(
                "算子组没有可配置参数；端口由算子组定义决定。"
                if group_operator
                else "该算子没有可配置参数。"
            )
            self.info_no_params.setVisible(True)
        ports = [
            _port_text(spec.inputs, "输入端口"),
            _port_text(spec.outputs, "输出端口"),
        ]
        self.info_ports.setText("\n".join(ports))

    def _position_info_panel(self) -> None:
        if not self.info_panel.isVisible():
            return
        width = max(240, self.width() - 16)
        height = min(360, max(170, int(self.height() * 0.58)))
        x = max(0, (self.width() - width) // 2)
        y = max(0, self.height() - height - 8)
        self.info_panel.setGeometry(x, y, width, height)

    def _hide_info_panel(self) -> None:
        if not self.info_panel.isVisible():
            return
        self.info_panel.hide()
        app = QApplication.instance()
        if app is not None and self._info_filter_installed:
            app.removeEventFilter(self)
            self._info_filter_installed = False

    def eventFilter(self, obj, event) -> bool:  # noqa: N802 - Qt 命名
        if self.info_panel.isVisible():
            event_type = event.type()
            if event_type == QEvent.Type.KeyPress and event.key() == Qt.Key.Key_Escape:
                self._hide_info_panel()
                return True
            if event_type in (
                QEvent.Type.MouseButtonPress,
                QEvent.Type.MouseButtonDblClick,
            ):
                try:
                    global_pos = event.globalPosition().toPoint()
                except AttributeError:
                    global_pos = QPoint(0, 0)
                target = QApplication.widgetAt(global_pos)
                inside = target is not None and (
                    target is self.info_panel or self.info_panel.isAncestorOf(target)
                )
                if not inside:
                    self._hide_info_panel()
                return False
        return super().eventFilter(obj, event)

    # ------------------------------------------------------------------ 拖拽
    def mime_data_for(self, op_id: str) -> QMimeData:
        """构造拖拽算子使用的 MIME 数据（便于测试与复用）。"""
        mime = QMimeData()
        payload = json.dumps({"id": op_id}, ensure_ascii=False).encode("utf-8")
        mime.setData(MIME_OPERATOR, payload)
        entry = self._registry.entry(op_id) if self._registry is not None else None
        if entry is not None:
            mime.setText(entry.name)
        return mime

    def begin_operator_drag(self, source: QWidget, op_id: str) -> None:
        """开始拖拽（使用 ``MIME_OPERATOR``）。"""
        entry = self._registry.entry(op_id) if self._registry is not None else None
        mime = self.mime_data_for(op_id)
        drag = QDrag(source)
        drag.setMimeData(mime)
        if entry is not None:
            drag.setPixmap(self._drag_pixmap(entry.name))
        drag.setHotSpot(QPoint(24, 16))
        drag.exec(Qt.DropAction.CopyAction)

    def _drag_pixmap(self, name: str) -> QPixmap:
        font = QFont(self.font())
        font.setBold(True)
        font.setPointSizeF(9.0)
        metrics = QFontMetrics(font)
        width = max(150, metrics.horizontalAdvance("＋ " + name) + 32)
        height = 34
        pixmap = QPixmap(width, height)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor("#2563eb"))
        painter.drawRoundedRect(QRectF(0.5, 0.5, width - 1.0, height - 1.0), 8.0, 8.0)
        painter.setPen(QColor("#ffffff"))
        painter.setFont(font)
        painter.drawText(
            QRectF(12.0, 0.0, width - 24.0, float(height)),
            Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            "＋ " + name,
        )
        painter.end()
        return pixmap

    # ------------------------------------------------------------------ 其他
    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        super().resizeEvent(event)
        if self.info_panel.isVisible():
            self._position_info_panel()

    def hideEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        super().hideEvent(event)
        self._hide_info_panel()

    def operator_count(self) -> int:
        return len(self._registry) if self._registry is not None else 0

    def load_error_count(self) -> int:
        if self._registry is None:
            return 0
        return len(self._registry.errors)

    def current_tree(self) -> _OperatorTree | None:
        if not self._category_items:
            return None
        return self.tree
