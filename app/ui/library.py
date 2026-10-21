"""左侧算子库：并排一级分类、嵌套折叠树、拖拽与说明浮层。

* 一级分类用横向并排的胶囊按钮（``#categoryChip``，2 列网格自动换行）展示，
  当前分类高亮，按钮文字水平且带算子数量；
* 更深层分类用 ``QTreeWidget`` 分组（支持任意层级），默认只展开一级分组，
  二级及更深分组默认折叠；当前分类提供“展开全部 / 折叠全部”；
* 搜索跨所有分类过滤，当前分类无命中时自动切换到第一个有命中的分类，
  并自动展开命中项；
* 将算子叶子拖到画布时使用 :data:`app.core.constants.MIME_OPERATOR`，
  payload 为 ``{"id": op_id}`` 的 UTF-8 JSON；
* 单击算子叶子在库底部弹出说明浮层（名称 / 分类 / 描述 / 参数 / 端口），
  点击浮层外部或按 Esc 关闭。
"""

from __future__ import annotations

import json
from typing import Any

from PySide6.QtCore import QEvent, QMimeData, QPoint, QRectF, Qt, Signal
from PySide6.QtGui import (
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
    QButtonGroup,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QScrollArea,
    QSizePolicy,
    QStackedWidget,
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
        accepts = f"[{port.accept_text()}]"
        optional = " · 可选" if port.optional else ""
        items.append(f"{name} {accepts}{optional}")
    return f"{label}：" + "；".join(items)


class _OperatorTree(QTreeWidget):
    """只允许拖拽算子叶子节点的树。"""

    def __init__(self, library: "OperatorLibrary", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._library = library
        self.setHeaderHidden(True)
        self.setRootIsDecorated(True)
        self.setIndentation(14)
        self.setAnimated(False)
        self.setUniformRowHeights(True)
        self.setExpandsOnDoubleClick(True)
        self.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.setDragEnabled(True)
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragOnly)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.NoContextMenu)
        self.itemClicked.connect(self._on_item_clicked)
        self.itemActivated.connect(self._on_item_activated)

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

    def _on_item_activated(self, item: QTreeWidgetItem, _column: int) -> None:
        op_id = item.data(0, Qt.ItemDataRole.UserRole)
        if op_id:
            self._library.show_operator_info(str(op_id))


class _CategoryPage(QWidget):
    """单个一级分类页：树 + 空状态。"""

    def __init__(
        self,
        library: "OperatorLibrary",
        category: str,
        entry_count: int,
    ) -> None:
        super().__init__(library)
        self.category = category
        self.entry_count = entry_count
        self.tree = _OperatorTree(library, self)
        self.empty = QLabel("该分类下暂无算子", self)
        self.empty.setObjectName("hintLabel")
        self.empty.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty.setWordWrap(True)
        self.stack = QStackedWidget(self)
        self.stack.addWidget(self.tree)
        self.stack.addWidget(self.empty)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 6, 4, 4)
        layout.addWidget(self.stack)


class OperatorLibrary(QWidget):
    """算子库面板。"""

    refresh_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMinimumWidth(220)
        self._registry: OperatorRegistry | None = None
        self._pages: dict[str, _CategoryPage] = {}
        self._current_category: str | None = None
        self._category_buttons: list[QToolButton] = []
        self._category_button_map: dict[str, QToolButton] = {}
        self._filter_text = ""
        self._info_filter_installed = False
        self._build_ui()
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
        self.search_edit.setToolTip("跨所有分类搜索算子；有命中时自动切换到匹配分类。")
        self.search_edit.textChanged.connect(self._on_search_changed)
        self.refresh_button = QToolButton(self)
        self.refresh_button.setText("刷新")
        self.refresh_button.setToolTip("重新扫描算子目录（F5）")
        self.refresh_button.clicked.connect(self._on_refresh_clicked)
        header.addWidget(self.search_edit, 1)
        header.addWidget(self.refresh_button)
        layout.addLayout(header)

        # 一级分类：2 列胶囊按钮，横向并排、自动换行（可滚动）。
        self._category_scroll = QScrollArea(self)
        self._category_scroll.setObjectName("categoryBar")
        self._category_scroll.setWidgetResizable(True)
        self._category_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._category_scroll.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        self._category_scroll.setVerticalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAsNeeded
        )
        self._category_scroll.setMaximumHeight(140)
        self._category_scroll.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        bar_content = QWidget(self._category_scroll)
        bar_content.setObjectName("categoryBarContent")
        self._category_bar_content = bar_content
        self._category_grid = QGridLayout(bar_content)
        self._category_grid.setContentsMargins(0, 0, 0, 0)
        self._category_grid.setHorizontalSpacing(6)
        self._category_grid.setVerticalSpacing(6)
        self._category_grid.setColumnStretch(0, 1)
        self._category_grid.setColumnStretch(1, 1)
        self._category_scroll.setWidget(bar_content)
        self._category_group = QButtonGroup(self)
        self._category_group.setExclusive(True)
        layout.addWidget(self._category_scroll)

        # 当前分类的工具条：分类名 + 展开 / 折叠全部。
        self.tree_bar = QWidget(self)
        tree_bar_layout = QHBoxLayout(self.tree_bar)
        tree_bar_layout.setContentsMargins(2, 0, 2, 0)
        tree_bar_layout.setSpacing(4)
        self.category_label = QLabel("", self.tree_bar)
        self.category_label.setObjectName("hintLabel")
        self.expand_all_button = QToolButton(self.tree_bar)
        self.expand_all_button.setObjectName("treeAction")
        self.expand_all_button.setText("展开全部")
        self.expand_all_button.setToolTip("展开当前分类的全部层级")
        self.expand_all_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.expand_all_button.clicked.connect(self._on_expand_all)
        self.collapse_all_button = QToolButton(self.tree_bar)
        self.collapse_all_button.setObjectName("treeAction")
        self.collapse_all_button.setText("折叠全部")
        self.collapse_all_button.setToolTip("折叠当前分类的全部分组")
        self.collapse_all_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.collapse_all_button.clicked.connect(self._on_collapse_all)
        tree_bar_layout.addWidget(self.category_label, 1)
        tree_bar_layout.addWidget(self.expand_all_button)
        tree_bar_layout.addWidget(self.collapse_all_button)
        layout.addWidget(self.tree_bar)

        self._content_stack = QStackedWidget(self)
        self.category_stack = QStackedWidget(self)

        self.empty_page = QWidget(self)
        empty_layout = QVBoxLayout(self.empty_page)
        empty_layout.setContentsMargins(16, 24, 16, 16)
        empty_layout.addStretch(1)
        empty_title = QLabel("尚未加载算子", self.empty_page)
        empty_title.setObjectName("sectionTitle")
        empty_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_hint = QLabel(
            "请检查「设置 → 算子目录」，\n然后点击右上角「刷新」重新扫描。",
            self.empty_page,
        )
        empty_hint.setObjectName("hintLabel")
        empty_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        empty_hint.setWordWrap(True)
        empty_layout.addWidget(empty_title)
        empty_layout.addWidget(empty_hint)
        empty_layout.addStretch(2)

        self._content_stack.addWidget(self.category_stack)
        self._content_stack.addWidget(self.empty_page)
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
        """用新的注册表重建全部分类（保留搜索与当前分类）。"""
        previous_category = self._current_category
        self._registry = registry
        self._hide_info_panel()
        self.setUpdatesEnabled(False)
        try:
            self._clear_categories()
            entries = registry.entries if registry is not None else {}
            if not entries or registry is None:
                self._category_scroll.setVisible(False)
                self.tree_bar.setVisible(False)
                self._content_stack.setCurrentWidget(self.empty_page)
                self._update_footer()
                return
            for index, (name, category_entries) in enumerate(registry.tabs().items()):
                if not category_entries:
                    continue
                page = _CategoryPage(self, name, len(category_entries))
                self._build_tree(page.tree, category_entries)
                self._pages[name] = page
                self.category_stack.addWidget(page)
                self._add_category_button(index, name, len(category_entries))
            self._update_category_bar_height()
            self._category_scroll.setVisible(True)
            self.tree_bar.setVisible(True)
            target = (
                previous_category
                if previous_category in self._pages
                else next(iter(self._pages), None)
            )
            if target is None:
                self._content_stack.setCurrentWidget(self.empty_page)
            else:
                self.select_category(target)
            self._apply_filter()
        finally:
            self.setUpdatesEnabled(True)

    def _clear_categories(self) -> None:
        """清空分类页与一级分类按钮。"""
        while self.category_stack.count():
            page = self.category_stack.widget(0)
            self.category_stack.removeWidget(page)
            if page is not None:
                page.deleteLater()
        self._pages.clear()
        for button in self._category_buttons:
            self._category_group.removeButton(button)
        self._category_buttons.clear()
        self._category_button_map.clear()
        while self._category_grid.count():
            layout_item = self._category_grid.takeAt(0)
            widget = layout_item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._current_category = None

    def _add_category_button(self, index: int, name: str, count: int) -> None:
        """在 2 列网格中添加一个一级分类胶囊按钮。"""
        button = QToolButton(self._category_bar_content)
        button.setObjectName("categoryChip")
        button.setCheckable(True)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        button.setText(f"{name} {count}")
        button.setToolTip(f"{name} · {count} 个算子")
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.setMinimumWidth(1)
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        button.clicked.connect(
            lambda _checked=False, category=name: self.select_category(category)
        )
        self._category_group.addButton(button)
        row, column = divmod(index, 2)
        self._category_grid.addWidget(button, row, column)
        self._category_buttons.append(button)
        self._category_button_map[name] = button

    def _update_category_bar_height(self) -> None:
        """按行数收紧分类栏高度；分类很多时封顶 140px 并允许滚动。"""
        if not self._category_buttons:
            return
        row_height = max(
            button.sizeHint().height() for button in self._category_buttons
        )
        rows = (len(self._category_buttons) + 1) // 2
        spacing = self._category_grid.verticalSpacing()
        height = rows * row_height + max(0, rows - 1) * spacing + 2
        self._category_scroll.setFixedHeight(min(140, height))

    # ------------------------------------------------------------------ 分类
    def category_names(self) -> list[str]:
        """按展示顺序返回全部一级分类名。"""
        return list(self._pages.keys())

    def category_buttons(self) -> list[QToolButton]:
        """返回全部一级分类按钮（与 :meth:`category_names` 同序）。"""
        return list(self._category_buttons)

    def current_category(self) -> str | None:
        return self._current_category

    def category_tree(self, name: str) -> QTreeWidget | None:
        page = self._pages.get(name)
        return page.tree if page is not None else None

    def select_category(self, name: str) -> bool:
        """切换一级分类；分类不存在时返回 False。"""
        page = self._pages.get(name)
        if page is None:
            return False
        if self._current_category != name:
            self._hide_info_panel()
        self._current_category = name
        button = self._category_button_map.get(name)
        if button is not None and not button.isChecked():
            button.setChecked(True)
        self.category_stack.setCurrentWidget(page)
        self._content_stack.setCurrentWidget(self.category_stack)
        self.category_label.setText(f"{name} · {page.entry_count} 个算子")
        self.expand_all_button.setEnabled(True)
        self.collapse_all_button.setEnabled(True)
        return True

    def _on_expand_all(self) -> None:
        tree = self.current_tree()
        if tree is not None:
            tree.expandAll()

    def _on_collapse_all(self) -> None:
        tree = self.current_tree()
        if tree is not None:
            tree.collapseAll()

    def _build_tree(self, tree: QTreeWidget, entries: list[OperatorEntry]) -> None:
        root: dict[str, Any] = {"groups": {}, "entries": []}
        for entry in entries:
            node = root
            for part in entry.subgroups:
                node = node["groups"].setdefault(part, {"groups": {}, "entries": []})
            node["entries"].append(entry)

        def add_nodes(
            parent: QTreeWidgetItem | QTreeWidget,
            node: dict[str, Any],
            path: tuple[str, ...],
        ) -> None:
            for group_name, child in node["groups"].items():
                item = QTreeWidgetItem([group_name])
                group_path = path + (group_name,)
                item.setToolTip(0, "分类：" + " / ".join(group_path))
                item.setData(0, Qt.ItemDataRole.UserRole, None)
                item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                group_font = item.font(0)
                group_font.setBold(True)
                item.setFont(0, group_font)
                if isinstance(parent, QTreeWidget):
                    parent.addTopLevelItem(item)
                else:
                    parent.addChild(item)
                add_nodes(item, child, group_path)

            for entry in node["entries"]:
                item = QTreeWidgetItem([entry.name])
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
                item.setToolTip(
                    0,
                    f"{entry.name}\n分类：{category_text}\n{entry.spec.description or '暂无描述'}",
                )
                item.setFlags(
                    Qt.ItemFlag.ItemIsEnabled
                    | Qt.ItemFlag.ItemIsSelectable
                    | Qt.ItemFlag.ItemIsDragEnabled
                )
                if isinstance(parent, QTreeWidget):
                    parent.addTopLevelItem(item)
                else:
                    parent.addChild(item)

        add_nodes(tree, root, ())
        self._reset_tree_expansion(tree)

    @staticmethod
    def _reset_tree_expansion(tree: QTreeWidget) -> None:
        """恢复默认折叠状态：只展开一级分组，更深层级折叠。"""
        tree.collapseAll()
        for index in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(index)
            if item.childCount():
                item.setExpanded(True)

    # ------------------------------------------------------------------ 过滤
    def _on_search_changed(self, _text: str) -> None:
        self._hide_info_panel()
        self._apply_filter()

    def _apply_filter(self) -> None:
        was_filtering = bool(self._filter_text)
        self._filter_text = self.search_edit.text().strip().lower()
        counts: list[int] = []
        for page in self._pages.values():
            matched = self._filter_tree(page.tree, self._filter_text)
            page.stack.setCurrentWidget(page.tree if matched else page.empty)
            page.empty.setText(
                "未找到匹配的算子\n试试其它关键字。"
                if self._filter_text
                else "该分类下暂无算子。"
            )
            counts.append(matched)
            if was_filtering and not self._filter_text:
                # 清空搜索后恢复默认折叠状态，避免整棵树保持全展开。
                self._reset_tree_expansion(page.tree)
        if self._filter_text and counts:
            names = list(self._pages)
            current = self._current_category
            current_index = names.index(current) if current in self._pages else -1
            if current_index < 0 or counts[current_index] == 0:
                for name, matched in zip(names, counts):
                    if matched:
                        self.select_category(name)
                        break
        self._update_footer(sum(counts) if self._filter_text else None)

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
        if text:
            tree.expandAll()
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
        category = " / ".join(spec.category) or "未分类"
        source = "内置" if entry.builtin else "用户"
        self.info_name.setText(entry.name)
        self.info_meta.setText(f"{category} · {source}")
        self.info_meta.setToolTip(f"算子 id：{entry.id}\n文件：{entry.path}")
        description = spec.description or "暂无描述"
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
            self.info_table.setVisible(False)
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
        if self._current_category is None:
            return None
        page = self._pages.get(self._current_category)
        return page.tree if page is not None else None
