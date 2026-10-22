"""替换算子对话框：按一级分类浏览 / 搜索可用算子，并排除算子组。

用于画布上「失效节点」或普通节点替换算子：

* 数据源为 OperatorRegistry，按 registry.tabs() 的分类顺序展示，
  「用户自定义算子」固定在最顶部；
* 默认排除算子组（GroupOperator），但允许当前算子保留在列表中；
* 搜索框跨全部分类过滤（名称 / id / 描述 / 分类 / 关键字），匹配项自动展开；
* 双击算子或点击「替换」确认；exec() 后通过 selected_op_id() 读取结果，
  取消或无选择时返回 None。
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QLineEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.operators.loader import OperatorEntry, OperatorRegistry


def _is_group_operator(operator: object) -> bool:
    """判断算子是否为算子组。

    延迟导入 app.operators.group，并用 group_definition 属性兜底：
    即使帧模块尚未就绪 / 导入异常，选择列表也不会因此崩溃。
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


class ReplaceOperatorDialog(QDialog):
    """选择替换算子的模态对话框。

    参数
    ----
    registry:
        当前算子注册表。
    current_op_id:
        被替换节点的算子 id；列表中会保留它（即使它是算子组），
        并在名称右侧标注「当前」。
    parent:
        Qt 父窗口。
    """

    def __init__(
        self,
        registry: OperatorRegistry | None,
        *,
        current_op_id: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._registry = registry
        self._current_op_id = current_op_id
        self._selected_op_id: str | None = None
        self._entry_items: dict[str, QTreeWidgetItem] = {}
        self._match_count = 0
        self.setObjectName("replaceOperatorDialog")
        self.setWindowTitle("替换算子")
        self.setModal(True)
        self.setMinimumSize(440, 500)
        self.resize(500, 580)
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)

        title = QLabel("替换算子", self)
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        current_text = ""
        entry = self._current_entry()
        if entry is not None:
            current_text = f"当前算子：{entry.name}（{entry.id}）"
        self.current_label = QLabel(current_text, self)
        self.current_label.setObjectName("hintLabel")
        self.current_label.setWordWrap(True)
        self.current_label.setVisible(bool(current_text))
        layout.addWidget(self.current_label)

        hint = QLabel(
            "选择一个普通算子替换当前节点；算子组不能作为替换目标。", self
        )
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.search_edit = QLineEdit(self)
        self.search_edit.setPlaceholderText("搜索算子（名称 / id / 说明 / 分类）")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.setToolTip("跨所有分类搜索；匹配项会自动展开。")
        self.search_edit.textChanged.connect(self._on_search_changed)
        layout.addWidget(self.search_edit)

        self.tree = QTreeWidget(self)
        self.tree.setObjectName("replaceOperatorTree")
        self.tree.setHeaderHidden(True)
        self.tree.setColumnCount(2)
        self.tree.setRootIsDecorated(True)
        self.tree.setIndentation(16)
        self.tree.setAnimated(False)
        self.tree.setUniformRowHeights(True)
        self.tree.setExpandsOnDoubleClick(False)
        self.tree.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.tree.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.tree.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        header = self.tree.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.tree.setColumnWidth(1, 52)
        self.tree.itemSelectionChanged.connect(self._on_selection_changed)
        self.tree.itemDoubleClicked.connect(self._on_item_double_clicked)
        layout.addWidget(self.tree, 1)

        self.empty_label = QLabel("", self)
        self.empty_label.setObjectName("hintLabel")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setWordWrap(True)
        self.empty_label.setVisible(False)
        layout.addWidget(self.empty_label)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        self.ok_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        self.ok_button.setText("替换")
        self.ok_button.setObjectName("primaryButton")
        self.ok_button.setEnabled(False)
        self.cancel_button = self.button_box.button(
            QDialogButtonBox.StandardButton.Cancel
        )
        self.cancel_button.setText("取消")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    # ------------------------------------------------------------------ 数据
    def refresh(self) -> None:
        """按最新注册表重建分类树，并重新应用当前搜索。"""

        previous = self._selected_op_id
        self._selected_op_id = None
        self._rebuild_tree()
        self._apply_filter(self.search_edit.text())
        if previous:
            self.select_operator(previous)
        self._update_ok_state()

    def filter(self, text: str) -> int:
        """设置搜索关键字并过滤列表，返回当前匹配的算子数量。"""

        text = "" if text is None else str(text)
        if self.search_edit.text() != text:
            self.search_edit.setText(text)
            return self._match_count
        return self._apply_filter(text)

    def operator_ids(self) -> list[str]:
        """按展示顺序返回列表中的全部算子 id（已排除算子组）。"""

        return list(self._entry_items.keys())

    def matched_operator_ids(self) -> list[str]:
        """返回当前搜索条件下命中的算子 id（忽略分类折叠状态）。"""

        return [
            op_id
            for op_id, item in self._entry_items.items()
            if not item.isHidden()
            and item.parent() is not None
            and not item.parent().isHidden()
        ]

    def category_names(self) -> list[str]:
        """按展示顺序返回列表中的一级分类名（空分类会被省略）。"""

        return [
            self.tree.topLevelItem(index).text(0)
            for index in range(self.tree.topLevelItemCount())
        ]

    def item_for(self, op_id: str) -> QTreeWidgetItem | None:
        """返回算子对应的叶子条目，便于测试与上层定位。"""

        return self._entry_items.get(op_id)

    def selected_op_id(self) -> str | None:
        """返回当前选中的算子 id；未选择或取消时为 None。"""

        return self._selected_op_id

    def select_operator(self, op_id: str) -> bool:
        """选中指定算子；它被搜索过滤掉时返回 False。"""

        item = self._entry_items.get(op_id)
        if item is None or item.isHidden() or item.parent().isHidden():
            return False
        self.tree.setCurrentItem(item)
        self.tree.scrollToItem(item, QAbstractItemView.ScrollHint.EnsureVisible)
        self._selected_op_id = op_id
        self._update_ok_state()
        return True

    # ------------------------------------------------------------------ 内部
    def _current_entry(self) -> OperatorEntry | None:
        if self._registry is None or not self._current_op_id:
            return None
        return self._registry.entry(self._current_op_id)

    @staticmethod
    def _entry_haystack(entry: OperatorEntry) -> str:
        category = " ".join(entry.spec.category)
        keywords = " ".join(str(keyword) for keyword in entry.spec.keywords)
        return " ".join(
            [
                entry.name,
                entry.id,
                entry.spec.description or "",
                category,
                keywords,
            ]
        ).lower()

    def _rebuild_tree(self) -> None:
        self.tree.blockSignals(True)
        try:
            self.tree.clear()
            self._entry_items.clear()
            tabs = self._registry.tabs() if self._registry is not None else {}
            for category, entries in tabs.items():
                available: list[OperatorEntry] = []
                for entry in entries:
                    if entry.id == self._current_op_id or not _is_group_operator(
                        entry.operator
                    ):
                        available.append(entry)
                if not available:
                    continue
                group = QTreeWidgetItem([category, str(len(available))])
                group.setData(0, Qt.ItemDataRole.UserRole, None)
                group.setFlags(Qt.ItemFlag.ItemIsEnabled)
                group.setData(
                    1,
                    Qt.ItemDataRole.TextAlignmentRole,
                    int(
                        Qt.AlignmentFlag.AlignRight
                        | Qt.AlignmentFlag.AlignVCenter
                    ),
                )
                group_font = group.font(0)
                group_font.setBold(True)
                group.setFont(0, group_font)
                group.setSizeHint(0, QSize(0, 30))
                group.setToolTip(0, f"{category} · 双击展开 / 折叠")
                self.tree.addTopLevelItem(group)
                for entry in available:
                    marker = "当前" if entry.id == self._current_op_id else ""
                    item = QTreeWidgetItem([entry.name, marker])
                    item.setData(0, Qt.ItemDataRole.UserRole, entry.id)
                    item.setData(
                        0, Qt.ItemDataRole.UserRole + 1, self._entry_haystack(entry)
                    )
                    item.setData(
                        1,
                        Qt.ItemDataRole.TextAlignmentRole,
                        int(
                            Qt.AlignmentFlag.AlignRight
                            | Qt.AlignmentFlag.AlignVCenter
                        ),
                    )
                    item.setFlags(
                        Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
                    )
                    item.setSizeHint(0, QSize(0, 26))
                    tooltip_lines = [
                        entry.name,
                        f"id：{entry.id}",
                        f"分类：{' / '.join(entry.spec.category) or '未分类'}",
                        entry.spec.description or "暂无描述",
                    ]
                    item.setToolTip(0, "\n".join(tooltip_lines))
                    group.addChild(item)
                    self._entry_items[entry.id] = item
        finally:
            self.tree.blockSignals(False)
        self.tree.collapseAll()

    def _on_search_changed(self, text: str) -> None:
        self._apply_filter(text)

    def _apply_filter(self, text: str) -> int:
        keyword = str(text or "").strip().lower()
        matched = 0
        for index in range(self.tree.topLevelItemCount()):
            group = self.tree.topLevelItem(index)
            group_hit = False
            for child_index in range(group.childCount()):
                child = group.child(child_index)
                haystack = str(
                    child.data(0, Qt.ItemDataRole.UserRole + 1) or ""
                )
                hit = not keyword or keyword in haystack
                child.setHidden(not hit)
                if hit:
                    group_hit = True
                    matched += 1
            group.setHidden(not group_hit)
        self._match_count = matched
        self.tree.blockSignals(True)
        try:
            if keyword:
                self.tree.expandAll()
            else:
                self.tree.collapseAll()
        finally:
            self.tree.blockSignals(False)

        self.empty_label.setText(
            "没有可替换的算子。"
            if self.tree.topLevelItemCount() == 0
            else "未找到匹配的算子，试试其它关键字。"
        )
        self.empty_label.setVisible(matched == 0)

        current = self._current_selection_id()
        if current is not None:
            item = self._entry_items.get(current)
            if item is None or item.isHidden() or item.parent().isHidden():
                self.tree.clearSelection()
                self.tree.setCurrentItem(None)
                self._selected_op_id = None
        self._update_ok_state()
        return matched

    def _current_selection_id(self) -> str | None:
        item = self.tree.currentItem()
        if item is None:
            return None
        op_id = item.data(0, Qt.ItemDataRole.UserRole)
        return str(op_id) if op_id else None

    def _on_selection_changed(self) -> None:
        self._selected_op_id = self._current_selection_id()
        self._update_ok_state()

    def _update_ok_state(self) -> None:
        self.ok_button.setEnabled(self._selected_op_id is not None)

    def _on_item_double_clicked(
        self, item: QTreeWidgetItem, _column: int
    ) -> None:
        op_id = item.data(0, Qt.ItemDataRole.UserRole)
        if not op_id:
            # 双击分类行直接展开 / 折叠。
            item.setExpanded(not item.isExpanded())
            return
        self.tree.setCurrentItem(item)
        self._selected_op_id = str(op_id)
        self.accept()

    # ------------------------------------------------------------------ 确认
    def accept(self) -> None:
        op_id = self._current_selection_id()
        item = self._entry_items.get(op_id) if op_id else None
        if item is None or item.isHidden() or item.parent().isHidden():
            return
        self._selected_op_id = op_id
        super().accept()

    def reject(self) -> None:
        self._selected_op_id = None
        super().reject()
