"""左侧算子库：行式一级分类手风琴、多级折叠树、拖拽与说明浮层。

* 一级分类是整行可折叠标题（左侧箭头 + 分类名 + 右侧数量），默认全部折叠；
* 点击某个一级分类就在该行下方原地展开，支持同时展开多个分类，展开后下方
  分类标题整体下移；二级、三级分类仍是折叠分组，可逐级展开 / 折叠；
* 搜索跨所有分类过滤，命中项自动展开并在列表中定位；
* 将算子叶子拖到画布时使用 :data:`app.core.constants.MIME_OPERATOR`，
  payload 为 ``{"id": op_id}`` 的 UTF-8 JSON；
* 用户算子叶子拖动时同时携带 :data:`app.core.constants.MIME_LIBRARY_MOVE`，
  可在算子库内拖到其它分类整理；磁盘上确实存在于用户扫描根下的用户分类
  （分组）也能拖动，内置算子 / 内置分类 / 仅由 ``spec.category`` 形成的虚拟
  分组不可拖动。移动只在算子文件 / 分类目录所属的用户扫描根内进行；
* 单击算子叶子在库底部弹出说明浮层（名称 / 分类 / 描述 / 参数 / 端口），
  点击浮层外部或按 Esc 关闭；
* 右键分类 / 分组 / 空白处可直接新建用户算子或分类，右键用户算子可原地改名、
  打开编辑器或删除（新建与改名后通过 :attr:`OperatorLibrary.refresh_requested`
  让上层同步重扫算子目录）。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from PySide6.QtCore import (
    QEvent,
    QMimeData,
    QModelIndex,
    QPoint,
    QRectF,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QDrag,
    QFont,
    QFontMetrics,
    QKeySequence,
    QPainter,
    QPen,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QFrame,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QLineEdit,
    QMenu,
    QMessageBox,
    QScrollArea,
    QStackedWidget,
    QStyle,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from app.config import AppSettings, default_user_operator_dir
from app.core.constants import (
    CATEGORY_ORDER,
    CUSTOM_OPERATOR_CATEGORY,
    MIME_LIBRARY_MOVE,
    MIME_OPERATOR,
)
from app.operators import loader as operator_loader
from app.operators.loader import OperatorEntry, OperatorRegistry
from app.operators.spec import ParamSpec

# 条目数据角色：前两个沿用旧值（UserRole / UserRole + 1），避免破坏现有拖拽与搜索。
ROLE_OP_ID = Qt.ItemDataRole.UserRole
ROLE_HAYSTACK = Qt.ItemDataRole.UserRole + 1
ROLE_KIND = Qt.ItemDataRole.UserRole + 2
ROLE_CATEGORY_PATH = Qt.ItemDataRole.UserRole + 3
ROLE_OPERATOR_PATH = Qt.ItemDataRole.UserRole + 4
ROLE_FILE_STEM = Qt.ItemDataRole.UserRole + 5

_KIND_CATEGORY = "category"
_KIND_GROUP = "group"
_KIND_OPERATOR = "operator"

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


def _normalize_category(
    category: str | tuple[str, ...] | list[str] | None,
) -> tuple[str, ...]:
    """把分类路径统一成去掉空段的 tuple；``None`` 表示没有分类。"""

    if category is None:
        return ()
    if isinstance(category, str):
        parts = category.replace("\\", "/").split("/")
    else:
        parts = [str(part) for part in category]
    return tuple(part.strip() for part in parts if part and part.strip())


def _is_same_or_descendant(
    candidate: tuple[str, ...], parent: tuple[str, ...]
) -> bool:
    """``candidate`` 是否等于 ``parent`` 或位于其下。"""

    return len(candidate) >= len(parent) and candidate[: len(parent)] == parent


def _category_sort_key(name: str) -> tuple[int, str]:
    """一级分类排序：CATEGORY_ORDER 内按声明顺序，未知分类按名称排最后。"""

    try:
        return (CATEGORY_ORDER.index(name), name)
    except ValueError:
        return (len(CATEGORY_ORDER), name)


class _OperatorNameDelegate(QStyledItemDelegate):
    """用户算子文件名的原地编辑器：只显示文件 stem，提交后交给算子库改名。

    Qt 默认会把编辑器文本写回模型；这里改由 :meth:`OperatorLibrary.rename_operator_to`
    统一校验并改名，避免条目显示名被未校验的文本覆盖。
    """

    def __init__(self, library: "OperatorLibrary", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._library = library

    def createEditor(  # noqa: N802 - Qt 命名
        self,
        parent: QWidget,
        option: QStyleOptionViewItem,
        index: QModelIndex,
    ) -> QWidget | None:
        if index.column() != 0 or not index.data(ROLE_FILE_STEM):
            return None
        editor = QLineEdit(parent)
        editor.setFrame(False)
        editor.setObjectName("operatorNameEditor")
        return editor

    def setEditorData(self, editor: QWidget, index: QModelIndex) -> None:  # noqa: N802
        if not isinstance(editor, QLineEdit):
            return
        editor.setText(str(index.data(ROLE_FILE_STEM) or ""))
        editor.selectAll()

    def setModelData(  # noqa: N802 - Qt 命名
        self,
        editor: QWidget,
        model,
        index: QModelIndex,
    ) -> None:
        if not isinstance(editor, QLineEdit):
            return
        op_id = str(index.data(ROLE_OP_ID) or "")
        old_stem = str(index.data(ROLE_FILE_STEM) or "")
        new_stem = editor.text().strip()
        if not op_id or not new_stem or new_stem == old_stem:
            return
        # 延后到编辑器关闭、Qt 退出数据提交后再重建算子树，避免删掉正在提交的条目。
        QTimer.singleShot(
            0, lambda: self._library.rename_operator_to(op_id, new_stem)
        )


class _OperatorTree(QTreeWidget):
    """算子库主树：一级分类为整行折叠标题，叶子可拖拽。

    * 第一列：分类 / 分组 / 算子名；第二列：分类或分组的算子数量；
    * 非叶子条目单击整行即可原地展开 / 折叠（箭头仍由 Qt 原生处理）；
    * 所有算子叶子拖动时携带 ``MIME_OPERATOR``（拖到画布创建节点）；用户
      算子 / 用户分类额外携带 ``MIME_LIBRARY_MOVE``，可在库内拖动整理。
      这里自行处理 drop，只接受 ``MIME_LIBRARY_MOVE``，不启用 Qt 默认的
      InternalMove（否则会由模型移动行）；
    * 拖动中在合法目标行上画虚线高亮，非法目标不接收。
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
        # 允许“拖出到画布”与“拖入库内整理”，但 drop 逻辑完全自行处理；
        # 不使用 InternalMove，避免 Qt 在模型层移动 / 复制行。
        self.setDragDropMode(QAbstractItemView.DragDropMode.DragDrop)
        self.setDefaultDropAction(Qt.DropAction.CopyAction)
        self.setDropIndicatorShown(False)
        self.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
        self.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.customContextMenuRequested.connect(self._on_context_menu_requested)
        header = self.header()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.setColumnWidth(1, 52)
        self.setMouseTracking(True)
        self._hover_item: QTreeWidgetItem | None = None
        self._drop_target_item: QTreeWidgetItem | None = None
        self.itemClicked.connect(self._on_item_clicked)
        self.itemActivated.connect(self._on_item_activated)

    # ------------------------------------------------------------------ 绘制
    def _row_background(self, item: QTreeWidgetItem | None) -> QColor | None:
        """整行底色：一级分类按展开状态，其余按选中 / 悬停。"""

        if item is None:
            return None
        if item is self._drop_target_item:
            return QColor("#dbeafe")
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

        # 库内拖动整理：合法目标行画虚线框，提示“松手会移到这里”。
        if item is not None and item is self._drop_target_item:
            painter.save()
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#2563eb"), 1.5, Qt.PenStyle.DashLine))
            painter.drawRoundedRect(
                QRectF(
                    1.0,
                    float(option.rect.top()) + 1.5,
                    float(self.viewport().width()) - 3.0,
                    float(option.rect.height()) - 3.0,
                ),
                6.0,
                6.0,
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
        # 算子叶子（含内置）允许拖到画布；带库内移动 payload 的用户算子 /
        # 用户分类才允许发起库内整理拖动。
        if not item.data(0, ROLE_OP_ID) and self._library.library_drag_payload(item) is None:
            return
        self._library.begin_library_drag(self, item)

    # ------------------------------------------------------------ 库内拖动整理
    @staticmethod
    def _payload_from_mime(mime: QMimeData) -> dict[str, Any] | None:
        if mime is None or not mime.hasFormat(MIME_LIBRARY_MOVE):
            return None
        try:
            raw = bytes(mime.data(MIME_LIBRARY_MOVE)).decode("utf-8")
            payload = json.loads(raw)
        except (UnicodeDecodeError, ValueError, TypeError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("kind") not in ("operator", "category"):
            return None
        return payload

    def _set_drop_target(self, item: QTreeWidgetItem | None) -> None:
        if item is self._drop_target_item:
            return
        self._drop_target_item = item
        self.viewport().update()

    def _auto_scroll_for_drag(self, pos: QPoint) -> None:
        """拖动到视口上下边缘时滚动，便于把算子拖到远处的分类。"""

        margin = 24
        step = 18
        bar = self.verticalScrollBar()
        if pos.y() < margin and bar.value() > bar.minimum():
            bar.setValue(max(bar.minimum(), bar.value() - step))
        elif (
            pos.y() > self.viewport().height() - margin
            and bar.value() < bar.maximum()
        ):
            bar.setValue(min(bar.maximum(), bar.value() + step))

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        payload = self._payload_from_mime(event.mimeData())
        item = self.itemAt(event.position().toPoint())
        if payload is not None and self._library.can_accept_library_drop(payload, item):
            self._set_drop_target(item)
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            return
        self._set_drop_target(None)
        event.ignore()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        payload = self._payload_from_mime(event.mimeData())
        pos = event.position().toPoint()
        if payload is not None:
            self._auto_scroll_for_drag(pos)
        item = self.itemAt(pos)
        if payload is not None and self._library.can_accept_library_drop(payload, item):
            self._set_drop_target(item)
            event.setDropAction(Qt.DropAction.CopyAction)
            event.accept()
            return
        self._set_drop_target(None)
        event.ignore()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        self._set_drop_target(None)
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        payload = self._payload_from_mime(event.mimeData())
        item = self.itemAt(event.position().toPoint())
        if payload is None or not self._library.can_accept_library_drop(payload, item):
            self._set_drop_target(None)
            event.ignore()
            return
        target = self._library.drop_target_category(item)
        self._set_drop_target(None)
        event.setDropAction(Qt.DropAction.CopyAction)
        event.accept()
        if payload.get("kind") == "operator":
            self._library.move_operator_to_category(str(payload.get("id") or ""), target)
        else:
            self._library.move_category_to_category(
                _normalize_category(payload.get("path")), target
            )

    def _on_context_menu_requested(self, pos: QPoint) -> None:
        item = self.itemAt(pos)
        self._library.show_context_menu(item, self.viewport().mapToGlobal(pos))

    def _on_item_clicked(self, item: QTreeWidgetItem, _column: int) -> None:
        op_id = item.data(0, ROLE_OP_ID)
        if op_id:
            self._library.show_operator_info(str(op_id))
            self._library.note_leaf_clicked(item)
            return
        # 一级分类 / 二级分组：单击整行原地展开或折叠（多开，不关闭其它分类）。
        item.setExpanded(not item.isExpanded())

    def _on_item_activated(self, item: QTreeWidgetItem, _column: int) -> None:
        op_id = item.data(0, ROLE_OP_ID)
        if op_id:
            self._library.show_operator_info(str(op_id))


class OperatorLibrary(QWidget):
    """算子库面板。"""

    refresh_requested = Signal()
    # 新建 / 改名 / 删除 / 打开编辑器失败时的简短中文提示（由主窗口转状态栏）。
    status_message = Signal(str)

    def __init__(
        self,
        parent: QWidget | None = None,
        *,
        settings: AppSettings | None = None,
        user_operator_dir: Path | str | None = None,
    ) -> None:
        super().__init__(parent)
        self.setMinimumWidth(220)
        self._settings = settings
        self._user_operator_dir = (
            Path(user_operator_dir) if user_operator_dir is not None else None
        )
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
        self._name_delegate = _OperatorNameDelegate(self, self.tree)
        # 委托覆盖整棵树，但 createEditor 只对第 0 列返回编辑器；这样用户算子的
        # 数量列不会被双击写入任意文本。
        self.tree.setItemDelegate(self._name_delegate)
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
            "右键此处可直接新建算子或分类；\n"
            "也可检查「设置 → 算子目录」后点击右上角刷新。",
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
        # 空列表 / 无匹配页面也要能右键新建算子或分类。
        for page in (self.empty_page, self.filter_empty_page):
            page.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
            page.customContextMenuRequested.connect(
                lambda pos, widget=page: self.show_context_menu(
                    None, widget.mapToGlobal(pos)
                )
            )

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
            known_categories = self._known_categories(registry)
            # 0 算子的空分类也要显示，便于右键新建算子。
            if not entries and not known_categories:
                self.tree_bar.setVisible(False)
                self._content_stack.setCurrentWidget(self.empty_page)
                self._update_footer()
                return
            self._build_tree(self.tree, registry.tabs(), known_categories)
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

    @staticmethod
    def _known_categories(
        registry: OperatorRegistry | None,
    ) -> tuple[tuple[str, ...], ...]:
        """读取 loader 记录的空分类（含仅有一级的分类）；旧 loader 无接口时为空。"""

        if registry is None:
            return ()
        getter = getattr(registry, "known_categories", None)
        if not callable(getter):
            return ()
        try:
            raw = getter()
        except Exception:  # noqa: BLE001 - 空分类只是增强信息，失败不应影响主树
            return ()
        result: list[tuple[str, ...]] = []
        for path in raw:
            normalized = _normalize_category(path)
            if normalized and normalized not in result:
                result.append(normalized)
        return tuple(result)

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
        known_categories: tuple[tuple[str, ...], ...] = (),
    ) -> None:
        tree.clear()
        # 先合并 registry.tabs()（非空分类）与 known_categories（含 0 算子空分类），
        # 再按 CATEGORY_ORDER 统一排序一级分类，避免空分类被追加到末尾。
        roots: dict[str, dict[str, Any]] = {}
        for name, entries in tabs.items():
            root = roots.setdefault(name, {"groups": {}, "entries": []})
            for entry in entries:
                node = root
                for part in entry.subgroups:
                    node = node["groups"].setdefault(part, {"groups": {}, "entries": []})
                node["entries"].append(entry)
        for path in known_categories:
            if not path:
                continue
            node = roots.setdefault(path[0], {"groups": {}, "entries": []})
            for part in path[1:]:
                node = node["groups"].setdefault(part, {"groups": {}, "entries": []})

        # 无论分类来自 tabs 还是空目录，一级分类都按 CATEGORY_ORDER 排序；
        # 未知分类按名称排在最后，保证「用户自定义算子」始终第一。
        for name in sorted(roots, key=_category_sort_key):
            root = roots[name]
            category = QTreeWidgetItem([name, ""])
            category.setData(0, ROLE_OP_ID, None)
            category.setData(0, ROLE_KIND, _KIND_CATEGORY)
            category.setData(0, ROLE_CATEGORY_PATH, (name,))
            category_draggable = self._user_category_root((name,)) is not None
            category_flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            if category_draggable:
                category_flags |= Qt.ItemFlag.ItemIsDragEnabled
            category.setFlags(category_flags)
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
            tooltip = f"{name} · 单击展开 / 折叠 · 右键新建"
            if category_draggable:
                tooltip += " · 可拖动到其它分类"
            category.setToolTip(0, tooltip)
            tree.addTopLevelItem(category)
            total = self._populate_tree(category, root, (), name)
            category.setText(1, str(total))
            category.setToolTip(1, f"{total} 个算子")
            self._category_items[name] = category
        tree.collapseAll()

    def _populate_tree(
        self,
        parent: QTreeWidgetItem,
        node: dict[str, Any],
        path: tuple[str, ...],
        top_name: str,
    ) -> int:
        total = 0
        for group_name, child in node["groups"].items():
            group = QTreeWidgetItem([group_name, ""])
            group_path = path + (group_name,)
            group.setData(0, ROLE_OP_ID, None)
            group.setData(0, ROLE_KIND, _KIND_GROUP)
            group.setData(0, ROLE_CATEGORY_PATH, (top_name,) + group_path)
            group_draggable = (
                self._user_category_root((top_name,) + group_path) is not None
            )
            group_flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            if group_draggable:
                group_flags |= Qt.ItemFlag.ItemIsDragEnabled
            group.setFlags(group_flags)
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
            group_tooltip = "分类：" + " / ".join((top_name,) + group_path)
            if group_draggable:
                group_tooltip += " · 可拖动到其它分类"
            group.setToolTip(0, group_tooltip)
            parent.addChild(group)
            count = self._populate_tree(group, child, group_path, top_name)
            group.setText(1, str(count))
            group.setToolTip(1, f"{count} 个算子")
            total += count

        for entry in node["entries"]:
            display_name = getattr(entry, "display_name", None) or entry.name
            item = QTreeWidgetItem([display_name, ""])
            category_text = " / ".join(entry.spec.category) or "未分类"
            haystack = " ".join(
                [
                    display_name,
                    entry.name,
                    entry.path.stem,
                    entry.path.name,
                    entry.spec.id,
                    entry.spec.description or "",
                    category_text,
                    " ".join(str(keyword) for keyword in entry.spec.keywords),
                ]
            ).lower()
            item.setData(0, ROLE_OP_ID, entry.id)
            item.setData(0, ROLE_HAYSTACK, haystack)
            item.setData(0, ROLE_KIND, _KIND_OPERATOR)
            item.setData(0, ROLE_CATEGORY_PATH, (top_name,) + path)
            item.setData(0, ROLE_OPERATOR_PATH, str(entry.path))
            tooltip_lines = [display_name, f"分类：{category_text}"]
            if _is_group_operator(entry.operator):
                tooltip_lines.append("类型：算子组")
            tooltip_lines.append(entry.spec.description or "暂无描述")
            item.setToolTip(0, "\n".join(tooltip_lines))
            flags = (
                Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
                | Qt.ItemFlag.ItemIsDragEnabled
            )
            if not entry.builtin:
                # 用户算子允许原地改名；内置算子保持只读。
                item.setData(0, ROLE_FILE_STEM, entry.path.stem)
                flags |= Qt.ItemFlag.ItemIsEditable
            item.setFlags(flags)
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
            op_id = item.data(0, ROLE_OP_ID)
            if op_id:
                haystack = str(item.data(0, ROLE_HAYSTACK) or item.text(0).lower())
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

    # ------------------------------------------------------------------ 右键菜单
    def build_context_menu(self, item: QTreeWidgetItem | None) -> QMenu:
        """按条目类型构造右键菜单（只构造、不弹出，便于测试）。"""

        menu = QMenu(self)
        kind = item.data(0, ROLE_KIND) if item is not None else None
        if item is None or kind in (_KIND_CATEGORY, _KIND_GROUP):
            category = self._category_path_for_item(item)
            self._add_menu_action(
                menu,
                "新建算子",
                lambda: self.create_operator(category),
                "create_operator",
                category,
            )
            self._add_menu_action(
                menu,
                "新建分类…",
                lambda: self._prompt_create_category(category),
                "create_category",
                category,
            )
            return menu
        entry = self._entry_for_item(item)
        if entry is None:
            return menu
        if entry.builtin:
            self._add_menu_action(
                menu,
                "查看说明",
                lambda: self.show_operator_info(entry.id),
                "show_info",
                entry.id,
            )
            return menu
        self._add_menu_action(
            menu,
            "重命名",
            lambda: self.begin_rename_operator(entry.id),
            "rename_operator",
            entry.id,
        )
        self._add_menu_action(
            menu,
            "打开编辑器",
            lambda: self.open_editor(entry.id),
            "open_editor",
            entry.id,
        )
        menu.addSeparator()
        self._add_menu_action(
            menu,
            "删除算子…",
            lambda: self.delete_operator(entry.id),
            "delete_operator",
            entry.id,
        )
        return menu

    @staticmethod
    def _add_menu_action(
        menu: QMenu,
        text: str,
        callback,
        action_name: str,
        payload,
    ) -> QAction:
        action = menu.addAction(text)
        action.setData({"action": action_name, "payload": payload})
        action.triggered.connect(lambda _checked=False: callback())
        return action

    @staticmethod
    def _category_path_for_item(
        item: QTreeWidgetItem | None,
    ) -> tuple[str, ...] | None:
        if item is None:
            return None
        path = item.data(0, ROLE_CATEGORY_PATH)
        if not path:
            return None
        return _normalize_category(path)

    def _entry_for_item(self, item: QTreeWidgetItem | None) -> OperatorEntry | None:
        if item is None:
            return None
        op_id = item.data(0, ROLE_OP_ID)
        if not op_id:
            return None
        return self._entry(str(op_id))

    def show_context_menu(self, item: QTreeWidgetItem | None, global_pos: QPoint) -> None:
        """弹出右键菜单（测试只调用 :meth:`build_context_menu`，不阻塞）。"""

        menu = self.build_context_menu(item)
        if menu.actions():
            menu.exec(global_pos)
        menu.deleteLater()

    # ------------------------------------------------------------------ 新建 / 改名 / 删除
    def user_operator_root(self) -> Path:
        """用户算子根目录：构造参数优先，否则使用默认用户算子目录。"""

        return self._user_operator_dir or default_user_operator_dir()

    def create_operator(
        self,
        category: str | tuple[str, ...] | list[str] | None = None,
    ) -> str | None:
        """在用户算子目录新建一个算子文件；返回新算子 id（失败返回 None）。

        新建后发出 :attr:`refresh_requested`，由上层同步重扫注册表；未指定分类
        （算子库空白处 / 空列表右键）时固定归入 :data:`CUSTOM_OPERATOR_CATEGORY`，
        分类 / 分组行右键会显式传入该分类路径。
        """

        category_path = _normalize_category(category) or (CUSTOM_OPERATOR_CATEGORY,)
        try:
            from app.core import operator_files
        except Exception as exc:  # noqa: BLE001 - 并行开发期文件服务可能尚未就绪
            self.status_message.emit(f"新建算子失败：文件服务不可用（{exc}）")
            return None
        try:
            created = operator_files.create_operator_file(
                self.user_operator_root(), category_path
            )
        except Exception as exc:  # noqa: BLE001 - 统一转换为状态栏提示
            self.status_message.emit(f"新建算子失败：{exc}")
            return None
        op_id = str(getattr(created, "operator_id", "") or "")
        # 搜索过滤可能把新条目隐藏；先清空过滤，让用户看到刚建的文件。
        self._clear_search_for_new_item()
        self.refresh_requested.emit()
        if op_id:
            self.select_operator(op_id)
            self.begin_rename_operator(op_id)
        return op_id or None

    def _clear_search_for_new_item(self) -> None:
        """新建后清空搜索过滤，避免新条目被当前过滤条件隐藏。"""

        if self.search_edit.text().strip():
            self.search_edit.clear()

    def create_category_named(
        self,
        parent: str | tuple[str, ...] | list[str] | None,
        name: str,
    ) -> Path | None:
        """在 ``parent`` 分类下创建目录并刷新；返回新目录路径（失败返回 None）。"""

        parent_path = _normalize_category(parent)
        raw_name = str(name or "").strip()
        if not raw_name:
            self.status_message.emit("新建分类失败：分类名称不能为空。")
            return None
        try:
            from app.core import operator_files
        except Exception as exc:  # noqa: BLE001 - 并行开发期文件服务可能尚未就绪
            self.status_message.emit(f"新建分类失败：文件服务不可用（{exc}）")
            return None
        try:
            created = operator_files.create_category_dir(
                self.user_operator_root(), parent_path, raw_name
            )
        except Exception as exc:  # noqa: BLE001 - 统一转换为状态栏提示
            self.status_message.emit(f"新建分类失败：{exc}")
            return None
        directory = Path(created)
        self._clear_search_for_new_item()
        self.refresh_requested.emit()
        self._select_category_path(
            self._relative_category(directory, parent_path, raw_name)
        )
        return directory

    def _prompt_create_category(
        self,
        parent: tuple[str, ...] | None,
    ) -> Path | None:
        parent_path = _normalize_category(parent)
        title = "新建分类" if not parent_path else "新建子分类"
        name, accepted = QInputDialog.getText(
            self,
            title,
            "分类名称：",
            QLineEdit.EchoMode.Normal,
            "",
        )
        if not accepted:
            return None
        return self.create_category_named(parent_path, name)

    def _relative_category(
        self,
        directory: Path,
        parent: tuple[str, ...],
        name: str,
    ) -> tuple[str, ...]:
        """优先按实际创建目录反推分类路径（名称可能被 sanitize）。"""

        root = self.user_operator_root()
        try:
            relative = directory.resolve().relative_to(root.resolve())
        except (OSError, ValueError):
            return parent + (name,)
        parts = tuple(part for part in relative.parts if part not in ("", "."))
        return parts or (parent + (name,))

    def rename_operator_to(self, op_id: str, new_stem: str) -> bool:
        """原地重命名用户算子文件；失败时只发 :attr:`status_message`，不弹窗。"""

        entry = self._entry(op_id)
        if entry is None:
            self.status_message.emit("重命名失败：找不到该算子。")
            return False
        if entry.builtin:
            self.status_message.emit("重命名失败：内置算子文件不能改名。")
            return False
        stem = str(new_stem or "").strip()
        if stem.lower().endswith(".py"):
            # 列表不显示扩展名；用户误输入 .py 时不要再叠一层。
            stem = stem[:-3].rstrip()
        if not stem:
            self.status_message.emit("重命名失败：文件名不能为空。")
            return False
        if stem == entry.path.stem:
            self.select_operator(op_id)
            return True
        try:
            from app.core import operator_files
        except Exception as exc:  # noqa: BLE001 - 并行开发期文件服务可能尚未就绪
            self.status_message.emit(f"重命名失败：文件服务不可用（{exc}）")
            return False
        try:
            # rename_operator_file 内部同步 UI_CREATED 模板的 name 行，无需重复调用。
            operator_files.rename_operator_file(entry.path, stem)
        except Exception as exc:  # noqa: BLE001 - 中文错误直接展示到状态栏
            self.status_message.emit(f"重命名失败：{exc}")
            return False
        self.refresh_requested.emit()
        self.select_operator(op_id)
        return True

    def delete_operator(self, op_id: str, confirm: bool = True) -> bool:
        """删除用户算子文件；``confirm=True`` 时先弹确认框。"""

        entry = self._entry(op_id)
        if entry is None:
            self.status_message.emit("删除失败：找不到该算子。")
            return False
        if entry.builtin:
            self.status_message.emit("删除失败：内置算子不能删除。")
            return False
        if confirm:
            answer = QMessageBox.question(
                self,
                "删除算子",
                f"确定删除用户算子文件“{entry.path.name}”吗？\n"
                f"{entry.path}\n删除后无法恢复。",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return False
        try:
            from app.core import operator_files
        except Exception as exc:  # noqa: BLE001 - 并行开发期文件服务可能尚未就绪
            self.status_message.emit(f"删除失败：文件服务不可用（{exc}）")
            return False
        try:
            operator_files.delete_operator_file(entry.path)
        except Exception as exc:  # noqa: BLE001 - 中文错误直接展示到状态栏
            self.status_message.emit(f"删除失败：{exc}")
            return False
        self.refresh_requested.emit()
        return True

    def open_editor(self, op_id: str) -> bool:
        """用设置中的默认编辑器打开用户算子文件。"""

        entry = self._entry(op_id)
        if entry is None:
            self.status_message.emit("打开编辑器失败：找不到该算子。")
            return False
        if not entry.path.exists():
            self.status_message.emit(f"打开编辑器失败：文件不存在（{entry.path}）。")
            return False
        editor_value = "auto"
        if self._settings is not None:
            editor_value = str(
                getattr(self._settings, "default_editor", "auto") or "auto"
            )
        try:
            from app.core import editor as editor_module
        except Exception as exc:  # noqa: BLE001 - 并行开发期编辑器模块可能尚未就绪
            self.status_message.emit(f"打开编辑器失败：编辑器模块不可用（{exc}）")
            return False
        try:
            launched, message = editor_module.launch_editor(
                str(entry.path), editor_value
            )
        except Exception as exc:  # noqa: BLE001 - 启动失败统一提示
            self.status_message.emit(f"打开编辑器失败：{exc}")
            return False
        if not launched:
            detail = str(message or "").strip() or "未找到可用编辑器。"
            self.status_message.emit(f"打开编辑器失败：{detail}")
            return False
        return True

    def select_operator(self, op_id: str) -> bool:
        """展开所属分类并选中算子条目。"""

        item = self._entry_items.get(str(op_id))
        if item is None:
            return False
        parent = item.parent()
        while parent is not None:
            parent.setExpanded(True)
            parent = parent.parent()
        self.tree.setCurrentItem(item)
        self.tree.scrollToItem(item, QAbstractItemView.ScrollHint.EnsureVisible)
        top = self._top_category(item)
        self._current_category = top.text(0)
        self._refresh_category_styles()
        self._update_tree_label()
        return True

    def begin_rename_operator(self, op_id: str) -> bool:
        """对用户算子条目第一列开始原地编辑（编辑框只显示文件 stem）。"""

        entry = self._entry(op_id)
        item = self._entry_items.get(str(op_id))
        if entry is None or item is None or entry.builtin:
            return False
        if not (item.flags() & Qt.ItemFlag.ItemIsEditable):
            return False
        self.select_operator(op_id)
        self.tree.setFocus()
        self.tree.editItem(item, 0)
        return True

    def _select_category_path(self, path: tuple[str, ...]) -> bool:
        if not path:
            return False
        item = self._category_items.get(path[0])
        if item is None:
            return False
        item.setExpanded(True)
        current = item
        for part in path[1:]:
            child: QTreeWidgetItem | None = None
            for index in range(current.childCount()):
                candidate = current.child(index)
                if (
                    candidate.data(0, ROLE_KIND) == _KIND_GROUP
                    and candidate.text(0) == part
                ):
                    child = candidate
                    break
            if child is None:
                break
            child.setExpanded(True)
            current = child
        self.tree.setCurrentItem(current)
        self.tree.scrollToItem(current, QAbstractItemView.ScrollHint.EnsureVisible)
        self._current_category = path[0]
        self._refresh_category_styles()
        self._update_tree_label()
        return True

    def _entry(self, op_id: str) -> OperatorEntry | None:
        if self._registry is None:
            return None
        return self._registry.entry(str(op_id))

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
        self.info_name.setText(getattr(entry, "display_name", entry.name))
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

    # ------------------------------------------------------------------ 库内拖动整理
    def _is_builtin_root(self, root: Path | str) -> bool:
        """判断扫描根是否为内置算子目录（与 loader 保持同一来源）。"""

        try:
            candidate = Path(root).expanduser().resolve()
            builtin = operator_loader.BUILTIN_DIR.expanduser().resolve()
        except OSError:
            return False
        return candidate == builtin

    def _user_scan_roots(self) -> list[Path]:
        """用户扫描根：``registry.scan_roots`` 中排除内置目录，保持扫描顺序。"""

        registry = self._registry
        if registry is not None:
            roots: list[Path] = []
            for raw in registry.scan_roots:
                path = Path(raw).expanduser()
                if self._is_builtin_root(path):
                    continue
                try:
                    resolved = path.resolve()
                except OSError:
                    resolved = path
                if resolved not in roots:
                    roots.append(resolved)
            return roots
        try:
            return [self.user_operator_root().expanduser().resolve()]
        except OSError:
            return [self.user_operator_root().expanduser()]

    def _user_root_for_path(self, path: Path | str) -> Path | None:
        """返回包含该文件 / 目录的用户扫描根（嵌套根取最具体的一个）。"""

        try:
            target = Path(path).expanduser().resolve()
        except OSError:
            return None
        matches: list[Path] = []
        for root in self._user_scan_roots():
            try:
                target.relative_to(root)
            except ValueError:
                continue
            matches.append(root)
        if not matches:
            return None
        return max(matches, key=lambda item: len(item.parts))

    @staticmethod
    def _category_path_from_directory(
        directory: Path | str, root: Path | str
    ) -> tuple[str, ...]:
        try:
            relative = Path(directory).expanduser().resolve().relative_to(
                Path(root).expanduser().resolve()
            )
        except (OSError, ValueError):
            return ()
        return tuple(part for part in relative.parts if part not in ("", "."))

    def _user_category_root(
        self, category: str | tuple[str, ...] | list[str] | None
    ) -> Path | None:
        """分类目录实际存在于哪个用户扫描根下；不存在 / 仅在内置根下返回 None。"""

        parts = _normalize_category(category)
        if not parts:
            return None
        for root in self._user_scan_roots():
            try:
                if root.joinpath(*parts).is_dir():
                    return root
            except OSError:
                continue
        return None

    def _category_in_builtin(
        self, category: str | tuple[str, ...] | list[str] | None
    ) -> bool:
        parts = _normalize_category(category)
        if not parts:
            return False
        try:
            builtin = operator_loader.BUILTIN_DIR.expanduser()
            return builtin.joinpath(*parts).is_dir()
        except OSError:
            return False

    @staticmethod
    def _same_path(left: Path | str, right: Path | str) -> bool:
        try:
            return Path(left).expanduser().resolve() == Path(right).expanduser().resolve()
        except OSError:
            return Path(left) == Path(right)

    @staticmethod
    def _category_display(category: str | tuple[str, ...] | list[str] | None) -> str:
        parts = _normalize_category(category)
        return " / ".join(parts) if parts else "根目录"

    @staticmethod
    def _move_success_message(base: str, outcome) -> str:
        """把核心返回的 warnings 合并进成功提示。"""

        warnings: list[str] = []
        for warning in getattr(outcome, "warnings", ()) or ():
            text = str(warning).strip().rstrip("。；; ")
            if text:
                warnings.append(text)
        if not warnings:
            return base
        return f"{base} 提示：{'；'.join(warnings)}。"

    def drop_target_category(
        self, item: QTreeWidgetItem | None
    ) -> tuple[str, ...]:
        """拖放到某条目时的目标分类路径；空白处返回 ``()``。"""

        if item is None:
            return ()
        path = self._category_path_for_item(item)
        return path or ()

    def library_drag_payload(
        self, item: QTreeWidgetItem | None
    ) -> dict[str, Any] | None:
        """条目可做库内整理拖动时的 payload；内置 / 虚拟条目返回 None。"""

        if item is None:
            return None
        op_id = item.data(0, ROLE_OP_ID)
        if op_id:
            entry = self._entry(str(op_id))
            if entry is None or entry.builtin:
                return None
            if self._user_root_for_path(entry.path) is None:
                return None
            return {"kind": "operator", "id": entry.id}
        kind = item.data(0, ROLE_KIND)
        if kind not in (_KIND_CATEGORY, _KIND_GROUP):
            return None
        path = self._category_path_for_item(item)
        if not path or self._user_category_root(path) is None:
            return None
        return {"kind": "category", "path": list(path)}

    def can_accept_library_drop(
        self,
        payload: dict[str, Any] | None,
        target_item: QTreeWidgetItem | None,
    ) -> bool:
        """判断某个库内拖动 payload 能否落在目标条目上。"""

        if not isinstance(payload, dict):
            return False
        kind = payload.get("kind")
        target = self.drop_target_category(target_item)
        if kind == "operator":
            entry = self._entry(str(payload.get("id") or ""))
            if entry is None or entry.builtin:
                return False
            return self._user_root_for_path(entry.path) is not None
        if kind == "category":
            path = _normalize_category(payload.get("path"))
            if not path or self._user_category_root(path) is None:
                return False
            # 不能把分类拖进自身或后代（含目标行就是自身分类）。
            return not _is_same_or_descendant(target, path)
        return False

    def move_operator_to_category(
        self,
        op_id: str,
        category: str | tuple[str, ...] | list[str] | None,
    ) -> bool:
        """把用户算子文件移动到目标分类目录；失败只发中文状态提示，不弹窗。"""

        entry = self._entry(str(op_id))
        if entry is None:
            self.status_message.emit("移动失败：找不到该算子。")
            return False
        if entry.builtin:
            self.status_message.emit("移动失败：内置算子不能移动。")
            return False
        root = self._user_root_for_path(entry.path)
        if root is None:
            self.status_message.emit("移动失败：找不到该算子所属的用户算子目录。")
            return False
        target = _normalize_category(category)
        try:
            from app.core import operator_files
        except Exception as exc:  # noqa: BLE001 - 并行开发期文件服务可能尚未就绪
            self.status_message.emit(f"移动失败：文件服务不可用（{exc}）")
            return False
        try:
            outcome = operator_files.move_operator_file(
                root,
                entry.path,
                target,
                current_category=tuple(entry.spec.category),
            )
        except Exception as exc:  # noqa: BLE001 - 中文错误直接展示到状态栏
            self.status_message.emit(f"移动失败：{exc}")
            return False
        moved_path = Path(getattr(outcome, "path", entry.path))
        if self._same_path(moved_path, entry.path):
            # 目标目录就是源目录：不刷新、不打扰树，只恢复选中并提示。
            self.select_operator(str(op_id))
            self.status_message.emit(
                f"用户算子“{entry.display_name}”已在目标分类中。"
            )
            return True
        self.refresh_requested.emit()
        self.select_operator(str(op_id))
        message = (
            f"已移动用户算子“{entry.display_name}”到"
            f"“{self._category_display(target)}”。"
        )
        self.status_message.emit(self._move_success_message(message, outcome))
        return True

    def move_category_to_category(
        self,
        category: str | tuple[str, ...] | list[str] | None,
        target_parent: str | tuple[str, ...] | list[str] | None = None,
    ) -> bool:
        """把用户分类目录移动到目标父分类下；失败只发中文状态提示，不弹窗。"""

        source_path = _normalize_category(category)
        if not source_path:
            self.status_message.emit("移动失败：分类路径不能为空。")
            return False
        target = _normalize_category(target_parent)
        root = self._user_category_root(source_path)
        if root is None:
            if self._category_in_builtin(source_path):
                self.status_message.emit("移动失败：内置分类不能移动。")
            else:
                self.status_message.emit(
                    "移动失败：该分类没有对应的用户目录，不能移动。"
                )
            return False
        if _is_same_or_descendant(target, source_path):
            self.status_message.emit("移动失败：不能把分类移动到自身或子分类中。")
            return False
        source_dir = root.joinpath(*source_path)
        try:
            from app.core import operator_files
        except Exception as exc:  # noqa: BLE001 - 并行开发期文件服务可能尚未就绪
            self.status_message.emit(f"移动失败：文件服务不可用（{exc}）")
            return False
        try:
            outcome = operator_files.move_category_dir(root, source_dir, target)
        except Exception as exc:  # noqa: BLE001 - 中文错误直接展示到状态栏
            self.status_message.emit(f"移动失败：{exc}")
            return False
        moved_path = Path(getattr(outcome, "path", source_dir))
        if self._same_path(moved_path, source_dir):
            self._select_category_path(source_path)
            self.status_message.emit(
                f"分类“{' / '.join(source_path)}”已在目标分类中。"
            )
            return True
        self.refresh_requested.emit()
        new_path = self._category_path_from_directory(moved_path, root)
        if not new_path:
            new_path = target + (source_path[-1],)
        self._select_category_path(new_path)
        message = (
            f"已移动分类“{' / '.join(source_path)}”到"
            f"“{self._category_display(target)}”。"
        )
        self.status_message.emit(self._move_success_message(message, outcome))
        return True

    # ------------------------------------------------------------------ 拖拽
    def mime_data_for(self, op_id: str) -> QMimeData:
        """构造拖拽算子使用的 MIME 数据（仅 ``MIME_OPERATOR``，保持兼容）。"""
        mime = QMimeData()
        payload = json.dumps({"id": op_id}, ensure_ascii=False).encode("utf-8")
        mime.setData(MIME_OPERATOR, payload)
        entry = self._registry.entry(op_id) if self._registry is not None else None
        if entry is not None:
            mime.setText(getattr(entry, "display_name", entry.name))
        return mime

    def mime_data_for_item(self, item: QTreeWidgetItem | None) -> QMimeData:
        """按条目构造拖拽 MIME：用户算子带两种格式，用户分类只带库内移动。"""

        mime = QMimeData()
        payload = self.library_drag_payload(item)
        if payload is not None:
            mime.setData(
                MIME_LIBRARY_MOVE,
                json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            )
            if payload.get("kind") == "operator":
                op_id = str(payload.get("id") or "")
                mime.setData(
                    MIME_OPERATOR,
                    json.dumps({"id": op_id}, ensure_ascii=False).encode("utf-8"),
                )
                entry = self._entry(op_id)
                if entry is not None:
                    mime.setText(getattr(entry, "display_name", entry.name))
            else:
                mime.setText(" / ".join(payload.get("path") or []))
            return mime
        # 内置算子不允许库内移动，但仍要能拖到画布创建节点。
        op_id = item.data(0, ROLE_OP_ID) if item is not None else None
        if op_id:
            return self.mime_data_for(str(op_id))
        return mime

    def begin_library_drag(self, source: QWidget, item: QTreeWidgetItem) -> None:
        """从树条目开始拖动：用户条目带库内 payload，内置算子只带画布 payload。"""

        mime = self.mime_data_for_item(item)
        if not (mime.hasFormat(MIME_OPERATOR) or mime.hasFormat(MIME_LIBRARY_MOVE)):
            return
        op_id = item.data(0, ROLE_OP_ID)
        if op_id:
            entry = self._entry(str(op_id))
            name = getattr(entry, "display_name", None) or item.text(0)
            prefix = "＋ "
        else:
            name = self._category_display(self._category_path_for_item(item) or ())
            prefix = "↦ "
        drag = QDrag(source)
        drag.setMimeData(mime)
        drag.setPixmap(self._drag_pixmap(name, prefix=prefix))
        drag.setHotSpot(QPoint(24, 16))
        drag.exec(Qt.DropAction.CopyAction)

    def begin_operator_drag(self, source: QWidget, op_id: str) -> None:
        """开始拖拽（使用 ``MIME_OPERATOR``）。"""
        entry = self._registry.entry(op_id) if self._registry is not None else None
        mime = self.mime_data_for(op_id)
        drag = QDrag(source)
        drag.setMimeData(mime)
        if entry is not None:
            drag.setPixmap(
                self._drag_pixmap(getattr(entry, "display_name", entry.name))
            )
        drag.setHotSpot(QPoint(24, 16))
        drag.exec(Qt.DropAction.CopyAction)

    def _drag_pixmap(self, name: str, *, prefix: str = "＋ ") -> QPixmap:
        font = QFont(self.font())
        font.setBold(True)
        font.setPointSizeF(9.0)
        metrics = QFontMetrics(font)
        width = max(150, metrics.horizontalAdvance(prefix + name) + 32)
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
            prefix + name,
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
