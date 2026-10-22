"""节点画布：QGraphicsView + 场景 + 撤销栈集成。

画布只把 :class:`GraphModel` 当作唯一事实来源：模型增删节点 / 连线时重建或
更新对应条目；用户交互则通过 ``app.graph.commands`` 中的 ``QUndoCommand``
修改模型。这样撤销 / 重做、打开工程和外部脚本修改都能保持一致。

对外接口见 ``docs/architecture.md``（GraphView 一节）。
"""

from __future__ import annotations

import copy
import json
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QKeySequence,
    QMouseEvent,
    QPainter,
    QUndoCommand,
    QUndoStack,
)
from PySide6.QtWidgets import QApplication, QFrame, QGraphicsItem, QGraphicsView, QInputDialog, QMenu

from app.config import AppSettings
from app.core.constants import (
    GROUP_OP_ID,
    MIME_IMAGE_PATHS,
    MIME_OPERATOR,
    SINGLE_CLICK_DELAY_MS,
)
from app.core.data_types import DataType
from app.core.group import INVALID_LEVEL, GroupModel
from app.core.model import ConnectionModel, GraphError, GraphModel, NodeModel, WorkspaceModel, new_id
from app.graph.collapse_item import CollapseBadgeItem
from app.graph.commands import (
    AddConnectionCommand,
    AddNodeCommand,
    DissolveGroupCommand,
    InsertNodeOnEdgeCommand,
    MoveNodeCommand,
    ReconnectGroupMoveCommand,
    RemoveConnectionCommand,
    RemoveNodeCommand,
    RemoveNodeTreeCommand,
    RenameNodeCommand,
    ReplaceOperatorCommand,
    SetParentGroupCommand,
)
from app.graph.connection_item import ConnectionItem
from app.graph.group_item import (
    DEFAULT_GROUP_HEIGHT,
    DEFAULT_GROUP_WIDTH,
    GROUP_PADDING_BOTTOM,
    GROUP_PADDING_TOP,
    GROUP_PADDING_X,
    GroupItem,
)
from app.graph.layout import compute_layout
from app.graph.node_item import NodeItem
from app.graph.port_item import PortItem
from app.graph.scene import GraphScene
from app.operators.loader import OperatorRegistry
from app.operators.spec import OperatorSpec, PortSpec, connection_error, param_defaults, port_types_compatible

# 算子组模块 / 替换对话框由其它协作方实现；延迟导入保证画布可独立加载。
try:  # pragma: no cover - 集成后始终可用
    from app.operators.group import (
        GroupOperator,
        is_group_operator,
        materialize_group_operator,
        save_group_operator,
    )
except Exception:  # noqa: BLE001
    GroupOperator = None  # type: ignore[assignment]
    is_group_operator = None  # type: ignore[assignment]
    materialize_group_operator = None  # type: ignore[assignment]
    save_group_operator = None  # type: ignore[assignment]

try:  # pragma: no cover - shell-ui 尚未完成时保持可导入
    from app.ui.replace_operator import ReplaceOperatorDialog
except Exception:  # noqa: BLE001
    ReplaceOperatorDialog = None  # type: ignore[assignment]


def _ensure_group_api() -> bool:
    """按需加载算子组接口（测试环境可提前 patch 模块级函数）。"""

    global GroupOperator, is_group_operator, materialize_group_operator, save_group_operator
    if all(
        item is not None
        for item in (is_group_operator, materialize_group_operator, save_group_operator)
    ):
        return True
    try:
        from app.operators.group import (
            GroupOperator as _GroupOperator,
            is_group_operator as _is_group_operator,
            materialize_group_operator as _materialize_group_operator,
            save_group_operator as _save_group_operator,
        )
    except Exception:  # noqa: BLE001
        return False
    if GroupOperator is None:
        GroupOperator = _GroupOperator
    if is_group_operator is None:
        is_group_operator = _is_group_operator
    if materialize_group_operator is None:
        materialize_group_operator = _materialize_group_operator
    if save_group_operator is None:
        save_group_operator = _save_group_operator
    return True


def _ensure_replace_dialog():
    global ReplaceOperatorDialog
    if ReplaceOperatorDialog is not None:
        return ReplaceOperatorDialog
    try:
        from app.ui.replace_operator import ReplaceOperatorDialog as _ReplaceOperatorDialog
    except Exception:  # noqa: BLE001
        return None
    ReplaceOperatorDialog = _ReplaceOperatorDialog
    return ReplaceOperatorDialog

_IMAGE_SUFFIXES = {
    ".png",
    ".jpg",
    ".jpeg",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
    ".ppm",
    ".pgm",
    ".npy",
}

#: 拖动时吸附到端口 / 连线的场景半径（像素）。
SNAP_RADIUS = 26.0
#: 自动连线时新节点端口与目标端口之间的垂直间距。
AUTO_CONNECT_GAP = 48.0
#: 节点复制粘贴的剪贴板数据标识。
NODE_CLIPBOARD_KIND = "mineimage/nodes"
#: 粘贴时相对原位置的级联偏移。
PASTE_CASCADE = 36.0
#: 从输出端口拖到空白处时自动创建的输出节点算子。
OUTPUT_NODE_OP_ID = "output.display"
#: 拖动节点 / 连线时，鼠标进入视口边缘多少像素内开始自动滚动。
EDGE_SCROLL_MARGIN = 36.0
#: 自动滚动每个 tick 的最大像素步长（越靠近边缘越快）。
EDGE_SCROLL_STEP = 16
#: 自动滚动定时器间隔（毫秒）。
EDGE_SCROLL_INTERVAL_MS = 30


@dataclass(frozen=True)
class _DropTarget:
    """库拖拽命中的目标：端口（kind=input/output）或已有连线。"""

    kind: str
    port: PortItem | None = None
    connection: ConnectionItem | None = None


@dataclass(frozen=True)
class _MoveTarget:
    """现有节点拖动过程中待执行的自动连线 / 插入动作。"""

    kind: str  # "connect" | "insert"
    src_node: str = ""
    src_port: str = ""
    dst_node: str = ""
    dst_port: str = ""
    ports: tuple[PortItem, ...] = ()
    connection_id: str = ""
    node_id: str = ""
    input_port: str = ""
    output_port: str = ""


class GraphView(QGraphicsView):
    """节点画布视图。"""

    selection_changed = Signal(str)  # "" 表示取消选择
    node_activated = Signal(str)  # 双击节点
    node_created = Signal(str)  # 画布内新增算子节点
    connection_changed = Signal(str)  # 目标节点 id
    node_removed = Signal(str)
    images_dropped = Signal(list)  # list[str]，拖入的图像路径
    status_message = Signal(str)
    operators_changed = Signal()  # 保存新的算子组自定义算子后通知主窗口刷新算子库

    def __init__(
        self,
        workspace: WorkspaceModel,
        registry: OperatorRegistry,
        undo_stack: QUndoStack,
        settings: AppSettings,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.workspace = workspace
        self.graph: GraphModel = workspace.graph
        self._registry = registry
        self._undo_stack = undo_stack
        self._settings = settings

        self._nodes: dict[str, NodeItem] = {}
        self._connections: dict[str, ConnectionItem] = {}
        self._collapse_badges: dict[str, CollapseBadgeItem] = {}
        self._collapsed: set[str] = set()
        self._results: dict[str, Any] = {}
        self._errors: dict[str, str] = {}
        self._thumbnails_enabled = bool(settings.show_node_thumbnails)
        self._syncing_selection = False
        self._panning = False
        self._pan_last = QPointF()
        self._pan_dragged = False
        self._last_drag_view_pos = QPointF()
        self._move_start: dict[str, QPointF] = {}
        self._pending_move_target: _MoveTarget | None = None
        self._pending_move_reason: str | None = None
        self._highlighted_ports: list[PortItem] = []
        self._highlighted_connection: ConnectionItem | None = None
        self._paste_count = 0
        self._zoom_min = 0.15
        self._zoom_max = 3.0
        #: 节点条目创建时的 (op_id, 是否算子组) 签名，用于替换算子后重建条目。
        self._item_signatures: dict[str, tuple[str, bool]] = {}
        #: 拖动算子组时，本帧上一次的条目位置（用于实时平移子节点）。
        self._group_live_last: dict[str, QPointF] = {}
        self._suppress_move_events = False
        self._syncing_group_geometry = False

        self._scene = GraphScene(self)
        self.setScene(self._scene)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setRenderHints(
            QPainter.RenderHint.Antialiasing
            | QPainter.RenderHint.TextAntialiasing
            | QPainter.RenderHint.SmoothPixmapTransform
        )
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setRubberBandSelectionMode(Qt.ItemSelectionMode.IntersectsItemShape)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAcceptDrops(True)
        self.setMouseTracking(True)
        # 隐藏滚动条，让画布看起来是一整块可自由拖动的画布；平移仍通过
        # 左键拖动空白处 / Shift+滚轮完成（滚动条只是不显示，范围仍有效）。
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._selection_timer = QTimer(self)
        self._selection_timer.setSingleShot(True)
        self._selection_timer.setInterval(int(SINGLE_CLICK_DELAY_MS))
        self._selection_timer.timeout.connect(self._emit_selection_changed)
        self._scene.selectionChanged.connect(self._on_scene_selection_changed)

        # 拖动节点 / 连线 / 框选时，靠近视口边缘自动滚动画布。
        self._edge_scroll_timer = QTimer(self)
        self._edge_scroll_timer.setInterval(EDGE_SCROLL_INTERVAL_MS)
        self._edge_scroll_timer.timeout.connect(self._auto_scroll_during_drag)

        self.graph.node_added.connect(self._on_node_added)
        self.graph.node_removed.connect(self._on_node_removed)
        self.graph.node_changed.connect(self._on_node_changed)
        self.graph.connection_added.connect(self._on_connection_added)
        self.graph.connection_removed.connect(self._on_connection_removed)
        self.graph.set_port_type_resolver(self._resolve_port_type)
        self._undo_stack.indexChanged.connect(self._on_undo_stack_changed)

        self.refresh_all()
        self._update_scene_rect()

    # ------------------------------------------------------------------ 设置
    def set_registry(self, registry: OperatorRegistry) -> None:
        self._registry = registry
        self.refresh_all()

    # --------------------------------------------------------------- 接口实现
    def selected_node_id(self) -> str | None:
        for item in self._scene.selectedItems():
            if isinstance(item, NodeItem):
                return item.node_id
        return None

    def set_selected_node(self, node_id: str | None) -> None:
        """程序化选中节点（不发出 ``selection_changed``，避免调用方循环）。"""

        self._selection_timer.stop()
        self._syncing_selection = True
        try:
            self._scene.clearSelection()
            item = self._nodes.get(node_id or "")
            if item is not None:
                item.setSelected(True)
                self.ensureVisible(item, 24, 24)
        finally:
            self._syncing_selection = False

    # ------------------------------------------------------------ 复制 / 粘贴
    def copy_selected(self) -> bool:
        """复制选中的节点及它们之间的连线到系统剪贴板。"""

        node_ids = [
            node.id
            for node in self.graph.nodes.values()
            if node.id in self._nodes and self._nodes[node.id].isSelected()
        ]
        if not node_ids:
            self.status_message.emit("请先选择要复制的节点")
            return False
        selected = set(node_ids)
        # 复制折叠点时连同被隐藏的下游节点一起复制。
        for node_id in list(selected):
            if node_id in self._collapsed:
                selected.update(self.graph.descendants(node_id))
        # 复制算子组时连同全部后代一起复制（parent_group 在粘贴时重映射）。
        for node_id in list(selected):
            node = self.graph.nodes.get(node_id)
            if node is not None and getattr(node, "group", None) is not None:
                selected.update(child.id for child in self.graph.group_subtree(node_id))
        node_ids = [node.id for node in self.graph.nodes.values() if node.id in selected]
        payload = {
            "kind": NODE_CLIPBOARD_KIND,
            "nodes": [copy.deepcopy(self.graph.nodes[node_id].to_dict()) for node_id in node_ids],
            "connections": [
                copy.deepcopy(connection.to_dict())
                for connection in self.graph.connections.values()
                if connection.src_node in selected and connection.dst_node in selected
            ],
        }
        QApplication.clipboard().setText(
            json.dumps(payload, ensure_ascii=False, default=str)
        )
        self._paste_count = 0
        self.status_message.emit(f"已复制 {len(node_ids)} 个节点")
        return True

    def paste_nodes(self) -> list[str]:
        """从剪贴板粘贴节点，保留副本之间的连线；整次粘贴是一步撤销。"""

        data = self._clipboard_node_payload()
        if data is None:
            self.status_message.emit("剪贴板中没有可粘贴的节点")
            return []

        source_nodes: list[NodeModel] = []
        skipped = 0
        for raw in data["nodes"]:
            try:
                node = NodeModel.from_dict(raw)
            except (KeyError, TypeError, ValueError):
                continue
            if (
                node.is_source
                and node.source_image_id
                and node.source_image_id not in self.workspace.images
            ):
                skipped += 1
                continue
            source_nodes.append(node)
        if not source_nodes:
            self.status_message.emit("剪贴板中的节点无法粘贴到此工作区")
            return []

        self._paste_count += 1
        min_x = min(node.pos[0] for node in source_nodes)
        min_y = min(node.pos[1] for node in source_nodes)
        max_x = max(node.pos[0] for node in source_nodes)
        max_y = max(node.pos[1] for node in source_nodes)
        origin = self._paste_origin(min_x, min_y, max_x - min_x, max_y - min_y)
        dx = origin.x() - min_x
        dy = origin.y() - min_y

        id_map: dict[str, str] = {node.id: new_id("n") for node in source_nodes}
        clones: list[NodeModel] = []
        for node in source_nodes:
            clone = NodeModel.from_dict(node.to_dict())
            clone.id = id_map[node.id]
            clone.pos = (round(node.pos[0] + dx, 2), round(node.pos[1] + dy, 2))
            old_parent = node.parent_group
            if node.is_source:
                clone.parent_group = None
            elif old_parent is not None and old_parent in id_map:
                clone.parent_group = id_map[old_parent]
            else:
                # 父算子组没有一起复制：粘贴为顶层，避免错误挂到原算子组。
                clone.parent_group = None
            clones.append(clone)
        cloned_by_id = {clone.id: clone for clone in clones}

        def _clone_depth(clone: NodeModel) -> int:
            depth = 0
            parent = clone.parent_group
            seen = {clone.id}
            while parent is not None and parent not in seen:
                seen.add(parent)
                depth += 1
                parent_model = cloned_by_id.get(parent)
                if parent_model is None:
                    break
                parent = parent_model.parent_group
            return depth

        clones.sort(key=_clone_depth)
        new_ids: list[str] = []
        self._undo_stack.beginMacro("粘贴节点")
        try:
            for clone in clones:
                new_ids.append(clone.id)
                self._undo_stack.push(AddNodeCommand(self.graph, clone))

            for raw in data.get("connections", []):
                try:
                    src_node = id_map[str(raw["src_node"])]
                    dst_node = id_map[str(raw["dst_node"])]
                    src_port = str(raw["src_port"])
                    dst_port = str(raw["dst_port"])
                except (KeyError, TypeError, ValueError):
                    continue
                try:
                    self._undo_stack.push(
                        AddConnectionCommand(
                            self.graph, src_node, src_port, dst_node, dst_port
                        )
                    )
                except GraphError as exc:
                    self.status_message.emit(f"粘贴连线失败：{exc}")
        finally:
            self._undo_stack.endMacro()

        self._sync_group_geometries()
        self._select_nodes(new_ids, new_ids[-1])
        self.node_created.emit(new_ids[-1])
        message = f"已粘贴 {len(new_ids)} 个节点"
        if skipped:
            message += f"（跳过 {skipped} 个源图像节点：当前工作区缺少对应图像）"
        self.status_message.emit(message)
        return new_ids

    def _paste_origin(
        self,
        min_x: float,
        min_y: float,
        width: float,
        height: float,
    ) -> QPointF:
        """粘贴位置：鼠标在画布内时以光标为中心，否则按原位置级联偏移。"""

        viewport_pos = self.viewport().mapFromGlobal(QCursor.pos())
        if self.viewport().underMouse() and self.viewport().rect().contains(viewport_pos):
            center = self.mapToScene(viewport_pos)
            cascade = PASTE_CASCADE * self._paste_count
            return QPointF(
                center.x() - width / 2.0 + cascade,
                center.y() - height / 2.0 + cascade,
            )
        offset = PASTE_CASCADE * self._paste_count
        return QPointF(min_x + offset, min_y + offset)

    def _clipboard_node_payload(self) -> dict[str, Any] | None:
        text = QApplication.clipboard().text().strip()
        if not text:
            return None
        try:
            data = json.loads(text)
        except (TypeError, ValueError):
            return None
        if not isinstance(data, dict) or data.get("kind") != NODE_CLIPBOARD_KIND:
            return None
        if not isinstance(data.get("nodes"), list) or not data["nodes"]:
            return None
        if not isinstance(data.get("connections"), list):
            data["connections"] = []
        return data

    def _select_nodes(self, node_ids: list[str], primary: str | None = None) -> None:
        self._selection_timer.stop()
        self._syncing_selection = True
        try:
            self._scene.clearSelection()
            for node_id in node_ids:
                item = self._nodes.get(node_id)
                if item is not None:
                    item.setSelected(True)
            primary_item = self._nodes.get(primary or "")
            if primary_item is not None:
                self.ensureVisible(primary_item, 24, 24)
        finally:
            self._syncing_selection = False
        self.selection_changed.emit(primary or (node_ids[-1] if node_ids else ""))

    def notify_result(self, node_id: str, result: Any) -> None:
        self._results[node_id] = result
        self._errors.pop(node_id, None)
        item = self._nodes.get(node_id)
        if item is not None:
            item.set_result(result, self._thumbnails_enabled)
            self._update_connections_for(node_id)

    def notify_error(self, node_id: str, message: str | None) -> None:
        if message:
            self._errors[node_id] = str(message)
        else:
            self._errors.pop(node_id, None)
        item = self._nodes.get(node_id)
        if item is not None:
            item.set_error(message)

    def set_thumbnails_enabled(self, enabled: bool) -> None:
        self._thumbnails_enabled = bool(enabled)
        box = self._thumbnail_box()
        for item in self._nodes.values():
            item.set_thumbnails_enabled(self._thumbnails_enabled, box)
        self._update_scene_rect()

    def add_operator_at(self, op_id: str, scene_pos: QPointF | None = None) -> str | None:
        if scene_pos is None:
            scene_pos = self.mapToScene(self.viewport().rect().center())
        scene_pos = QPointF(scene_pos)
        # 自定义算子组：展开成算子组 + 内部节点 + 连线，整次一步撤销。
        if self._is_group_operator_id(op_id):
            return self._add_group_operator_at(op_id, scene_pos)
        spec = self._registry.spec(op_id)
        if spec is None:
            self.status_message.emit(f"未找到算子：{op_id}")
            return None

        target = self._drop_target_at(scene_pos, spec)
        if target is not None:
            if target.kind == "edge" and target.connection is not None:
                if (
                    spec.inputs
                    and spec.outputs
                    and self._insertion_ports(spec, target.connection) is not None
                ):
                    node_id = self._insert_node_on_edge(target.connection, op_id, spec, scene_pos)
                    if node_id is not None:
                        return node_id
            elif target.port is not None:
                node_id = self._add_operator_with_port_connection(op_id, spec, target)
                if node_id is not None:
                    return node_id

        # 自动连接失败（例如目标输入已被占用）时，普通放置要避让目标节点，
        # 不能把新节点直接压在原节点上：放到目标节点右侧并留出间距。
        fallback_pos = scene_pos
        if target is not None and target.port is not None:
            target_item = target.port.node_item
            target_rect = QRectF(target_item.pos(), target_item.size())
            fallback_pos = QPointF(target_rect.right() + 32.0, target_rect.top())
        if target is not None and target.port is not None:
            parent_group = self._scope_for_port_target(target.port, target.kind)
        else:
            parent_group = self.group_at(fallback_pos)
        node = NodeModel(
            id=new_id("n"),
            op_id=op_id,
            title=spec.name,
            pos=(fallback_pos.x(), fallback_pos.y()),
            params=param_defaults(spec),
            parent_group=parent_group,
        )
        self._undo_stack.push(AddNodeCommand(self.graph, node))
        self._sync_group_geometries()
        self.set_selected_node(node.id)
        self.node_created.emit(node.id)
        self.status_message.emit(f"已添加节点“{spec.name}”")
        return node.id

    def drop_connection_on_blank(self, source: PortItem, scene_pos: QPointF) -> str | None:
        """输出端口拖到空白处：自动创建输出节点并连接。"""

        if not self.viewport().rect().contains(self.mapFromScene(scene_pos)):
            return None
        if self.item_at_scene(scene_pos) is not None:
            self.status_message.emit(
                "未找到可连接的输入端口：请拖到目标节点的输入端口上，或拖到空白处自动创建输出节点"
            )
            return None
        return self.add_output_node_at(source.node_item.node_id, source.key, scene_pos)

    def add_output_node_at(
        self, src_node_id: str, src_port: str, scene_pos: QPointF
    ) -> str | None:
        """在指定场景位置创建输出节点，并接到该输出端口上（整次一步撤销）。"""

        spec = self._registry.spec(OUTPUT_NODE_OP_ID)
        src_node = self.graph.nodes.get(src_node_id)
        if spec is None or src_node is None:
            self.status_message.emit("未找到“输出预览”算子，无法自动创建输出节点")
            return None
        if not self._port_exists(src_node, src_port, is_input=False):
            return None
        src_spec = self.port_spec(src_node_id, src_port, False)
        input_key = next(
            (
                port.key
                for port in spec.inputs
                if src_spec is None or port_types_compatible(src_spec, port)
            ),
            None,
        )
        if input_key is None:
            self.status_message.emit("输出节点没有能接收该数据的输入端口，未自动创建")
            return None

        scene_pos = QPointF(scene_pos)
        parent_group = self.group_at(scene_pos)
        source_level = self.graph.port_level(src_node_id, src_port, "src")
        if source_level != INVALID_LEVEL and source_level != parent_group:
            self.status_message.emit(
                "只能在同一算子组层级内连接；请通过算子组的输入 / 输出端口连接跨组数据"
            )
            return None
        node = NodeModel(
            id=new_id("n"),
            op_id=OUTPUT_NODE_OP_ID,
            title=spec.name,
            # 输入端口在节点上边缘且单端口居中：让落点正好落在输入端口上。
            pos=(scene_pos.x() - NodeItem.WIDTH / 2.0, scene_pos.y()),
            params=param_defaults(spec),
            parent_group=parent_group,
        )
        self._undo_stack.beginMacro("添加输出节点并连接")
        try:
            self._undo_stack.push(AddNodeCommand(self.graph, node))
            self._undo_stack.push(
                AddConnectionCommand(self.graph, src_node_id, src_port, node.id, input_key)
            )
        except GraphError as exc:
            self._undo_stack.endMacro()
            self.status_message.emit(f"自动创建输出节点失败：{exc}")
            return None
        self._undo_stack.endMacro()

        self._sync_group_geometries()
        self._update_scene_rect()
        self._select_nodes([node.id], node.id)
        self.status_message.emit(
            f"已添加输出节点并连接：{src_node.display_title()} → {spec.name}"
        )
        return node.id

    def refresh_all(self) -> None:
        selected = self.selected_node_id()
        self._clear_items()
        for node in list(self.graph.nodes.values()):
            self._create_node_item(node)
        for connection in list(self.graph.connections.values()):
            self._create_connection_item(connection)
        self._sync_group_geometries()
        self._refresh_collapse_state()
        if selected and selected in self._nodes and self._nodes[selected].isVisible():
            self.set_selected_node(selected)

    # ------------------------------------------------------------ 算子组基础设施
    def is_operator_registered(self, op_id: str | None) -> bool:
        """算子 id 是否已注册（失效节点 / 缺失依赖判断用）。"""

        if not op_id:
            return False
        try:
            return self._registry.spec(op_id) is not None
        except Exception:  # noqa: BLE001 - 最小化测试注册表
            return False

    def _resolve_port_type(self, node_id: str, port_key: str, is_input: bool) -> DataType | None:
        """注入 GraphModel 的端口类型解析器：由注册表 + 算子组端口决定类型。"""

        spec = self.port_spec(node_id, port_key, is_input)
        if spec is None or spec.any_type or spec.data_type is None:
            return None
        return spec.data_type

    def _group_nodes(self) -> list[NodeModel]:
        return [node for node in self.graph.nodes.values() if getattr(node, "group", None) is not None]

    def _group_depth(self, group_id: str) -> int:
        depth = 0
        current = self.graph.nodes.get(group_id)
        seen: set[str] = set()
        while current is not None and current.parent_group:
            if current.parent_group in seen:
                break
            seen.add(current.parent_group)
            parent = self.graph.nodes.get(current.parent_group)
            if parent is None or getattr(parent, "group", None) is None:
                break
            depth += 1
            current = parent
        return depth

    def _group_ancestors(self, node_id: str) -> list[str]:
        ancestors: list[str] = []
        current = self.graph.nodes.get(node_id)
        seen: set[str] = set()
        while current is not None and current.parent_group:
            parent_id = current.parent_group
            if parent_id in seen:
                break
            seen.add(parent_id)
            parent = self.graph.nodes.get(parent_id)
            if parent is None:
                break
            ancestors.append(parent_id)
            current = parent
        return ancestors

    def _group_item_rect(self, group_id: str) -> QRectF | None:
        item = self._nodes.get(group_id)
        if item is None:
            return None
        return QRectF(item.pos(), item.size())

    def group_at(self, scene_pos: QPointF, *, exclude: Iterable[str] = ()) -> str | None:
        """返回场景坐标命中的最深算子组 id（支持嵌套；忽略排除项）。"""

        excluded = set(exclude)
        best: str | None = None
        best_depth = -1
        for node in self._group_nodes():
            if node.id in excluded:
                continue
            rect = self._group_item_rect(node.id)
            if rect is None or not rect.contains(scene_pos):
                continue
            depth = self._group_depth(node.id)
            if depth > best_depth:
                best = node.id
                best_depth = depth
        return best

    def _scope_for_port_target(self, port: PortItem, kind: str) -> str | None:
        """库拖拽吸附到端口时，新节点应处的算子组层级。"""

        role = "dst" if kind == "input" else "src"
        level = self.graph.port_level(port.node_item.node_id, port.key, role)
        if level is None or level == INVALID_LEVEL:
            return None
        return level

    def _sync_group_geometries(self) -> None:
        """按直接子节点包围盒反推算子组位置 / 尺寸（不写入撤销栈）。"""

        if self._syncing_group_geometry:
            return
        self._syncing_group_geometry = True
        self._suppress_move_events = True
        try:
            groups = self._group_nodes()
            groups.sort(key=lambda node: self._group_depth(node.id), reverse=True)
            for group_node in groups:
                item = self._nodes.get(group_node.id)
                if not isinstance(item, GroupItem):
                    continue
                children = self.graph.direct_children(group_node.id)
                if not children:
                    item.set_group_geometry(DEFAULT_GROUP_WIDTH, DEFAULT_GROUP_HEIGHT)
                    continue
                content = QRectF()
                for child in children:
                    child_item = self._nodes.get(child.id)
                    if child_item is None:
                        continue
                    child_rect = QRectF(child_item.pos(), child_item.size())
                    content = child_rect if content.isNull() else content.united(child_rect)
                if content.isNull():
                    item.set_group_geometry(DEFAULT_GROUP_WIDTH, DEFAULT_GROUP_HEIGHT)
                    continue
                width = content.width() + 2.0 * GROUP_PADDING_X
                height = content.height() + GROUP_PADDING_TOP + GROUP_PADDING_BOTTOM
                new_pos = (
                    round(content.left() - GROUP_PADDING_X, 2),
                    round(content.top() - GROUP_PADDING_TOP, 2),
                )
                if (
                    abs(group_node.pos[0] - new_pos[0]) > 0.01
                    or abs(group_node.pos[1] - new_pos[1]) > 0.01
                ):
                    self.graph.set_node_pos(group_node.id, new_pos, touch=False)
                item.sync_position(new_pos)
                item.set_group_geometry(width, height)
                self._update_connections_for(group_node.id)
                self._position_collapse_badge(group_node.id)
        finally:
            self._suppress_move_events = False
            self._syncing_group_geometry = False
        self._update_scene_rect()

    def _refresh_group_geometry(self, group_id: str | None = None) -> None:
        self._sync_group_geometries()

    def on_group_item_geometry_changed(self, item: GroupItem) -> None:
        self._update_connections_for(item.node_id)
        self._position_collapse_badge(item.node_id)

    def _on_undo_stack_changed(self, _index: int = 0) -> None:
        # 撤销 / 重做后重新按子节点反推算子组几何，保证模型与条目一致。
        self._sync_group_geometries()

    # ------------------------------------------------------------ 新建算子组
    def add_group_at(
        self,
        scene_pos: QPointF | None = None,
        *,
        parent_group: str | None = None,
        title: str = "算子组",
    ) -> str:
        """在场景坐标中心位置新建一个算子组（可指定父算子组实现嵌套）。"""

        if scene_pos is None:
            scene_pos = self.mapToScene(self.viewport().rect().center())
        scene_pos = QPointF(scene_pos)
        if parent_group is not None:
            parent = self.graph.nodes.get(parent_group)
            if parent is None or getattr(parent, "group", None) is None:
                self.status_message.emit("父算子组不存在，已在顶层创建算子组")
                parent_group = None
        node = NodeModel(
            id=new_id("n"),
            op_id=GROUP_OP_ID,
            title=str(title or "算子组"),
            pos=(
                round(scene_pos.x() - DEFAULT_GROUP_WIDTH / 2.0, 2),
                round(scene_pos.y() - DEFAULT_GROUP_HEIGHT / 2.0, 2),
            ),
            group=GroupModel.create_default(),
            parent_group=parent_group,
        )
        try:
            self._undo_stack.push(AddNodeCommand(self.graph, node))
        except GraphError as exc:
            self.status_message.emit(f"添加算子组失败：{exc}")
            return ""
        self._sync_group_geometries()
        self.set_selected_node(node.id)
        self.node_created.emit(node.id)
        self.status_message.emit(f"已添加算子组“{node.display_title()}”")
        return node.id

    def _is_group_operator_id(self, op_id: str) -> bool:
        if not _ensure_group_api() or is_group_operator is None:
            return False
        candidates: list[Any] = []
        getter = getattr(self._registry, "get", None)
        if callable(getter):
            try:
                candidates.append(getter(op_id))
            except Exception:  # noqa: BLE001
                pass
        entry_getter = getattr(self._registry, "entry", None)
        if callable(entry_getter):
            try:
                entry = entry_getter(op_id)
                candidates.append(getattr(entry, "operator", entry))
            except Exception:  # noqa: BLE001
                pass
        candidates.append(self._registry.spec(op_id))
        for candidate in candidates:
            if candidate is None:
                continue
            try:
                if is_group_operator(candidate):
                    return True
            except Exception:  # noqa: BLE001
                continue
        return False

    def _add_group_operator_at(self, op_id: str, scene_pos: QPointF) -> str | None:
        """从算子库拖入算子组自定义算子：一步撤销生成算子组 + 内部节点 + 连线。"""

        if not _ensure_group_api() or materialize_group_operator is None:
            self.status_message.emit("当前环境未提供算子组展开接口")
            return None
        parent_group = self.group_at(scene_pos)
        try:
            materialization = materialize_group_operator(
                self._registry,
                op_id,
                pos=(float(scene_pos.x()), float(scene_pos.y())),
                parent_group=parent_group,
            )
        except Exception as exc:  # noqa: BLE001 - 定义错误需要提示给用户
            self.status_message.emit(f"展开算子组失败：{exc}")
            return None
        nodes = list(materialization.nodes)
        connections = list(materialization.connections)
        by_id = {node.id: node for node in nodes}
        nodeset = set(by_id)

        def local_depth(node: NodeModel) -> int:
            depth = 0
            parent = node.parent_group
            seen = {node.id}
            while parent in nodeset and parent not in seen:
                seen.add(parent)
                depth += 1
                parent_model = by_id.get(parent)
                if parent_model is None:
                    break
                parent = parent_model.parent_group
            return depth

        ordered = sorted(nodes, key=local_depth)
        self._undo_stack.beginMacro("添加算子组")
        try:
            for node in ordered:
                self._undo_stack.push(AddNodeCommand(self.graph, node))
            for connection in connections:
                self._undo_stack.push(
                    AddConnectionCommand(
                        self.graph,
                        connection.src_node,
                        connection.src_port,
                        connection.dst_node,
                        connection.dst_port,
                    )
                )
        finally:
            self._undo_stack.endMacro()
        group_id = materialization.group_node.id
        self._sync_group_geometries()
        if group_id in self._nodes:
            self.set_selected_node(group_id)
        self.node_created.emit(group_id)
        self.status_message.emit(
            f"已添加算子组“{materialization.group_node.display_title()}”"
        )
        return group_id

    # -------------------------------------------------------- 保存为自定义算子
    def save_group_as_operator(self, group_id: str) -> None:
        """右键菜单：把算子组保存为自定义算子。"""

        node = self.graph.nodes.get(group_id)
        if node is None or not node.is_group or node.group is None:
            self.status_message.emit("只能把算子组保存为新算子")
            return
        sources = [
            item
            for item in [node, *self.graph.group_subtree(group_id)]
            if getattr(item, "is_source", False)
        ]
        if sources:
            self.status_message.emit(
                "算子组内包含原始图像节点，无法保存为新算子；请先把源图像节点移出算子组"
            )
            return
        if not _ensure_group_api() or save_group_operator is None:
            self.status_message.emit("当前环境未提供算子组保存接口")
            return
        name, ok = QInputDialog.getText(
            self,
            "保存为新算子",
            "算子名称：",
            text=node.display_title(),
        )
        if not ok:
            return
        name = str(name or "").strip()
        if not name:
            self.status_message.emit("算子名称不能为空")
            return
        try:
            saved = save_group_operator(self.graph, group_id, name, registry=self._registry)
        except Exception as exc:  # noqa: BLE001 - 保存失败需要提示给用户
            self.status_message.emit(f"保存为新算子失败：{exc}")
            return
        self.status_message.emit(f"已保存算子“{saved.name}”（{saved.op_id}）")
        self.operators_changed.emit()

    # ------------------------------------------------------------ 替换失效算子
    def replace_operator(self, node_id: str, op_id: str) -> bool:
        """把失效 / 占位节点替换成普通算子；尽量保留可复用连线，一步撤销。"""

        node = self.graph.nodes.get(node_id)
        if node is None:
            return False
        spec = self._registry.spec(op_id)
        if spec is None:
            self.status_message.emit(f"未找到算子：{op_id}")
            return False
        if self._is_group_operator_id(op_id):
            self.status_message.emit("算子组不能用于替换节点，请从算子库直接拖入画布")
            return False

        old_connections = [
            connection
            for connection in self.graph.connections.values()
            if node_id in (connection.src_node, connection.dst_node)
        ]
        reusable: list[tuple[str, str, str, str]] = []
        used_inputs: set[str] = set()
        for connection in old_connections:
            if connection.dst_node == node_id:
                source_spec = self.port_spec(connection.src_node, connection.src_port, False)
                for port in spec.inputs:
                    if port.key in used_inputs:
                        continue
                    if (
                        source_spec is None
                        or source_spec.any_type
                        or port.any_type
                        or port_types_compatible(source_spec, port)
                    ):
                        reusable.append(
                            (connection.src_node, connection.src_port, node_id, port.key)
                        )
                        used_inputs.add(port.key)
                        break
            elif connection.src_node == node_id:
                target_spec = self.port_spec(connection.dst_node, connection.dst_port, True)
                for port in spec.outputs:
                    if (
                        target_spec is None
                        or target_spec.any_type
                        or port.any_type
                        or port_types_compatible(port, target_spec)
                    ):
                        reusable.append(
                            (node_id, port.key, connection.dst_node, connection.dst_port)
                        )
                        break
        command = ReplaceOperatorCommand(
            self.graph,
            node_id,
            op_id=op_id,
            title=spec.name,
            params=param_defaults(spec),
            reusable_connections=reusable,
        )
        self._undo_stack.push(command)
        self._sync_group_geometries()
        self.status_message.emit(f"已将节点替换为“{spec.name}”")
        return True

    def _prompt_replace_operator(self, node_id: str) -> None:
        node = self.graph.nodes.get(node_id)
        if node is None:
            return
        dialog_cls = _ensure_replace_dialog()
        if dialog_cls is None:
            self.status_message.emit("替换算子对话框不可用")
            return
        try:
            dialog = dialog_cls(self._registry, current_op_id=node.op_id, parent=self.window())
        except TypeError:
            dialog = dialog_cls(self._registry, current_op_id=node.op_id)
        if not dialog.exec():
            return
        try:
            selected = dialog.selected_op_id()
        except Exception:  # noqa: BLE001
            selected = None
        if selected:
            self.replace_operator(node_id, str(selected))

    def _recreate_node_item(self, node_id: str) -> None:
        node = self.graph.nodes.get(node_id)
        old = self._nodes.pop(node_id, None)
        was_selected = bool(old is not None and old.isSelected())
        if old is not None and old.scene() is self._scene:
            self._scene.removeItem(old)
        self._item_signatures.pop(node_id, None)
        if node is None:
            return
        item = self._create_node_item(node)
        if was_selected:
            self._syncing_selection = True
            try:
                item.setSelected(True)
            finally:
                self._syncing_selection = False
        for connection in self._connections.values():
            if node_id in (connection.src_node, connection.dst_node):
                connection.update_path()
        self._update_connections_for(node_id)

    # ------------------------------------------------------ 算子组拖动实时联动
    def _move_group_descendants_live(self, item: NodeItem) -> None:
        start = self._move_start.get(item.node_id)
        if start is None or self._suppress_move_events:
            return
        last = self._group_live_last.get(item.node_id, start)
        delta = item.pos() - last
        if abs(delta.x()) < 1e-9 and abs(delta.y()) < 1e-9:
            return
        self._group_live_last[item.node_id] = QPointF(item.pos())
        self._suppress_move_events = True
        try:
            for child in self.graph.group_subtree(item.node_id):
                child_item = self._nodes.get(child.id)
                if child_item is None:
                    continue
                child_item.setPos(child_item.pos() + delta)
                for connection in self._connections.values():
                    if child.id in (connection.src_node, connection.dst_node):
                        connection.update_path()
                self._position_collapse_badge(child.id)
        finally:
            self._suppress_move_events = False
        self._update_scene_rect()

    def move_out_of_group(self, node_id: str) -> None:
        """把组内节点移出到顶层（经组端口的连线改接为直连，一步撤销）。"""

        node = self.graph.nodes.get(node_id)
        if node is None or not node.parent_group:
            return
        command = self._reparent_command(node_id, None, "移出算子组")
        if command is not None:
            self._undo_stack.push(command)
        self._sync_group_geometries()
        self.status_message.emit(f"已把“{node.display_title()}”移出算子组")

    # ------------------------------------------------- 拖动 / 移出时自动重连
    def _reparent_command(
        self, node_id: str, new_parent: str | None, text: str
    ) -> QUndoCommand | None:
        """返回调整所属算子组的命令；能经组端口重连时返回组合命令。"""

        node = self.graph.nodes.get(node_id)
        if node is None or node.parent_group == new_parent:
            return None
        changes = [(node_id, node.parent_group, new_parent)]
        command = self._build_reconnect_command(changes, text)
        if command is not None:
            return command
        return SetParentGroupCommand(self.graph, node_id, node.parent_group, new_parent, text)

    def _build_reconnect_command(
        self,
        changes: list[tuple[str, str | None, str | None]],
        text: str = "移动节点并调整算子组",
    ) -> QUndoCommand | None:
        """规划一次所属算子组变化带来的连线改接；无需改接时返回 None。

        * 节点原来的直连若在新位置跨层，则在公共祖先与目标层之间的各级算子组
          上补端口，把连线引到组边界的内 / 外侧；
        * 从旧父组端口接出来的连线先“展开”为真实的外部来源 / 目标，再按新层级
          重新走线；
        * 任何一步无法规划时返回 None，由调用方退回只调整所属关系。
        """

        parent_map = {node_id: new_parent for node_id, _old, new_parent in changes}
        old_parent_map = {node_id: old_parent for node_id, old_parent, _new in changes}
        removed: dict[str, ConnectionModel] = {}
        edges: list[tuple[str, str, str, str]] = []
        seen_edges: set[tuple[str, str, str, str]] = set()
        processed: set[str] = set()

        def add_edge(edge: tuple[str, str, str, str]) -> None:
            if edge in seen_edges:
                return
            seen_edges.add(edge)
            edges.append(edge)

        for connection in list(self.graph.connections.values()):
            if connection.id in processed:
                continue
            src_changed = connection.src_node in parent_map
            dst_changed = connection.dst_node in parent_map
            if not src_changed and not dst_changed:
                continue
            processed.add(connection.id)
            if dst_changed and not src_changed:
                for edge in self._logical_incoming_edges(
                    connection, connection.dst_node, old_parent_map[connection.dst_node], removed
                ):
                    add_edge(edge)
            elif src_changed and not dst_changed:
                for edge in self._logical_outgoing_edges(
                    connection, connection.src_node, old_parent_map[connection.src_node], removed
                ):
                    add_edge(edge)
            else:
                removed[connection.id] = connection
                add_edge(
                    (connection.src_node, connection.src_port, connection.dst_node, connection.dst_port)
                )

        plan: dict[str, list] = {"removed": [], "added": [], "ports": [], "used_ports": set()}
        # 端口分配顺序跟随节点自身的输入 / 输出顺序，避免多输入节点的跨层连线
        # 因为连接创建顺序不同而交叉；同一节点内按端口声明顺序处理。
        node_order = {node_id: index for index, node_id in enumerate(self.graph.nodes)}

        def edge_order(edge: tuple[str, str, str, str]) -> tuple[int, int, int]:
            src_node, src_port, dst_node, dst_port = edge
            dst_moved = dst_node in parent_map
            src_moved = src_node in parent_map
            if dst_moved and not src_moved:
                return (
                    0,
                    node_order.get(dst_node, 0),
                    self._port_declaration_index(dst_node, dst_port, True),
                )
            if src_moved and not dst_moved:
                return (
                    1,
                    node_order.get(src_node, 0),
                    self._port_declaration_index(src_node, src_port, False),
                )
            return (2, node_order.get(src_node, 0), 0)

        for edge in sorted(edges, key=edge_order):
            if not self._route_logical_edge(edge, parent_map, plan):
                return None
        for connection in removed.values():
            plan["removed"].append(connection)
        if not plan["removed"] and not plan["added"] and not plan["ports"]:
            return None
        added = [(new_id("c"), *edge) for edge in plan["added"]]
        return ReconnectGroupMoveCommand(
            self.graph,
            changes,
            removed=plan["removed"],
            added=added,
            ports=plan["ports"],
            text=text,
        )

    def _logical_incoming_edges(
        self,
        connection: ConnectionModel,
        node_id: str,
        old_parent: str | None,
        removed: dict[str, ConnectionModel],
    ) -> list[tuple[str, str, str, str]]:
        """把一条入边还原到真实来源；旧父组输入端口的内侧连线会被展开。"""

        removed[connection.id] = connection
        if old_parent and connection.src_node == old_parent:
            group_node = self.graph.nodes.get(old_parent)
            group = group_node.group if group_node is not None and group_node.is_group else None
            if group is not None and connection.src_port in group.input_keys():
                outer = self.graph.input_connection(old_parent, connection.src_port)
                other_inner = [
                    item
                    for item in self.graph.outgoing(old_parent)
                    if item.src_port == connection.src_port and item.id != connection.id
                ]
                if outer is not None and not other_inner:
                    removed[outer.id] = outer
                if outer is None:
                    return []
                return [(outer.src_node, outer.src_port, node_id, connection.dst_port)]
        return [(connection.src_node, connection.src_port, node_id, connection.dst_port)]

    def _logical_outgoing_edges(
        self,
        connection: ConnectionModel,
        node_id: str,
        old_parent: str | None,
        removed: dict[str, ConnectionModel],
    ) -> list[tuple[str, str, str, str]]:
        """把一条出边还原到真实目标；旧父组输出端口的外侧连线会被展开。"""

        removed[connection.id] = connection
        if old_parent and connection.dst_node == old_parent:
            group_node = self.graph.nodes.get(old_parent)
            group = group_node.group if group_node is not None and group_node.is_group else None
            if group is not None and connection.dst_port in group.output_keys():
                outer_edges = [
                    item
                    for item in self.graph.outgoing(old_parent)
                    if item.src_port == connection.dst_port
                ]
                results: list[tuple[str, str, str, str]] = []
                for outer in outer_edges:
                    removed[outer.id] = outer
                    results.append(
                        (node_id, connection.src_port, outer.dst_node, outer.dst_port)
                    )
                return results
        return [(node_id, connection.src_port, connection.dst_node, connection.dst_port)]

    def _route_logical_edge(
        self,
        edge: tuple[str, str, str, str],
        parent_map: dict[str, str | None],
        plan: dict[str, list],
    ) -> bool:
        """把一条逻辑连线接到两端所在层级；跨层时在各级算子组上补端口。"""

        src_node, src_port, dst_node, dst_port = edge
        src_level = self._final_port_level(src_node, src_port, "src", parent_map)
        dst_level = self._final_port_level(dst_node, dst_port, "dst", parent_map)
        if src_level == INVALID_LEVEL or dst_level == INVALID_LEVEL:
            return False
        data_type = self._edge_data_type(src_node, src_port, dst_node, dst_port)
        common = self._common_level(src_level, dst_level, parent_map)
        current_node, current_port = src_node, src_port
        current_level = src_level
        guard = 0
        while current_level != common:
            guard += 1
            if guard > 64 or current_level is None:
                return False
            group_node = self.graph.nodes.get(current_level)
            if group_node is None or not group_node.is_group:
                return False
            key = self._plan_group_port(plan, current_level, False, data_type)
            if key is None:
                return False
            plan["added"].append((current_node, current_port, current_level, key))
            current_node, current_port = current_level, key
            current_level = self._final_level(current_level, parent_map)
        chain: list[str] = []
        level = dst_level
        guard = 0
        while level != common:
            guard += 1
            if guard > 64 or level is None:
                return False
            group_node = self.graph.nodes.get(level)
            if group_node is None or not group_node.is_group:
                return False
            chain.append(level)
            level = self._final_level(level, parent_map)
        for group_id in reversed(chain):
            key = self._plan_group_port(plan, group_id, True, data_type)
            if key is None:
                return False
            plan["added"].append((current_node, current_port, group_id, key))
            current_node, current_port = group_id, key
        plan["added"].append((current_node, current_port, dst_node, dst_port))
        return True

    def _port_declaration_index(self, node_id: str, key: str, is_input: bool) -> int:
        """端口在节点声明中的序号（用于按节点输入 / 输出顺序分配组端口）。"""

        node = self.graph.nodes.get(node_id)
        if node is None:
            return 999
        group = getattr(node, "group", None)
        if group is not None:
            ports = group.inputs if is_input else group.outputs
        elif node.is_source:
            return 0
        else:
            try:
                spec = self._registry.spec(node.op_id)
            except Exception:  # noqa: BLE001 - 未注册算子按末尾处理
                spec = None
            if spec is None:
                return 999
            ports = spec.inputs if is_input else spec.outputs
        for index, port in enumerate(ports):
            if port.key == key:
                return index
        return 999

    def _plan_group_port(
        self, plan: dict[str, list], group_id: str, is_input: bool, data_type
    ) -> str | None:
        """在计划里给算子组分配一个端口 key（不直接修改图模型）。

        优先复用第一个两侧都没有连线的空闲端口：空算子组拖入节点时，第一条跨层
        连线应使用已有的「输入 1」并把它确定类型，而不是跳过它往后追加新点；
        旧版本遗留的已定类型空闲端口也会被直接复用。没有空闲端口时才创建新端口。
        """

        node = self.graph.nodes.get(group_id)
        group = node.group if node is not None and node.is_group else None
        if group is None:
            return None
        used = plan.setdefault("used_ports", set())
        for port in group.ports(is_input):
            token = (group_id, is_input, port.key)
            if token in used:
                continue
            if not self.graph.group_port_free(group_id, port.key, is_input):
                continue
            # 已定类型但两侧都没有连线的端口也视为空闲（旧版本删除连线时不会
            # 回收端口，这里直接复用第一个空闲点，避免首个端口悬空导致连线交叉）。
            used.add(token)
            plan["ports"].append((group_id, is_input, port.key, port.label, data_type))
            return port.key
        keys = set(group.input_keys() if is_input else group.output_keys())
        keys.update(
            key
            for planned_group, planned_input, key, _label, _dtype in plan["ports"]
            if planned_group == group_id and planned_input == is_input
        )
        index = 1
        while f"{'input' if is_input else 'output'}_{index}" in keys:
            index += 1
        key = f"{'input' if is_input else 'output'}_{index}"
        label = f"{'输入' if is_input else '输出'} {index}"
        plan["ports"].append((group_id, is_input, key, label, data_type))
        return key

    def _edge_data_type(self, src_node: str, src_port: str, dst_node: str, dst_port: str):
        for node_id, key, is_input in (
            (src_node, src_port, False),
            (dst_node, dst_port, True),
        ):
            spec = self.port_spec(node_id, key, is_input)
            if spec is not None and not spec.any_type and spec.data_type is not None:
                return spec.data_type
        return None

    def _final_level(self, node_id: str, parent_map: dict[str, str | None]) -> str | None:
        if node_id in parent_map:
            return parent_map[node_id]
        node = self.graph.nodes.get(node_id)
        return node.parent_group if node is not None else None

    def _level_chain(
        self, level: str | None, parent_map: dict[str, str | None]
    ) -> list[str | None]:
        chain: list[str | None] = []
        current = level
        seen: set[str | None] = set()
        while True:
            chain.append(current)
            if current is None:
                break
            node = self.graph.nodes.get(current)
            if node is None:
                break
            current = parent_map.get(current, node.parent_group)
            if current in seen:
                break
            seen.add(current)
        return chain

    def _common_level(
        self,
        first: str | None,
        second: str | None,
        parent_map: dict[str, str | None],
    ) -> str | None:
        second_chain = set(self._level_chain(second, parent_map))
        for level in self._level_chain(first, parent_map):
            if level in second_chain:
                return level
        return None

    def _final_port_level(
        self,
        node_id: str,
        port_key: str,
        role: str,
        parent_map: dict[str, str | None],
    ) -> str | None:
        node = self.graph.nodes.get(node_id)
        if node is None:
            return INVALID_LEVEL
        if node.is_group and node.group is not None:
            if port_key in node.group.input_keys():
                return node.id if role == "src" else self._final_level(node_id, parent_map)
            if port_key in node.group.output_keys():
                return self._final_level(node_id, parent_map) if role == "src" else node.id
            return INVALID_LEVEL
        return self._final_level(node_id, parent_map)

    def dissolve_group(self, node_id: str) -> None:
        """解散算子组：内部节点留在画布，边界连线改成内外直连（一步撤销）。"""

        node = self.graph.nodes.get(node_id)
        if node is None or not node.is_group:
            self.status_message.emit("只能解散算子组")
            return
        children = len(self.graph.direct_children(node_id))
        direct = self._dissolve_direct_connections(node_id)
        command = DissolveGroupCommand(self.graph, node_id, direct)
        self._undo_stack.push(command)
        self._sync_group_geometries()
        self._update_scene_rect()
        self._selection_timer.stop()
        self.selection_changed.emit(self.selected_node_id() or "")
        message = f"已解散算子组，{children} 个内部节点已放回画布"
        if command.failed_connections:
            message += f"；{command.failed_connections} 条边界连线无法直连，已丢弃"
        self.status_message.emit(message)

    def _dissolve_direct_connections(
        self, group_id: str
    ) -> list[tuple[str, str, str, str]]:
        """把算子组边界上的连线两两配对成内部 / 外部直连。"""

        node = self.graph.nodes.get(group_id)
        group = node.group if node is not None else None
        if group is None:
            return []
        pairs: list[tuple[str, str, str, str]] = []
        seen: set[tuple[str, str, str, str]] = set()

        def add(pair: tuple[str, str, str, str]) -> None:
            if pair in seen:
                return
            seen.add(pair)
            pairs.append(pair)

        output_keys = group.output_keys()
        for key in group.input_keys():
            outer = self.graph.input_connection(group_id, key)
            if outer is None:
                continue
            for inner in self.graph.outgoing(group_id):
                if inner.src_port != key:
                    continue
                if inner.dst_node == group_id and inner.dst_port in output_keys:
                    # 组输入直接接到组输出：继续展开到该输出端口的外部下游。
                    for outer_edge in self.graph.outgoing(group_id):
                        if outer_edge.src_port == inner.dst_port:
                            add(
                                (
                                    outer.src_node,
                                    outer.src_port,
                                    outer_edge.dst_node,
                                    outer_edge.dst_port,
                                )
                            )
                    continue
                if inner.dst_node == group_id:
                    continue
                add((outer.src_node, outer.src_port, inner.dst_node, inner.dst_port))
        for key in output_keys:
            inner = self.graph.input_connection(group_id, key)
            if inner is None or inner.src_node == group_id:
                continue
            for outer_edge in self.graph.outgoing(group_id):
                if outer_edge.src_port != key:
                    continue
                add(
                    (
                        inner.src_node,
                        inner.src_port,
                        outer_edge.dst_node,
                        outer_edge.dst_port,
                    )
                )
        return pairs

    # ------------------------------------------------------------ 折叠 / 展开
    def _branch_node_ids(self) -> set[str]:
        """有至少两个不同下游子节点的节点（多分支连接点）。"""

        return {
            node_id
            for node_id in self.graph.nodes
            if len(self.graph.successors(node_id)) >= 2
        }

    def is_node_collapsed(self, node_id: str) -> bool:
        return node_id in self._collapsed

    def hidden_node_ids(self) -> set[str]:
        """当前被折叠隐藏的全部下游节点。"""

        hidden: set[str] = set()
        for node_id in self._collapsed:
            hidden.update(self.graph.descendants(node_id))
        return hidden

    def toggle_node_collapsed(self, node_id: str) -> None:
        if node_id in self._collapsed:
            self.expand_node(node_id)
        else:
            self.collapse_node(node_id)

    def collapse_node(self, node_id: str) -> None:
        """折叠多分支节点的全部下游子节点。"""

        node = self.graph.nodes.get(node_id)
        if node is None or node_id in self._collapsed:
            return
        if len(self.graph.successors(node_id)) < 2:
            return
        count = len(self.graph.descendants(node_id))
        if count == 0:
            return
        self._collapsed.add(node_id)
        self._refresh_collapse_state()
        self.status_message.emit(f"已折叠“{node.display_title()}”的 {count} 个下游节点")

    def expand_node(self, node_id: str, *, announce: bool = True) -> None:
        if node_id not in self._collapsed:
            return
        self._collapsed.discard(node_id)
        self._refresh_collapse_state()
        node = self.graph.nodes.get(node_id)
        if announce and node is not None:
            self.status_message.emit(f"已展开“{node.display_title()}”的子节点")

    def _auto_expand_on_insert(self, node_id: str) -> None:
        """向折叠点插入新连线 / 算子时自动展开，避免新内容被藏起来。"""

        if node_id not in self._collapsed:
            return
        self._collapsed.discard(node_id)
        self._refresh_collapse_state()
        node = self.graph.nodes.get(node_id)
        if node is not None:
            self.status_message.emit(
                f"检测到向折叠点插入新节点，已自动展开“{node.display_title()}”"
            )

    def _refresh_collapse_state(self) -> None:
        """同步折叠状态：隐藏下游条目、维护按钮并刷新画布范围。"""

        for node_id in list(self._collapsed):
            if node_id not in self.graph.nodes or not self.graph.descendants(node_id):
                self._collapsed.discard(node_id)
        hidden = self.hidden_node_ids()
        for node_id, item in self._nodes.items():
            visible = node_id not in hidden
            if not visible and item.isSelected():
                item.setSelected(False)
            item.setVisible(visible)
        for item in self._connections.values():
            visible = item.src_node not in hidden and item.dst_node not in hidden
            if not visible and item.isSelected():
                item.setSelected(False)
            item.setVisible(visible)

        wanted = (self._branch_node_ids() | self._collapsed) & set(self._nodes)
        for node_id in list(self._collapse_badges):
            if node_id in wanted:
                continue
            badge = self._collapse_badges.pop(node_id)
            if badge.scene() is self._scene:
                self._scene.removeItem(badge)
        for node_id in wanted:
            node_item = self._nodes[node_id]
            badge = self._collapse_badges.get(node_id)
            if badge is None:
                badge = CollapseBadgeItem(self, node_id)
                self._scene.addItem(badge)
                self._collapse_badges[node_id] = badge
            badge.set_state(node_id in self._collapsed, len(self.graph.descendants(node_id)))
            badge.setVisible(node_item.isVisible())
            self._position_collapse_badge(node_id)

        if self._highlighted_ports and any(
            port.node_item.node_id in hidden for port in self._highlighted_ports
        ):
            self._clear_target_highlight()
        if (
            self._highlighted_connection is not None
            and not self._highlighted_connection.isVisible()
        ):
            self._clear_target_highlight()
        self._update_scene_rect()

    def _position_collapse_badge(self, node_id: str) -> None:
        badge = self._collapse_badges.get(node_id)
        item = self._nodes.get(node_id)
        if badge is None or item is None:
            return
        size = item.size()
        badge.setPos(
            item.pos().x() + size.width() / 2.0,
            item.pos().y() + size.height() + CollapseBadgeItem.BOTTOM_OFFSET,
        )

    def _clear_pending_interaction(self) -> None:
        """清掉拖动中的临时状态，避免删除后仍引用已移除的条目。"""

        self._pending_move_target = None
        self._pending_move_reason = None
        self._clear_target_highlight()

    def delete_selected(self) -> None:
        self._clear_pending_interaction()
        node_items = [item for item in self._scene.selectedItems() if isinstance(item, NodeItem)]
        connection_items = [item for item in self._scene.selectedItems() if isinstance(item, ConnectionItem)]
        if not node_items and not connection_items:
            return
        selected_ids = {item.node_id for item in node_items}
        node_ids = set(selected_ids)
        # 删除折叠点时连同被隐藏的下游节点一起删除，避免留下孤立分支。
        for node_id in selected_ids:
            if node_id in self._collapsed:
                node_ids.update(self.graph.descendants(node_id))
        # 删除算子组时连同全部后代一起删除（一步撤销）。
        for node_id in selected_ids:
            node = self.graph.nodes.get(node_id)
            if node is not None and getattr(node, "group", None) is not None:
                node_ids.update(child.id for child in self.graph.group_subtree(node_id))

        def _removal_depth(nid: str) -> int:
            depth = 0
            current = self.graph.nodes.get(nid)
            seen: set[str] = set()
            while current is not None and current.parent_group and current.parent_group not in seen:
                seen.add(current.parent_group)
                depth += 1
                current = self.graph.nodes.get(current.parent_group)
            return depth

        # 先删最深的后代、最后删算子组本身；撤销时算子组先恢复，子节点才能挂回去。
        ordered_deletes = sorted(node_ids, key=_removal_depth, reverse=True)
        connections_to_remove: list[ConnectionModel] = []
        for item in connection_items:
            connection = self.graph.connections.get(item.connection_id)
            if connection is None:
                continue
            if connection.src_node in node_ids or connection.dst_node in node_ids:
                continue
            connections_to_remove.append(connection)

        self._undo_stack.beginMacro("删除选中对象")
        try:
            for node_id in ordered_deletes:
                if node_id in self.graph.nodes:
                    self._undo_stack.push(RemoveNodeCommand(self.graph, node_id))
            for connection in connections_to_remove:
                if connection.id in self.graph.connections:
                    self._undo_stack.push(RemoveConnectionCommand(self.graph, connection))
        finally:
            self._undo_stack.endMacro()

        self._sync_group_geometries()
        hidden_extra = len(node_ids) - len(selected_ids)
        self._update_scene_rect()
        self._selection_timer.stop()
        self.selection_changed.emit(self.selected_node_id() or "")
        if hidden_extra:
            self.status_message.emit(f"已删除选中对象（含 {hidden_extra} 个关联节点）")
        else:
            self.status_message.emit("已删除选中对象")

    def delete_node_with_children(self, node_id: str) -> None:
        """删除指定节点及其全部下游子节点，整次删除一步撤销。"""

        if node_id not in self.graph.nodes:
            return
        self._clear_pending_interaction()
        node = self.graph.nodes[node_id]
        if getattr(node, "group", None) is not None:
            # 算子组：删除算子组本身 + 全部内部节点 + 相关连线，整次一步撤销。
            group_ids = [node_id] + [child.id for child in self.graph.group_subtree(node_id)]

            def _depth(nid: str) -> int:
                depth = 0
                current = self.graph.nodes.get(nid)
                seen: set[str] = set()
                while current is not None and current.parent_group and current.parent_group not in seen:
                    seen.add(current.parent_group)
                    depth += 1
                    current = self.graph.nodes.get(current.parent_group)
                return depth

            ordered = sorted(group_ids, key=_depth, reverse=True)
            self._undo_stack.beginMacro("删除算子组及内部节点")
            try:
                for nid in ordered:
                    if nid in self.graph.nodes:
                        self._undo_stack.push(RemoveNodeCommand(self.graph, nid))
            finally:
                self._undo_stack.endMacro()
            self._sync_group_geometries()
            self._update_scene_rect()
            self._selection_timer.stop()
            self.selection_changed.emit(self.selected_node_id() or "")
            children = len(group_ids) - 1
            self.status_message.emit(f"已删除算子组及 {children} 个内部节点")
            return
        command = RemoveNodeTreeCommand(self.graph, node_id)
        self._undo_stack.push(command)
        self._update_scene_rect()
        self._selection_timer.stop()
        self.selection_changed.emit(self.selected_node_id() or "")
        children = max(len(command.node_ids) - 1, 0)
        if children:
            self.status_message.emit(f"已删除节点及 {children} 个子节点")
        else:
            self.status_message.emit("已删除节点")

    # --------------------------------------------------------------- 自动布局
    def _layout_changes(self) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
        """计算自动布局带来的节点位置变化；图中有环时抛出 :class:`GraphError`。

        先布局顶层节点，再按深度从外到内布局每个算子组的直接子节点。组内节点
        与普通节点使用同一套布局规则（只考虑同一层内部的连线），并以算子组
        内容区左上角作为布局锚点。
        """

        sizes: dict[str, tuple[float, float]] = {}
        for node_id, item in self._nodes.items():
            size = item.size()
            sizes[node_id] = (float(size.width()), float(size.height()))
        top_level_ids = [
            node_id
            for node_id, node in self.graph.nodes.items()
            if node.parent_group is None
        ]
        targets: dict[str, tuple[float, float]] = dict(
            compute_layout(self.graph, sizes, node_ids=top_level_ids)
        )
        groups = sorted(
            (node_id for node_id, node in self.graph.nodes.items() if node.is_group),
            key=self._group_depth,
        )
        for group_id in groups:
            child_ids = [child.id for child in self.graph.direct_children(group_id)]
            if not child_ids:
                continue
            anchor = targets.get(group_id)
            if anchor is None:
                group_node = self.graph.nodes.get(group_id)
                if group_node is None:
                    continue
                anchor = (float(group_node.pos[0]), float(group_node.pos[1]))
            positions = compute_layout(self.graph, sizes, node_ids=child_ids)
            if not positions:
                continue
            anchor_x = anchor[0] + GROUP_PADDING_X
            anchor_y = anchor[1] + GROUP_PADDING_TOP
            min_x = min(x for x, _y in positions.values())
            min_y = min(y for _x, y in positions.values())
            for child_id, (x, y) in positions.items():
                targets[child_id] = (anchor_x + (x - min_x), anchor_y + (y - min_y))
        changes: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
        for node_id, (x, y) in targets.items():
            node = self.graph.nodes.get(node_id)
            if node is None:
                continue
            if abs(node.pos[0] - x) < 0.5 and abs(node.pos[1] - y) < 0.5:
                continue
            changes.append((node_id, (float(node.pos[0]), float(node.pos[1])), (float(x), float(y))))
        return changes

    def _push_layout_commands(
        self,
        changes: list[tuple[str, tuple[float, float], tuple[float, float]]],
    ) -> None:
        """把布局变化写入当前 undo 宏；不做 begin/endMacro，便于与插入命令合并。

        ``_layout_changes()`` 已包含算子组内部节点的目标位置，因此这里不再额外
        平移算子组的后代，避免同一次布局出现两套坐标。
        """

        for node_id, old_pos, new_pos in changes:
            self._undo_stack.push(
                MoveNodeCommand(self.graph, node_id, old_pos, new_pos, "自动布局")
            )

    def auto_layout(self) -> None:
        if not self.graph.nodes:
            return
        try:
            changes = self._layout_changes()
        except GraphError as exc:
            self.status_message.emit(f"自动布局失败：{exc}")
            return
        if changes:
            self._undo_stack.beginMacro("自动布局")
            try:
                self._push_layout_commands(changes)
            finally:
                self._undo_stack.endMacro()
        self._sync_group_geometries()
        self._update_scene_rect()
        self.fit_graph()

    def fit_graph(self) -> None:
        rect = self._content_bounding_rect()
        if rect.isNull() or not rect.isValid() or rect.isEmpty():
            self.resetTransform()
            self.centerOn(0.0, 0.0)
            return
        margin = 48.0
        rect = rect.adjusted(-margin, -margin, margin, margin)
        if rect.width() <= 0.0 or rect.height() <= 0.0:
            self.resetTransform()
            return
        self.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        scale = self.transform().m11()
        if scale > 1.3 and scale > 0.0:
            factor = 1.3 / scale
            self.scale(factor, factor)

    # ------------------------------------------------------------- 模型 -> 画布
    def _create_node_item(self, node: NodeModel) -> NodeItem:
        signature = (node.op_id, getattr(node, "group", None) is not None)
        self._item_signatures[node.id] = signature
        if getattr(node, "group", None) is not None:
            item = GroupItem(
                self,
                self.graph,
                node,
                self._registry.spec(node.op_id),
                self._thumbnails_enabled,
                self._thumbnail_box(),
            )
        else:
            item = NodeItem(
                self,
                self.graph,
                node,
                self._registry.spec(node.op_id),
                self._thumbnails_enabled,
                self._thumbnail_box(),
            )
        self._scene.addItem(item)
        self._nodes[node.id] = item
        result = self._results.get(node.id)
        if result is not None:
            item.set_result(result, self._thumbnails_enabled)
        error = self._errors.get(node.id)
        if error:
            item.set_error(error)
        item.update_port_states()
        return item

    def _create_connection_item(self, connection: ConnectionModel) -> ConnectionItem:
        item = ConnectionItem(self, connection)
        self._scene.addItem(item)
        self._connections[connection.id] = item
        self._update_connections_for(connection.src_node)
        self._update_connections_for(connection.dst_node)
        return item

    def _on_node_added(self, node: NodeModel) -> None:
        if node.id in self._nodes:
            return
        self._create_node_item(node)
        self._refresh_collapse_state()
        if node.parent_group is not None:
            self._sync_group_geometries()

    def _on_node_removed(self, node_id: str) -> None:
        if any(port.node_item.node_id == node_id for port in self._highlighted_ports):
            self._clear_target_highlight()
        if self._pending_move_target is not None and node_id in (
            self._pending_move_target.node_id,
            self._pending_move_target.src_node,
            self._pending_move_target.dst_node,
        ):
            self._pending_move_target = None
            self._pending_move_reason = None
        item = self._nodes.pop(node_id, None)
        if item is not None and item.scene() is self._scene:
            self._scene.removeItem(item)
        for connection_id, connection in list(self._connections.items()):
            if node_id in (connection.src_node, connection.dst_node):
                if connection.scene() is self._scene:
                    self._scene.removeItem(connection)
                self._connections.pop(connection_id, None)
        self._collapsed.discard(node_id)
        badge = self._collapse_badges.pop(node_id, None)
        if badge is not None and badge.scene() is self._scene:
            self._scene.removeItem(badge)
        self._results.pop(node_id, None)
        self._errors.pop(node_id, None)
        self._item_signatures.pop(node_id, None)
        self._group_live_last.pop(node_id, None)
        self.node_removed.emit(node_id)
        self._refresh_collapse_state()
        self._sync_group_geometries()

    def _on_node_changed(self, node_id: str) -> None:
        node = self.graph.nodes.get(node_id)
        if node is None:
            return
        expected = (node.op_id, getattr(node, "group", None) is not None)
        if self._highlighted_ports and any(
            port.node_item.node_id == node_id for port in self._highlighted_ports
        ):
            # 算子组端口重建后旧 PortItem 会失效，先清掉高亮引用。
            self._clear_target_highlight()
        item = self._nodes.get(node_id)
        if item is None or self._item_signatures.get(node_id) != expected:
            self._recreate_node_item(node_id)
            item = self._nodes.get(node_id)
        if item is None:
            return
        item.sync_position(node.pos)
        item.refresh_from_model()
        item.update_port_states()
        self._update_connections_for(node_id)
        self._position_collapse_badge(node_id)
        if getattr(node, "group", None) is not None or self._group_ancestors(node_id):
            self._sync_group_geometries()

    def _on_connection_added(self, connection: ConnectionModel) -> None:
        if connection.id not in self._connections:
            self._create_connection_item(connection)
        if connection.src_node in self._collapsed:
            self._auto_expand_on_insert(connection.src_node)
        else:
            self._refresh_collapse_state()
        self.connection_changed.emit(connection.dst_node)

    def _on_connection_removed(self, connection: ConnectionModel) -> None:
        item = self._connections.pop(connection.id, None)
        if self._highlighted_connection is item:
            self._highlighted_connection = None
        if self._pending_move_target is not None and (
            self._pending_move_target.connection_id == connection.id
        ):
            self._pending_move_target = None
            self._pending_move_reason = None
        if item is not None and item.scene() is self._scene:
            self._scene.removeItem(item)
        self._update_connections_for(connection.src_node)
        self._update_connections_for(connection.dst_node)
        self._refresh_collapse_state()
        self.connection_changed.emit(connection.dst_node)

    def _update_connections_for(self, node_id: str) -> None:
        self._refresh_port_states(node_id)
        for connection in self._connections.values():
            if node_id in (connection.src_node, connection.dst_node):
                connection.update_path()

    def _refresh_port_states(self, node_id: str) -> None:
        item = self._nodes.get(node_id)
        if item is not None:
            item.update_port_states()

    def on_node_item_moved(self, item: NodeItem) -> None:
        """NodeItem 拖动时刷新相连连线，并检测端口 / 连线的自动吸附目标。"""

        for connection in self._connections.values():
            if item.node_id in (connection.src_node, connection.dst_node):
                connection.update_path()
        if self._suppress_move_events:
            return
        node = self.graph.nodes.get(item.node_id)
        if node is not None and getattr(node, "group", None) is not None:
            # 拖动算子组本体：实时平移全部后代节点，释放时统一写入移动命令。
            self._move_group_descendants_live(item)
            self._position_collapse_badge(item.node_id)
            if not self._scene.sceneRect().contains(item.sceneBoundingRect()):
                self._update_scene_rect()
            return
        self._position_collapse_badge(item.node_id)
        if not self._scene.sceneRect().contains(item.sceneBoundingRect()):
            self._update_scene_rect()
        # 仅跟踪单节点拖动，多选整体移动时不做自动连接，避免歧义。
        if len(self._move_start) != 1 or item.node_id not in self._move_start:
            return
        if item.node_id not in self.graph.nodes:
            return
        target, reason = self._detect_move_target(item)
        self._pending_move_target = target
        self._pending_move_reason = reason if target is None else None
        self._set_target_highlight(target)

    # ------------------------------------------------------------ 自动连线检测
    def _reachable(self, start: str, target: str, ignore_connection: str | None = None) -> bool:
        """沿连线方向判断 ``start`` 能否到达 ``target``（可忽略一条旧连线）。"""

        if start == target:
            return True
        queue = deque([start])
        visited = {start}
        while queue:
            current = queue.popleft()
            for connection in self.graph.connections.values():
                if ignore_connection is not None and connection.id == ignore_connection:
                    continue
                if connection.src_node != current:
                    continue
                nxt = connection.dst_node
                if nxt == target:
                    return True
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    def _connection_exists(
        self, src_node: str, src_port: str, dst_node: str, dst_port: str
    ) -> bool:
        return any(
            connection.src_node == src_node
            and connection.src_port == src_port
            and connection.dst_node == dst_node
            and connection.dst_port == dst_port
            for connection in self.graph.connections.values()
        )

    @staticmethod
    def _port_distance(first: QPointF, second: QPointF) -> float:
        return math.hypot(first.x() - second.x(), first.y() - second.y())

    def _detect_move_target(self, item: NodeItem) -> tuple[_MoveTarget | None, str | None]:
        """返回节点拖动时应执行的自动动作，以及无法执行时的提示原因。"""

        reason: str | None = None
        best: _MoveTarget | None = None
        best_distance = SNAP_RADIUS
        moving_id = item.node_id

        # 移动节点的输入 ← 其它节点的输出。
        for key, in_port in item.input_ports.items():
            if self.graph.input_connection(moving_id, key) is not None:
                continue
            in_center = in_port.center_scene_pos()
            for other in self._nodes.values():
                if other is item or not other.isVisible() or isinstance(other, GroupItem):
                    continue
                for out_port in other.output_ports.values():
                    distance = self._port_distance(out_port.center_scene_pos(), in_center)
                    if distance > SNAP_RADIUS:
                        continue
                    if self._connection_exists(other.node_id, out_port.key, moving_id, key):
                        continue
                    if not self.ports_compatible(other.node_id, out_port.key, moving_id, key):
                        continue
                    if self._reachable(moving_id, other.node_id):
                        reason = reason or "该连线会形成环路（当前版本不支持回环）"
                        continue
                    if distance <= best_distance:
                        best_distance = distance
                        best = _MoveTarget(
                            "connect",
                            src_node=other.node_id,
                            src_port=out_port.key,
                            dst_node=moving_id,
                            dst_port=key,
                            ports=(in_port, out_port),
                        )

        # 移动节点的输出 → 其它节点的输入。
        for key, out_port in item.output_ports.items():
            out_center = out_port.center_scene_pos()
            for other in self._nodes.values():
                if other is item or not other.isVisible() or isinstance(other, GroupItem):
                    continue
                for in_port in other.input_ports.values():
                    distance = self._port_distance(out_center, in_port.center_scene_pos())
                    if distance > SNAP_RADIUS:
                        continue
                    if self.graph.input_connection(other.node_id, in_port.key) is not None:
                        reason = reason or f"输入端口“{in_port.label}”已有连线，未自动连接"
                        continue
                    if self._connection_exists(moving_id, key, other.node_id, in_port.key):
                        continue
                    if not self.ports_compatible(moving_id, key, other.node_id, in_port.key):
                        continue
                    if self._reachable(other.node_id, moving_id):
                        reason = reason or "该连线会形成环路（当前版本不支持回环）"
                        continue
                    if distance <= best_distance:
                        best_distance = distance
                        best = _MoveTarget(
                            "connect",
                            src_node=moving_id,
                            src_port=key,
                            dst_node=other.node_id,
                            dst_port=in_port.key,
                            ports=(out_port, in_port),
                        )

        if best is not None:
            return best, None

        # 移动节点中心落在其它连线上 → 插入到连线中间。
        if item.input_ports and item.output_ports:
            first_input = next(iter(item.input_ports))
            if self.graph.input_connection(moving_id, first_input) is None:
                edge = self._connection_at(item.sceneBoundingRect().center())
                if edge is not None and moving_id not in (edge.src_node, edge.dst_node):
                    connection = self.graph.connections.get(edge.connection_id)
                    if connection is not None:
                        if self._reachable(moving_id, connection.src_node, connection.id) or self._reachable(
                            connection.dst_node, moving_id, connection.id
                        ):
                            reason = reason or "插入节点会形成环路（当前版本不支持回环）"
                        else:
                            first_output = next(iter(item.output_ports))
                            return (
                                _MoveTarget(
                                    "insert",
                                    connection_id=connection.id,
                                    node_id=moving_id,
                                    input_port=first_input,
                                    output_port=first_output,
                                    ports=(item.input_ports[first_input],),
                                ),
                                None,
                            )
        return None, reason

    def _set_target_highlight(self, target: _DropTarget | _MoveTarget | None) -> None:
        """高亮当前吸附目标（端口虚线高亮 / 连线高亮），并清理旧高亮。"""

        ports: tuple[PortItem, ...] = ()
        connection: ConnectionItem | None = None
        if isinstance(target, _MoveTarget):
            ports = target.ports
            if target.kind == "insert":
                connection = self._connections.get(target.connection_id)
        elif isinstance(target, _DropTarget):
            if target.port is not None:
                ports = (target.port,)
            connection = target.connection

        if tuple(self._highlighted_ports) == ports and self._highlighted_connection is connection:
            return
        self._clear_target_highlight()
        for port in ports:
            port.set_highlight(True)
        self._highlighted_ports = list(ports)
        if connection is not None:
            setter = getattr(connection, "set_highlight", None)
            if callable(setter):
                setter(True)
        self._highlighted_connection = connection

    def _clear_target_highlight(self) -> None:
        for port in self._highlighted_ports:
            port.set_highlight(False)
        self._highlighted_ports = []
        connection = self._highlighted_connection
        self._highlighted_connection = None
        if connection is not None:
            setter = getattr(connection, "set_highlight", None)
            if callable(setter):
                setter(False)

    # ------------------------------------------------------------------ 连线
    def port_items(self, is_input: bool) -> Iterable[PortItem]:
        for node_item in self._nodes.values():
            # 折叠隐藏的节点不参与连线吸附 / 高亮。
            if not node_item.isVisible():
                continue
            ports = node_item.input_ports if is_input else node_item.output_ports
            for port in ports.values():
                yield port

    def port_item(self, node_id: str, key: str, is_input: bool) -> PortItem | None:
        node_item = self._nodes.get(node_id)
        if node_item is None:
            return None
        return node_item.port_item(key, is_input)

    def connection_drag_source(self, port: PortItem) -> PortItem | None:
        """返回按下端口时应拖拽的实际输出端口。

        算子组输入端口的外侧圆点与内侧输出圆点共用同一个连接点：从外侧输入端口
        按下时改用同 key 的内侧输出端口作为拖拽起点，方便把组输入接到组内节点。
        普通输入端口不是拖拽起点，返回 None。
        """

        if not port.is_input:
            return port
        node_item = port.node_item
        inner_ports = getattr(node_item, "inner_output_ports", None)
        outer_ports = getattr(node_item, "outer_input_ports", None)
        if not isinstance(inner_ports, dict) or not isinstance(outer_ports, dict):
            return None
        if outer_ports.get(port.key) is not port:
            return None
        return inner_ports.get(port.key)

    def port_spec(self, node_id: str, key: str, is_input: bool) -> PortSpec | None:
        """查询端口声明；源节点动态生成图像输出端口，未注册算子返回 ``None``。"""

        node = self.graph.nodes.get(node_id)
        if node is None:
            return None
        group = getattr(node, "group", None)
        if group is not None:
            if key in group.input_keys():
                return group.port_spec(key, True)
            if key in group.output_keys():
                return group.port_spec(key, False)
            return None
        if node.is_source:
            if not is_input and key == "image":
                return PortSpec("image", "图像", description="原始图像输出")
            return None
        spec = self._registry.spec(node.op_id)
        if spec is None:
            return None
        ports = spec.inputs if is_input else spec.outputs
        for port in ports:
            if port.key == key:
                return port
        return None

    def port_data_type(self, node_id: str, key: str, is_input: bool) -> DataType | None:
        port = self.port_spec(node_id, key, is_input)
        return port.data_type if port is not None else None

    def ports_compatible(
        self,
        src_node_id: str,
        src_port: str,
        dst_node_id: str,
        dst_port: str,
    ) -> bool:
        """输出端口能否连接到输入端口；未知算子 / 源节点保持宽松兼容。"""

        allowed, _reason = self.graph.connection_allowed(
            src_node_id, src_port, dst_node_id, dst_port
        )
        if not allowed:
            return False
        output_spec = self.port_spec(src_node_id, src_port, False)
        input_spec = self.port_spec(dst_node_id, dst_port, True)
        if output_spec is None or input_spec is None:
            return True
        return port_types_compatible(output_spec, input_spec)

    def _compatible_output_key(self, spec: OperatorSpec, target: PortItem) -> str | None:
        target_spec = self.port_spec(target.node_item.node_id, target.key, True)
        for port in spec.outputs:
            if target_spec is not None:
                if not port_types_compatible(port, target_spec):
                    continue
            elif (
                not port.any_type
                and not target.any_type
                and port.data_type is not target.data_type
            ):
                continue
            return port.key
        return None

    def _compatible_input_key(self, spec: OperatorSpec, target: PortItem) -> str | None:
        target_spec = self.port_spec(target.node_item.node_id, target.key, False)
        for port in spec.inputs:
            if target_spec is not None:
                if not port_types_compatible(target_spec, port):
                    continue
            elif (
                not port.any_type
                and not target.any_type
                and port.data_type is not target.data_type
            ):
                continue
            return port.key
        return None

    def _insertion_ports(
        self, spec: OperatorSpec, connection: ConnectionModel
    ) -> tuple[str, str] | None:
        """选择在连线上插入节点时用于上下游连接的输入 / 输出端口 key。

        原连线两端的数据类型必须都能在待插入算子上找到同类型端口；未知类型时
        退化为第一个端口，兼容旧工程与未注册算子。
        """

        source_type = self.port_data_type(connection.src_node, connection.src_port, False)
        target_type = self.port_data_type(connection.dst_node, connection.dst_port, True)
        source_spec = self.port_spec(connection.src_node, connection.src_port, False)
        target_spec = self.port_spec(connection.dst_node, connection.dst_port, True)

        def input_ok(port: PortSpec) -> bool:
            if source_spec is not None:
                return port_types_compatible(source_spec, port)
            if (
                source_type is not None
                and not port.any_type
                and port.data_type is not source_type
            ):
                return False
            return True

        def output_ok(port: PortSpec) -> bool:
            if target_spec is not None:
                return port_types_compatible(port, target_spec)
            if (
                target_type is not None
                and not port.any_type
                and port.data_type is not target_type
            ):
                return False
            return True

        input_key = next((port.key for port in spec.inputs if input_ok(port)), None)
        output_key = next((port.key for port in spec.outputs if output_ok(port)), None)
        if input_key is None or output_key is None:
            return None
        return input_key, output_key

    def connect_ports(self, src_node_id: str, src_port: str, dst_node_id: str, dst_port: str) -> None:
        src_node = self.graph.nodes.get(src_node_id)
        dst_node = self.graph.nodes.get(dst_node_id)
        if src_node is None or dst_node is None:
            return
        if src_node_id == dst_node_id:
            self.status_message.emit("不能把节点连接到自身")
            return
        if not self._port_exists(src_node, src_port, is_input=False):
            self.status_message.emit(f"输出端口不存在：{src_port}")
            return
        if not self._port_exists(dst_node, dst_port, is_input=True):
            self.status_message.emit(f"输入端口不存在：{dst_port}")
            return
        allowed, reason = self.graph.connection_allowed(
            src_node_id, src_port, dst_node_id, dst_port
        )
        if not allowed:
            self.status_message.emit(reason)
            return
        output_spec = self.port_spec(src_node_id, src_port, False)
        input_spec = self.port_spec(dst_node_id, dst_port, True)
        if (
            output_spec is not None
            and input_spec is not None
            and not port_types_compatible(output_spec, input_spec)
        ):
            self.status_message.emit(connection_error(output_spec, input_spec))
            return
        existing = self.graph.input_connection(dst_node_id, dst_port)
        if existing is not None and existing.src_node == src_node_id and existing.src_port == src_port:
            self.status_message.emit("该连线已经存在")
            return
        if self.graph._reachable(dst_node_id, src_node_id):  # noqa: SLF001 - 图模型的内部可达性检查
            self.status_message.emit("该连线会形成环路（当前版本不支持回环）")
            return
        try:
            if existing is not None:
                self._undo_stack.beginMacro("替换连线")
                try:
                    self._undo_stack.push(RemoveConnectionCommand(self.graph, existing))
                    self._undo_stack.push(
                        AddConnectionCommand(self.graph, src_node_id, src_port, dst_node_id, dst_port)
                    )
                finally:
                    self._undo_stack.endMacro()
                self.status_message.emit("已替换输入端口上的连线")
            else:
                self._undo_stack.push(
                    AddConnectionCommand(self.graph, src_node_id, src_port, dst_node_id, dst_port)
                )
                self.status_message.emit(
                    f"已连接：{src_node.display_title()} → {dst_node.display_title()}"
                )
        except GraphError as exc:
            self.status_message.emit(str(exc))

    def _port_exists(self, node: NodeModel, port: str, is_input: bool) -> bool:
        group = getattr(node, "group", None)
        if group is not None:
            return port in (group.input_keys() | group.output_keys())
        spec = self._registry.spec(node.op_id)
        if spec is not None:
            keys = {item.key for item in (spec.inputs if is_input else spec.outputs)}
            return port in keys
        if node.is_source:
            return (not is_input) and port == "image"
        return True

    def _connection_at(self, scene_pos: QPointF) -> ConnectionItem | None:
        best: ConnectionItem | None = None
        best_distance = float("inf")
        for item in self._connections.values():
            if not item.isVisible() or not item.hit_test(scene_pos):
                continue
            center = item.boundingRect().center()
            distance = (center - scene_pos).manhattanLength()
            if distance < best_distance:
                best = item
                best_distance = distance
        return best

    def _badge_at(self, scene_pos: QPointF) -> CollapseBadgeItem | None:
        for badge in self._collapse_badges.values():
            if badge.isVisible() and badge.hit_test(scene_pos):
                return badge
        return None

    def _drop_target_at(
        self, scene_pos: QPointF, spec: OperatorSpec | None
    ) -> _DropTarget | None:
        """库拖拽落点命中检测：优先折叠按钮，其次端口，最后已有连线。"""

        if spec is not None and spec.inputs:
            badge = self._badge_at(scene_pos)
            if badge is not None:
                node_item = self._nodes.get(badge.node_id)
                if node_item is not None and node_item.isVisible():
                    for out_port in node_item.output_ports.values():
                        if self._compatible_input_key(spec, out_port) is not None:
                            return _DropTarget("output", port=out_port)
        if spec is not None:
            best_port: PortItem | None = None
            best_kind = ""
            best_distance = SNAP_RADIUS
            if spec.outputs:
                for port in self.port_items(is_input=True):
                    if self._compatible_output_key(spec, port) is None:
                        continue
                    distance = self._port_distance(port.center_scene_pos(), scene_pos)
                    if distance <= best_distance:
                        best_port = port
                        best_kind = "input"
                        best_distance = distance
            if spec.inputs:
                for port in self.port_items(is_input=False):
                    if self._compatible_input_key(spec, port) is None:
                        continue
                    distance = self._port_distance(port.center_scene_pos(), scene_pos)
                    if distance <= best_distance:
                        best_port = port
                        best_kind = "output"
                        best_distance = distance
            if best_port is not None:
                return _DropTarget(best_kind, port=best_port)
        edge = self._connection_at(scene_pos)
        if edge is not None:
            return _DropTarget("edge", connection=edge)
        return None

    def _highlightable_drop_target(
        self,
        target: _DropTarget | None,
        spec: OperatorSpec | None = None,
    ) -> _DropTarget | None:
        """过滤掉拖拽时不应高亮的目标，避免误导用户。

        * 已占用的输入端口不会自动连接，拖动时也不高亮；
        * 只有同时有输入 / 输出端口的算子才能插入到连线上，否则不高亮连线。
        """

        if target is None:
            return None
        if target.kind == "edge":
            if spec is None or target.connection is None:
                return None
            if self._insertion_ports(spec, target.connection) is None:
                return None
            return target
        if target.kind not in ("input", "output") or target.port is None:
            return target
        port = target.port
        if target.kind == "input":
            if spec is not None and self._compatible_output_key(spec, port) is None:
                return None
            if self.graph.input_connection(port.node_item.node_id, port.key) is not None:
                return None
        else:
            if spec is not None and self._compatible_input_key(spec, port) is None:
                return None
        return target

    def _add_operator_with_port_connection(
        self,
        op_id: str,
        spec: OperatorSpec,
        target: _DropTarget,
    ) -> str | None:
        """在端口附近放下新算子：创建节点并自动连线（同一 undo 宏）。"""

        port = target.port
        if port is None:
            return None
        anchor = port.center_scene_pos()
        if target.kind == "input":
            if not spec.outputs:
                return None
            if self.graph.input_connection(port.node_item.node_id, port.key) is not None:
                self.status_message.emit(
                    f"输入端口“{port.label}”已有连线，未自动建立连接"
                )
                return None
            port_key = self._compatible_output_key(spec, port)
            if port_key is None:
                self.status_message.emit(
                    f"算子“{spec.name}”没有与输入端口“{port.label}”相同类型（{port.data_type.label}）的输出"
                )
                return None
            node_is_input = False
            rough = QPointF(anchor.x() - NodeItem.WIDTH / 2.0, anchor.y() - AUTO_CONNECT_GAP - 80.0)
            desired = QPointF(anchor.x(), anchor.y() - AUTO_CONNECT_GAP)
            done_message = f"已添加节点“{spec.name}”并接入目标节点输入"
        else:
            if not spec.inputs:
                return None
            port_key = self._compatible_input_key(spec, port)
            if port_key is None:
                self.status_message.emit(
                    f"算子“{spec.name}”没有与输出端口“{port.label}”相同类型（{port.data_type.label}）的输入"
                )
                return None
            node_is_input = True
            rough = QPointF(anchor.x() - NodeItem.WIDTH / 2.0, anchor.y() + AUTO_CONNECT_GAP)
            desired = QPointF(anchor.x(), anchor.y() + AUTO_CONNECT_GAP)
            done_message = f"已添加节点“{spec.name}”并接在目标节点输出之后"

        parent_group = self._scope_for_port_target(port, target.kind)
        node = NodeModel(
            id=new_id("n"),
            op_id=op_id,
            title=spec.name,
            pos=(rough.x(), rough.y()),
            params=param_defaults(spec),
            parent_group=parent_group,
        )
        connected = True
        self._undo_stack.beginMacro("添加节点并连接")
        try:
            self._undo_stack.push(AddNodeCommand(self.graph, node))
            self._align_auto_node(self._nodes.get(node.id), port_key, node_is_input, desired)
            if target.kind == "input":
                self._undo_stack.push(
                    AddConnectionCommand(
                        self.graph, node.id, port_key, port.node_item.node_id, port.key
                    )
                )
            else:
                self._undo_stack.push(
                    AddConnectionCommand(
                        self.graph, port.node_item.node_id, port.key, node.id, port_key
                    )
                )
        except GraphError as exc:
            connected = False
            self.status_message.emit(f"自动连接失败：{exc}")
        finally:
            self._undo_stack.endMacro()

        self._sync_group_geometries()
        self.set_selected_node(node.id)
        self.node_created.emit(node.id)
        if connected:
            self.status_message.emit(done_message)
        return node.id

    def _align_auto_node(
        self,
        item: NodeItem | None,
        port_key: str,
        is_input: bool,
        desired: QPointF,
    ) -> None:
        """创建节点后按真实端口位置微调，保证端口对齐且垂直间距约为 48px。"""

        if item is None:
            return
        port = item.port_item(port_key, is_input)
        node = self.graph.nodes.get(item.node_id)
        if port is None or node is None:
            return
        delta = desired - port.center_scene_pos()
        new_pos = (
            round(item.pos().x() + delta.x(), 2),
            round(item.pos().y() + delta.y(), 2),
        )
        if abs(node.pos[0] - new_pos[0]) < 0.01 and abs(node.pos[1] - new_pos[1]) < 0.01:
            return
        self._undo_stack.push(MoveNodeCommand(self.graph, item.node_id, node.pos, new_pos))

    def _insert_node_on_edge(
        self,
        edge: ConnectionItem,
        op_id: str,
        spec: OperatorSpec,
        scene_pos: QPointF,
    ) -> str | None:
        connection = self.graph.connections.get(edge.connection_id)
        if connection is None:
            return None
        ports = self._insertion_ports(spec, connection)
        if ports is None:
            self.status_message.emit(
                f"算子“{spec.name}”的输入 / 输出数据类型无法接续这条连线，未插入节点"
            )
            return None
        parent_group = self.graph.port_level(connection.src_node, connection.src_port, "src")
        if parent_group == INVALID_LEVEL:
            parent_group = self.group_at(scene_pos)
        node = NodeModel(
            id=new_id("n"),
            op_id=op_id,
            title=spec.name,
            pos=(scene_pos.x(), scene_pos.y()),
            params=param_defaults(spec),
            parent_group=parent_group,
        )
        command = InsertNodeOnEdgeCommand(
            self.graph,
            node,
            connection,
            ports[0],
            ports[1],
        )
        # 插入节点与随后的自动布局必须同属一步撤销，否则第一次 Ctrl+Z 只会
        # 回退布局、节点仍留在画布上，容易让用户误以为撤销失效。
        self._undo_stack.beginMacro("在连线上插入节点")
        try:
            self._undo_stack.push(command)
            try:
                changes = self._layout_changes()
            except GraphError as exc:
                self.status_message.emit(f"自动布局失败：{exc}")
            else:
                self._push_layout_commands(changes)
        finally:
            self._undo_stack.endMacro()
        self._sync_group_geometries()
        self.set_selected_node(node.id)
        self.node_created.emit(node.id)
        self.status_message.emit(f"已在连线上插入节点“{spec.name}”")
        self._update_scene_rect()
        self.fit_graph()
        return node.id

    # --------------------------------------------------------------- 选中逻辑
    def _on_scene_selection_changed(self) -> None:
        if self._syncing_selection:
            return
        # 单击后才延迟发出 selection_changed；双击会取消该定时器。
        self._selection_timer.start()

    def schedule_selection_signal(self) -> None:
        """（重新）开始单击延迟；用于点击已选中的节点也能触发预览。"""

        if self._syncing_selection:
            return
        self._selection_timer.start()

    def cancel_selection_timer(self) -> None:
        self._selection_timer.stop()

    def item_at_scene(self, scene_pos: QPointF):
        return self.itemAt(self.mapFromScene(scene_pos))

    def _emit_selection_changed(self) -> None:
        self.selection_changed.emit(self.selected_node_id() or "")

    # ------------------------------------------------------------------ 移动
    def begin_move_tracking(self) -> None:
        self._pending_move_target = None
        self._pending_move_reason = None
        self._group_live_last = {}
        self._clear_target_highlight()
        self._move_start = {
            item.node_id: QPointF(item.pos())
            for item in self._scene.selectedItems()
            if isinstance(item, NodeItem)
            and bool(item.flags() & QGraphicsItem.GraphicsItemFlag.ItemIsMovable)
        }

    def finish_move_tracking(self) -> None:
        starts = self._move_start
        self._move_start = {}
        target = self._pending_move_target
        reason = self._pending_move_reason
        self._pending_move_target = None
        self._pending_move_reason = None
        self._clear_target_highlight()
        if not starts:
            return
        moves: dict[str, tuple[str, tuple[float, float], tuple[float, float]]] = {}
        for node_id, _start in starts.items():
            item = self._nodes.get(node_id)
            node = self.graph.nodes.get(node_id)
            if item is None or node is None:
                continue
            new_pos = item.pos()
            if abs(node.pos[0] - new_pos.x()) < 0.01 and abs(node.pos[1] - new_pos.y()) < 0.01:
                continue
            moves[node_id] = (
                node_id,
                (float(node.pos[0]), float(node.pos[1])),
                (round(new_pos.x(), 2), round(new_pos.y(), 2)),
            )
        if not moves:
            return

        # 折叠点被拖动时，隐藏的下游节点一起平移，展开后仍保持相对布局。
        for node_id, (_nid, old_pos, new_pos) in list(moves.items()):
            if node_id not in self._collapsed:
                continue
            dx = new_pos[0] - old_pos[0]
            dy = new_pos[1] - old_pos[1]
            for descendant in self.graph.descendants(node_id):
                if descendant in moves:
                    continue
                child = self.graph.nodes.get(descendant)
                if child is None:
                    continue
                moves[descendant] = (
                    descendant,
                    (float(child.pos[0]), float(child.pos[1])),
                    (round(child.pos[0] + dx, 2), round(child.pos[1] + dy, 2)),
                )

        # 拖动算子组本体：全部后代（含嵌套算子组）整体平移，释放时写成一个宏。
        group_deltas: dict[str, tuple[float, float]] = {}
        for node_id, (_nid, old_pos, new_pos) in list(moves.items()):
            node = self.graph.nodes.get(node_id)
            if node is None or getattr(node, "group", None) is None:
                continue
            group_deltas[node_id] = (new_pos[0] - old_pos[0], new_pos[1] - old_pos[1])
        for group_id, (dx, dy) in group_deltas.items():
            if abs(dx) < 0.01 and abs(dy) < 0.01:
                continue
            for child in self.graph.group_subtree(group_id):
                if child.id in moves:
                    continue
                child_item = self._nodes.get(child.id)
                if child_item is None:
                    continue
                moves[child.id] = (
                    child.id,
                    (float(child.pos[0]), float(child.pos[1])),
                    (round(child.pos[0] + dx, 2), round(child.pos[1] + dy, 2)),
                )

        # 按移动后的中心点更新所属算子组（取最深命中；原始图像节点始终在顶层）。
        parent_changes: list[tuple[str, str | None, str | None]] = []
        for node_id, (_nid, _old_pos, new_pos) in moves.items():
            node = self.graph.nodes.get(node_id)
            item = self._nodes.get(node_id)
            if node is None or item is None:
                continue
            if node.is_source:
                new_parent = None
            else:
                size = item.size()
                center = QPointF(
                    new_pos[0] + size.width() / 2.0,
                    new_pos[1] + size.height() / 2.0,
                )
                if getattr(node, "group", None) is not None:
                    excluded = {node_id} | {
                        child.id for child in self.graph.group_subtree(node_id)
                    }
                else:
                    excluded = set()
                new_parent = self.group_at(center, exclude=excluded)
                if new_parent == node_id:
                    new_parent = node.parent_group
                if new_parent in ("", INVALID_LEVEL):
                    new_parent = node.parent_group
            if new_parent != node.parent_group:
                parent_changes.append((node_id, node.parent_group, new_parent))

        # 节点进入 / 移出算子组时不再自动吸附端口，避免跨层级连线。
        reconnect_command: QUndoCommand | None = None
        if parent_changes:
            # 进入 / 移出算子组：自动经组端口重连，不再吸附其它端口。
            reconnect_command = self._build_reconnect_command(parent_changes)
            target = None
            reason = None

        connection_commands, action_message = self._build_move_connection_commands(target)
        if not connection_commands and reason is not None and target is None:
            self.status_message.emit(reason)

        move_items = list(moves.values())
        if len(move_items) == 1 and not connection_commands and not parent_changes:
            node_id, old_pos, new_pos = move_items[0]
            self._undo_stack.push(MoveNodeCommand(self.graph, node_id, old_pos, new_pos))
        else:
            if connection_commands:
                text = "移动并连接节点"
            elif parent_changes:
                text = "移动节点并调整算子组"
            else:
                text = "移动节点"
            self._undo_stack.beginMacro(text)
            try:
                for node_id, old_pos, new_pos in move_items:
                    self._undo_stack.push(MoveNodeCommand(self.graph, node_id, old_pos, new_pos))
                if reconnect_command is not None:
                    self._undo_stack.push(reconnect_command)
                else:
                    for node_id, old_parent, new_parent in parent_changes:
                        self._undo_stack.push(
                            SetParentGroupCommand(self.graph, node_id, old_parent, new_parent)
                        )
                for command in connection_commands:
                    self._undo_stack.push(command)
            except GraphError as exc:
                self.status_message.emit(f"自动连接失败：{exc}")
            finally:
                self._undo_stack.endMacro()
            if parent_changes and reconnect_command is not None:
                action_message = "已调整所属算子组，并把跨层连线改接到组端口"
                failed = getattr(reconnect_command, "failed_connections", 0)
                if failed:
                    action_message += f"；{failed} 条连线无法重连，已丢弃"
        if action_message is not None:
            self.status_message.emit(action_message)
        self._sync_group_geometries()
        self._update_scene_rect()

    def _build_move_connection_commands(
        self, target: _MoveTarget | None
    ) -> tuple[list[QUndoCommand], str | None]:
        """把拖动吸附目标转换为与移动命令同处一个 undo 宏的连线命令。"""

        if target is None:
            return [], None
        if target.kind == "connect":
            if target.src_node not in self.graph.nodes or target.dst_node not in self.graph.nodes:
                return [], None
            if self._connection_exists(
                target.src_node, target.src_port, target.dst_node, target.dst_port
            ):
                return [], "该连线已经存在"
            if self.graph.input_connection(target.dst_node, target.dst_port) is not None:
                return [], "目标输入端口已有连线，未自动连接"
            if self._reachable(target.dst_node, target.src_node):
                return [], "该连线会形成环路（当前版本不支持回环）"
            command = AddConnectionCommand(
                self.graph, target.src_node, target.src_port, target.dst_node, target.dst_port
            )
            src_title = self.graph.nodes[target.src_node].display_title()
            dst_title = self.graph.nodes[target.dst_node].display_title()
            return [command], f"已自动连接：{src_title} → {dst_title}"

        connection = self.graph.connections.get(target.connection_id)
        if connection is None or target.node_id not in self.graph.nodes:
            return [], None
        item = self._nodes.get(target.node_id)
        if item is None or not item.input_ports or not item.output_ports:
            return [], None
        input_port = target.input_port or next(iter(item.input_ports))
        output_port = target.output_port or next(iter(item.output_ports))
        if self.graph.input_connection(target.node_id, input_port) is not None:
            return [], "该节点的第一输入端口已被占用，未插入连线"
        if self._reachable(
            target.node_id, connection.src_node, connection.id
        ) or self._reachable(connection.dst_node, target.node_id, connection.id):
            return [], "插入节点会形成环路（当前版本不支持回环）"
        commands: list[QUndoCommand] = [
            RemoveConnectionCommand(self.graph, connection),
            AddConnectionCommand(
                self.graph,
                connection.src_node,
                connection.src_port,
                target.node_id,
                input_port,
                text="连接上游",
            ),
            AddConnectionCommand(
                self.graph,
                target.node_id,
                output_port,
                connection.dst_node,
                connection.dst_port,
                text="连接下游",
            ),
        ]
        title = self.graph.nodes[target.node_id].display_title()
        return commands, f"已将节点“{title}”插入到连线中"

    # --------------------------------------------------------------- 鼠标事件
    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        # 左键从空白处按下 = 拖动画布；按住 Ctrl 时仍交给视图做框选。
        if (
            event.button() == Qt.MouseButton.LeftButton
            and not (event.modifiers() & Qt.KeyboardModifier.ControlModifier)
            and self.itemAt(event.position().toPoint()) is None
        ):
            self.setFocus(Qt.FocusReason.MouseFocusReason)
            self._panning = True
            self._pan_last = QPointF(event.position())
            self._pan_dragged = False
            self._selection_timer.stop()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._last_drag_view_pos = QPointF(event.position())

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        if self._panning:
            if not (event.buttons() & Qt.MouseButton.LeftButton):
                self._finish_canvas_pan()
                super().mouseMoveEvent(event)
                return
            position = QPointF(event.position())
            delta = position - self._pan_last
            self._pan_last = position
            if delta.x() or delta.y():
                self._pan_dragged = True
                self._scroll_canvas(
                    -int(round(delta.x())),
                    -int(round(delta.y())),
                )
            event.accept()
            return
        self._last_drag_view_pos = QPointF(event.position())
        if event.buttons() & Qt.MouseButton.LeftButton:
            if not self._edge_scroll_timer.isActive():
                self._edge_scroll_timer.start()
        elif self._edge_scroll_timer.isActive():
            self._edge_scroll_timer.stop()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        if self._panning and event.button() == Qt.MouseButton.LeftButton:
            dragged = self._pan_dragged
            self._finish_canvas_pan()
            if not dragged:
                # 空白处单击：保持“点击空白清空选择”的旧行为。
                self._scene.clearSelection()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._edge_scroll_timer.stop()
        super().mouseReleaseEvent(event)

    def _finish_canvas_pan(self) -> None:
        self._panning = False
        self._pan_dragged = False
        self._edge_scroll_timer.stop()
        self.viewport().unsetCursor()

    def _scroll_canvas(self, dx: int, dy: int) -> None:
        """按视口像素平移画布；滚动条单位与视口像素一致。"""

        hbar = self.horizontalScrollBar()
        vbar = self.verticalScrollBar()
        hbar.setValue(hbar.value() + dx)
        vbar.setValue(vbar.value() + dy)

    # ----------------------------------------------------------- 边缘自动滚动
    def _auto_scroll_during_drag(self) -> None:
        """拖动状态下鼠标靠近视口边缘时持续滚动画布。"""

        if self._panning or not (QApplication.mouseButtons() & Qt.MouseButton.LeftButton):
            self._edge_scroll_timer.stop()
            return
        rect = self.viewport().rect()
        position = self._last_drag_view_pos
        if rect.isNull():
            return
        # 鼠标可以拖出画布范围：只要左键没松开就继续滚动，越往外速度越快
        # （最多按边缘的满速滚动），避免可用触发区域太小。
        margin = min(EDGE_SCROLL_MARGIN, max(8.0, min(rect.width(), rect.height()) / 3.0))
        dx = self._edge_scroll_delta(position.x() - rect.left(), rect.width(), margin)
        dy = self._edge_scroll_delta(position.y() - rect.top(), rect.height(), margin)
        if not dx and not dy:
            return
        hbar = self.horizontalScrollBar()
        vbar = self.verticalScrollBar()
        before = (hbar.value(), vbar.value())
        self._scroll_canvas(dx, dy)
        if (hbar.value(), vbar.value()) == before:
            return
        # 画布滚动后补发移动事件，让节点 / 拖出中的连线继续跟随光标。
        self._dispatch_drag_move()

    @staticmethod
    def _edge_scroll_delta(offset: float, extent: float, margin: float) -> int:
        """按鼠标离边缘的深度返回滚动像素；负值向左 / 上，正值向右 / 下。"""

        if offset < margin:
            depth = min(1.0, (margin - offset) / margin)
            return -max(1, int(round(EDGE_SCROLL_STEP * depth)))
        if offset > extent - margin:
            depth = min(1.0, (offset - (extent - margin)) / margin)
            return max(1, int(round(EDGE_SCROLL_STEP * depth)))
        return 0

    def _dispatch_drag_move(self) -> None:
        position = self._last_drag_view_pos
        event = QMouseEvent(
            QEvent.Type.MouseMove,
            position,
            self.viewport().mapToGlobal(position.toPoint()),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            QApplication.keyboardModifiers(),
        )
        QApplication.sendEvent(self.viewport(), event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        if event.button() == Qt.MouseButton.LeftButton:
            item = self._node_item_at(event.position().toPoint())
            if item is not None:
                self._selection_timer.stop()
                self._syncing_selection = True
                try:
                    self._scene.clearSelection()
                    item.setSelected(True)
                finally:
                    self._syncing_selection = False
                self.node_activated.emit(item.node_id)
                event.accept()
                return
        super().mouseDoubleClickEvent(event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        if event.matches(QKeySequence.StandardKey.Copy):
            self.copy_selected()
            event.accept()
            return
        if event.matches(QKeySequence.StandardKey.Paste):
            self.paste_nodes()
            event.accept()
            return
        if event.key() in (Qt.Key.Key_Delete, Qt.Key.Key_Backspace):
            self.delete_selected()
            event.accept()
            return
        if event.key() == Qt.Key.Key_Escape:
            self._pending_move_target = None
            self._pending_move_reason = None
            self._clear_target_highlight()
            self._scene.cancel_connection_drag()
            self._scene.clearSelection()
            self._edge_scroll_timer.stop()
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        modifiers = event.modifiers()
        if modifiers & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
            scale = self.transform().m11() * factor
            if self._zoom_min <= scale <= self._zoom_max:
                self.scale(factor, factor)
            event.accept()
            return
        if modifiers & Qt.KeyboardModifier.ShiftModifier:
            self._scroll_canvas_horizontally(event)
            event.accept()
            return
        super().wheelEvent(event)

    def _scroll_canvas_horizontally(self, event) -> None:
        """Shift+滚轮：把纵向滚轮量换算成横向滚动。"""

        pixel_delta = event.pixelDelta()
        angle_delta = event.angleDelta()
        if pixel_delta.x() or pixel_delta.y():
            amount = pixel_delta.x() or pixel_delta.y()
        else:
            delta = angle_delta.y() or angle_delta.x()
            if not delta:
                return
            lines = QApplication.wheelScrollLines()
            bar = self.horizontalScrollBar()
            # 与 Qt 默认滚轮一致：一格 = 120，滚动 wheelScrollLines 行；
            # 返回 -1 表示整页。高精度滚轮按比例换算。
            step = bar.pageStep() if lines <= 0 else bar.singleStep() * lines
            amount = int(round(delta / 120.0 * step))
        if not amount:
            return
        bar = self.horizontalScrollBar()
        bar.setValue(bar.value() - amount)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        super().resizeEvent(event)
        self._update_scene_rect()

    # ------------------------------------------------------------------ 绘制
    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802 - Qt 接口
        super().drawForeground(painter, rect)
        if self.graph.nodes:
            return
        painter.save()
        painter.resetTransform()
        painter.setPen(QColor("#9aa4b2"))
        font = QFont("Microsoft YaHei UI")
        font.setPointSizeF(11.0)
        painter.setFont(font)
        viewport_rect = self.viewport().rect()
        hint = "画布为空\n从左侧算子库拖入算子开始搭建\n也可以直接拖入图像文件创建原始图像节点"
        painter.drawText(
            viewport_rect,
            int(Qt.AlignmentFlag.AlignCenter),
            hint,
        )
        painter.restore()

    # --------------------------------------------------------------- 右键菜单
    def contextMenuEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        item = self.itemAt(event.pos())
        node_item = self._walk_to_node(item)
        menu = QMenu(self)
        if node_item is not None:
            if not node_item.isSelected():
                self.set_selected_node(node_item.node_id)
            menu.addAction("复制节点", self.copy_selected)
            menu.addAction("删除节点", self.delete_selected)
            menu.addAction(
                "删除节点及子节点",
                lambda: self.delete_node_with_children(node_item.node_id),
            )
            menu.addAction("重命名节点", lambda: self.rename_node(node_item.node_id))
            if (
                node_item.node_id in self._collapsed
                or len(self.graph.successors(node_item.node_id)) >= 2
            ):
                label = (
                    "展开子节点"
                    if node_item.node_id in self._collapsed
                    else "折叠子节点"
                )
                menu.addAction(
                    label,
                    lambda nid=node_item.node_id: self.toggle_node_collapsed(nid),
                )
            node_model = self.graph.nodes.get(node_item.node_id)
            if node_model is not None and getattr(node_model, "group", None) is not None:
                menu.addAction(
                    "保存为新算子…",
                    lambda nid=node_item.node_id: self.save_group_as_operator(nid),
                )
                menu.addAction(
                    "解散算子组",
                    lambda nid=node_item.node_id: self.dissolve_group(nid),
                )
            if node_model is not None and node_model.parent_group:
                menu.addAction(
                    "移出算子组",
                    lambda nid=node_item.node_id: self.move_out_of_group(nid),
                )
            if getattr(node_item, "is_stale", False) or getattr(node_item, "stale", False):
                menu.addAction(
                    "替换算子…",
                    lambda nid=node_item.node_id: self._prompt_replace_operator(nid),
                )
            menu.addSeparator()
            menu.addAction("自动布局", self.auto_layout)
            menu.addAction("适应视图", self.fit_graph)
        elif isinstance(item, ConnectionItem):
            if not item.isSelected():
                self._syncing_selection = True
                try:
                    self._scene.clearSelection()
                    item.setSelected(True)
                finally:
                    self._syncing_selection = False
            menu.addAction("删除连线", self.delete_selected)
            menu.addSeparator()
            menu.addAction("自动布局", self.auto_layout)
            menu.addAction("适应视图", self.fit_graph)
        else:
            scene_pos = self.mapToScene(event.pos())
            menu.addAction(
                "添加算子组",
                lambda pos=scene_pos: self.add_group_at(pos),
            )
            menu.addAction("粘贴节点", self.paste_nodes)
            menu.addSeparator()
            menu.addAction("自动布局", self.auto_layout)
            menu.addAction("适应视图", self.fit_graph)
        menu.exec(event.globalPos())

    def rename_node(self, node_id: str) -> None:
        node = self.graph.nodes.get(node_id)
        if node is None:
            return
        title, ok = QInputDialog.getText(self, "重命名节点", "节点名称：", text=node.display_title())
        if not ok:
            return
        title = str(title).strip()
        if not title or title == node.title:
            return
        self._undo_stack.push(RenameNodeCommand(self.graph, node_id, node.title, title))
        self.status_message.emit("节点已重命名")

    # ------------------------------------------------------------------ 拖放
    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        self._clear_target_highlight()
        mime = event.mimeData()
        if mime.hasFormat(MIME_OPERATOR) or self._image_paths_from_mime(mime):
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        mime = event.mimeData()
        if mime.hasFormat(MIME_OPERATOR):
            raw = bytes(mime.data(MIME_OPERATOR)).decode("utf-8", "replace")
            op_id = self._parse_operator_payload(raw)
            spec = self._registry.spec(op_id) if op_id else None
            scene_pos = self.mapToScene(event.position().toPoint())
            target = self._drop_target_at(scene_pos, spec)
            self._set_target_highlight(self._highlightable_drop_target(target, spec))
            event.acceptProposedAction()
            return
        if self._image_paths_from_mime(mime):
            self._clear_target_highlight()
            event.acceptProposedAction()
            return
        self._clear_target_highlight()
        super().dragMoveEvent(event)

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        self._clear_target_highlight()
        event.accept()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        self._clear_target_highlight()
        mime = event.mimeData()
        if mime.hasFormat(MIME_OPERATOR):
            raw = bytes(mime.data(MIME_OPERATOR)).decode("utf-8", "replace")
            op_id = self._parse_operator_payload(raw)
            if not op_id:
                self.status_message.emit("无法识别拖入的算子数据")
                event.acceptProposedAction()
                return
            scene_pos = self.mapToScene(event.position().toPoint())
            self.add_operator_at(op_id, scene_pos)
            event.acceptProposedAction()
            return
        paths = self._image_paths_from_mime(mime)
        if paths:
            self.images_dropped.emit(paths)
            event.acceptProposedAction()
            return
        super().dropEvent(event)

    # ------------------------------------------------------------------ 工具
    def _node_item_at(self, viewport_pos: QPoint) -> NodeItem | None:
        return self._walk_to_node(self.itemAt(viewport_pos))

    def _walk_to_node(self, item) -> NodeItem | None:
        while item is not None and not isinstance(item, NodeItem):
            item = item.parentItem()
        return item if isinstance(item, NodeItem) else None

    @staticmethod
    def _parse_operator_payload(raw: str) -> str | None:
        raw = (raw or "").strip()
        if not raw:
            return None
        try:
            data = json.loads(raw)
        except (TypeError, ValueError):
            return raw or None
        if isinstance(data, dict):
            op_id = data.get("id")
            return str(op_id) if op_id else None
        if isinstance(data, str):
            return data or None
        return None

    @staticmethod
    def _image_paths_from_mime(mime) -> list[str]:
        paths: list[str] = []
        if mime.hasUrls():
            for url in mime.urls():
                if not url.isLocalFile():
                    continue
                path = url.toLocalFile()
                if path and Path(path).suffix.lower() in _IMAGE_SUFFIXES:
                    paths.append(path)
        if not paths and mime.hasFormat(MIME_IMAGE_PATHS):
            raw = bytes(mime.data(MIME_IMAGE_PATHS)).decode("utf-8", "replace")
            try:
                data = json.loads(raw)
            except (TypeError, ValueError):
                data = None
            if isinstance(data, list):
                paths = [str(item) for item in data if str(item).strip()]
            elif data is None:
                paths = [line.strip() for line in raw.splitlines() if line.strip()]
        seen: set[str] = set()
        unique: list[str] = []
        for path in paths:
            if path not in seen:
                seen.add(path)
                unique.append(path)
        return unique

    def _thumbnail_box(self) -> int:
        try:
            return max(24, min(int(self._settings.node_thumbnail_size), 96))
        except Exception:  # noqa: BLE001 - 设置异常时使用默认值
            return 72

    def _clear_items(self) -> None:
        self._selection_timer.stop()
        self._pending_move_target = None
        self._pending_move_reason = None
        self._clear_target_highlight()
        self._syncing_selection = True
        try:
            self._scene.cancel_connection_drag()
            for badge in self._collapse_badges.values():
                if badge.scene() is self._scene:
                    self._scene.removeItem(badge)
            for connection in self._connections.values():
                if connection.scene() is self._scene:
                    self._scene.removeItem(connection)
            for node in self._nodes.values():
                if node.scene() is self._scene:
                    self._scene.removeItem(node)
        finally:
            self._syncing_selection = False
        self._collapse_badges.clear()
        self._connections.clear()
        self._nodes.clear()
        self._item_signatures.clear()
        self._group_live_last.clear()

    def _content_bounding_rect(self) -> QRectF:
        """可见内容的包围盒；被折叠隐藏的节点不参与适应视图 / 画布范围计算。"""

        rect = QRectF()
        for item in self._nodes.values():
            if not item.isVisible():
                continue
            item_rect = item.sceneBoundingRect()
            rect = item_rect if rect.isNull() else rect.united(item_rect)
        for badge in self._collapse_badges.values():
            if not badge.isVisible():
                continue
            badge_rect = badge.sceneBoundingRect()
            rect = badge_rect if rect.isNull() else rect.united(badge_rect)
        return rect

    def _update_scene_rect(self) -> None:
        items_rect = self._content_bounding_rect()
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        if items_rect.isNull() or not items_rect.isValid() or items_rect.isEmpty():
            base_rect = QRectF(-500.0, -350.0, 1000.0, 700.0)
        else:
            # 内容外留 180px 操作余量；再加一屏空白，保证左键拖动画布时
            # 即使流程图刚好放得下也始终有平移空间。
            pad_x = 180.0 + (visible.width() if visible.isValid() else 0.0)
            pad_y = 180.0 + (visible.height() if visible.isValid() else 0.0)
            base_rect = items_rect.adjusted(-pad_x, -pad_y, pad_x, pad_y)
        if visible.isValid() and not visible.isEmpty():
            union = base_rect.united(visible)
        else:
            union = base_rect
        if not math.isfinite(union.width()) or not math.isfinite(union.height()):
            union = QRectF(-500.0, -350.0, 1000.0, 700.0)
        self._scene.setSceneRect(union)


__all__ = ["GraphView"]
