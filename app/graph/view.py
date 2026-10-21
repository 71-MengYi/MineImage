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

from PySide6.QtCore import QPoint, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import (
    QColor,
    QCursor,
    QFont,
    QKeySequence,
    QPainter,
    QUndoCommand,
    QUndoStack,
)
from PySide6.QtWidgets import QApplication, QFrame, QGraphicsItem, QGraphicsView, QInputDialog, QMenu

from app.config import AppSettings
from app.core.constants import MIME_IMAGE_PATHS, MIME_OPERATOR, SINGLE_CLICK_DELAY_MS
from app.core.data_types import DataType
from app.core.model import ConnectionModel, GraphError, GraphModel, NodeModel, WorkspaceModel, new_id
from app.graph.commands import (
    AddConnectionCommand,
    AddNodeCommand,
    InsertNodeOnEdgeCommand,
    MoveNodeCommand,
    RemoveConnectionCommand,
    RemoveNodeCommand,
    RenameNodeCommand,
)
from app.graph.connection_item import ConnectionItem
from app.graph.layout import compute_layout
from app.graph.node_item import NodeItem
from app.graph.port_item import PortItem
from app.graph.scene import GraphScene
from app.operators.loader import OperatorRegistry
from app.operators.spec import OperatorSpec, PortSpec, connection_error, param_defaults, port_types_compatible

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
        self._results: dict[str, Any] = {}
        self._errors: dict[str, str] = {}
        self._thumbnails_enabled = bool(settings.show_node_thumbnails)
        self._syncing_selection = False
        self._move_start: dict[str, QPointF] = {}
        self._pending_move_target: _MoveTarget | None = None
        self._pending_move_reason: str | None = None
        self._highlighted_ports: list[PortItem] = []
        self._highlighted_connection: ConnectionItem | None = None
        self._paste_count = 0
        self._zoom_min = 0.15
        self._zoom_max = 3.0

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

        self._selection_timer = QTimer(self)
        self._selection_timer.setSingleShot(True)
        self._selection_timer.setInterval(int(SINGLE_CLICK_DELAY_MS))
        self._selection_timer.timeout.connect(self._emit_selection_changed)
        self._scene.selectionChanged.connect(self._on_scene_selection_changed)

        self.graph.node_added.connect(self._on_node_added)
        self.graph.node_removed.connect(self._on_node_removed)
        self.graph.node_changed.connect(self._on_node_changed)
        self.graph.connection_added.connect(self._on_connection_added)
        self.graph.connection_removed.connect(self._on_connection_removed)

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

        id_map: dict[str, str] = {}
        new_ids: list[str] = []
        self._undo_stack.beginMacro("粘贴节点")
        try:
            for node in source_nodes:
                clone = NodeModel(
                    id=new_id("n"),
                    op_id=node.op_id,
                    title=node.title,
                    pos=(round(node.pos[0] + dx, 2), round(node.pos[1] + dy, 2)),
                    params=copy.deepcopy(node.params),
                    source_image_id=node.source_image_id,
                    source_path=node.source_path,
                )
                id_map[node.id] = clone.id
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
        spec = self._registry.spec(op_id)
        if spec is None:
            self.status_message.emit(f"未找到算子：{op_id}")
            return None
        if scene_pos is None:
            scene_pos = self.mapToScene(self.viewport().rect().center())
        scene_pos = QPointF(scene_pos)

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
        node = NodeModel(
            id=new_id("n"),
            op_id=op_id,
            title=spec.name,
            pos=(fallback_pos.x(), fallback_pos.y()),
            params=param_defaults(spec),
        )
        self._undo_stack.push(AddNodeCommand(self.graph, node))
        self.set_selected_node(node.id)
        self.node_created.emit(node.id)
        self.status_message.emit(f"已添加节点“{spec.name}”")
        return node.id

    def refresh_all(self) -> None:
        selected = self.selected_node_id()
        self._clear_items()
        for node in list(self.graph.nodes.values()):
            self._create_node_item(node)
        for connection in list(self.graph.connections.values()):
            self._create_connection_item(connection)
        self._update_scene_rect()
        if selected and selected in self._nodes:
            self.set_selected_node(selected)

    def delete_selected(self) -> None:
        self._pending_move_target = None
        self._pending_move_reason = None
        self._clear_target_highlight()
        node_items = [item for item in self._scene.selectedItems() if isinstance(item, NodeItem)]
        connection_items = [item for item in self._scene.selectedItems() if isinstance(item, ConnectionItem)]
        if not node_items and not connection_items:
            return
        node_ids = {item.node_id for item in node_items}
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
            for item in node_items:
                if item.node_id in self.graph.nodes:
                    self._undo_stack.push(RemoveNodeCommand(self.graph, item.node_id))
            for connection in connections_to_remove:
                if connection.id in self.graph.connections:
                    self._undo_stack.push(RemoveConnectionCommand(self.graph, connection))
        finally:
            self._undo_stack.endMacro()

        self._update_scene_rect()
        self._selection_timer.stop()
        self.selection_changed.emit(self.selected_node_id() or "")
        self.status_message.emit("已删除选中对象")

    # --------------------------------------------------------------- 自动布局
    def _layout_changes(self) -> list[tuple[str, tuple[float, float], tuple[float, float]]]:
        """计算自动布局带来的节点位置变化；图中有环时抛出 :class:`GraphError`。"""

        sizes: dict[str, tuple[float, float]] = {}
        for node_id, item in self._nodes.items():
            size = item.size()
            sizes[node_id] = (float(size.width()), float(size.height()))
        positions = compute_layout(self.graph, sizes)
        changes: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
        for node_id, (x, y) in positions.items():
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
        """把布局变化写入当前 undo 宏；不做 begin/endMacro，便于与插入命令合并。"""

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
        self._update_scene_rect()
        self.fit_graph()

    def fit_graph(self) -> None:
        rect = self._scene.itemsBoundingRect()
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
        self._update_scene_rect()

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
        self._results.pop(node_id, None)
        self._errors.pop(node_id, None)
        self.node_removed.emit(node_id)
        self._update_scene_rect()

    def _on_node_changed(self, node_id: str) -> None:
        node = self.graph.nodes.get(node_id)
        item = self._nodes.get(node_id)
        if node is None or item is None:
            return
        item.sync_position(node.pos)
        item.refresh_from_model()
        item.update_port_states()
        self._update_connections_for(node_id)

    def _on_connection_added(self, connection: ConnectionModel) -> None:
        if connection.id not in self._connections:
            self._create_connection_item(connection)
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
                if other is item:
                    continue
                for out_port in other.output_ports.values():
                    distance = self._port_distance(out_port.center_scene_pos(), in_center)
                    if distance > SNAP_RADIUS:
                        continue
                    if self._connection_exists(other.node_id, out_port.key, moving_id, key):
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
                if other is item:
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
            ports = node_item.input_ports if is_input else node_item.output_ports
            for port in ports.values():
                yield port

    def port_item(self, node_id: str, key: str, is_input: bool) -> PortItem | None:
        node_item = self._nodes.get(node_id)
        if node_item is None:
            return None
        return node_item.port_item(key, is_input)

    def port_spec(self, node_id: str, key: str, is_input: bool) -> PortSpec | None:
        """查询端口声明；源节点动态生成图像输出端口，未注册算子返回 ``None``。"""

        node = self.graph.nodes.get(node_id)
        if node is None:
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

        output_spec = self.port_spec(src_node_id, src_port, False)
        input_spec = self.port_spec(dst_node_id, dst_port, True)
        if output_spec is None or input_spec is None:
            return True
        return port_types_compatible(output_spec, input_spec)

    def _compatible_output_key(self, spec: OperatorSpec, target: PortItem) -> str | None:
        target_spec = self.port_spec(target.node_item.node_id, target.key, True)
        for port in spec.outputs:
            if port.data_type is not target.data_type:
                continue
            if target_spec is not None and not port_types_compatible(port, target_spec):
                continue
            return port.key
        return None

    def _compatible_input_key(self, spec: OperatorSpec, target: PortItem) -> str | None:
        target_spec = self.port_spec(target.node_item.node_id, target.key, False)
        for port in spec.inputs:
            if port.data_type is not target.data_type:
                continue
            if target_spec is not None and not port_types_compatible(target_spec, port):
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
            if source_type is not None and port.data_type is not source_type:
                return False
            return source_spec is None or port_types_compatible(source_spec, port)

        def output_ok(port: PortSpec) -> bool:
            if target_type is not None and port.data_type is not target_type:
                return False
            return target_spec is None or port_types_compatible(port, target_spec)

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
            if not item.hit_test(scene_pos):
                continue
            center = item.boundingRect().center()
            distance = (center - scene_pos).manhattanLength()
            if distance < best_distance:
                best = item
                best_distance = distance
        return best

    def _drop_target_at(
        self, scene_pos: QPointF, spec: OperatorSpec | None
    ) -> _DropTarget | None:
        """库拖拽落点命中检测：优先吸附端口，其次命中已有连线。"""

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

        node = NodeModel(
            id=new_id("n"),
            op_id=op_id,
            title=spec.name,
            pos=(rough.x(), rough.y()),
            params=param_defaults(spec),
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
        node = NodeModel(
            id=new_id("n"),
            op_id=op_id,
            title=spec.name,
            pos=(scene_pos.x(), scene_pos.y()),
            params=param_defaults(spec),
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
        moves: list[tuple[str, tuple[float, float], tuple[float, float]]] = []
        for node_id, _start in starts.items():
            item = self._nodes.get(node_id)
            node = self.graph.nodes.get(node_id)
            if item is None or node is None:
                continue
            new_pos = item.pos()
            if abs(node.pos[0] - new_pos.x()) < 0.01 and abs(node.pos[1] - new_pos.y()) < 0.01:
                continue
            moves.append(
                (
                    node_id,
                    (float(node.pos[0]), float(node.pos[1])),
                    (round(new_pos.x(), 2), round(new_pos.y(), 2)),
                )
            )
        if not moves:
            return

        connection_commands, action_message = self._build_move_connection_commands(target)
        if not connection_commands and reason is not None and target is None:
            self.status_message.emit(reason)

        if len(moves) == 1 and not connection_commands:
            node_id, old_pos, new_pos = moves[0]
            self._undo_stack.push(MoveNodeCommand(self.graph, node_id, old_pos, new_pos))
        else:
            text = "移动并连接节点" if connection_commands else "移动节点"
            self._undo_stack.beginMacro(text)
            try:
                for node_id, old_pos, new_pos in moves:
                    self._undo_stack.push(MoveNodeCommand(self.graph, node_id, old_pos, new_pos))
                for command in connection_commands:
                    self._undo_stack.push(command)
            except GraphError as exc:
                self.status_message.emit(f"自动连接失败：{exc}")
            finally:
                self._undo_stack.endMacro()
        if action_message is not None:
            self.status_message.emit(action_message)
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
            event.accept()
            return
        super().keyPressEvent(event)

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
            scale = self.transform().m11() * factor
            if self._zoom_min <= scale <= self._zoom_max:
                self.scale(factor, factor)
            event.accept()
            return
        super().wheelEvent(event)

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
            menu.addAction("重命名节点", lambda: self.rename_node(node_item.node_id))
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
            for connection in self._connections.values():
                if connection.scene() is self._scene:
                    self._scene.removeItem(connection)
            for node in self._nodes.values():
                if node.scene() is self._scene:
                    self._scene.removeItem(node)
        finally:
            self._syncing_selection = False
        self._connections.clear()
        self._nodes.clear()

    def _update_scene_rect(self) -> None:
        items_rect = self._scene.itemsBoundingRect()
        if items_rect.isNull() or not items_rect.isValid() or items_rect.isEmpty():
            base_rect = QRectF(-500.0, -350.0, 1000.0, 700.0)
        else:
            base_rect = items_rect.adjusted(-180.0, -180.0, 180.0, 180.0)
        visible = self.mapToScene(self.viewport().rect()).boundingRect()
        if visible.isValid() and not visible.isEmpty():
            union = base_rect.united(visible)
        else:
            union = base_rect
        if not math.isfinite(union.width()) or not math.isfinite(union.height()):
            union = QRectF(-500.0, -350.0, 1000.0, 700.0)
        self._scene.setSceneRect(union)


__all__ = ["GraphView"]
