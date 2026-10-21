"""工作区 / DAG 图的数据模型。

模型层不依赖算子实现，只保存 ``op_id``、参数和连线；端口合法性、参数类型
由算子注册表与执行引擎校验。所有会改变图的方法都会发出信号，供画布与引擎响应。
"""

from __future__ import annotations

import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from PySide6.QtCore import QObject, Signal

from app.core.constants import SOURCE_OP_ID
from app.core.types import ImageData


def new_id(prefix: str = "") -> str:
    value = uuid.uuid4().hex[:10]
    return f"{prefix}{value}" if prefix else value


class GraphError(RuntimeError):
    """非法的图操作（重名、环路、重复连线等）。"""


@dataclass
class NodeModel:
    id: str
    op_id: str
    title: str
    pos: tuple[float, float] = (0.0, 0.0)
    params: dict[str, Any] = field(default_factory=dict)
    source_image_id: str | None = None
    source_path: str | None = None
    revision: int = 0

    @property
    def is_source(self) -> bool:
        return self.op_id == SOURCE_OP_ID

    def touch(self) -> None:
        self.revision += 1

    def display_title(self) -> str:
        return self.title or self.op_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "op_id": self.op_id,
            "title": self.title,
            "pos": [float(self.pos[0]), float(self.pos[1])],
            "params": dict(self.params),
            "source_image_id": self.source_image_id,
            "source_path": self.source_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeModel":
        pos = data.get("pos") or [0.0, 0.0]
        return cls(
            id=str(data["id"]),
            op_id=str(data["op_id"]),
            title=str(data.get("title") or data.get("op_id") or ""),
            pos=(float(pos[0]), float(pos[1])),
            params=dict(data.get("params") or {}),
            source_image_id=data.get("source_image_id"),
            source_path=data.get("source_path"),
        )


@dataclass
class ConnectionModel:
    id: str
    src_node: str
    src_port: str
    dst_node: str
    dst_port: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "src_node": self.src_node,
            "src_port": self.src_port,
            "dst_node": self.dst_node,
            "dst_port": self.dst_port,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConnectionModel":
        return cls(
            id=str(data["id"]),
            src_node=str(data["src_node"]),
            src_port=str(data["src_port"]),
            dst_node=str(data["dst_node"]),
            dst_port=str(data["dst_port"]),
        )


class GraphModel(QObject):
    """有向无环图。输入端口最多一条连线；输出端口可以扇出。"""

    node_added = Signal(object)  # NodeModel
    node_removed = Signal(str)
    node_changed = Signal(str)  # node id：参数 / 位置 / 标题
    connection_added = Signal(object)  # ConnectionModel
    connection_removed = Signal(object)
    structure_changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.nodes: "OrderedDict[str, NodeModel]" = OrderedDict()
        self.connections: "OrderedDict[str, ConnectionModel]" = OrderedDict()

    # ------------------------------------------------------------------ 节点
    def add_node(self, node: NodeModel) -> NodeModel:
        if node.id in self.nodes:
            raise GraphError(f"节点 id 已存在：{node.id}")
        self.nodes[node.id] = node
        self.node_added.emit(node)
        self.structure_changed.emit()
        return node

    def create_operator_node(
        self,
        op_id: str,
        title: str,
        params: dict[str, Any] | None = None,
        pos: tuple[float, float] = (0.0, 0.0),
    ) -> NodeModel:
        return self.add_node(
            NodeModel(
                id=new_id("n"),
                op_id=op_id,
                title=title,
                pos=pos,
                params=dict(params or {}),
            )
        )

    def create_source_node(
        self,
        image_id: str,
        title: str,
        source_path: str | None = None,
        pos: tuple[float, float] = (0.0, 0.0),
    ) -> NodeModel:
        return self.add_node(
            NodeModel(
                id=new_id("n"),
                op_id=SOURCE_OP_ID,
                title=title,
                pos=pos,
                source_image_id=image_id,
                source_path=source_path,
            )
        )

    def node(self, node_id: str) -> NodeModel:
        try:
            return self.nodes[node_id]
        except KeyError as exc:
            raise GraphError(f"节点不存在：{node_id}") from exc

    def remove_node(self, node_id: str) -> list[ConnectionModel]:
        self.node(node_id)
        removed = [c for c in self.connections.values() if c.src_node == node_id or c.dst_node == node_id]
        for connection in removed:
            self.remove_connection(connection.id)
        del self.nodes[node_id]
        self.node_removed.emit(node_id)
        self.structure_changed.emit()
        return removed

    def rename_node(self, node_id: str, title: str) -> None:
        node = self.node(node_id)
        title = title.strip()
        if not title or node.title == title:
            return
        node.title = title
        node.touch()
        self.node_changed.emit(node_id)

    def set_node_pos(self, node_id: str, pos: tuple[float, float], touch: bool = True) -> None:
        node = self.node(node_id)
        node.pos = (float(pos[0]), float(pos[1]))
        if touch:
            self.node_changed.emit(node_id)

    def set_node_params(self, node_id: str, params: dict[str, Any], touch: bool = True) -> None:
        node = self.node(node_id)
        node.params = dict(params)
        node.touch()
        if touch:
            self.node_changed.emit(node_id)

    # ------------------------------------------------------------------ 连线
    def connection(self, connection_id: str) -> ConnectionModel:
        try:
            return self.connections[connection_id]
        except KeyError as exc:
            raise GraphError(f"连线不存在：{connection_id}") from exc

    def input_connection(self, node_id: str, port: str) -> ConnectionModel | None:
        for connection in self.connections.values():
            if connection.dst_node == node_id and connection.dst_port == port:
                return connection
        return None

    def incoming(self, node_id: str) -> list[ConnectionModel]:
        return [c for c in self.connections.values() if c.dst_node == node_id]

    def outgoing(self, node_id: str) -> list[ConnectionModel]:
        return [c for c in self.connections.values() if c.src_node == node_id]

    def add_connection(
        self,
        src_node: str,
        src_port: str,
        dst_node: str,
        dst_port: str,
        connection_id: str | None = None,
    ) -> ConnectionModel:
        self.node(src_node)
        self.node(dst_node)
        if src_node == dst_node:
            raise GraphError("不能把节点连接到自身")
        if not src_port or not dst_port:
            raise GraphError("端口名不能为空")
        existing = self.input_connection(dst_node, dst_port)
        if existing is not None:
            if existing.src_node == src_node and existing.src_port == src_port:
                raise GraphError("该连线已经存在")
            raise GraphError("该输入端口已连接，请先断开旧连线")
        for connection in self.connections.values():
            if (
                connection.src_node == src_node
                and connection.src_port == src_port
                and connection.dst_node == dst_node
                and connection.dst_port == dst_port
            ):
                raise GraphError("该连线已经存在")
        if self._reachable(dst_node, src_node):
            raise GraphError("该连线会形成环路（当前版本不支持回环）")
        connection = ConnectionModel(connection_id or new_id("c"), src_node, src_port, dst_node, dst_port)
        self.connections[connection.id] = connection
        self.connection_added.emit(connection)
        self.structure_changed.emit()
        return connection

    def remove_connection(self, connection_id: str) -> ConnectionModel:
        try:
            connection = self.connections.pop(connection_id)
        except KeyError as exc:
            raise GraphError(f"连线不存在：{connection_id}") from exc
        self.connection_removed.emit(connection)
        self.structure_changed.emit()
        return connection

    def remove_connections_to(self, node_id: str, port: str) -> list[ConnectionModel]:
        removed = [c for c in self.connections.values() if c.dst_node == node_id and c.dst_port == port]
        for connection in list(removed):
            self.remove_connection(connection.id)
        return removed

    # ------------------------------------------------------------------ 遍历
    def successors(self, node_id: str) -> list[str]:
        seen: list[str] = []
        for connection in self.connections.values():
            if connection.src_node == node_id and connection.dst_node not in seen:
                seen.append(connection.dst_node)
        return seen

    def predecessors(self, node_id: str) -> list[str]:
        seen: list[str] = []
        for connection in self.connections.values():
            if connection.dst_node == node_id and connection.src_node not in seen:
                seen.append(connection.src_node)
        return seen

    def descendants(self, node_id: str, include_self: bool = False) -> list[str]:
        result: list[str] = [node_id] if include_self else []
        queue = deque(self.successors(node_id))
        visited = {node_id}
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            result.append(current)
            queue.extend(self.successors(current))
        return result

    def ancestors(self, node_id: str, include_self: bool = False) -> list[str]:
        result: list[str] = [node_id] if include_self else []
        queue = deque(self.predecessors(node_id))
        visited = {node_id}
        while queue:
            current = queue.popleft()
            if current in visited:
                continue
            visited.add(current)
            result.append(current)
            queue.extend(self.predecessors(current))
        return result

    def _reachable(self, start: str, target: str) -> bool:
        """从 start 出发（沿连接方向）能否到达 target。"""
        queue = deque([start])
        visited = {start}
        while queue:
            current = queue.popleft()
            if current == target:
                return True
            for nxt in self.successors(current):
                if nxt not in visited:
                    visited.add(nxt)
                    queue.append(nxt)
        return False

    def topological_order(self, node_ids: Iterable[str] | None = None) -> list[str]:
        """Kahn 拓扑排序；若存在环将抛出 :class:`GraphError`。

        同一对节点之间可能存在多条端口连线（例如通道拆分/合并），入度按
        “唯一的前驱节点对”计算，避免把并行端口连线误判为环。
        """
        selected = set(node_ids) if node_ids is not None else set(self.nodes)
        indegree = {node_id: 0 for node_id in selected}
        edges = {
            (connection.src_node, connection.dst_node)
            for connection in self.connections.values()
            if connection.src_node in selected and connection.dst_node in selected
        }
        for _src, dst in edges:
            indegree[dst] += 1
        queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
        result: list[str] = []
        while queue:
            node_id = queue.popleft()
            result.append(node_id)
            for nxt in self.successors(node_id):
                if nxt in indegree:
                    indegree[nxt] -= 1
                    if indegree[nxt] == 0:
                        queue.append(nxt)
        if len(result) != len(selected):
            raise GraphError("图中存在环路，无法排序")
        return result

    def source_nodes(self) -> list[NodeModel]:
        return [node for node in self.nodes.values() if node.is_source]

    # --------------------------------------------------------------- 序列化
    def to_dict(self) -> dict[str, Any]:
        return {
            "nodes": [node.to_dict() for node in self.nodes.values()],
            "connections": [connection.to_dict() for connection in self.connections.values()],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GraphModel":
        graph = cls()
        for node_data in data.get("nodes") or []:
            graph.add_node(NodeModel.from_dict(node_data))
        for connection_data in data.get("connections") or []:
            connection = ConnectionModel.from_dict(connection_data)
            if connection.src_node not in graph.nodes or connection.dst_node not in graph.nodes:
                continue
            graph.connections[connection.id] = connection
        return graph


class WorkspaceModel(QObject):
    """一个工程工作区：一张 DAG + 若干源图像。"""

    name_changed = Signal(str)
    image_added = Signal(str)  # image id

    def __init__(self, name: str = "未命名工作区", parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._name = name
        self.graph = GraphModel(self)
        self.images: "OrderedDict[str, ImageData]" = OrderedDict()
        self.image_paths: dict[str, str | None] = {}

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        value = value.strip() or "未命名工作区"
        if value != self._name:
            self._name = value
            self.name_changed.emit(value)

    def add_source_image(
        self,
        image: ImageData,
        path: str | None = None,
        pos: tuple[float, float] = (0.0, 0.0),
        title: str | None = None,
    ) -> NodeModel:
        image_id = new_id("img")
        display_name = title or (Path(path).name if path else image.name) or "原始图像"
        image.name = display_name
        self.images[image_id] = image
        self.image_paths[image_id] = path
        node = self.graph.create_source_node(image_id, display_name, path, pos)
        self.image_added.emit(image_id)
        return node

    def image_for_node(self, node: NodeModel) -> ImageData | None:
        if not node.source_image_id:
            return None
        return self.images.get(node.source_image_id)

    def prune_unused_images(self) -> None:
        used = {node.source_image_id for node in self.graph.nodes.values() if node.source_image_id}
        for image_id in list(self.images.keys()):
            if image_id not in used:
                self.images.pop(image_id, None)
                self.image_paths.pop(image_id, None)

    def next_source_position(self) -> tuple[float, float]:
        count = len(self.graph.source_nodes())
        return (0.0, float(count * 160))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "graph": self.graph.to_dict(),
            "images": [
                {
                    "id": image_id,
                    "name": image.name,
                    "path": self.image_paths.get(image_id),
                }
                for image_id, image in self.images.items()
            ],
        }
