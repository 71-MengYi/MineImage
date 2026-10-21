"""工作区 / DAG 图的数据模型。

模型层不依赖算子实现，只保存 ``op_id``、参数和连线；端口合法性、参数类型
由算子注册表与执行引擎校验。所有会改变图的方法都会发出信号，供画布与引擎响应。
"""

from __future__ import annotations

import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from PySide6.QtCore import QObject, Signal

from app.core.constants import FRAME_OP_ID, SOURCE_OP_ID
from app.core.frame import INVALID_LEVEL, FrameModel
from app.core.types import ImageData


def new_id(prefix: str = "") -> str:
    value = uuid.uuid4().hex[:10]
    return f"{prefix}{value}" if prefix else value


def is_frame_op_id(op_id: str) -> bool:
    return op_id == FRAME_OP_ID


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
    #: 所属流程框的节点 id；None 表示顶层节点。
    parent_frame: str | None = None
    #: 流程框节点独有的端口 / 来源信息；普通算子节点为 None。
    frame: FrameModel | None = None

    @property
    def is_source(self) -> bool:
        return self.op_id == SOURCE_OP_ID

    @property
    def is_frame(self) -> bool:
        return self.op_id == FRAME_OP_ID and self.frame is not None

    def touch(self) -> None:
        self.revision += 1

    def display_title(self) -> str:
        return self.title or self.op_id

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "op_id": self.op_id,
            "title": self.title,
            "pos": [float(self.pos[0]), float(self.pos[1])],
            "params": dict(self.params),
            "source_image_id": self.source_image_id,
            "source_path": self.source_path,
        }
        if self.parent_frame is not None:
            data["parent_frame"] = self.parent_frame
        if self.frame is not None:
            data["frame"] = self.frame.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeModel":
        pos = data.get("pos") or [0.0, 0.0]
        raw_frame = data.get("frame")
        frame = (
            FrameModel.from_dict(raw_frame)
            if isinstance(raw_frame, dict) or is_frame_op_id(str(data.get("op_id") or ""))
            else None
        )
        parent = data.get("parent_frame")
        return cls(
            id=str(data["id"]),
            op_id=str(data["op_id"]),
            title=str(data.get("title") or data.get("op_id") or ""),
            pos=(float(pos[0]), float(pos[1])),
            params=dict(data.get("params") or {}),
            source_image_id=data.get("source_image_id"),
            source_path=data.get("source_path"),
            parent_frame=str(parent) if parent else None,
            frame=frame,
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
        # 由界面层注入：fn(node_id, port_key, is_input) -> DataType | None
        # 用于在流程框端口被连接时确定类型；模型层不依赖算子注册表。
        self._port_type_resolver: Callable[[str, str, bool], Any] | None = None

    # ------------------------------------------------------------------ 节点
    def add_node(self, node: NodeModel) -> NodeModel:
        if node.id in self.nodes:
            raise GraphError(f"节点 id 已存在：{node.id}")
        if node.parent_frame is not None:
            parent = self.nodes.get(node.parent_frame)
            if parent is None or not parent.is_frame:
                raise GraphError(f"父流程框不存在：{node.parent_frame}")
        self.nodes[node.id] = node
        self.node_added.emit(node)
        self.structure_changed.emit()
        return node

    def create_frame_node(
        self,
        title: str = "流程框",
        pos: tuple[float, float] = (0.0, 0.0),
        frame: FrameModel | None = None,
        parent_frame: str | None = None,
    ) -> NodeModel:
        return self.add_node(
            NodeModel(
                id=new_id("n"),
                op_id=FRAME_OP_ID,
                title=title,
                pos=pos,
                frame=frame.copy() if frame is not None else FrameModel.create_default(),
                parent_frame=parent_frame,
            )
        )

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

    def set_node_frame(self, node_id: str, frame: FrameModel, touch: bool = True) -> None:
        node = self.node(node_id)
        if not node.is_frame:
            raise GraphError(f"节点不是流程框：{node_id}")
        node.frame = frame
        if touch:
            node.touch()
            self.node_changed.emit(node_id)

    def set_node_parent(self, node_id: str, parent_frame: str | None, touch: bool = True) -> None:
        node = self.node(node_id)
        if parent_frame is not None:
            parent = self.nodes.get(parent_frame)
            if parent is None or not parent.is_frame:
                raise GraphError(f"父流程框不存在：{parent_frame}")
            if node_id == parent_frame or self.is_inside(parent_frame, node_id):
                raise GraphError("不能把流程框移动到它自己的内部")
        node.parent_frame = parent_frame
        if touch:
            node.touch()
            self.node_changed.emit(node_id)

    # ------------------------------------------------------------ 流程框语义
    def set_port_type_resolver(self, resolver: Callable[[str, str, bool], Any] | None) -> None:
        self._port_type_resolver = resolver

    def node_container(self, node_id: str) -> str | None:
        node = self.nodes.get(node_id)
        return node.parent_frame if node is not None else None

    def direct_children(self, frame_id: str) -> list[NodeModel]:
        return [node for node in self.nodes.values() if node.parent_frame == frame_id]

    def frame_subtree(self, frame_id: str, *, include_self: bool = False) -> list[NodeModel]:
        """返回流程框的全部后代节点（默认不含框自身），按节点创建顺序。"""

        result: list[NodeModel] = []
        if include_self and frame_id in self.nodes:
            result.append(self.nodes[frame_id])
        queue: deque[str] = deque([frame_id])
        seen = {frame_id}
        while queue:
            current = queue.popleft()
            for node in self.nodes.values():
                if node.parent_frame != current or node.id in seen:
                    continue
                seen.add(node.id)
                result.append(node)
                queue.append(node.id)
        return result

    def is_inside(self, node_id: str, frame_id: str) -> bool:
        """node_id 是否位于 frame_id 的（多层）内部；框自身不算。"""

        if node_id == frame_id:
            return False
        current = self.nodes.get(node_id)
        seen: set[str] = set()
        while current is not None and current.parent_frame:
            parent = current.parent_frame
            if parent == frame_id:
                return True
            if parent in seen:
                return False
            seen.add(parent)
            current = self.nodes.get(parent)
        return False

    def port_level(self, node_id: str, port_key: str, role: str) -> str | None:
        """端口所在的数据层级。

        普通节点端口在其 parent_frame 层；流程框的输入/输出端口有内外两侧：
        * 作为框内数据源 / 目标时属于框内部（层级 = 框 id）；
        * 作为框外连接的起点 / 终点时属于框所在的层级（= 框的 parent_frame）。
        两条端口层级相同才允许连线。
        """

        if role not in ("src", "dst"):
            raise ValueError(f"非法端口角色：{role}")
        node = self.nodes.get(node_id)
        if node is None:
            return INVALID_LEVEL
        if node.is_frame and node.frame is not None:
            frame = node.frame
            if port_key in frame.input_keys():
                return node.id if role == "src" else node.parent_frame
            if port_key in frame.output_keys():
                return node.parent_frame if role == "src" else node.id
            return INVALID_LEVEL
        return node.parent_frame

    def connection_role(
        self,
        src_node: str,
        src_port: str,
        dst_node: str,
        dst_port: str,
    ) -> str:
        """返回连线的边界语义，供层级校验与回环检测使用。"""

        src = self.nodes.get(src_node)
        dst = self.nodes.get(dst_node)
        if src is None or dst is None:
            return "invalid"
        if src.is_frame and src.frame is not None:
            if src_port in src.frame.input_keys():
                return "frame_input_inner"
            if src_port in src.frame.output_keys():
                return "frame_output_outer"
            return "invalid"
        if dst.is_frame and dst.frame is not None:
            if dst_port in dst.frame.output_keys():
                return "frame_output_inner"
            if dst_port in dst.frame.input_keys():
                return "frame_input_outer"
            return "invalid"
        return "normal"

    def connection_allowed(
        self,
        src_node: str,
        src_port: str,
        dst_node: str,
        dst_port: str,
    ) -> tuple[bool, str]:
        """流程框感知的连线合法性：端口层级必须一致。"""

        if src_node not in self.nodes or dst_node not in self.nodes:
            return False, "节点不存在"
        if src_node == dst_node:
            return False, "不能把节点连接到自身"
        if not src_port or not dst_port:
            return False, "端口名不能为空"
        if self.connection_role(src_node, src_port, dst_node, dst_port) == "invalid":
            return False, "流程框端口不存在"
        src_level = self.port_level(src_node, src_port, "src")
        dst_level = self.port_level(dst_node, dst_port, "dst")
        if src_level == INVALID_LEVEL or dst_level == INVALID_LEVEL:
            return False, "端口不存在"
        if src_level != dst_level:
            return (
                False,
                "只能连接同一流程框层级内的节点；框内节点请通过流程框的输入/输出端口连接",
            )
        return True, ""

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
        allowed, reason = self.connection_allowed(src_node, src_port, dst_node, dst_port)
        if not allowed:
            raise GraphError(reason)
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
        role = self.connection_role(src_node, src_port, dst_node, dst_port)
        # 框输入 / 框输出边界边本身不构成节点级依赖，跳过回环检查，避免把
        # 「框输入 -> 框内节点」和「框内节点 -> 框输出」误判为环。
        if role not in ("frame_input_inner", "frame_output_inner") and self._reachable(
            dst_node, src_node
        ):
            raise GraphError("该连线会形成环路（当前版本不支持回环）")
        connection = ConnectionModel(connection_id or new_id("c"), src_node, src_port, dst_node, dst_port)
        self.connections[connection.id] = connection
        self._apply_frame_port_binding(connection)
        self.connection_added.emit(connection)
        self.structure_changed.emit()
        return connection

    def _apply_frame_port_binding(self, connection: ConnectionModel) -> None:
        """连线落到流程框输入/输出端口时，确定端口类型并追加新的待定端口。"""

        frame_node = self.nodes.get(connection.dst_node)
        if frame_node is None or not frame_node.is_frame or frame_node.frame is None:
            return
        frame = frame_node.frame
        if connection.dst_port in frame.input_keys():
            is_input = True
        elif connection.dst_port in frame.output_keys():
            is_input = False
        else:
            return
        data_type = None
        resolver = self._port_type_resolver
        if resolver is not None:
            try:
                data_type = resolver(connection.src_node, connection.src_port, False)
            except Exception:  # noqa: BLE001 - 类型解析失败时保持待定
                data_type = None
        if frame.bind(connection.dst_port, is_input, data_type):
            frame_node.touch()
            self.node_changed.emit(frame_node.id)

    # -------------------------------------------------------- 流程框边界解析
    def resolve_source(
        self,
        src_node: str,
        src_port: str,
        _seen: set[tuple[str, str]] | None = None,
    ) -> tuple[str, str] | None:
        """把可能位于流程框边界上的源端口解析成真实的普通节点端口。

        * 源是流程框输入端口 -> 沿该输入端口的外部入边继续解析；
        * 源是流程框输出端口 -> 沿该输出端口的内部入边继续解析；
        * 普通节点直接返回。遇到断开的边界返回 None。
        """

        node = self.nodes.get(src_node)
        if node is None or not src_port:
            return None
        seen = set() if _seen is None else _seen
        key = (src_node, src_port)
        if key in seen:
            return None
        seen.add(key)
        if node.is_frame and node.frame is not None:
            frame = node.frame
            if src_port not in frame.input_keys() and src_port not in frame.output_keys():
                return None
            incoming = self.input_connection(src_node, src_port)
            if incoming is None:
                return None
            return self.resolve_source(incoming.src_node, incoming.src_port, seen)
        return (src_node, src_port)

    def frame_output_sources(self, frame_id: str) -> dict[str, tuple[str, str]]:
        """流程框每个输出端口真实的数据来源（已跨过嵌套边界解析）。"""

        result: dict[str, tuple[str, str]] = {}
        node = self.nodes.get(frame_id)
        if node is None or not node.is_frame or node.frame is None:
            return result
        for key in node.frame.output_keys():
            incoming = self.input_connection(frame_id, key)
            if incoming is None:
                continue
            resolved = self.resolve_source(incoming.src_node, incoming.src_port)
            if resolved is not None:
                result[key] = resolved
        return result

    def frame_state_snapshot(self) -> dict[str, dict[str, Any]]:
        """所有流程框的端口快照，供连线命令撤销时恢复自动增长的端口。"""

        return {
            node_id: node.frame.to_dict()
            for node_id, node in self.nodes.items()
            if node.is_frame and node.frame is not None
        }

    def restore_frame_state(self, state: dict[str, dict[str, Any]] | None) -> None:
        for node_id, data in (state or {}).items():
            node = self.nodes.get(node_id)
            if node is None or not node.is_frame:
                continue
            node.frame = FrameModel.from_dict(data)
            node.touch()
            self.node_changed.emit(node_id)

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

    def effective_successors(self, node_id: str) -> list[str]:
        """用于回环检测的后继：忽略「流程框输入 -> 框内节点」边界边。

        框输入来自框外上游，不是流程框节点自己产生的数据；把它算作流程框的
        出边会在「框输入 -> 框内节点 -> 框输出」时误判出环。
        """

        seen: list[str] = []
        for connection in self.connections.values():
            if connection.src_node != node_id:
                continue
            if (
                self.connection_role(
                    connection.src_node,
                    connection.src_port,
                    connection.dst_node,
                    connection.dst_port,
                )
                == "frame_input_inner"
            ):
                continue
            if connection.dst_node not in seen:
                seen.append(connection.dst_node)
        return seen

    def _reachable(self, start: str, target: str) -> bool:
        """从 start 出发（沿流程框感知后的连接方向）能否到达 target。"""

        queue = deque([start])
        visited = {start}
        while queue:
            current = queue.popleft()
            if current == target:
                return True
            for nxt in self.effective_successors(current):
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
            try:
                node = NodeModel.from_dict(node_data)
            except (KeyError, TypeError, ValueError):
                continue
            if node.id in graph.nodes:
                continue
            graph.nodes[node.id] = node
        # 修复悬空的 parent_frame，避免旧工程 / 手工编辑数据让流程框失效。
        for node in graph.nodes.values():
            parent = node.parent_frame
            if parent is not None:
                parent_node = graph.nodes.get(parent)
                if parent_node is None or not parent_node.is_frame:
                    node.parent_frame = None
        for connection_data in data.get("connections") or []:
            try:
                connection = ConnectionModel.from_dict(connection_data)
            except (KeyError, TypeError, ValueError):
                continue
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
