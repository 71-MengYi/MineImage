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

from app.core.constants import (
    DEFAULT_IMAGE_DISPLAY_MODE,
    GROUP_OP_ID,
    IMAGE_DISPLAY_MODES,
    SOURCE_OP_ID,
)
from app.core.group import INVALID_LEVEL, GroupModel, GroupPort
from app.core.types import ImageData


def new_id(prefix: str = "") -> str:
    value = uuid.uuid4().hex[:10]
    return f"{prefix}{value}" if prefix else value


def is_group_op_id(op_id: str) -> bool:
    return op_id == GROUP_OP_ID


class GraphError(RuntimeError):
    """非法的图操作（重名、环路、重复连线等）。"""


@dataclass
class NodeModel:
    id: str
    op_id: str
    title: str
    pos: tuple[float, float] = (0.0, 0.0)
    params: dict[str, Any] = field(default_factory=dict)
    #: 源节点引用的第一张图像（兼容旧工程 / 单图打开）。
    source_image_id: str | None = None
    #: 源节点引用的全部图像；为空时回退到 source_image_id。
    source_image_ids: list[str] = field(default_factory=list)
    source_path: str | None = None
    revision: int = 0
    #: 所属算子组的节点 id；None 表示顶层节点。
    parent_group: str | None = None
    #: 算子组节点独有的端口 / 来源信息；普通算子节点为 None。
    group: GroupModel | None = None

    @property
    def is_source(self) -> bool:
        return self.op_id == SOURCE_OP_ID

    @property
    def is_group(self) -> bool:
        return self.op_id == GROUP_OP_ID and self.group is not None

    @property
    def source_ids(self) -> list[str]:
        """源节点引用的全部图像 id；旧工程只有一个 source_image_id。"""

        ids = [str(image_id) for image_id in self.source_image_ids if image_id]
        if not ids and self.source_image_id:
            ids = [str(self.source_image_id)]
        return ids

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
        ids = self.source_ids
        if ids:
            data["source_image_ids"] = ids
        if self.parent_group is not None:
            data["parent_group"] = self.parent_group
        if self.group is not None:
            data["group"] = self.group.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NodeModel":
        pos = data.get("pos") or [0.0, 0.0]
        raw_group = data.get("group")
        group = (
            GroupModel.from_dict(raw_group)
            if isinstance(raw_group, dict) or is_group_op_id(str(data.get("op_id") or ""))
            else None
        )
        parent = data.get("parent_group")
        raw_ids = data.get("source_image_ids")
        image_ids = [str(item) for item in raw_ids if item] if isinstance(raw_ids, (list, tuple)) else []
        single = data.get("source_image_id")
        if single:
            single = str(single)
            if single not in image_ids:
                image_ids.insert(0, single)
        return cls(
            id=str(data["id"]),
            op_id=str(data["op_id"]),
            title=str(data.get("title") or data.get("op_id") or ""),
            pos=(float(pos[0]), float(pos[1])),
            params=dict(data.get("params") or {}),
            source_image_id=image_ids[0] if image_ids else None,
            source_image_ids=image_ids,
            source_path=data.get("source_path"),
            parent_group=str(parent) if parent else None,
            group=group,
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
        # 用于在算子组端口被连接时确定类型；模型层不依赖算子注册表。
        self._port_type_resolver: Callable[[str, str, bool], Any] | None = None

    # ------------------------------------------------------------------ 节点
    def add_node(self, node: NodeModel) -> NodeModel:
        if node.id in self.nodes:
            raise GraphError(f"节点 id 已存在：{node.id}")
        if node.parent_group is not None:
            parent = self.nodes.get(node.parent_group)
            if parent is None or not parent.is_group:
                raise GraphError(f"父算子组不存在：{node.parent_group}")
        self.nodes[node.id] = node
        self.node_added.emit(node)
        self.structure_changed.emit()
        return node

    def create_group_node(
        self,
        title: str = "算子组",
        pos: tuple[float, float] = (0.0, 0.0),
        group: GroupModel | None = None,
        parent_group: str | None = None,
    ) -> NodeModel:
        return self.add_node(
            NodeModel(
                id=new_id("n"),
                op_id=GROUP_OP_ID,
                title=title,
                pos=pos,
                group=group.copy() if group is not None else GroupModel.create_default(),
                parent_group=parent_group,
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
        image_ids: Iterable[str] | None = None,
    ) -> NodeModel:
        ids = [str(item) for item in (image_ids or ()) if item]
        if image_id and image_id not in ids:
            ids.insert(0, image_id)
        return self.add_node(
            NodeModel(
                id=new_id("n"),
                op_id=SOURCE_OP_ID,
                title=title,
                pos=pos,
                source_image_id=ids[0] if ids else image_id,
                source_image_ids=ids,
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

    def set_node_group(self, node_id: str, group: GroupModel, touch: bool = True) -> None:
        node = self.node(node_id)
        if not node.is_group:
            raise GraphError(f"节点不是算子组：{node_id}")
        node.group = group
        if touch:
            node.touch()
            self.node_changed.emit(node_id)

    def set_node_parent(self, node_id: str, parent_group: str | None, touch: bool = True) -> None:
        node = self.node(node_id)
        if parent_group is not None:
            parent = self.nodes.get(parent_group)
            if parent is None or not parent.is_group:
                raise GraphError(f"父算子组不存在：{parent_group}")
            if node_id == parent_group or self.is_inside(parent_group, node_id):
                raise GraphError("不能把算子组移动到它自己的内部")
        node.parent_group = parent_group
        if touch:
            node.touch()
            self.node_changed.emit(node_id)

    # ------------------------------------------------------------ 算子组语义
    def set_port_type_resolver(self, resolver: Callable[[str, str, bool], Any] | None) -> None:
        self._port_type_resolver = resolver

    def node_container(self, node_id: str) -> str | None:
        node = self.nodes.get(node_id)
        return node.parent_group if node is not None else None

    def direct_children(self, group_id: str) -> list[NodeModel]:
        return [node for node in self.nodes.values() if node.parent_group == group_id]

    def group_subtree(self, group_id: str, *, include_self: bool = False) -> list[NodeModel]:
        """返回算子组的全部后代节点（默认不含算子组自身），按节点创建顺序。"""

        result: list[NodeModel] = []
        if include_self and group_id in self.nodes:
            result.append(self.nodes[group_id])
        queue: deque[str] = deque([group_id])
        seen = {group_id}
        while queue:
            current = queue.popleft()
            for node in self.nodes.values():
                if node.parent_group != current or node.id in seen:
                    continue
                seen.add(node.id)
                result.append(node)
                queue.append(node.id)
        return result

    def is_inside(self, node_id: str, group_id: str) -> bool:
        """node_id 是否位于 group_id 的（多层）内部；算子组自身不算。"""

        if node_id == group_id:
            return False
        current = self.nodes.get(node_id)
        seen: set[str] = set()
        while current is not None and current.parent_group:
            parent = current.parent_group
            if parent == group_id:
                return True
            if parent in seen:
                return False
            seen.add(parent)
            current = self.nodes.get(parent)
        return False

    def port_level(self, node_id: str, port_key: str, role: str) -> str | None:
        """端口所在的数据层级。

        普通节点端口在其 parent_group 层；算子组的输入/输出端口有内外两侧：
        * 作为组内数据源 / 目标时属于组内部（层级 = 算子组 id）；
        * 作为组外连接的起点 / 终点时属于算子组所在的层级（= 组的 parent_group）。
        两条端口层级相同才允许连线。
        """

        if role not in ("src", "dst"):
            raise ValueError(f"非法端口角色：{role}")
        node = self.nodes.get(node_id)
        if node is None:
            return INVALID_LEVEL
        if node.is_group and node.group is not None:
            group = node.group
            if port_key in group.input_keys():
                return node.id if role == "src" else node.parent_group
            if port_key in group.output_keys():
                return node.parent_group if role == "src" else node.id
            return INVALID_LEVEL
        return node.parent_group

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
        if src.is_group and src.group is not None:
            if src_port in src.group.input_keys():
                return "group_input_inner"
            if src_port in src.group.output_keys():
                return "group_output_outer"
            return "invalid"
        if dst.is_group and dst.group is not None:
            if dst_port in dst.group.output_keys():
                return "group_output_inner"
            if dst_port in dst.group.input_keys():
                return "group_input_outer"
            return "invalid"
        return "normal"

    def connection_allowed(
        self,
        src_node: str,
        src_port: str,
        dst_node: str,
        dst_port: str,
    ) -> tuple[bool, str]:
        """算子组感知的连线合法性：端口层级必须一致。"""

        if src_node not in self.nodes or dst_node not in self.nodes:
            return False, "节点不存在"
        if src_node == dst_node:
            return False, "不能把节点连接到自身"
        if not src_port or not dst_port:
            return False, "端口名不能为空"
        if self.connection_role(src_node, src_port, dst_node, dst_port) == "invalid":
            return False, "算子组端口不存在"
        src_level = self.port_level(src_node, src_port, "src")
        dst_level = self.port_level(dst_node, dst_port, "dst")
        if src_level == INVALID_LEVEL or dst_level == INVALID_LEVEL:
            return False, "端口不存在"
        if src_level != dst_level:
            return (
                False,
                "只能连接同一算子组层级内的节点；组内节点请通过算子组的输入/输出端口连接",
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
        # 组输入 / 组输出边界边本身不构成节点级依赖，跳过回环检查，避免把
        # 「组输入 -> 组内节点」和「组内节点 -> 组输出」误判为环。
        if role not in ("group_input_inner", "group_output_inner") and self._reachable(
            dst_node, src_node
        ):
            raise GraphError("该连线会形成环路（当前版本不支持回环）")
        connection = ConnectionModel(connection_id or new_id("c"), src_node, src_port, dst_node, dst_port)
        self.connections[connection.id] = connection
        self._apply_group_port_binding(connection)
        self.connection_added.emit(connection)
        self.structure_changed.emit()
        return connection

    def _apply_group_port_binding(self, connection: ConnectionModel) -> None:
        """连线落到算子组输入/输出端口时，确定端口类型并追加新的待定端口。"""

        group_node = self.nodes.get(connection.dst_node)
        if group_node is None or not group_node.is_group or group_node.group is None:
            return
        group = group_node.group
        if connection.dst_port in group.input_keys():
            is_input = True
        elif connection.dst_port in group.output_keys():
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
        if group.bind(connection.dst_port, is_input, data_type):
            group_node.touch()
            self.node_changed.emit(group_node.id)

    # -------------------------------------------------------- 算子组边界解析
    def resolve_source(
        self,
        src_node: str,
        src_port: str,
        _seen: set[tuple[str, str]] | None = None,
    ) -> tuple[str, str] | None:
        """把可能位于算子组边界上的源端口解析成真实的普通节点端口。

        * 源是算子组输入端口 -> 沿该输入端口的外部入边继续解析；
        * 源是算子组输出端口 -> 沿该输出端口的内部入边继续解析；
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
        if node.is_group and node.group is not None:
            group = node.group
            if src_port not in group.input_keys() and src_port not in group.output_keys():
                return None
            incoming = self.input_connection(src_node, src_port)
            if incoming is None:
                return None
            return self.resolve_source(incoming.src_node, incoming.src_port, seen)
        return (src_node, src_port)

    def group_output_sources(self, group_id: str) -> dict[str, tuple[str, str]]:
        """算子组每个输出端口真实的数据来源（已跨过嵌套边界解析）。"""

        result: dict[str, tuple[str, str]] = {}
        node = self.nodes.get(group_id)
        if node is None or not node.is_group or node.group is None:
            return result
        for key in node.group.output_keys():
            incoming = self.input_connection(group_id, key)
            if incoming is None:
                continue
            resolved = self.resolve_source(incoming.src_node, incoming.src_port)
            if resolved is not None:
                result[key] = resolved
        return result

    def group_port_free(self, group_id: str, key: str, is_input: bool) -> bool:
        """算子组边界端口是否已经完全空闲（内外两侧都没有连线）。

        只有空闲端口才允许被自动连线复用或回收；只要任一侧仍有连线，端口就
        继续保留，避免删除一条线时错误地拆掉另一侧的连接。
        """

        node = self.nodes.get(group_id)
        group = node.group if node is not None and node.is_group else None
        if group is None:
            return False
        keys = group.input_keys() if is_input else group.output_keys()
        if key not in keys:
            return False
        for connection in self.connections.values():
            if connection.src_node == group_id and connection.src_port == key:
                return False
            if connection.dst_node == group_id and connection.dst_port == key:
                return False
        return True

    def _group_boundary_ports(
        self, connection: ConnectionModel
    ) -> list[tuple[str, str, bool]]:
        """返回连线两端属于算子组边界端口的 ``(组 id, key, 是否输入)``。"""

        result: list[tuple[str, str, bool]] = []
        for node_id, key in (
            (connection.src_node, connection.src_port),
            (connection.dst_node, connection.dst_port),
        ):
            node = self.nodes.get(node_id)
            group = node.group if node is not None and node.is_group else None
            if group is None:
                continue
            if key in group.input_keys():
                result.append((node_id, key, True))
            elif key in group.output_keys():
                result.append((node_id, key, False))
        return result

    def _reclaim_group_port(self, group_id: str, key: str, is_input: bool) -> bool:
        """连线删除后回收已经不再使用的已定类型端口。

        待定端口是"下一条连线"的入口，不参与回收；已定类型端口一旦两侧都没有
        连线，就从算子组上删除，并让待定端口顶替它的 key，保证「下一条连线仍
        落在待定端点上」且端口编号不会随连接增删不断增长。
        """

        node = self.nodes.get(group_id)
        group = node.group if node is not None and node.is_group else None
        if group is None:
            return False
        port = group.port(key, is_input)
        if port is None or port.pending:
            return False
        if not self.group_port_free(group_id, key, is_input):
            return False
        group.remove_port(key, is_input)
        free_pending = next(
            (
                item
                for item in group.ports(is_input)
                if item.pending and self.group_port_free(group_id, item.key, is_input)
            ),
            None,
        )
        if free_pending is not None:
            # 空闲待定端口顶替被回收的 key，下一条连线自然会接在这里。
            free_pending.key = key
        else:
            group.ports(is_input).append(
                GroupPort(key, f"{'输入' if is_input else '输出'} {len(group.ports(is_input)) + 1}", None)
            )
        group.ensure_pending(is_input)
        group.relabel_auto_ports()
        node.touch()
        return True

    def group_state_snapshot(self) -> dict[str, dict[str, Any]]:
        """所有算子组的端口快照，供连线命令撤销时恢复自动增长的端口。"""

        return {
            node_id: node.group.to_dict()
            for node_id, node in self.nodes.items()
            if node.is_group and node.group is not None
        }

    def restore_group_state(self, state: dict[str, dict[str, Any]] | None) -> None:
        for node_id, data in (state or {}).items():
            node = self.nodes.get(node_id)
            if node is None or not node.is_group:
                continue
            node.group = GroupModel.from_dict(data)
            node.touch()
            self.node_changed.emit(node_id)

    def remove_connection(self, connection_id: str) -> ConnectionModel:
        try:
            connection = self.connections.pop(connection_id)
        except KeyError as exc:
            raise GraphError(f"连线不存在：{connection_id}") from exc
        reclaimed = [
            group_id
            for group_id, key, is_input in self._group_boundary_ports(connection)
            if self._reclaim_group_port(group_id, key, is_input)
        ]
        self.connection_removed.emit(connection)
        self.structure_changed.emit()
        for group_id in dict.fromkeys(reclaimed):
            self.node_changed.emit(group_id)
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
        """用于回环检测的后继：忽略「算子组输入 -> 组内节点」边界边。

        组输入来自组外上游，不是算子组节点自己产生的数据；把它算作算子组的
        出边会在「组输入 -> 组内节点 -> 组输出」时误判出环。
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
                == "group_input_inner"
            ):
                continue
            if connection.dst_node not in seen:
                seen.append(connection.dst_node)
        return seen

    def _reachable(self, start: str, target: str) -> bool:
        """从 start 出发（沿算子组感知后的连接方向）能否到达 target。"""

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
        # 修复悬空的 parent_group，避免旧工程 / 手工编辑数据让算子组失效。
        for node in graph.nodes.values():
            parent = node.parent_group
            if parent is not None:
                parent_node = graph.nodes.get(parent)
                if parent_node is None or not parent_node.is_group:
                    node.parent_group = None
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
        self._display_mode = DEFAULT_IMAGE_DISPLAY_MODE

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        value = value.strip() or "未命名工作区"
        if value != self._name:
            self._name = value
            self.name_changed.emit(value)

    @property
    def display_mode(self) -> str:
        """预览显示模式；每个工作区独立保存，切换节点不会改变。"""

        return self._display_mode

    @display_mode.setter
    def display_mode(self, mode: str) -> None:
        normalized = str(mode or "").strip().lower()
        if normalized not in IMAGE_DISPLAY_MODES:
            normalized = DEFAULT_IMAGE_DISPLAY_MODE
        self._display_mode = normalized

    def add_source_image(
        self,
        image: ImageData,
        path: str | None = None,
        pos: tuple[float, float] = (0.0, 0.0),
        title: str | None = None,
    ) -> NodeModel:
        return self.add_source_images([image], [path], pos=pos, title=title)

    def add_source_images(
        self,
        images: Iterable[ImageData],
        paths: Iterable[str | None] | None = None,
        pos: tuple[float, float] = (0.0, 0.0),
        title: str | None = None,
    ) -> NodeModel:
        """创建一个引用多张图像的源节点（图像矩阵）。

        每个工作区图像仍是独立条目，节点用 source_image_ids 记录顺序；这样单图
        工程格式无需改变，多图工程只是多存几个 npz。
        """

        image_list = list(images)
        if not image_list:
            raise ValueError("至少需要一张源图像")
        path_list = list(paths) if paths is not None else []
        if len(path_list) < len(image_list):
            path_list.extend([None] * (len(image_list) - len(path_list)))
        first_path = path_list[0]
        display_name = title or (
            Path(first_path).name if first_path else image_list[0].name
        ) or "原始图像"
        image_ids: list[str] = []
        for index, image in enumerate(image_list):
            path = path_list[index]
            image_id = new_id("img")
            name = Path(path).name if path else image.name
            image.name = name or f"图像 {index + 1}"
            self.images[image_id] = image
            self.image_paths[image_id] = path
            image_ids.append(image_id)
        node = self.graph.create_source_node(
            image_ids[0], display_name, first_path, pos, image_ids=image_ids
        )
        for image_id in image_ids:
            self.image_added.emit(image_id)
        return node

    def store_image(self, image: ImageData, path: str | None = None) -> str:
        """把一张图像登记到工作区并返回新的 image id。

        拖动图像副本到画布 / 合并到其它源节点时使用；登记后的图像即使暂时没有
        节点引用也会保留在内存中，保存工程时由 prune_unused_images 统一清理。
        """

        image_id = new_id("img")
        name = Path(path).name if path else image.name
        image.name = name or image.name or "图像"
        self.images[image_id] = image
        self.image_paths[image_id] = path
        self.image_added.emit(image_id)
        return image_id

    def set_source_image_ids(
        self,
        node_id: str,
        image_ids: Iterable[str],
        *,
        touch: bool = True,
    ) -> None:
        """替换源节点引用的图像顺序；用于合并图像 / 撤销恢复。"""

        node = self.graph.nodes.get(node_id)
        if node is None:
            raise GraphError(f"节点不存在：{node_id}")
        if not node.is_source:
            raise GraphError(f"节点不是原始图像节点：{node_id}")
        ids: list[str] = []
        for raw in image_ids:
            image_id = str(raw)
            if image_id and image_id not in ids:
                ids.append(image_id)
        for image_id in ids:
            if image_id not in self.images:
                raise GraphError(f"图像不存在：{image_id}")
        node.source_image_ids = ids
        node.source_image_id = ids[0] if ids else None
        if touch:
            node.touch()
            self.graph.node_changed.emit(node_id)

    def append_source_image_ids(
        self,
        node_id: str,
        image_ids: Iterable[str],
        *,
        touch: bool = True,
    ) -> None:
        """把若干已有图像追加到源节点的图像矩阵末尾。"""

        node = self.graph.nodes.get(node_id)
        if node is None:
            raise GraphError(f"节点不存在：{node_id}")
        combined = list(node.source_ids)
        for raw in image_ids:
            image_id = str(raw)
            if image_id and image_id not in combined:
                combined.append(image_id)
        self.set_source_image_ids(node_id, combined, touch=touch)

    def images_for_node(self, node: NodeModel) -> list[ImageData]:
        """按源节点声明的顺序返回其全部源图像。"""

        result: list[ImageData] = []
        for image_id in node.source_ids:
            image = self.images.get(image_id)
            if image is not None:
                result.append(image)
        return result

    def image_for_node(self, node: NodeModel) -> ImageData | None:
        """源节点的第一张图像（兼容单图调用方）。"""

        images = self.images_for_node(node)
        return images[0] if images else None

    def prune_unused_images(self) -> None:
        used = {
            image_id
            for node in self.graph.nodes.values()
            for image_id in node.source_ids
        }
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
            "display_mode": self.display_mode,
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
