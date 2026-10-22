"""算子组节点（Group Node）的端口与定义模型。

算子组是一个可嵌套的容器节点：自身带动态数量的输入 / 输出端口，内部节点可以连接
组的输入端口（作为内部数据源）和输出端口（作为内部数据终点）。每个算子组始终保留一个
"待定"（未知类型）端口；当某条连线确定了端口类型后，自动追加一个新的待定端口。

本模块只依赖 app.core.data_types 与 app.operators.spec，不导入 GraphModel，
避免循环导入；需要图模型的函数在调用时按需使用鸭子类型接口。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.core.constants import GROUP_OP_ID
from app.core.data_types import DataType, coerce_data_type
from app.operators.spec import OperatorSpec, PortSpec

GROUP_DEFINITION_FORMAT = "mineimage-group-definition"
GROUP_DEFINITION_VERSION = 1
#: 算子组在自身定义文件中的哨兵 id（用于表示组的输入 / 输出边界）。
GROUP_BOUNDARY_ID = "__group__"
#: 无效端口的层级返回值（GraphModel.port_level）。
INVALID_LEVEL = "__invalid__"


class GroupError(ValueError):
    """算子组模型 / 算子生成错误。"""


class GroupDefinitionError(GroupError):
    """算子组定义无法生成或解析。"""


def _port_key(index: int, is_input: bool) -> str:
    return f"{'input' if is_input else 'output'}_{index}"


def _port_label(index: int, is_input: bool) -> str:
    return f"{'输入' if is_input else '输出'} {index}"


def _jsonable(value: Any) -> Any:
    """把参数值转成可写入 JSON / Python 字面量的结构。"""

    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, bool) or value is None or isinstance(value, (int, float, str)):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    try:
        import numpy as np

        if isinstance(value, np.integer):
            return int(value)
        if isinstance(value, np.floating):
            return float(value)
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:  # noqa: BLE001 - 可选项，不因 numpy 异常影响保存
        pass
    # 其它对象（例如 Path）按字符串保存，保证生成文件一定可加载。
    return str(value)


@dataclass
class GroupPort:
    """算子组边界上的一个数据端口。

    data_type 为 None 表示"待定"：类型由连到该端口的第一条数据连线决定，
    在确定之前兼容所有数据类型。
    """

    key: str
    label: str
    data_type: DataType | None = None

    def __post_init__(self) -> None:
        key = str(self.key or "").strip()
        if not key:
            raise GroupError("算子组端口 key 不能为空")
        object.__setattr__(self, "key", key)
        object.__setattr__(self, "label", str(self.label or key).strip() or key)
        data_type = self.data_type
        if data_type is not None and not isinstance(data_type, DataType):
            data_type = coerce_data_type(data_type)
        object.__setattr__(self, "data_type", data_type)

    @property
    def pending(self) -> bool:
        """是否为待定（未知）类型端口。"""

        return self.data_type is None

    @property
    def type_text(self) -> str:
        return "任意类型（待定）" if self.pending else self.data_type.label

    def port_spec(self) -> PortSpec:
        if self.data_type is None:
            return PortSpec(
                self.key,
                self.label,
                any_type=True,
                description="待定端口：连接后自动确定类型",
            )
        return PortSpec(self.key, self.label, data_type=self.data_type)

    def copy(self) -> "GroupPort":
        return GroupPort(self.key, self.label, self.data_type)

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "data_type": self.data_type.value if self.data_type is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "GroupPort":
        if not isinstance(data, dict):
            raise GroupError("算子组端口数据无效")
        raw_type = data.get("data_type")
        if isinstance(raw_type, DataType):
            data_type: DataType | None = raw_type
        elif raw_type in (None, ""):
            data_type = None
        else:
            try:
                data_type = coerce_data_type(str(raw_type))
            except ValueError:
                data_type = None
        return cls(str(data.get("key") or ""), str(data.get("label") or ""), data_type)


@dataclass
class GroupModel:
    """算子组节点自有的端口与来源信息。"""

    inputs: list[GroupPort] = field(default_factory=list)
    outputs: list[GroupPort] = field(default_factory=list)
    operator_id: str | None = None

    def __post_init__(self) -> None:
        self.inputs = list(self.inputs)
        self.outputs = list(self.outputs)
        for ports in (self.inputs, self.outputs):
            keys = [port.key for port in ports]
            if len(keys) != len(set(keys)):
                raise GroupError("算子组端口 key 重复")
            for port in ports:
                if not isinstance(port, GroupPort):
                    raise GroupError("算子组端口必须是 GroupPort")
        if self.operator_id is not None:
            self.operator_id = str(self.operator_id).strip() or None

    # ------------------------------------------------------------------ 构造
    @classmethod
    def create_default(cls, *, operator_id: str | None = None) -> "GroupModel":
        """默认各一个待定端口。"""

        return cls(
            inputs=[GroupPort(_port_key(1, True), _port_label(1, True))],
            outputs=[GroupPort(_port_key(1, False), _port_label(1, False))],
            operator_id=operator_id,
        )

    def copy(self) -> "GroupModel":
        return GroupModel(
            inputs=[port.copy() for port in self.inputs],
            outputs=[port.copy() for port in self.outputs],
            operator_id=self.operator_id,
        )

    # ------------------------------------------------------------------ 查询
    def ports(self, is_input: bool) -> list[GroupPort]:
        return self.inputs if is_input else self.outputs

    def input_keys(self) -> set[str]:
        return {port.key for port in self.inputs}

    def output_keys(self) -> set[str]:
        return {port.key for port in self.outputs}

    def port(self, key: str, is_input: bool) -> GroupPort | None:
        for port in self.ports(is_input):
            if port.key == key:
                return port
        return None

    def has_pending(self, is_input: bool) -> bool:
        return any(port.pending for port in self.ports(is_input))

    def next_index(self, is_input: bool) -> int:
        keys = {port.key for port in self.ports(is_input)}
        index = len(keys) + 1
        while _port_key(index, is_input) in keys:
            index += 1
        return index

    # ------------------------------------------------------------ 端口增长
    def ensure_pending(self, is_input: bool) -> GroupPort:
        """确保该侧至少有一个待定端口（已有则直接返回）。"""

        for port in self.ports(is_input):
            if port.pending:
                return port
        index = self.next_index(is_input)
        port = GroupPort(_port_key(index, is_input), _port_label(index, is_input))
        self.ports(is_input).append(port)
        return port

    def ensure_pending_ports(self) -> None:
        self.ensure_pending(True)
        self.ensure_pending(False)

    def bind(self, key: str, is_input: bool, data_type: DataType | str | None) -> bool:
        """把待定端口绑定为具体类型；绑定成功时自动追加新的待定端口。

        返回是否发生了变化（端口不存在、已是具体类型或类型未知时返回 False）。
        """

        if data_type is None:
            return False
        try:
            dtype = data_type if isinstance(data_type, DataType) else coerce_data_type(data_type)
        except ValueError:
            return False
        port = self.port(key, is_input)
        if port is None or not port.pending:
            return False
        port.data_type = dtype
        self.ensure_pending(is_input)
        return True

    def bind_from_spec(self, key: str, is_input: bool, spec: PortSpec | None) -> bool:
        if spec is None or spec.any_type or spec.data_type is None:
            return False
        return self.bind(key, is_input, spec.data_type)

    # ------------------------------------------------------------------ 规范
    def port_spec(self, key: str, is_input: bool) -> PortSpec | None:
        port = self.port(key, is_input)
        return port.port_spec() if port is not None else None

    def spec(self, title: str) -> OperatorSpec:
        """生成用于节点渲染 / 连线校验的动态算子声明。"""

        self.ensure_pending_ports()
        return OperatorSpec(
            id=GROUP_OP_ID,
            name=str(title or "算子组"),
            category=("算子组",),
            description="算子组：把多个算子节点组合成一个带输入输出的整体。",
            inputs=tuple(port.port_spec() for port in self.inputs),
            outputs=tuple(port.port_spec() for port in self.outputs),
        )

    # -------------------------------------------------------------- 序列化
    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "inputs": [port.to_dict() for port in self.inputs],
            "outputs": [port.to_dict() for port in self.outputs],
        }
        if self.operator_id:
            data["operator_id"] = self.operator_id
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "GroupModel":
        if not isinstance(data, dict):
            return cls.create_default()
        inputs = [
            GroupPort.from_dict(raw)
            for raw in (data.get("inputs") or [])
            if isinstance(raw, dict)
        ]
        outputs = [
            GroupPort.from_dict(raw)
            for raw in (data.get("outputs") or [])
            if isinstance(raw, dict)
        ]
        operator_id = data.get("operator_id")
        group = cls(
            inputs=inputs,
            outputs=outputs,
            operator_id=str(operator_id) if operator_id else None,
        )
        group.ensure_pending_ports()
        return group


# ---------------------------------------------------------------------------
# 算子组定义（保存为自定义算子 / 复制粘贴共用）
# ---------------------------------------------------------------------------
def _included_nodes(graph: Any, group_node_id: str) -> list[Any]:
    """返回属于该算子组的内部节点（排除被引用自定义算子内部的节点）。"""

    included: list[Any] = []
    for node in graph.nodes.values():
        if node.id == group_node_id:
            continue
        current = node.parent_group
        inside = False
        referenced = False
        seen: set[str] = set()
        while current and current not in seen:
            seen.add(current)
            if current == group_node_id:
                inside = True
                break
            parent = graph.nodes.get(current)
            if parent is None:
                break
            if parent.is_group and parent.group is not None and parent.group.operator_id:
                referenced = True
                break
            current = parent.parent_group
        if inside and not referenced:
            included.append(node)
    return included


def build_group_definition(graph: Any, group_node_id: str, name: str) -> dict[str, Any]:
    """把一个算子组及其内部节点导出为可保存/复用的定义字典。

    * 节点位置转换为相对算子组左上角的坐标；
    * 边界连线使用哨兵 "__group__" 表示算子组自身；
    * 若内部包含"引用自定义算子"的算子组，只保存引用（不重复内部实现）；
    * 算子组内包含原始图像节点时抛出 GroupDefinitionError。
    """

    root = graph.nodes.get(group_node_id)
    if root is None or not getattr(root, "is_group", False) or root.group is None:
        raise GroupDefinitionError("指定的节点不是算子组")
    name = str(name or "").strip() or root.display_title()
    root_x, root_y = float(root.pos[0]), float(root.pos[1])
    included = _included_nodes(graph, group_node_id)
    for node in included:
        if getattr(node, "is_source", False):
            raise GroupDefinitionError(
                "算子组内包含原始图像节点，无法保存为算子；请先把源图像节点移出算子组。"
            )

    included_ids = {node.id for node in included}
    nodes_payload: list[dict[str, Any]] = []
    for node in included:
        parent = node.parent_group
        parent_ref = None if parent == group_node_id else (parent if parent in included_ids else None)
        entry: dict[str, Any] = {
            "id": node.id,
            "op_id": node.op_id,
            "title": node.display_title(),
            "pos": [
                round(float(node.pos[0]) - root_x, 3),
                round(float(node.pos[1]) - root_y, 3),
            ],
            "params": _jsonable(dict(node.params or {})),
            "parent_group": parent_ref,
        }
        if node.is_group and node.group is not None:
            if node.group.operator_id:
                # 引用其它自定义算子：只保存引用与端口，避免重复代码。
                entry["op_id"] = node.group.operator_id
                entry["group_ref"] = {
                    "operator_id": node.group.operator_id,
                    "name": node.display_title(),
                    "inputs": [port.to_dict() for port in node.group.inputs],
                    "outputs": [port.to_dict() for port in node.group.outputs],
                }
            else:
                entry["group"] = node.group.to_dict()
        nodes_payload.append(entry)

    allowed_ids = included_ids | {group_node_id}
    connections_payload: list[dict[str, Any]] = []

    def map_node(node_id: str) -> str:
        return GROUP_BOUNDARY_ID if node_id == group_node_id else node_id

    for connection in graph.connections.values():
        if connection.src_node not in allowed_ids or connection.dst_node not in allowed_ids:
            continue
        connections_payload.append(
            {
                "src_node": map_node(connection.src_node),
                "src_port": connection.src_port,
                "dst_node": map_node(connection.dst_node),
                "dst_port": connection.dst_port,
            }
        )

    return {
        "format": GROUP_DEFINITION_FORMAT,
        "version": GROUP_DEFINITION_VERSION,
        "name": name,
        "inputs": [port.to_dict() for port in root.group.inputs],
        "outputs": [port.to_dict() for port in root.group.outputs],
        "nodes": nodes_payload,
        "connections": connections_payload,
    }


def normalize_group_definition(data: Any) -> dict[str, Any]:
    """校验并规范化一个算子组定义（用于生成算子文件 / 反序列化）。"""

    if not isinstance(data, dict):
        raise GroupDefinitionError("算子组定义必须是字典")
    if data.get("format") not in (None, GROUP_DEFINITION_FORMAT):
        raise GroupDefinitionError("无法识别的算子组定义格式")
    version = int(data.get("version") or GROUP_DEFINITION_VERSION)
    if version > GROUP_DEFINITION_VERSION:
        raise GroupDefinitionError("算子组定义由更高版本创建，当前版本无法加载")
    try:
        inputs = [
            GroupPort.from_dict(raw)
            for raw in (data.get("inputs") or [])
            if isinstance(raw, dict)
        ]
        outputs = [
            GroupPort.from_dict(raw)
            for raw in (data.get("outputs") or [])
            if isinstance(raw, dict)
        ]
    except GroupError as exc:
        raise GroupDefinitionError(f"算子组端口数据无效：{exc}") from exc
    if not outputs:
        raise GroupDefinitionError("算子组至少需要一个输出端口")
    nodes: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw in data.get("nodes") or []:
        if not isinstance(raw, dict):
            continue
        node_id = str(raw.get("id") or "").strip()
        if not node_id or node_id == GROUP_BOUNDARY_ID or node_id in seen:
            continue
        op_id = str(raw.get("op_id") or "").strip()
        if not op_id:
            continue
        seen.add(node_id)
        pos = raw.get("pos") or [0.0, 0.0]
        try:
            pos_x, pos_y = float(pos[0]), float(pos[1])
        except (TypeError, ValueError, IndexError):
            pos_x, pos_y = 0.0, 0.0
        entry: dict[str, Any] = {
            "id": node_id,
            "op_id": op_id,
            "title": str(raw.get("title") or op_id),
            "pos": [pos_x, pos_y],
            "params": _jsonable(raw.get("params") or {}),
            "parent_group": raw.get("parent_group"),
        }
        group_raw = raw.get("group")
        if op_id == GROUP_OP_ID and isinstance(group_raw, dict):
            entry["group"] = GroupModel.from_dict(group_raw).to_dict()
        group_ref = raw.get("group_ref")
        if isinstance(group_ref, dict) and group_ref.get("operator_id"):
            entry["group_ref"] = {
                "operator_id": str(group_ref.get("operator_id")),
                "name": str(group_ref.get("name") or entry["title"]),
                "inputs": [
                    GroupPort.from_dict(item).to_dict()
                    for item in (group_ref.get("inputs") or [])
                    if isinstance(item, dict)
                ],
                "outputs": [
                    GroupPort.from_dict(item).to_dict()
                    for item in (group_ref.get("outputs") or [])
                    if isinstance(item, dict)
                ],
            }
        nodes.append(entry)

    known = {entry["id"] for entry in nodes}
    connections: list[dict[str, Any]] = []
    for raw in data.get("connections") or []:
        if not isinstance(raw, dict):
            continue
        src_node = str(raw.get("src_node") or "")
        dst_node = str(raw.get("dst_node") or "")
        src_port = str(raw.get("src_port") or "")
        dst_port = str(raw.get("dst_port") or "")
        if not src_port or not dst_port:
            continue
        if src_node not in known and src_node != GROUP_BOUNDARY_ID:
            continue
        if dst_node not in known and dst_node != GROUP_BOUNDARY_ID:
            continue
        connections.append(
            {
                "src_node": src_node,
                "src_port": src_port,
                "dst_node": dst_node,
                "dst_port": dst_port,
            }
        )
    return {
        "format": GROUP_DEFINITION_FORMAT,
        "version": version,
        "name": str(data.get("name") or ""),
        "inputs": [port.to_dict() for port in inputs],
        "outputs": [port.to_dict() for port in outputs],
        "nodes": nodes,
        "connections": connections,
    }


__all__ = [
    "GROUP_BOUNDARY_ID",
    "GROUP_DEFINITION_FORMAT",
    "GROUP_DEFINITION_VERSION",
    "GroupDefinitionError",
    "GroupError",
    "GroupModel",
    "GroupPort",
    "INVALID_LEVEL",
    "build_group_definition",
    "normalize_group_definition",
]
