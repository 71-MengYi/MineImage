"""算子组（GroupOperator）：把算子组保存成可复用的自定义算子。

生成的算子文件只保存内部节点的 operator id / 参数 / 连线（``group_definition``），
不复制其它算子的实现代码；从算子库拖入画布时由 :func:`materialize_group_operator`
展开成可继续编辑的算子组。

依赖算子缺失时算子文件仍然可以正常加载（它只有数据，不 import 依赖），展开后的
算子组里对应节点会以"失效"状态显示，用户可以右键替换。
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ClassVar

from app.config import default_user_operator_dir
from app.core.constants import CUSTOM_OPERATOR_CATEGORY, GROUP_OP_ID
from app.core.group import (
    GROUP_BOUNDARY_ID,
    GroupDefinitionError,
    GroupError,
    GroupModel,
    GroupPort,
    normalize_group_definition,
)
from app.core.model import ConnectionModel, NodeModel, new_id
from app.operators.spec import Operator, OperatorError, OperatorSpec, PortSpec


class GroupOperator(Operator):
    """算子组自定义算子的基类；``group_definition`` 保存可展开的内部图。"""

    group_definition: ClassVar[dict[str, Any]] = {}

    def process(self, **kwargs: Any):  # pragma: no cover - 算子组由引擎直接执行内部图
        raise OperatorError(
            "由算子组保存的算子不能作为普通算子直接执行；请把它从算子库拖入画布，内部节点会展开成算子组。"
        )


@dataclass
class GroupMaterialization:
    """一次算子组展开的结果（尚未写入图模型）。"""

    group_node: NodeModel
    nodes: list[NodeModel]
    connections: list[ConnectionModel]

    @property
    def group_id(self) -> str:
        return self.group_node.id


@dataclass
class SavedGroupOperator:
    """保存为新算子的结果。"""

    path: Path
    op_id: str
    name: str
    definition: dict[str, Any]


# ---------------------------------------------------------------------------
# 识别 / 查询
# ---------------------------------------------------------------------------
def _operator_of(value: Any) -> Any:
    return getattr(value, "operator", value)


def is_group_operator(value: Any) -> bool:
    """判断算子实例 / 类 / 注册表条目是否是算子组。"""

    obj = _operator_of(value)
    if isinstance(obj, GroupOperator):
        return True
    definition = getattr(obj, "group_definition", None)
    return isinstance(definition, dict) and bool(definition)


def group_definition_of(value: Any) -> dict[str, Any] | None:
    obj = _operator_of(value)
    definition = getattr(obj, "group_definition", None)
    return definition if isinstance(definition, dict) and definition else None


def _operator_for(registry: Any, op_id: str) -> Any | None:
    if registry is None or not op_id:
        return None
    entry = None
    getter = getattr(registry, "entry", None)
    if callable(getter):
        try:
            entry = getter(op_id)
        except Exception:  # noqa: BLE001 - 兼容最小化测试注册表
            entry = None
    if entry is not None:
        return _operator_of(entry)
    getter = getattr(registry, "get", None)
    if callable(getter):
        try:
            return getter(op_id)
        except Exception:  # noqa: BLE001
            return None
    return None


def _registration_exists(registry: Any, op_id: str, directory: Path) -> bool:
    if registry is not None:
        contains = getattr(registry, "__contains__", None)
        if callable(contains):
            try:
                if op_id in registry:
                    return True
            except Exception:  # noqa: BLE001
                pass
        if _operator_for(registry, op_id) is not None:
            return True
    filename = f"{op_id.replace('.', '_')}.py"
    return (directory / filename).exists()


# ---------------------------------------------------------------------------
# 展开自定义算子 -> 算子组
# ---------------------------------------------------------------------------
def materialize_group_operator(
    registry: Any,
    op_id: str,
    *,
    pos: tuple[float, float] = (0.0, 0.0),
    parent_group: str | None = None,
    title: str | None = None,
) -> GroupMaterialization:
    """把注册表中的算子组展开成可以放进画布的节点集合。"""

    operator = _operator_for(registry, op_id)
    if operator is None:
        raise GroupError(f"找不到算子组：{op_id}")
    definition = group_definition_of(operator)
    if definition is None:
        raise GroupError(f"算子不是算子组：{op_id}")
    try:
        normalized = normalize_group_definition(definition)
    except GroupDefinitionError as exc:
        raise GroupError(f"算子组定义无效：{exc}") from exc
    spec = getattr(operator, "spec", None)
    display_name = (
        str(title or "").strip()
        or str(getattr(spec, "name", "") or "").strip()
        or str(normalized.get("name") or "").strip()
        or op_id
    )
    nodes: list[NodeModel] = []
    connections: list[ConnectionModel] = []
    root = _materialize_definition(
        normalized,
        origin=(float(pos[0]), float(pos[1])),
        parent_group=parent_group,
        operator_id=op_id,
        title=display_name,
        registry=registry,
        nodes=nodes,
        connections=connections,
    )
    return GroupMaterialization(group_node=root, nodes=nodes, connections=connections)


def _materialize_definition(
    definition: dict[str, Any],
    *,
    origin: tuple[float, float],
    parent_group: str | None,
    operator_id: str | None,
    title: str,
    registry: Any,
    nodes: list[NodeModel],
    connections: list[ConnectionModel],
) -> NodeModel:
    group = GroupModel(
        inputs=[GroupPort.from_dict(raw) for raw in definition.get("inputs") or []],
        outputs=[GroupPort.from_dict(raw) for raw in definition.get("outputs") or []],
        operator_id=operator_id,
    )
    group.ensure_pending_ports()
    root = NodeModel(
        id=new_id("n"),
        op_id=GROUP_OP_ID,
        title=title,
        pos=(float(origin[0]), float(origin[1])),
        group=group,
        parent_group=parent_group,
    )
    nodes.append(root)
    id_map: dict[str, str] = {GROUP_BOUNDARY_ID: root.id}
    deferred: list[tuple[dict[str, Any], NodeModel]] = []

    for raw in definition.get("nodes") or []:
        if not isinstance(raw, dict):
            continue
        old_id = str(raw.get("id") or "")
        if not old_id or old_id in id_map:
            continue
        rel = raw.get("pos") or [0.0, 0.0]
        try:
            node_pos = (origin[0] + float(rel[0]), origin[1] + float(rel[1]))
        except (TypeError, ValueError, IndexError):
            node_pos = origin
        group_ref = raw.get("group_ref")
        if isinstance(group_ref, dict) and group_ref.get("operator_id"):
            ref_id = str(group_ref["operator_id"])
            dependency = _operator_for(registry, ref_id)
            dependency_definition = group_definition_of(dependency)
            if dependency_definition is not None:
                try:
                    nested_definition = normalize_group_definition(dependency_definition)
                except GroupDefinitionError:
                    nested_definition = None
                if nested_definition is not None:
                    nested_title = (
                        str(raw.get("title") or "").strip()
                        or str(group_ref.get("name") or "").strip()
                        or ref_id
                    )
                    nested = _materialize_definition(
                        nested_definition,
                        origin=node_pos,
                        parent_group=root.id,
                        operator_id=ref_id,
                        title=nested_title,
                        registry=registry,
                        nodes=nodes,
                        connections=connections,
                    )
                    id_map[old_id] = nested.id
                    continue
            # 依赖缺失：生成带同样端口的占位算子组，标记失效但保持可编辑 / 可替换。
            try:
                placeholder = GroupModel(
                    inputs=[
                        GroupPort.from_dict(item)
                        for item in (group_ref.get("inputs") or [])
                        if isinstance(item, dict)
                    ],
                    outputs=[
                        GroupPort.from_dict(item)
                        for item in (group_ref.get("outputs") or [])
                        if isinstance(item, dict)
                    ],
                    operator_id=ref_id,
                )
            except GroupError:
                placeholder = GroupModel.create_default(operator_id=ref_id)
            placeholder.ensure_pending_ports()
            node = NodeModel(
                id=new_id("n"),
                op_id=GROUP_OP_ID,
                title=str(raw.get("title") or group_ref.get("name") or ref_id),
                pos=node_pos,
                group=placeholder,
                parent_group=root.id,
            )
        else:
            op_id = str(raw.get("op_id") or "")
            if not op_id:
                continue
            node = NodeModel(
                id=new_id("n"),
                op_id=op_id,
                title=str(raw.get("title") or op_id),
                pos=node_pos,
                params=dict(raw.get("params") or {}),
                parent_group=root.id,
            )
            if op_id == GROUP_OP_ID:
                node.group = GroupModel.from_dict(raw.get("group"))
        nodes.append(node)
        id_map[old_id] = node.id
        if raw.get("parent_group"):
            deferred.append((raw, node))

    for raw, node in deferred:
        parent_old = str(raw.get("parent_group") or "")
        parent_new = id_map.get(parent_old)
        if parent_new:
            node.parent_group = parent_new

    for raw in definition.get("connections") or []:
        if not isinstance(raw, dict):
            continue
        src_node = id_map.get(str(raw.get("src_node") or ""))
        dst_node = id_map.get(str(raw.get("dst_node") or ""))
        src_port = str(raw.get("src_port") or "")
        dst_port = str(raw.get("dst_port") or "")
        if not src_node or not dst_node or not src_port or not dst_port:
            continue
        connections.append(
            ConnectionModel(
                id=new_id("c"),
                src_node=src_node,
                src_port=src_port,
                dst_node=dst_node,
                dst_port=dst_port,
            )
        )
    return root


# ---------------------------------------------------------------------------
# 保存为新算子
# ---------------------------------------------------------------------------
def _slugify(name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_]+", "_", str(name or "")).strip("_").lower()
    return text or "group"


def _class_name(slug: str) -> str:
    parts = [part for part in slug.split("_") if part]
    name = "".join(part[:1].upper() + part[1:] for part in parts) or "Group"
    if name[0].isdigit():
        name = "Group" + name
    return name


def save_group_operator(
    graph: Any,
    group_node_id: str,
    name: str,
    *,
    registry: Any = None,
    directory: str | Path | None = None,
) -> SavedGroupOperator:
    """把算子组保存为 ``用户自定义算子``；返回写入的文件与定义。"""

    display_name = str(name or "").strip()
    if not display_name:
        raise GroupError("算子名称不能为空")
    if len(display_name) > 64:
        raise GroupError("算子名称不能超过 64 个字符")

    target_dir = Path(directory) if directory is not None else default_user_operator_dir() / "custom"
    target_dir.mkdir(parents=True, exist_ok=True)

    from app.core.group import build_group_definition

    definition = build_group_definition(graph, group_node_id, display_name)
    slug = _slugify(display_name)
    base_id = f"group.{slug}"
    op_id = base_id
    counter = 2
    while _registration_exists(registry, op_id, target_dir) or (
        target_dir / f"{op_id.replace('.', '_')}.py"
    ).exists():
        op_id = f"{base_id}_{counter}"
        counter += 1
    path = target_dir / f"{op_id.replace('.', '_')}.py"

    content = _render_operator_module(display_name, op_id, definition)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}_", suffix=".py.tmp", dir=str(target_dir), delete=False
    )
    temp_path = Path(handle.name)
    try:
        handle.write(content.encode("utf-8"))
        handle.close()
        os.replace(temp_path, path)
    except Exception:
        try:
            handle.close()
        except Exception:  # noqa: BLE001
            pass
        temp_path.unlink(missing_ok=True)
        raise
    return SavedGroupOperator(path=path, op_id=op_id, name=display_name, definition=definition)


def _port_repr(port: dict[str, Any]) -> str:
    key = repr(str(port.get("key") or ""))
    label = repr(str(port.get("label") or ""))
    data_type = port.get("data_type")
    if not data_type:
        return f"PortSpec({key}, {label}, any_type=True)"
    return f"PortSpec({key}, {label}, data_type=DataType.{str(data_type).upper()})"


def _render_operator_module(name: str, op_id: str, definition: dict[str, Any]) -> str:
    inputs = [port for port in definition.get("inputs") or [] if isinstance(port, dict)]
    outputs = [port for port in definition.get("outputs") or [] if isinstance(port, dict)]
    typed = any(port.get("data_type") for port in inputs + outputs)
    json_text = json.dumps(definition, ensure_ascii=False, separators=(",", ":"))
    class_name = _class_name(_slugify(name))
    node_count = len(definition.get("nodes") or [])
    description = f"由算子组“{name}”保存的自定义算子，内部包含 {node_count} 个节点。"

    def tuple_block(ports: list[dict[str, Any]]) -> str:
        if not ports:
            return "()"
        lines = "\n".join(f"            {_port_repr(port)}," for port in ports)
        return f"(\n{lines}\n        )"

    imports = [
        "import json",
        "",
        "from app.operators.group import GroupOperator",
        "from app.operators.spec import OperatorSpec, PortSpec",
    ]
    if typed:
        imports.insert(2, "from app.core.data_types import DataType")
    header = "\n".join(imports)
    return f'''"""{description}

本文件由 MineImage「算子组 → 保存为新算子」自动生成。
内部节点以 operator id 的形式引用，不重复其它算子的实现代码；依赖的算子缺失时
本文件仍可正常加载，展开后的算子组内对应节点会显示为失效，可右键替换。
"""

{header}

GROUP_DEFINITION = json.loads({json_text!r})


class {class_name}(GroupOperator):
    spec = OperatorSpec(
        id={op_id!r},
        name={name!r},
        category=({CUSTOM_OPERATOR_CATEGORY!r},),
        description={description!r},
        version="1.0.0",
        inputs={tuple_block(inputs)},
        outputs={tuple_block(outputs)},
    )
    group_definition = GROUP_DEFINITION


OPERATOR = {class_name}
'''


__all__ = [
    "GroupMaterialization",
    "GroupOperator",
    "SavedGroupOperator",
    "group_definition_of",
    "is_group_operator",
    "materialize_group_operator",
    "save_group_operator",
]
