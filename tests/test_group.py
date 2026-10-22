"""算子组模型 / 自定义算子保存与展开的单元测试。

运行方式（仓库根目录）::

    $env:PYTHONPATH = "."
    python -m unittest tests.test_group -v
"""

from __future__ import annotations

import os
import shutil
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from app.core.constants import CATEGORY_ORDER, CUSTOM_OPERATOR_CATEGORY, GROUP_OP_ID
from app.core.data_types import DataType
from app.core.group import (
    GROUP_BOUNDARY_ID,
    GroupDefinitionError,
    GroupModel,
    GroupPort,
    build_group_definition,
    normalize_group_definition,
)
from app.core.model import GraphModel, GraphError, NodeModel, WorkspaceModel, new_id
from app.core.project import load_project, save_project
from app.operators.group import (
    GroupOperator,
    is_group_operator,
    materialize_group_operator,
    save_group_operator,
)
from app.operators.loader import OperatorRegistry
from app.operators.spec import OperatorSpec, PortSpec

_APP: QApplication | None = None
#: 沙箱环境下系统临时目录不可写，测试统一把临时文件放在工作区 .test_tmp 下。
_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".test_tmp" / "group-tests"


def _temp_dir(prefix: str = "case-") -> Path:
    path = _TEMP_ROOT / f"{prefix}{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _cleanup_dir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def setUpModule() -> None:
    global _APP
    # 与其它 GUI 测试共用同一个 QApplication：QCoreApplication 会让后续
    # QWidget 测试直接 abort。
    _APP = QApplication.instance() or QApplication([])


def tearDownModule() -> None:
    global _APP
    if _APP is not None:
        _APP.processEvents()
        _APP = None


def _node(graph: GraphModel, op_id: str, title: str, parent: str | None = None, pos=(0, 0)) -> NodeModel:
    return graph.add_node(
        NodeModel(id=new_id("n"), op_id=op_id, title=title, pos=pos, parent_group=parent)
    )


def _resolver(mapping: dict[str, DataType] | None = None):
    mapping = mapping or {}

    def resolve(node_id: str, port_key: str, is_input: bool) -> DataType | None:
        return mapping.get(node_id)

    return resolve


class GroupModelTests(unittest.TestCase):
    def test_default_ports_are_pending(self) -> None:
        group = GroupModel.create_default()
        self.assertEqual([port.key for port in group.inputs], ["input_1"])
        self.assertEqual([port.key for port in group.outputs], ["output_1"])
        self.assertTrue(group.inputs[0].pending)
        self.assertTrue(group.outputs[0].pending)
        spec = group.spec("测试算子组")
        self.assertEqual(spec.id, GROUP_OP_ID)
        self.assertTrue(spec.inputs[0].any_type)
        self.assertTrue(spec.outputs[0].any_type)
        self.assertEqual(spec.outputs[0].type_text(), "任意数据")

    def test_bind_typed_port_adds_pending_port(self) -> None:
        group = GroupModel.create_default()
        self.assertTrue(group.bind("input_1", True, DataType.IMAGE))
        self.assertEqual(group.inputs[0].data_type, DataType.IMAGE)
        self.assertEqual(len(group.inputs), 2)
        self.assertTrue(group.inputs[1].pending)
        # 已确定的端口不会被再次改写
        self.assertFalse(group.bind("input_1", True, DataType.NUMBER))
        self.assertEqual(group.inputs[0].data_type, DataType.IMAGE)
        # 未知类型不会绑定
        self.assertFalse(group.bind("input_2", True, None))
        self.assertTrue(group.inputs[1].pending)

    def test_bind_from_spec(self) -> None:
        group = GroupModel.create_default()
        self.assertFalse(group.bind_from_spec("input_1", True, PortSpec("x", "x", any_type=True)))
        self.assertTrue(
            group.bind_from_spec("output_1", False, PortSpec("y", "y", data_type=DataType.NUMBER))
        )
        self.assertEqual(group.outputs[0].data_type, DataType.NUMBER)
        self.assertEqual(len(group.outputs), 2)

    def test_serialization_roundtrip(self) -> None:
        group = GroupModel.create_default(operator_id="group.demo")
        group.bind("input_1", True, DataType.CONTOURS)
        data = group.to_dict()
        restored = GroupModel.from_dict(data)
        self.assertEqual(restored.operator_id, "group.demo")
        self.assertEqual(restored.inputs[0].data_type, DataType.CONTOURS)
        self.assertEqual(len(restored.inputs), 2)

    def test_invalid_port_rejected(self) -> None:
        with self.assertRaises(Exception):
            GroupPort("", "空 key")


class GroupGraphTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = GraphModel()

    def test_connection_binds_input_and_adds_pending(self) -> None:
        self.graph.set_port_type_resolver(lambda _node, _port, _is_input: DataType.IMAGE)
        group = self.graph.create_group_node("算子组", (0, 0))
        source = self.graph.add_node(NodeModel(id="n_src", op_id="fake.source", title="源"))
        inner = _node(self.graph, "fake.inner", "内部", parent=group.id)
        self.graph.add_connection(source.id, "image", group.id, "input_1")
        self.assertEqual(group.group.inputs[0].data_type, DataType.IMAGE)
        self.assertEqual(len(group.group.inputs), 2)
        self.assertTrue(group.group.inputs[1].pending)
        # 组输入 -> 组内节点
        self.graph.add_connection(group.id, "input_1", inner.id, "image")
        # 组内节点 -> 组输出
        self.graph.add_connection(inner.id, "image", group.id, "output_1")
        self.assertEqual(group.group.outputs[0].data_type, DataType.IMAGE)
        self.assertEqual(len(group.group.outputs), 2)
        self.assertEqual(group.group.outputs[1].key, "output_2")
        # 端口变化后 revision 增加
        self.assertGreater(group.revision, 0)

    def test_scope_rules(self) -> None:
        group = self.graph.create_group_node("算子组", (0, 0))
        inner = _node(self.graph, "fake.inner", "内部", parent=group.id)
        outer = _node(self.graph, "fake.outer", "外部")
        other = self.graph.create_group_node("另一个算子组", (400, 0))
        self.assertTrue(self.graph.connection_allowed(outer.id, "image", group.id, "input_1")[0])
        self.assertTrue(self.graph.connection_allowed(group.id, "input_1", inner.id, "image")[0])
        self.assertTrue(self.graph.connection_allowed(inner.id, "image", group.id, "output_1")[0])
        self.assertTrue(self.graph.connection_allowed(group.id, "output_1", outer.id, "image")[0])
        # 外部节点不能直接连到组输出
        self.assertFalse(self.graph.connection_allowed(outer.id, "image", group.id, "output_1")[0])
        # 组内节点不能直接使用组输入作为终点
        self.assertFalse(self.graph.connection_allowed(inner.id, "image", group.id, "input_1")[0])
        # 组输入不能直接连到组外节点
        self.assertFalse(self.graph.connection_allowed(group.id, "input_1", outer.id, "image")[0])
        # 跨算子组节点
        inner2 = _node(self.graph, "fake.inner2", "内部2", parent=other.id)
        self.assertFalse(self.graph.connection_allowed(inner.id, "image", inner2.id, "image")[0])
        # 自连
        self.assertFalse(self.graph.connection_allowed(group.id, "output_1", group.id, "input_1")[0])

    def test_resolve_source_through_nested_groups(self) -> None:
        graph = self.graph
        graph.set_port_type_resolver(lambda _n, _p, _i: DataType.IMAGE)
        outer = graph.create_group_node("外层", (0, 0))
        inner_group = graph.create_group_node("内层", (20, 60), parent_group=outer.id)
        leaf = _node(graph, "fake.leaf", "叶节点", parent=inner_group.id)
        source = _node(graph, "fake.source", "源")
        sink = _node(graph, "fake.sink", "汇")
        graph.add_connection(source.id, "image", outer.id, "input_1")
        graph.add_connection(outer.id, "input_1", inner_group.id, "input_1")
        graph.add_connection(inner_group.id, "input_1", leaf.id, "image")
        graph.add_connection(leaf.id, "image", inner_group.id, "output_1")
        graph.add_connection(inner_group.id, "output_1", outer.id, "output_1")
        graph.add_connection(outer.id, "output_1", sink.id, "image")

        self.assertEqual(graph.resolve_source(outer.id, "input_1"), (source.id, "image"))
        self.assertEqual(graph.resolve_source(outer.id, "output_1"), (leaf.id, "image"))
        self.assertEqual(
            graph.group_output_sources(outer.id), {"output_1": (leaf.id, "image")}
        )
        # 算子组边界的 2-cycle 不应阻止合法连线
        self.assertIn(leaf.id, graph.descendants(inner_group.id))

    def test_group_ports_are_not_bound_without_resolver(self) -> None:
        group = self.graph.create_group_node("算子组", (0, 0))
        inner = _node(self.graph, "fake.inner", "内部", parent=group.id)
        source = _node(self.graph, "fake.source", "源")
        self.graph.add_connection(source.id, "image", group.id, "input_1")
        self.assertTrue(group.group.inputs[0].pending)
        self.graph.add_connection(inner.id, "image", group.id, "output_1")
        self.assertTrue(group.group.outputs[0].pending)

    def test_group_snapshot_restore(self) -> None:
        self.graph.set_port_type_resolver(lambda _n, _p, _i: DataType.IMAGE)
        group = self.graph.create_group_node("算子组", (0, 0))
        source = _node(self.graph, "fake.source", "源")
        self.graph.add_connection(source.id, "image", group.id, "input_1")
        snapshot = self.graph.group_state_snapshot()
        self.assertEqual(len(group.group.inputs), 2)
        self.graph.restore_group_state(
            {
                group.id: {
                    "inputs": [{"key": "input_1", "label": "输入 1", "data_type": "image"}],
                    "outputs": [{"key": "output_1", "label": "输出 1", "data_type": "number"}],
                }
            }
        )
        # from_dict 会保证每侧至少保留一个待定端口
        self.assertEqual(len(group.group.inputs), 2)
        self.assertEqual(group.group.inputs[0].data_type, DataType.IMAGE)
        self.assertTrue(group.group.inputs[1].pending)
        self.assertEqual(group.group.outputs[0].data_type, DataType.NUMBER)
        self.assertTrue(group.group.outputs[1].pending)
        self.assertIn(group.id, snapshot)

    def test_remove_connection_reclaims_only_fully_free_group_port(self) -> None:
        self.graph.set_port_type_resolver(lambda _n, _p, _i: DataType.IMAGE)
        group = self.graph.create_group_node("算子组", (0, 0))
        inner = _node(self.graph, "fake.inner", "内部", parent=group.id)
        source = _node(self.graph, "fake.source", "源")
        outer = self.graph.add_connection(source.id, "image", group.id, "input_1")
        inner_edge = self.graph.add_connection(group.id, "input_1", inner.id, "image")
        self.assertEqual([port.key for port in group.group.inputs], ["input_1", "input_2"])

        # 只断开外侧：内侧仍在使用，端口继续保留（且不会误删另一侧连线）。
        self.graph.remove_connection(outer.id)
        self.assertEqual([port.key for port in group.group.inputs], ["input_1", "input_2"])
        self.assertFalse(group.group.inputs[0].pending)
        self.assertIsNotNone(self.graph.outgoing(group.id))

        # 再断开内侧：两侧都空闲，端口回收，空闲待定端口顶替同一 key。
        self.graph.remove_connection(inner_edge.id)
        self.assertEqual([port.key for port in group.group.inputs], ["input_1"])
        self.assertTrue(group.group.inputs[0].pending)
        self.assertEqual(group.group.inputs[0].label, "输入 1")

    def test_remove_connection_reclaims_group_output_port(self) -> None:
        self.graph.set_port_type_resolver(lambda _n, _p, _i: DataType.IMAGE)
        group = self.graph.create_group_node("算子组", (0, 0))
        inner = _node(self.graph, "fake.inner", "内部", parent=group.id)
        sink = _node(self.graph, "fake.sink", "下游")
        outer = self.graph.add_connection(group.id, "output_1", sink.id, "image")
        inner_edge = self.graph.add_connection(inner.id, "image", group.id, "output_1")
        self.assertEqual([port.key for port in group.group.outputs], ["output_1", "output_2"])

        # 只有外侧连线时先断开：端口仍被内侧占用，保留。
        self.graph.remove_connection(inner_edge.id)
        self.assertEqual([port.key for port in group.group.outputs], ["output_1", "output_2"])
        self.assertFalse(group.group.outputs[0].pending)
        # 内外两侧都断开后回收为待定端口。
        self.graph.remove_connection(outer.id)
        self.assertEqual([port.key for port in group.group.outputs], ["output_1"])
        self.assertTrue(group.group.outputs[0].pending)

    def test_node_serialization_preserves_group(self) -> None:
        group = self.graph.create_group_node("算子组", (10, 20))
        inner = _node(self.graph, "fake.inner", "内部", parent=group.id)
        data = self.graph.to_dict()
        restored = GraphModel.from_dict(data)
        group2 = restored.nodes[group.id]
        self.assertTrue(group2.is_group)
        self.assertEqual(group2.group.inputs[0].key, "input_1")
        self.assertEqual(restored.nodes[inner.id].parent_group, group.id)

    def test_project_roundtrip_with_group(self) -> None:
        workspace = WorkspaceModel("算子组工程")
        group = workspace.graph.create_group_node("算子组", (0, 0))
        _node(workspace.graph, "fake.inner", "内部", parent=group.id)
        tmp = _temp_dir("project-")
        try:
            path = save_project(tmp / "group.mi", workspace)
            loaded = load_project(path)
        finally:
            _cleanup_dir(tmp)
        group2 = loaded.graph.nodes[group.id]
        self.assertTrue(group2.is_group)
        self.assertEqual(len(loaded.graph.group_subtree(group.id)), 1)


class GroupDefinitionTests(unittest.TestCase):
    def setUp(self) -> None:
        self.graph = GraphModel()

    def test_build_definition(self) -> None:
        graph = self.graph
        group = graph.create_group_node("外层", (100, 200))
        group.group.bind("input_1", True, DataType.IMAGE)
        inner = _node(graph, "fake.inner", "内部", parent=group.id, pos=(160, 260))
        nested = graph.create_group_node("内层", (300, 300), parent_group=group.id)
        nested_child = _node(graph, "fake.nested_child", "内层节点", parent=nested.id, pos=(320, 340))
        graph.add_connection(group.id, "input_1", inner.id, "image")
        graph.add_connection(inner.id, "image", group.id, "output_1")
        graph.add_connection(group.id, "input_1", nested.id, "input_1")
        graph.add_connection(nested.id, "input_1", nested_child.id, "image")

        definition = build_group_definition(graph, group.id, "外层")
        self.assertEqual(definition["format"], "mineimage-group-definition")
        self.assertEqual(definition["name"], "外层")
        self.assertEqual(definition["inputs"][0]["data_type"], "image")
        ids = {node["id"] for node in definition["nodes"]}
        self.assertIn(inner.id, ids)
        self.assertIn(nested.id, ids)
        self.assertIn(nested_child.id, ids)
        inner_entry = next(node for node in definition["nodes"] if node["id"] == inner.id)
        self.assertEqual(inner_entry["pos"], [60.0, 60.0])
        self.assertEqual(inner_entry["parent_group"], None)
        nested_entry = next(node for node in definition["nodes"] if node["id"] == nested.id)
        self.assertIn("group", nested_entry)
        ports = {(item["src_node"], item["src_port"], item["dst_node"], item["dst_port"]) for item in definition["connections"]}
        self.assertIn((GROUP_BOUNDARY_ID, "input_1", inner.id, "image"), ports)
        self.assertIn((nested.id, "input_1", nested_child.id, "image"), ports)

    def test_source_node_in_group_rejected(self) -> None:
        group = self.graph.create_group_node("算子组", (0, 0))
        source = NodeModel(id="n_src", op_id="__source__", title="源", source_image_id="img", parent_group=group.id)
        self.graph.add_node(source)
        with self.assertRaises(GroupDefinitionError):
            build_group_definition(self.graph, group.id, "算子组")

    def test_reference_group_saved_without_children(self) -> None:
        graph = self.graph
        root = graph.create_group_node("根", (0, 0))
        referenced = graph.create_group_node("引用", (40, 40), parent_group=root.id)
        referenced.group.operator_id = "group.dependency"
        _node(graph, "fake.hidden", "不应保存", parent=referenced.id)
        definition = build_group_definition(graph, root.id, "根")
        entry = next(node for node in definition["nodes"] if node["id"] == referenced.id)
        self.assertIn("group_ref", entry)
        self.assertEqual(entry["group_ref"]["operator_id"], "group.dependency")
        self.assertNotIn("fake.hidden", {node["op_id"] for node in definition["nodes"]})

    def test_normalize_definition(self) -> None:
        definition = normalize_group_definition(
            {
                "name": "测试",
                "inputs": [{"key": "input_1", "label": "输入", "data_type": None}],
                "outputs": [{"key": "output_1", "label": "输出", "data_type": "number"}],
                "nodes": [
                    {"id": "a", "op_id": "fake.a", "pos": [1, 2], "parent_group": None},
                    {"id": "b", "op_id": "fake.b", "pos": "bad", "parent_group": "missing"},
                ],
                "connections": [
                    {"src_node": "__group__", "src_port": "input_1", "dst_node": "a", "dst_port": "image"},
                    {"src_node": "a", "src_port": "image", "dst_node": "not-exist", "dst_port": "image"},
                ],
            }
        )
        self.assertEqual(len(definition["nodes"]), 2)
        self.assertEqual(len(definition["connections"]), 1)
        self.assertEqual(definition["outputs"][0]["data_type"], "number")


class GroupOperatorFileTests(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = _temp_dir("operator-")
        self.graph = GraphModel()
        self.graph.set_port_type_resolver(lambda _n, _p, _i: DataType.IMAGE)

    def tearDown(self) -> None:
        _cleanup_dir(self.directory)

    def test_save_and_load_operator_file(self) -> None:
        group = self.graph.create_group_node("保存测试", (0, 0))
        inner = _node(self.graph, "fake.inner", "内部", parent=group.id, pos=(60, 80))
        self.graph.add_connection(group.id, "input_1", inner.id, "image")
        self.graph.add_connection(inner.id, "image", group.id, "output_1")
        saved = save_group_operator(self.graph, group.id, "保存测试", directory=self.directory)
        self.assertTrue(saved.path.exists())
        self.assertTrue(saved.op_id.startswith("group."))
        source_text = saved.path.read_text(encoding="utf-8")
        compile(source_text, str(saved.path), "exec")
        self.assertIn("GroupOperator", source_text)

        registry = OperatorRegistry.build([self.directory])
        entry = registry.entry(saved.op_id)
        self.assertIsNotNone(entry)
        self.assertTrue(is_group_operator(entry))
        self.assertIsInstance(entry.operator, GroupOperator)
        self.assertEqual(entry.spec.category, (CUSTOM_OPERATOR_CATEGORY,))
        self.assertEqual(CUSTOM_OPERATOR_CATEGORY, CATEGORY_ORDER[0])
        self.assertEqual(next(iter(registry.tabs())), CUSTOM_OPERATOR_CATEGORY)

        materialized = materialize_group_operator(registry, saved.op_id, pos=(400, 300))
        self.assertTrue(materialized.group_node.is_group)
        self.assertEqual(materialized.group_node.group.operator_id, saved.op_id)
        self.assertEqual(materialized.group_node.pos, (400.0, 300.0))
        self.assertEqual(len(materialized.nodes), 2)
        self.assertEqual(len(materialized.connections), 2)
        child = next(node for node in materialized.nodes if node.id != materialized.group_node.id)
        self.assertEqual(child.parent_group, materialized.group_node.id)
        self.assertEqual(child.pos, (460.0, 380.0))

    def test_duplicate_name_gets_unique_id(self) -> None:
        group = self.graph.create_group_node("重名", (0, 0))
        first = save_group_operator(self.graph, group.id, "重名", directory=self.directory)
        second = save_group_operator(self.graph, group.id, "重名", directory=self.directory)
        self.assertNotEqual(first.op_id, second.op_id)
        self.assertTrue(first.path.exists() and second.path.exists())

    def test_reference_operator_is_not_inlined(self) -> None:
        graph = self.graph
        # 先保存内层算子 A
        inner_group = graph.create_group_node("内层算子", (0, 0))
        inner_child = _node(graph, "fake.inner_child", "A 内部", parent=inner_group.id)
        graph.add_connection(inner_group.id, "input_1", inner_child.id, "image")
        graph.add_connection(inner_child.id, "image", inner_group.id, "output_1")
        saved_a = save_group_operator(graph, inner_group.id, "内层算子", directory=self.directory)
        inner_group.group.operator_id = saved_a.op_id

        # 外层算子组 B 引用 A
        outer = graph.create_group_node("外层算子", (600, 0))
        ref_node = graph.create_group_node("内层算子", (40, 40), parent_group=outer.id)
        ref_node.group.operator_id = saved_a.op_id
        sibling = _node(graph, "fake.sibling", "外层节点", parent=outer.id)
        sink = _node(graph, "fake.sink2", "外层汇", parent=outer.id)
        graph.add_connection(group_id := outer.id, "input_1", sibling.id, "image")
        graph.add_connection(sibling.id, "image", outer.id, "output_1")
        graph.add_connection(outer.id, "input_1", ref_node.id, "input_1")
        graph.add_connection(ref_node.id, "output_1", sink.id, "image")
        saved_b = save_group_operator(graph, outer.id, "外层算子", directory=self.directory)
        definition = saved_b.definition
        ref_entries = [node for node in definition["nodes"] if "group_ref" in node]
        self.assertEqual(len(ref_entries), 1)
        self.assertEqual(ref_entries[0]["group_ref"]["operator_id"], saved_a.op_id)
        # 引用算子组的内部节点没有被重复保存
        self.assertNotIn(inner_child.id, {node["id"] for node in definition["nodes"]})

        # 依赖存在时递归展开
        registry = OperatorRegistry.build([self.directory])
        materialized = materialize_group_operator(registry, saved_b.op_id, pos=(0, 0))
        op_ids = {node.op_id for node in materialized.nodes}
        self.assertIn("fake.inner_child", op_ids)
        self.assertIn("fake.sibling", op_ids)

        # 依赖缺失时仍能加载，并生成带端口的占位算子组
        missing_dir = self.directory.parent / f"missing-{uuid.uuid4().hex[:8]}"
        missing_dir.mkdir()
        b_file = saved_b.path
        target = missing_dir / b_file.name
        target.write_text(b_file.read_text(encoding="utf-8"), encoding="utf-8")
        registry_missing = OperatorRegistry.build([missing_dir])
        entry = registry_missing.entry(saved_b.op_id)
        self.assertIsNotNone(entry)
        materialized_missing = materialize_group_operator(registry_missing, saved_b.op_id, pos=(0, 0))
        placeholders = [
            node
            for node in materialized_missing.nodes
            if node.is_group and node.group and node.group.operator_id == saved_a.op_id
        ]
        self.assertEqual(len(placeholders), 1)
        self.assertEqual(placeholders[0].title, "内层算子")
        self.assertGreaterEqual(len(placeholders[0].group.inputs), 1)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
