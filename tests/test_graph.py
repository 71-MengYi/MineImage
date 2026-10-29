"""节点画布测试：布局、撤销命令、拖放与点击时序。

运行方式（项目根目录）::

    $env:PYTHONPATH = "."
    python -m unittest discover -s tests -p "test_graph.py" -v
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QMimeData, QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QDropEvent, QImage, QPainter, QUndoStack, QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.config import AppSettings
from app.core.constants import (
    CUSTOM_OPERATOR_CATEGORY,
    GROUP_OP_ID,
    MIME_OPERATOR,
    SINGLE_CLICK_DELAY_MS,
    SOURCE_OP_ID,
)
from app.core.data_types import DataType
from app.core.group import GroupModel
from app.core.model import GraphError, NodeModel, WorkspaceModel, new_id
from app.core.types import ImageData
from app.engine.engine import NodeResult
from app.graph.commands import (
    AddConnectionCommand,
    AddNodeCommand,
    InsertNodeOnEdgeCommand,
    MoveNodeCommand,
    RemoveConnectionCommand,
    RemoveNodeCommand,
    RemoveNodeTreeCommand,
    SetNodeParamsCommand,
)
from app.graph.group_item import (
    DEFAULT_GROUP_HEIGHT,
    DEFAULT_GROUP_WIDTH,
    GROUP_PADDING_X,
    INNER_OFFSET,
    GroupItem,
)
from app.graph.layout import GRID, compute_layout
from app.graph.node_item import NodeItem
from app.graph.view import GraphView
from app.operators.group import GroupOperator
from app.operators.spec import OperatorSpec, ParamSpec, PortSpec, param_defaults

_APP: QApplication | None = None

BLUR_SPEC = OperatorSpec(
    id="test.blur",
    name="模糊",
    category=("滤波与平滑", "平滑"),
    description="测试用模糊算子。",
    inputs=(PortSpec("image", "图像"),),
    outputs=(PortSpec("image", "图像"),),
    params=(ParamSpec("ksize", "核大小", "int", default=5, min=1, max=99, odd=True),),
)

MERGE_SPEC = OperatorSpec(
    id="test.merge",
    name="合并",
    category=("其他",),
    inputs=(PortSpec("a", "图像 A"), PortSpec("b", "图像 B")),
    outputs=(PortSpec("image", "图像"),),
)

SPLIT_SPEC = OperatorSpec(
    id="test.split",
    name="分离",
    category=("通道与色彩",),
    inputs=(PortSpec("image", "图像"),),
    outputs=(PortSpec("left", "左"), PortSpec("right", "右")),
)

OUTPUT_SPEC = OperatorSpec(
    id="output.display",
    name="输出预览",
    category=("其他", "可视化"),
    inputs=(PortSpec("data", "任意数据", any_type=True),),
    outputs=(PortSpec("data", "任意数据", any_type=True),),
)

NOTE_SPEC = OperatorSpec(
    id="note.sticky",
    name="便签",
    category=("其他", "可视化"),
    inputs=(PortSpec("data", "任意数据", any_type=True, optional=True),),
    outputs=(PortSpec("data", "任意数据", any_type=True),),
    params=(ParamSpec("text", "便签文字", "string", default="", multiline=True),),
    canvas_text_param="text",
    allow_empty_output=True,
)


class FakeRegistry:
    """仅实现 GraphView 需要的 ``spec`` 查询接口。"""

    def __init__(self, *specs: OperatorSpec) -> None:
        self._specs = {spec.id: spec for spec in specs}

    def spec(self, op_id: str) -> OperatorSpec | None:
        return self._specs.get(op_id)

    def get(self, op_id: str):
        return None

    def entry(self, op_id: str):
        return None


def make_registry() -> FakeRegistry:
    return FakeRegistry(BLUR_SPEC, MERGE_SPEC, SPLIT_SPEC, OUTPUT_SPEC, NOTE_SPEC)


#: 测试用算子组定义：一个输入 -> 一个模糊节点 -> 一个输出。
DEMO_GROUP_DEFINITION = {
    "format": "mineimage-group-definition",
    "version": 1,
    "name": "演示流程",
    "inputs": [{"key": "input_1", "label": "输入 1", "data_type": "image"}],
    "outputs": [{"key": "output_1", "label": "输出 1", "data_type": "image"}],
    "nodes": [
        {
            "id": "n1",
            "op_id": "test.blur",
            "title": "组内模糊",
            "pos": [60.0, 80.0],
            "params": {"ksize": 5},
            "parent_group": None,
        }
    ],
    "connections": [
        {
            "src_node": "__group__",
            "src_port": "input_1",
            "dst_node": "n1",
            "dst_port": "image",
        },
        {
            "src_node": "n1",
            "src_port": "image",
            "dst_node": "__group__",
            "dst_port": "output_1",
        },
    ],
}


class GroupDemoOperator(GroupOperator):
    """测试用算子组算子（由定义直接展开）。"""

    spec = OperatorSpec(
        id="group.demo",
        name="演示流程",
        category=(CUSTOM_OPERATOR_CATEGORY,),
        inputs=(PortSpec("input_1", "输入 1"),),
        outputs=(PortSpec("output_1", "输出 1"),),
    )
    group_definition = DEMO_GROUP_DEFINITION


class GroupRegistry(FakeRegistry):
    """带算子组算子实例的注册表替身。"""

    def __init__(self, *specs: OperatorSpec, operator=None) -> None:
        super().__init__(*specs)
        self._operator = operator

    def get(self, op_id: str):
        if self._operator is not None and self._operator.spec.id == op_id:
            return self._operator
        return None


def make_group_registry() -> GroupRegistry:
    operator = GroupDemoOperator()
    return GroupRegistry(
        BLUR_SPEC,
        MERGE_SPEC,
        SPLIT_SPEC,
        OUTPUT_SPEC,
        operator.spec,
        operator=operator,
    )


class _FakeContextMenuEvent:
    """右键菜单事件替身：只实现 contextMenuEvent 用到的接口。"""

    def __init__(self, pos: QPoint) -> None:
        self._pos = pos

    def pos(self) -> QPoint:
        return self._pos

    def globalPos(self) -> QPoint:
        return QPoint(0, 0)


def setUpModule() -> None:
    global _APP
    _APP = QApplication.instance() or QApplication([])


def tearDownModule() -> None:
    global _APP
    if _APP is not None:
        _APP.processEvents()
        _APP = None


class GraphViewTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory(prefix="mineimage-graph-")
        self.settings = AppSettings(Path(self._temp_dir.name) / "test.ini")
        self.workspace = WorkspaceModel("测试工作区")
        self.graph = self.workspace.graph
        self.undo = QUndoStack()
        self.registry = make_registry()
        self.view = GraphView(self.workspace, self.registry, self.undo, self.settings)
        self.view.resize(1000, 700)

    def tearDown(self) -> None:
        self.view.close()
        self.view.deleteLater()
        if _APP is not None:
            _APP.processEvents()
        # QSettings 在 Windows 上可能仍持有 ini 文件句柄，清理失败不影响测试结果。
        try:
            self._temp_dir.cleanup()
        except OSError:
            pass

    # ------------------------------------------------------------------ 工具
    def add_node(
        self,
        op_id: str = "test.blur",
        title: str | None = None,
        pos: tuple[float, float] = (0.0, 0.0),
        params: dict | None = None,
    ) -> NodeModel:
        spec = self.registry.spec(op_id)
        assert spec is not None
        node = NodeModel(
            id=new_id("n"),
            op_id=op_id,
            title=title or spec.name,
            pos=pos,
            params=dict(param_defaults(spec)) if params is None else dict(params),
        )
        self.graph.add_node(node)
        return node

    def add_source(self, title: str = "原始图像", pos: tuple[float, float] = (0.0, 0.0)) -> NodeModel:
        node = NodeModel(id=new_id("n"), op_id=SOURCE_OP_ID, title=title, pos=pos)
        self.graph.add_node(node)
        return node

    def connect(
        self,
        src: NodeModel,
        dst: NodeModel,
        src_port: str = "image",
        dst_port: str = "image",
    ):
        return self.graph.add_connection(src.id, src_port, dst.id, dst_port)

    def item(self, node: NodeModel):
        return self.view._nodes[node.id]

    def make_branch(
        self,
    ) -> tuple[NodeModel, NodeModel, NodeModel, NodeModel, tuple]:
        """构造一个两分支 + 一个孙节点的画布：root -> left/right，left -> leaf。"""

        root = self.add_node(pos=(0.0, 0.0))
        left = self.add_node(pos=(-280.0, 220.0))
        right = self.add_node(pos=(280.0, 220.0))
        leaf = self.add_node(pos=(-280.0, 440.0))
        connections = (
            self.connect(root, left),
            self.connect(root, right),
            self.connect(left, leaf),
        )
        self.view.show()
        if _APP is not None:
            _APP.processEvents()
        self.view.fit_graph()
        if _APP is not None:
            _APP.processEvents()
        return root, left, right, leaf, connections

    # ---------------------------------------------------------------- 自动布局
    def test_auto_layout_branching_dag(self) -> None:
        source = self.add_source(pos=(600.0, 120.0))
        left = self.add_node(pos=(-300.0, 500.0))
        right = self.add_node(pos=(900.0, -200.0))
        merge = self.add_node("test.merge", pos=(0.0, 0.0))
        tail = self.add_node(pos=(250.0, 900.0))
        self.connect(source, left)
        self.connect(source, right)
        self.connect(left, merge, dst_port="a")
        self.connect(right, merge, dst_port="b")
        self.connect(merge, tail)

        self.view.auto_layout()

        items = {
            node_id: self.view._nodes[node_id]
            for node_id in (source.id, left.id, right.id, merge.id, tail.id)
        }
        for node_id, item in items.items():
            pos = item.pos()
            self.assertTrue(math.isfinite(pos.x()), f"{node_id} x 非有限值")
            self.assertTrue(math.isfinite(pos.y()), f"{node_id} y 非有限值")
            self.assertGreaterEqual(pos.x(), -0.001)
            self.assertGreaterEqual(pos.y(), -0.001)

        # 深度自上而下递增
        self.assertLess(items[source.id].pos().y(), items[left.id].pos().y())
        self.assertLess(items[source.id].pos().y(), items[right.id].pos().y())
        self.assertLess(items[left.id].pos().y(), items[merge.id].pos().y())
        self.assertLess(items[right.id].pos().y(), items[merge.id].pos().y())
        self.assertLess(items[merge.id].pos().y(), items[tail.id].pos().y())
        # 同层节点纵向对齐
        self.assertAlmostEqual(items[left.id].pos().y(), items[right.id].pos().y(), places=1)
        # 兄弟节点横向分开
        self.assertGreater(abs(items[left.id].pos().x() - items[right.id].pos().x()), 100.0)
        # 父节点尽量居中于子节点；坐标会被吸附到网格，允许一格以内的量化误差。
        left_center = items[left.id].sceneBoundingRect().center().x()
        right_center = items[right.id].sceneBoundingRect().center().x()
        merge_center = items[merge.id].sceneBoundingRect().center().x()
        self.assertAlmostEqual(merge_center, (left_center + right_center) / 2.0, delta=GRID)

        # 任意两个节点包围盒不重叠
        rects = [(node_id, item.sceneBoundingRect()) for node_id, item in items.items()]
        for index, (node_id, rect) in enumerate(rects):
            for other_id, other_rect in rects[index + 1 :]:
                self.assertFalse(
                    rect.intersects(other_rect),
                    f"节点重叠：{node_id} / {other_id}",
                )

        # 布局命令可撤销
        self.assertTrue(self.undo.canUndo())
        self.undo.undo()
        self.assertEqual(self.graph.nodes[source.id].pos, (600.0, 120.0))

    def test_auto_layout_empty_and_multiple_outputs(self) -> None:
        self.view.auto_layout()  # 空图不应崩溃
        split = self.add_node("test.split", pos=(10.0, 10.0))
        target_a = self.add_node(pos=(500.0, 500.0))
        target_b = self.add_node(pos=(500.0, -500.0))
        self.connect(split, target_a, src_port="left")
        self.connect(split, target_b, src_port="right")
        self.view.auto_layout()
        positions = {node_id: self.view._nodes[node_id].pos() for node_id in self.graph.nodes}
        for pos in positions.values():
            self.assertTrue(math.isfinite(pos.x()) and math.isfinite(pos.y()))
        self.assertGreater(self.view._nodes[target_a.id].pos().y(), self.view._nodes[split.id].pos().y())

    def test_auto_layout_parallel_edges_between_same_nodes(self) -> None:
        split = self.add_node("test.split", pos=(50.0, 50.0))
        merge = self.add_node("test.merge", pos=(50.0, 50.0))
        self.connect(split, merge, src_port="left", dst_port="a")
        self.connect(split, merge, src_port="right", dst_port="b")
        # 模型自带的拓扑排序会把同一对节点的重复边误判成环，布局必须自行处理。
        positions = compute_layout(
            self.graph,
            {split.id: (222.0, 184.0), merge.id: (222.0, 184.0)},
        )
        self.assertEqual(len(positions), 2)
        self.assertGreater(positions[merge.id][1], positions[split.id][1])

        self.view.auto_layout()
        self.assertGreater(self.view._nodes[merge.id].pos().y(), self.view._nodes[split.id].pos().y())
        self.assertTrue(self.undo.canUndo())

    def test_auto_layout_isolated_root_keeps_parent_centered(self) -> None:
        self.add_source("孤立的源", pos=(0.0, 0.0))
        split = self.add_node("test.split", pos=(0.0, 0.0))
        merge = self.add_node("test.merge", pos=(0.0, 0.0))
        self.connect(split, merge, src_port="left", dst_port="a")
        self.connect(split, merge, src_port="right", dst_port="b")
        self.view.auto_layout()
        split_center = self.view._nodes[split.id].sceneBoundingRect().center().x()
        merge_center = self.view._nodes[merge.id].sceneBoundingRect().center().x()
        self.assertAlmostEqual(split_center, merge_center, delta=2.0)

    def test_compute_layout_raises_without_nodes(self) -> None:
        self.assertEqual(compute_layout(self.graph), {})

    # ------------------------------------------------------------------ 环路
    def test_cycle_rejection(self) -> None:
        first = self.add_node("test.blur", pos=(0.0, 0.0))
        second = self.add_node("test.blur", pos=(0.0, 200.0))
        self.connect(first, second)
        with self.assertRaises(GraphError):
            self.graph.add_connection(second.id, "image", first.id, "image")
        with self.assertRaises(GraphError):
            self.graph.add_connection(first.id, "image", first.id, "image")
        self.assertEqual(len(self.graph.connections), 1)

    def test_view_cycle_connection_reports_status(self) -> None:
        first = self.add_node("test.blur", pos=(0.0, 0.0))
        second = self.add_node("test.blur", pos=(0.0, 200.0))
        self.connect(first, second)
        messages: list[str] = []
        self.view.status_message.connect(messages.append)
        self.view.connect_ports(second.id, "image", first.id, "image")
        self.assertTrue(messages)
        self.assertIn("环路", messages[-1])
        self.assertEqual(len(self.graph.connections), 1)

    def test_replace_input_connection_uses_undo_macro(self) -> None:
        first = self.add_source("源 1")
        second = self.add_source("源 2")
        target = self.add_node("test.blur", pos=(0.0, 300.0))
        old_connection = self.connect(first, target)

        self.view.connect_ports(second.id, "image", target.id, "image")
        self.assertNotIn(old_connection.id, self.graph.connections)
        self.assertEqual(len(self.graph.connections), 1)
        current = next(iter(self.graph.connections.values()))
        self.assertEqual(current.src_node, second.id)

        self.undo.undo()
        self.assertIn(old_connection.id, self.graph.connections)
        self.assertEqual(self.graph.connections[old_connection.id].src_node, first.id)
        self.assertEqual(len(self.graph.connections), 1)

        self.undo.redo()
        self.assertNotIn(old_connection.id, self.graph.connections)
        self.assertEqual(
            next(iter(self.graph.connections.values())).src_node,
            second.id,
        )

    # ------------------------------------------------------------------ 命令
    def test_add_node_command_undo_redo_keeps_items(self) -> None:
        node = NodeModel(
            id=new_id("n"),
            op_id="test.blur",
            title="模糊",
            pos=(30.0, 40.0),
            params=param_defaults(BLUR_SPEC),
        )
        command = AddNodeCommand(self.graph, node)
        self.undo.push(command)
        self.assertIn(node.id, self.graph.nodes)
        self.assertIn(node.id, self.view._nodes)

        self.undo.undo()
        self.assertNotIn(node.id, self.graph.nodes)
        self.assertNotIn(node.id, self.view._nodes)

        self.undo.redo()
        self.assertIn(node.id, self.graph.nodes)
        self.assertIn(node.id, self.view._nodes)
        self.assertEqual(self.view._nodes[node.id].pos(), QPointF(30.0, 40.0))

    def test_add_and_remove_connection_commands(self) -> None:
        first = self.add_node("test.blur", pos=(0.0, 0.0))
        second = self.add_node("test.blur", pos=(0.0, 200.0))
        command = AddConnectionCommand(self.graph, first.id, "image", second.id, "image")
        self.undo.push(command)
        connection_id = command.connection_id
        self.assertIn(connection_id, self.graph.connections)
        self.assertIn(connection_id, self.view._connections)

        self.undo.undo()
        self.assertNotIn(connection_id, self.graph.connections)
        self.assertNotIn(connection_id, self.view._connections)

        self.undo.redo()
        self.assertIn(connection_id, self.graph.connections)
        self.assertIn(connection_id, self.view._connections)

        connection = self.graph.connections[connection_id]
        remove = RemoveConnectionCommand(self.graph, connection)
        self.undo.push(remove)
        self.assertNotIn(connection_id, self.graph.connections)
        self.assertNotIn(connection_id, self.view._connections)
        self.undo.undo()
        self.assertIn(connection_id, self.graph.connections)
        self.assertIn(connection_id, self.view._connections)

    def test_move_node_command_merges_and_undoes(self) -> None:
        node = self.add_node(pos=(10.0, 10.0))
        self.undo.push(MoveNodeCommand(self.graph, node.id, (10.0, 10.0), (20.0, 20.0)))
        self.undo.push(MoveNodeCommand(self.graph, node.id, (20.0, 20.0), (30.0, 30.0)))
        self.assertEqual(self.undo.count(), 1)
        self.assertEqual(self.graph.nodes[node.id].pos, (30.0, 30.0))
        self.assertEqual(self.view._nodes[node.id].pos(), QPointF(30.0, 30.0))

        self.undo.undo()
        self.assertEqual(self.graph.nodes[node.id].pos, (10.0, 10.0))
        self.assertEqual(self.view._nodes[node.id].pos(), QPointF(10.0, 10.0))
        self.undo.redo()
        self.assertEqual(self.graph.nodes[node.id].pos, (30.0, 30.0))
        self.assertEqual(self.view._nodes[node.id].pos(), QPointF(30.0, 30.0))

    def test_copy_paste_nodes_preserves_internal_connections_and_undo(self) -> None:
        first = self.add_node("test.blur", pos=(0.0, 0.0), params={"ksize": 9})
        second = self.add_node("test.blur", pos=(0.0, 240.0))
        original = self.connect(first, second)
        self.view._nodes[first.id].setSelected(True)
        self.view._nodes[second.id].setSelected(True)

        self.assertTrue(self.view.copy_selected())
        new_ids = self.view.paste_nodes()
        self.assertEqual(len(new_ids), 2)
        self.assertEqual(len(self.graph.nodes), 4)
        self.assertTrue(all(self.view._nodes[node_id].isSelected() for node_id in new_ids))
        new_nodes = [self.graph.nodes[node_id] for node_id in new_ids]
        self.assertTrue(all(node.op_id == "test.blur" for node in new_nodes))
        self.assertEqual(new_nodes[0].params.get("ksize"), 9)
        self.assertNotEqual(new_nodes[0].pos, first.pos)
        self.assertNotEqual(new_nodes[1].pos, second.pos)
        # 原节点之间的连线只在副本之间复制一条，不复制到原有节点。
        copied_connections = [
            connection
            for connection in self.graph.connections.values()
            if connection.id != original.id
        ]
        self.assertEqual(len(copied_connections), 1)
        pasted_connection = copied_connections[0]
        self.assertIn(pasted_connection.src_node, new_ids)
        self.assertIn(pasted_connection.dst_node, new_ids)
        self.assertEqual(pasted_connection.src_port, "image")
        self.assertEqual(pasted_connection.dst_port, "image")

        # 整次粘贴一步撤销，节点与副本连线一起回退。
        self.assertEqual(self.undo.count(), 1)
        self.undo.undo()
        self.assertEqual(len(self.graph.nodes), 2)
        self.assertEqual(len(self.graph.connections), 1)
        self.assertIn(original.id, self.graph.connections)
        self.undo.redo()
        self.assertEqual(len(self.graph.nodes), 4)
        self.assertEqual(len(self.graph.connections), 2)

    def test_copy_paste_one_node_does_not_copy_dangling_connection(self) -> None:
        first = self.add_node("test.blur", pos=(0.0, 0.0))
        second = self.add_node("test.blur", pos=(0.0, 240.0))
        self.connect(first, second)
        self.view._nodes[first.id].setSelected(True)
        self.assertTrue(self.view.copy_selected())
        new_ids = self.view.paste_nodes()
        self.assertEqual(len(new_ids), 1)
        self.assertEqual(len(self.graph.nodes), 3)
        self.assertEqual(len(self.graph.connections), 1)

    def test_copy_paste_keyboard_shortcuts(self) -> None:
        first = self.add_node("test.blur", pos=(0.0, 0.0))
        self.view.show()
        self.view.setFocus()
        if _APP is not None:
            _APP.processEvents()
        self.view.set_selected_node(first.id)
        QTest.keyClick(
            self.view,
            Qt.Key.Key_C,
            Qt.KeyboardModifier.ControlModifier,
        )
        QTest.keyClick(
            self.view,
            Qt.Key.Key_V,
            Qt.KeyboardModifier.ControlModifier,
        )
        if _APP is not None:
            _APP.processEvents()
        self.assertEqual(len(self.graph.nodes), 2)
        self.assertEqual(len(self.graph.connections), 0)

    def test_move_tracking_pushes_command(self) -> None:
        node = self.add_node(pos=(10.0, 10.0))
        item = self.view._nodes[node.id]
        item.setSelected(True)
        self.view.begin_move_tracking()
        item.setPos(120.0, 80.0)
        self.view.finish_move_tracking()
        self.assertEqual(self.graph.nodes[node.id].pos, (120.0, 80.0))
        self.assertTrue(self.undo.canUndo())
        self.undo.undo()
        self.assertEqual(self.graph.nodes[node.id].pos, (10.0, 10.0))

    def test_remove_node_command_restores_connections(self) -> None:
        first = self.add_node("test.blur", pos=(0.0, 0.0))
        second = self.add_node("test.blur", pos=(0.0, 200.0))
        connection = self.connect(first, second)
        command = RemoveNodeCommand(self.graph, second.id)
        self.undo.push(command)
        self.assertNotIn(second.id, self.graph.nodes)
        self.assertNotIn(connection.id, self.graph.connections)
        self.assertNotIn(second.id, self.view._nodes)

        self.undo.undo()
        self.assertIn(second.id, self.graph.nodes)
        self.assertIn(connection.id, self.graph.connections)
        self.assertIn(second.id, self.view._nodes)
        self.assertIn(connection.id, self.view._connections)
        self.assertEqual(self.graph.nodes[second.id].pos, (0.0, 200.0))

        self.undo.redo()
        self.assertNotIn(second.id, self.graph.nodes)
        self.assertNotIn(connection.id, self.graph.connections)

    def test_set_node_params_command(self) -> None:
        node = self.add_node(params={"ksize": 5})
        old_params = {"ksize": 5}
        new_params = {"ksize": 9}
        command = SetNodeParamsCommand(self.graph, node.id, old_params, new_params)
        self.undo.push(command)
        self.assertEqual(self.graph.nodes[node.id].params, new_params)
        self.undo.undo()
        self.assertEqual(self.graph.nodes[node.id].params, old_params)
        self.undo.redo()
        self.assertEqual(self.graph.nodes[node.id].params, new_params)

    def test_insert_node_on_edge_command(self) -> None:
        first = self.add_node("test.blur", pos=(0.0, 0.0))
        second = self.add_node("test.blur", pos=(0.0, 300.0))
        connection = self.connect(first, second)
        inserted = NodeModel(
            id=new_id("n"),
            op_id="test.blur",
            title="插入的模糊",
            pos=(0.0, 150.0),
            params=param_defaults(BLUR_SPEC),
        )
        command = InsertNodeOnEdgeCommand(self.graph, inserted, connection, "image", "image")
        self.undo.push(command)

        self.assertIn(inserted.id, self.graph.nodes)
        self.assertIn(inserted.id, self.view._nodes)
        self.assertNotIn(connection.id, self.graph.connections)
        outgoing = self.graph.outgoing(first.id)
        incoming = self.graph.incoming(second.id)
        self.assertEqual(len(outgoing), 1)
        self.assertEqual(outgoing[0].dst_node, inserted.id)
        self.assertEqual(len(incoming), 1)
        self.assertEqual(incoming[0].src_node, inserted.id)
        self.assertEqual(len(self.graph.connections), 2)

        self.undo.undo()
        self.assertNotIn(inserted.id, self.graph.nodes)
        self.assertNotIn(inserted.id, self.view._nodes)
        self.assertIn(connection.id, self.graph.connections)
        self.assertEqual(len(self.graph.connections), 1)

        self.undo.redo()
        self.assertIn(inserted.id, self.graph.nodes)
        self.assertNotIn(connection.id, self.graph.connections)
        self.assertEqual(len(self.graph.connections), 2)

    def test_delete_selected_nodes_and_connections_undo(self) -> None:
        first = self.add_node("test.blur", pos=(0.0, 0.0))
        second = self.add_node("test.blur", pos=(0.0, 200.0))
        connection = self.connect(first, second)
        self.view._nodes[first.id].setSelected(True)
        self.view._nodes[second.id].setSelected(True)
        selection_events: list[str] = []
        self.view.selection_changed.connect(selection_events.append)

        self.view.delete_selected()
        self.assertFalse(self.graph.nodes)
        self.assertFalse(self.view._nodes)
        self.assertFalse(self.graph.connections)
        self.assertEqual(selection_events, [""])

        while self.undo.canUndo():
            self.undo.undo()
        self.assertIn(first.id, self.graph.nodes)
        self.assertIn(second.id, self.graph.nodes)
        self.assertIn(connection.id, self.graph.connections)
        self.assertEqual(len(self.view._nodes), 2)
        self.assertEqual(len(self.view._connections), 1)

    def test_remove_node_tree_command_removes_descendants_and_undo(self) -> None:
        upstream = self.add_node(pos=(0.0, -200.0))
        root = self.add_node(pos=(0.0, 0.0))
        child = self.add_node(pos=(0.0, 200.0))
        grandchild = self.add_node(pos=(0.0, 400.0))
        branch = self.add_node("test.split", pos=(400.0, 400.0))
        kept = self.add_node(pos=(600.0, 0.0))
        upstream_link = self.connect(upstream, root)
        first_link = self.connect(root, child)
        second_link = self.connect(child, grandchild)
        branch_link = self.connect(child, branch)

        command = RemoveNodeTreeCommand(self.graph, root.id)
        self.undo.push(command)

        self.assertEqual(
            set(command.node_ids),
            {root.id, child.id, grandchild.id, branch.id},
        )
        self.assertEqual(set(self.graph.nodes), {upstream.id, kept.id})
        self.assertFalse(self.graph.connections)
        self.assertEqual(set(self.view._nodes), {upstream.id, kept.id})
        self.assertFalse(self.view._connections)

        self.undo.undo()
        self.assertEqual(
            set(self.graph.nodes),
            {upstream.id, root.id, child.id, grandchild.id, branch.id, kept.id},
        )
        self.assertEqual(
            set(self.graph.connections),
            {upstream_link.id, first_link.id, second_link.id, branch_link.id},
        )
        self.assertEqual(len(self.view._nodes), 6)
        self.assertEqual(len(self.view._connections), 4)

        self.undo.redo()
        self.assertEqual(set(self.graph.nodes), {upstream.id, kept.id})
        self.assertFalse(self.graph.connections)

    def test_delete_node_with_children_via_context_menu(self) -> None:
        root = self.add_node(pos=(0.0, 0.0))
        child = self.add_node(pos=(0.0, 200.0))
        grandchild = self.add_node(pos=(0.0, 400.0))
        kept = self.add_node(pos=(500.0, 0.0))
        first_link = self.connect(root, child)
        self.connect(child, grandchild)

        point = self._show_and_focus_node(root)
        with patch("app.graph.view.QMenu") as menu_cls:
            self.view.contextMenuEvent(_FakeContextMenuEvent(point))
        actions = [
            call.args[0]
            for call in menu_cls.return_value.addAction.call_args_list
            if call.args
        ]
        self.assertEqual(
            actions.index("删除节点及子节点"),
            actions.index("删除节点") + 1,
        )

        status: list[str] = []
        self.view.status_message.connect(status.append)
        self.view.delete_node_with_children(root.id)

        self.assertEqual(set(self.graph.nodes), {kept.id})
        self.assertFalse(self.graph.connections)
        self.assertIn("已删除节点及 2 个子节点", status)

        self.undo.undo()
        self.assertEqual(
            set(self.graph.nodes),
            {root.id, child.id, grandchild.id, kept.id},
        )
        self.assertEqual(len(self.graph.connections), 2)
        self.assertIn(first_link.id, self.graph.connections)
        self.assertEqual(len(self.view._nodes), 4)
        self.assertEqual(len(self.view._connections), 2)

    # ------------------------------------------------------------ 折叠 / 展开
    def test_collapse_badge_appears_only_for_multi_branch_nodes(self) -> None:
        root = self.add_node(pos=(0.0, 0.0))
        first = self.add_node(pos=(-200.0, 200.0))
        second = self.add_node(pos=(200.0, 200.0))
        first_link = self.connect(root, first)
        self.assertNotIn(root.id, self.view._collapse_badges)

        self.connect(root, second)
        self.assertIn(root.id, self.view._collapse_badges)
        badge = self.view._collapse_badges[root.id]
        self.assertTrue(badge.isVisible())
        self.assertFalse(badge.collapsed)
        self.assertEqual(badge.count, 2)
        self.assertNotIn(first.id, self.view._collapse_badges)

        # 少一个分支后按钮消失。
        self.graph.remove_connection(first_link.id)
        self.assertNotIn(root.id, self.view._collapse_badges)

    def test_collapse_hides_subtree_and_badge_shows_count(self) -> None:
        root, left, right, leaf, connections = self.make_branch()
        extra = self.add_node(pos=(700.0, 0.0))
        if _APP is not None:
            _APP.processEvents()
        badge = self.view._collapse_badges[root.id]
        self.assertEqual(badge.count, 3)

        self.view.toggle_node_collapsed(root.id)
        if _APP is not None:
            _APP.processEvents()

        self.assertTrue(self.view.is_node_collapsed(root.id))
        self.assertEqual(self.view.hidden_node_ids(), {left.id, right.id, leaf.id})
        for node in (left, right, leaf):
            self.assertFalse(self.view._nodes[node.id].isVisible())
        self.assertTrue(self.view._nodes[root.id].isVisible())
        self.assertTrue(self.view._nodes[extra.id].isVisible())
        for connection in connections:
            self.assertFalse(self.view._connections[connection.id].isVisible())
        self.assertTrue(badge.isVisible())
        self.assertTrue(badge.collapsed)
        self.assertEqual(badge.count, 3)
        self.assertEqual(len(self.graph.nodes), 5)
        visible_inputs = {port.node_item.node_id for port in self.view.port_items(True)}
        self.assertNotIn(left.id, visible_inputs)
        self.assertNotIn(leaf.id, visible_inputs)
        self.assertIn(extra.id, visible_inputs)

        self.view.toggle_node_collapsed(root.id)
        if _APP is not None:
            _APP.processEvents()
        self.assertFalse(self.view.is_node_collapsed(root.id))
        for node in (left, right, leaf):
            self.assertTrue(self.view._nodes[node.id].isVisible())
        for connection in connections:
            self.assertTrue(self.view._connections[connection.id].isVisible())
        self.assertFalse(badge.collapsed)

    def test_nested_collapse_keeps_inner_state_when_expanded(self) -> None:
        outer = self.add_node(pos=(0.0, 0.0))
        left = self.add_node(pos=(-280.0, 220.0))
        right = self.add_node(pos=(280.0, 220.0))
        deep = self.add_node(pos=(-420.0, 440.0))
        deep2 = self.add_node(pos=(-140.0, 440.0))
        self.connect(outer, left)
        self.connect(outer, right)
        self.connect(left, deep)
        self.connect(left, deep2)
        self.view.show()
        if _APP is not None:
            _APP.processEvents()

        self.view.collapse_node(left.id)
        self.view.collapse_node(outer.id)
        if _APP is not None:
            _APP.processEvents()
        self.assertEqual(self.view.hidden_node_ids(), {left.id, right.id, deep.id, deep2.id})
        outer_badge = self.view._collapse_badges[outer.id]
        left_badge = self.view._collapse_badges[left.id]
        self.assertTrue(outer_badge.isVisible())
        self.assertTrue(outer_badge.collapsed)
        self.assertEqual(outer_badge.count, 4)
        self.assertFalse(left_badge.isVisible())

        self.view.expand_node(outer.id)
        if _APP is not None:
            _APP.processEvents()
        self.assertTrue(self.view.is_node_collapsed(left.id))
        self.assertTrue(self.view._nodes[left.id].isVisible())
        self.assertFalse(self.view._nodes[deep.id].isVisible())
        self.assertTrue(left_badge.isVisible())
        self.assertEqual(left_badge.count, 2)

        self.view.expand_node(left.id)
        if _APP is not None:
            _APP.processEvents()
        self.assertFalse(self.view.hidden_node_ids())
        self.assertTrue(self.view._nodes[deep.id].isVisible())
        self.assertTrue(self.view._nodes[deep2.id].isVisible())

    def test_click_collapse_badge_toggles(self) -> None:
        root, left, right, leaf, connections = self.make_branch()
        badge = self.view._collapse_badges[root.id]
        point = self.view.mapFromScene(badge.scenePos())

        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            point,
        )
        if _APP is not None:
            _APP.processEvents()
        self.assertTrue(self.view.is_node_collapsed(root.id))
        self.assertTrue(badge.collapsed)

        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            point,
        )
        if _APP is not None:
            _APP.processEvents()
        self.assertFalse(self.view.is_node_collapsed(root.id))
        self.assertFalse(badge.collapsed)

    def test_collapsed_branch_auto_expands_on_new_connection(self) -> None:
        root, left, right, leaf, connections = self.make_branch()
        extra = self.add_node(pos=(700.0, 300.0))
        self.view.collapse_node(root.id)
        self.assertTrue(self.view.is_node_collapsed(root.id))
        status: list[str] = []
        self.view.status_message.connect(status.append)

        self.view.connect_ports(root.id, "image", extra.id, "image")
        if _APP is not None:
            _APP.processEvents()

        self.assertFalse(self.view.is_node_collapsed(root.id))
        for node in (left, right, leaf):
            self.assertTrue(self.view._nodes[node.id].isVisible())
        link = self.graph.input_connection(extra.id, "image")
        self.assertIsNotNone(link)
        self.assertIn(link.id, self.view._connections)
        self.assertTrue(self.view._connections[link.id].isVisible())
        self.assertTrue(any("自动展开" in message for message in status))

    def test_drop_operator_on_collapsed_badge_expands_and_connects(self) -> None:
        root, _left, _right, _leaf, _connections = self.make_branch()
        self.view.collapse_node(root.id)
        badge = self.view._collapse_badges[root.id]

        new_node_id = self.view.add_operator_at("test.blur", QPointF(badge.scenePos()))
        if _APP is not None:
            _APP.processEvents()

        self.assertIsNotNone(new_node_id)
        assert new_node_id is not None
        self.assertFalse(self.view.is_node_collapsed(root.id))
        self.assertIn(new_node_id, self.graph.nodes)
        self.assertTrue(self.view._nodes[new_node_id].isVisible())
        link = self.graph.input_connection(new_node_id, "image")
        self.assertIsNotNone(link)
        self.assertEqual(link.src_node, root.id)
        self.assertIn(link.id, self.view._connections)
        self.assertTrue(self.view._connections[link.id].isVisible())

    def test_collapsed_group_moves_with_root(self) -> None:
        root, left, right, leaf, _connections = self.make_branch()
        self.view.collapse_node(root.id)
        node_ids = (root.id, left.id, right.id, leaf.id)
        before = {node_id: self.graph.nodes[node_id].pos for node_id in node_ids}
        item = self.view._nodes[root.id]
        self.view._scene.clearSelection()
        item.setSelected(True)

        self.view.begin_move_tracking()
        item.setPos(item.pos().x() + 120.0, item.pos().y() + 60.0)
        self.view.finish_move_tracking()
        if _APP is not None:
            _APP.processEvents()

        after = {node_id: self.graph.nodes[node_id].pos for node_id in node_ids}
        dx = after[root.id][0] - before[root.id][0]
        dy = after[root.id][1] - before[root.id][1]
        self.assertAlmostEqual(dx, 120.0, places=1)
        self.assertAlmostEqual(dy, 60.0, places=1)
        for node_id in (left.id, right.id, leaf.id):
            self.assertAlmostEqual(after[node_id][0], before[node_id][0] + dx, places=1)
            self.assertAlmostEqual(after[node_id][1], before[node_id][1] + dy, places=1)

        self.undo.undo()
        if _APP is not None:
            _APP.processEvents()
        for node_id in node_ids:
            self.assertEqual(self.graph.nodes[node_id].pos, before[node_id])

    def test_collapsed_group_copy_and_delete(self) -> None:
        root, left, right, leaf, _connections = self.make_branch()
        self.view.collapse_node(root.id)
        self.view._scene.clearSelection()
        self.view._nodes[root.id].setSelected(True)

        self.assertTrue(self.view.copy_selected())
        payload = json.loads(QApplication.clipboard().text())
        self.assertEqual(len(payload["nodes"]), 4)
        self.assertEqual(len(payload["connections"]), 3)

        self.view.delete_selected()
        if _APP is not None:
            _APP.processEvents()
        self.assertFalse(self.graph.nodes)
        self.assertFalse(self.view._nodes)

        self.undo.undo()
        if _APP is not None:
            _APP.processEvents()
        self.assertEqual(
            set(self.graph.nodes),
            {root.id, left.id, right.id, leaf.id},
        )
        self.assertEqual(len(self.graph.connections), 3)
        self.assertEqual(len(self.view._nodes), 4)
        self.assertEqual(len(self.view._connections), 3)

    # ------------------------------------------------------------------ 拖放
    def test_mime_operator_drop_creates_node(self) -> None:
        mime = QMimeData()
        mime.setData(MIME_OPERATOR, json.dumps({"id": "test.blur"}).encode("utf-8"))
        event = QDropEvent(
            QPointF(200.0, 150.0),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        before = set(self.graph.nodes)
        self.view.dropEvent(event)
        self.assertTrue(event.isAccepted())
        created = [node_id for node_id in self.graph.nodes if node_id not in before]
        self.assertEqual(len(created), 1)
        node = self.graph.nodes[created[0]]
        self.assertEqual(node.op_id, "test.blur")
        self.assertEqual(node.params, param_defaults(BLUR_SPEC))
        self.assertIn(node.id, self.view._nodes)
        self.assertEqual(self.view.selected_node_id(), node.id)

    def test_port_drag_creates_connection(self) -> None:
        first = self.add_node(pos=(0.0, 0.0))
        second = self.add_node(pos=(500.0, 500.0))
        self._show_and_focus_node(first)
        out_port = self.view._nodes[first.id].output_ports["image"]
        in_port = self.view._nodes[second.id].input_ports["image"]
        start = self.view.mapFromScene(out_port.center_scene_pos())
        end = self.view.mapFromScene(in_port.center_scene_pos())
        middle = QPoint((start.x() + end.x()) // 2, (start.y() + end.y()) // 2)

        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            start,
        )
        QTest.mouseMove(self.view.viewport(), middle)
        QTest.mouseMove(self.view.viewport(), end)
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            end,
        )
        if _APP is not None:
            _APP.processEvents()

        self.assertEqual(len(self.graph.connections), 1)
        connection = next(iter(self.graph.connections.values()))
        self.assertEqual(connection.src_node, first.id)
        self.assertEqual(connection.dst_node, second.id)
        self.assertIn(connection.id, self.view._connections)

        self.undo.undo()
        self.assertFalse(self.graph.connections)
        self.assertFalse(self.view._connections)

    def test_output_drag_to_blank_creates_output_node(self) -> None:
        """从输出端口拖到空白处：自动创建输出节点并连接，整次一步撤销。"""

        source = self.add_source("源", pos=(0.0, 0.0))
        self._show_and_focus_node(source)
        out_port = self.view._nodes[source.id].output_ports["image"]
        start = self.view.mapFromScene(out_port.center_scene_pos())
        drop_scene = out_port.center_scene_pos() + QPointF(220.0, 160.0)
        end = self.view.mapFromScene(drop_scene)
        self.assertIsNone(self.view.itemAt(end))
        middle = QPoint((start.x() + end.x()) // 2, (start.y() + end.y()) // 2)

        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            start,
        )
        QTest.mouseMove(self.view.viewport(), middle)
        QTest.mouseMove(self.view.viewport(), end)
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            end,
        )
        if _APP is not None:
            _APP.processEvents()

        new_nodes = [node for node in self.graph.nodes.values() if node.id != source.id]
        self.assertEqual(len(new_nodes), 1)
        node = new_nodes[0]
        self.assertEqual(node.op_id, "output.display")
        self.assertEqual(len(self.graph.connections), 1)
        connection = next(iter(self.graph.connections.values()))
        self.assertEqual((connection.src_node, connection.src_port), (source.id, "image"))
        self.assertEqual((connection.dst_node, connection.dst_port), (node.id, "data"))
        self.assertIn(node.id, self.view._nodes)
        self.assertEqual(self.view.selected_node_id(), node.id)
        # 屏幕坐标取整会让落点有不到 1px 的偏差。
        self.assertAlmostEqual(node.pos[0], drop_scene.x() - NodeItem.WIDTH / 2.0, delta=1.0)
        self.assertAlmostEqual(node.pos[1], drop_scene.y(), delta=1.0)

        # 创建节点 + 连线属于同一步撤销。
        self.assertEqual(self.undo.count(), 1)
        self.undo.undo()
        self.assertNotIn(node.id, self.graph.nodes)
        self.assertFalse(self.graph.connections)
        self.undo.redo()
        self.assertIn(node.id, self.graph.nodes)
        self.assertEqual(len(self.graph.connections), 1)

    def test_port_drag_onto_node_body_does_not_create_output_node(self) -> None:
        """拖到已有节点上（不是输入端口）时不应自动创建输出节点。"""

        first = self.add_node(pos=(0.0, 0.0))
        second = self.add_node(pos=(500.0, 500.0))
        self._show_and_focus_node(first)
        out_port = self.view._nodes[first.id].output_ports["image"]
        start = self.view.mapFromScene(out_port.center_scene_pos())
        end_scene = self.view._nodes[second.id].sceneBoundingRect().center()
        end = self.view.mapFromScene(end_scene)
        middle = QPoint((start.x() + end.x()) // 2, (start.y() + end.y()) // 2)

        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            start,
        )
        QTest.mouseMove(self.view.viewport(), middle)
        QTest.mouseMove(self.view.viewport(), end)
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            end,
        )
        if _APP is not None:
            _APP.processEvents()

        self.assertEqual(len(self.graph.nodes), 2)
        self.assertFalse(self.graph.connections)

    def test_drag_node_near_edge_auto_scrolls_canvas(self) -> None:
        """拖动节点到视口边缘时，画布应持续朝该方向滚动。"""

        node = self.add_node(pos=(0.0, 0.0))
        self.view.resize(900, 600)
        self.view.show()
        if _APP is not None:
            _APP.processEvents()
        item = self.view._nodes[node.id]
        self.view.centerOn(item.sceneBoundingRect().center())
        if _APP is not None:
            _APP.processEvents()
        hbar = self.view.horizontalScrollBar()
        h_before = hbar.value()
        start = self.view.mapFromScene(item.sceneBoundingRect().center())
        edge = QPoint(self.view.viewport().width() - 4, start.y())

        QTest.mousePress(
            self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start
        )
        QTest.mouseMove(self.view.viewport(), QPoint(start.x() + 20, start.y()))
        QTest.qWait(120)
        self.assertTrue(self.view._edge_scroll_timer.isActive())
        self.assertEqual(hbar.value(), h_before, "鼠标在画布中间时不应自动滚动")

        QTest.mouseMove(self.view.viewport(), edge)
        QTest.qWait(180)
        self.assertGreater(hbar.value(), h_before, "鼠标靠近右边缘时应自动向右滚动")

        QTest.mouseRelease(
            self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, edge
        )
        if _APP is not None:
            _APP.processEvents()
        self.assertFalse(self.view._edge_scroll_timer.isActive())
        self.assertTrue(self.undo.canUndo())

    def test_drag_beyond_canvas_keeps_auto_scrolling(self) -> None:
        """鼠标拖出画布范围后，只要左键未松开就继续自动滚动。"""

        node = self.add_node(pos=(0.0, 0.0))
        self.view.resize(900, 600)
        self.view.show()
        if _APP is not None:
            _APP.processEvents()
        item = self.view._nodes[node.id]
        self.view.centerOn(item.sceneBoundingRect().center())
        if _APP is not None:
            _APP.processEvents()
        hbar = self.view.horizontalScrollBar()
        start = self.view.mapFromScene(item.sceneBoundingRect().center())
        outside = QPoint(self.view.viewport().width() + 120, start.y())

        QTest.mousePress(
            self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start
        )
        QTest.mouseMove(self.view.viewport(), QPoint(start.x() + 20, start.y()))
        QTest.mouseMove(self.view.viewport(), outside)
        h_before = hbar.value()
        QTest.qWait(200)
        self.assertGreater(hbar.value(), h_before, "鼠标在画布外仍应继续滚动")
        self.assertTrue(self.view._edge_scroll_timer.isActive())

        # 回到画布中间（仍在拖动）后不再继续滚动。
        center = QPoint(self.view.viewport().width() // 2, self.view.viewport().height() // 2)
        QTest.mouseMove(self.view.viewport(), center)
        h_paused = hbar.value()
        QTest.qWait(120)
        self.assertEqual(hbar.value(), h_paused)

        QTest.mouseRelease(
            self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, center
        )
        if _APP is not None:
            _APP.processEvents()
        self.assertFalse(self.view._edge_scroll_timer.isActive())

    def test_connection_drag_near_edge_auto_scrolls_canvas(self) -> None:
        """拖动连线到视口边缘时画布自动滚动，临时连线末端跟随光标。"""

        source = self.add_source("源", pos=(0.0, 0.0))
        self.view.resize(900, 600)
        self.view.show()
        if _APP is not None:
            _APP.processEvents()
        item = self.view._nodes[source.id]
        self.view.centerOn(item.sceneBoundingRect().center())
        if _APP is not None:
            _APP.processEvents()
        out_port = item.output_ports["image"]
        start = self.view.mapFromScene(out_port.center_scene_pos())
        hbar = self.view.horizontalScrollBar()
        h_before = hbar.value()
        edge = QPoint(self.view.viewport().width() - 4, start.y() + 30)

        QTest.mousePress(
            self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start
        )
        QTest.mouseMove(self.view.viewport(), QPoint(start.x() + 20, start.y() + 30))
        QTest.mouseMove(self.view.viewport(), edge)
        QTest.qWait(180)
        self.assertTrue(self.view._scene.dragging_connection)
        self.assertGreater(hbar.value(), h_before)

        temp = self.view._scene._temp_connection
        self.assertIsNotNone(temp)
        cursor_scene = self.view.mapToScene(edge)
        self.assertLess(
            (temp.path().pointAtPercent(1.0) - cursor_scene).manhattanLength(),
            2.0,
            "自动滚动后临时连线末端应仍停在光标处",
        )

        # 松手在源节点上取消连线，避免落点触发“空白处自动建输出节点”。
        release_pos = self.view.mapFromScene(item.sceneBoundingRect().center())
        QTest.mouseRelease(
            self.view.viewport(), Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, release_pos
        )
        if _APP is not None:
            _APP.processEvents()
        self.assertFalse(self.view._edge_scroll_timer.isActive())
        self.assertFalse(self.view._scene.dragging_connection)
        self.assertEqual(len(self.graph.nodes), 1)
        self.assertFalse(self.graph.connections)

    def test_add_operator_at_inserts_on_edge_and_layouts(self) -> None:
        first = self.add_node("test.blur", pos=(0.0, 0.0))
        second = self.add_node("test.blur", pos=(0.0, 320.0))
        connection = self.connect(first, second)
        self.view._update_scene_rect()
        connection_item = self.view._connections[connection.id]
        midpoint = connection_item.path().pointAtPercent(0.5)

        new_node_id = self.view.add_operator_at("test.blur", midpoint)
        self.assertIsNotNone(new_node_id)
        self.assertIn(new_node_id, self.graph.nodes)
        self.assertNotIn(connection.id, self.graph.connections)
        self.assertEqual(self.graph.successors(first.id), [new_node_id])
        self.assertEqual(self.graph.predecessors(second.id), [new_node_id])
        self.assertEqual(len(self.graph.connections), 2)
        # 自动布局后节点位置应发生变化
        self.assertNotEqual(self.graph.nodes[second.id].pos, (0.0, 320.0))
        # 插入节点 + 自动布局必须合并为一步撤销。
        self.assertEqual(self.undo.count(), 1)

        self.undo.undo()
        self.assertNotIn(new_node_id, self.graph.nodes)
        self.assertIn(connection.id, self.graph.connections)
        self.assertEqual(self.graph.nodes[first.id].pos, (0.0, 0.0))
        self.assertEqual(self.graph.nodes[second.id].pos, (0.0, 320.0))

        self.undo.redo()
        self.assertIn(new_node_id, self.graph.nodes)
        self.assertNotIn(connection.id, self.graph.connections)
        self.assertEqual(len(self.graph.connections), 2)

    def test_add_operator_near_output_port_auto_connects(self) -> None:
        first = self.add_node(pos=(100.0, 100.0))
        output_item = self.view._nodes[first.id].output_ports["image"]
        output_center = output_item.center_scene_pos()
        before = set(self.graph.nodes)

        new_node_id = self.view.add_operator_at("test.blur", output_center)
        self.assertIsNotNone(new_node_id)
        self.assertEqual([node_id for node_id in self.graph.nodes if node_id not in before], [new_node_id])

        new_item = self.view._nodes[new_node_id]
        input_center = new_item.input_ports["image"].center_scene_pos()
        # 新节点放在已有输出端口下方，端口水平对齐且垂直间距约 48px。
        self.assertAlmostEqual(input_center.x(), output_center.x(), delta=1.0)
        self.assertAlmostEqual(input_center.y(), output_center.y() + 48.0, delta=1.0)

        self.assertEqual(len(self.graph.connections), 1)
        connection = next(iter(self.graph.connections.values()))
        self.assertEqual((connection.src_node, connection.src_port), (first.id, "image"))
        self.assertEqual((connection.dst_node, connection.dst_port), (new_node_id, "image"))

        # 添加节点 + 自动连线在同一个 undo 宏里，一次撤销全部回退。
        self.assertEqual(self.undo.count(), 1)
        self.undo.undo()
        self.assertNotIn(new_node_id, self.graph.nodes)
        self.assertFalse(self.graph.connections)
        self.undo.redo()
        self.assertIn(new_node_id, self.graph.nodes)
        self.assertEqual(len(self.graph.connections), 1)

    def test_add_operator_near_input_port_auto_connects(self) -> None:
        target = self.add_node(pos=(100.0, 400.0))
        input_item = self.view._nodes[target.id].input_ports["image"]
        input_center = input_item.center_scene_pos()
        before = set(self.graph.nodes)

        new_node_id = self.view.add_operator_at("test.blur", input_center)
        self.assertIsNotNone(new_node_id)
        self.assertEqual([node_id for node_id in self.graph.nodes if node_id not in before], [new_node_id])

        new_item = self.view._nodes[new_node_id]
        output_center = new_item.output_ports["image"].center_scene_pos()
        # 新节点放在已有输入端口上方，端口水平对齐且垂直间距约 48px。
        self.assertAlmostEqual(output_center.x(), input_center.x(), delta=1.0)
        self.assertAlmostEqual(output_center.y(), input_center.y() - 48.0, delta=1.0)

        self.assertEqual(len(self.graph.connections), 1)
        connection = next(iter(self.graph.connections.values()))
        self.assertEqual((connection.src_node, connection.src_port), (new_node_id, "image"))
        self.assertEqual((connection.dst_node, connection.dst_port), (target.id, "image"))

        self.assertEqual(self.undo.count(), 1)
        self.undo.undo()
        self.assertNotIn(new_node_id, self.graph.nodes)
        self.assertFalse(self.graph.connections)

    def test_add_operator_near_occupied_input_keeps_existing_connection(self) -> None:
        source = self.add_source("源", pos=(0.0, 0.0))
        target = self.add_node(pos=(0.0, 300.0))
        existing = self.connect(source, target)
        input_center = self.view._nodes[target.id].input_ports["image"].center_scene_pos()
        messages: list[str] = []
        self.view.status_message.connect(messages.append)

        new_node_id = self.view.add_operator_at("test.blur", input_center)
        self.assertIsNotNone(new_node_id)
        self.assertIn(new_node_id, self.graph.nodes)
        # 已占用的输入端口不能被替换：旧连线保留，也没有新增连线。
        self.assertEqual(list(self.graph.connections), [existing.id])
        self.assertEqual(self.graph.connections[existing.id].src_node, source.id)
        self.assertTrue(any("已有连线" in message for message in messages))
        # 自动连接失败后普通放置应避让目标节点，不能直接压在原节点上。
        self.assertFalse(
            self.view._nodes[new_node_id]
            .sceneBoundingRect()
            .intersects(self.view._nodes[target.id].sceneBoundingRect())
        )

    def test_node_drag_near_port_auto_connects_with_single_undo(self) -> None:
        source = self.add_source("源", pos=(0.0, 0.0))
        moved = self.add_node(pos=(500.0, 600.0))
        moved_item = self.view._nodes[moved.id]
        source_output = self.view._nodes[source.id].output_ports["image"]
        moved_input = moved_item.input_ports["image"]
        local = moved_input.center_scene_pos() - moved_item.pos()
        anchor = source_output.center_scene_pos() + QPointF(0.0, 10.0)

        moved_item.setSelected(True)
        self.view.begin_move_tracking()
        moved_item.setPos(anchor.x() - local.x(), anchor.y() - local.y())
        # 拖动过程中应高亮吸附端口。
        self.assertIsNotNone(self.view._pending_move_target)
        self.assertTrue(self.view._highlighted_ports)
        self.view.finish_move_tracking()

        self.assertEqual(len(self.graph.connections), 1)
        connection = next(iter(self.graph.connections.values()))
        self.assertEqual((connection.src_node, connection.src_port), (source.id, "image"))
        self.assertEqual((connection.dst_node, connection.dst_port), (moved.id, "image"))
        moved_pos = self.graph.nodes[moved.id].pos
        self.assertFalse(self.view._highlighted_ports)

        # 一次撤销同时回退移动 + 自动连线。
        self.assertEqual(self.undo.count(), 1)
        self.undo.undo()
        self.assertFalse(self.graph.connections)
        self.assertEqual(self.graph.nodes[moved.id].pos, (500.0, 600.0))
        self.undo.redo()
        self.assertEqual(len(self.graph.connections), 1)
        self.assertEqual(self.graph.nodes[moved.id].pos, moved_pos)

    def test_real_mouse_drag_near_port_auto_connects(self) -> None:
        """用真实 QMouseEvent 走一遍场景事件链，防止只测到内部方法。"""

        source = self.add_source("源", pos=(0.0, 0.0))
        moved = self.add_node(pos=(300.0, 300.0))
        self.view.fit_graph()
        self.view.show()
        if _APP is not None:
            _APP.processEvents()
        source_output = self.view._nodes[source.id].output_ports["image"]
        moved_item = self.view._nodes[moved.id]
        moved_input = moved_item.input_ports["image"]
        local = moved_input.center_scene_pos() - moved_item.pos()
        anchor = source_output.center_scene_pos()
        start = self.view.mapFromScene(moved_item.sceneBoundingRect().center())
        end_scene = QPointF(
            anchor.x() - local.x() + moved_item.size().width() / 2.0,
            anchor.y() - local.y() + moved_item.size().height() / 2.0,
        )
        end = self.view.mapFromScene(end_scene)

        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            start,
        )
        QTest.mouseMove(self.view.viewport(), QPoint((start.x() + end.x()) // 2, (start.y() + end.y()) // 2))
        QTest.mouseMove(self.view.viewport(), end)
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            end,
        )
        if _APP is not None:
            _APP.processEvents()

        self.assertEqual(len(self.graph.connections), 1)
        connection = next(iter(self.graph.connections.values()))
        self.assertEqual(connection.src_node, source.id)
        self.assertEqual(connection.dst_node, moved.id)
        self.assertEqual(self.undo.count(), 1)
        self.undo.undo()
        self.assertFalse(self.graph.connections)
        self.assertEqual(self.graph.nodes[moved.id].pos, (300.0, 300.0))

    def test_node_drag_onto_connection_inserts_with_single_undo(self) -> None:
        first = self.add_node("test.blur", pos=(0.0, 0.0))
        second = self.add_node("test.blur", pos=(0.0, 420.0))
        original = self.connect(first, second)
        self.view._update_scene_rect()
        connection_item = self.view._connections[original.id]
        midpoint = connection_item.path().pointAtPercent(0.5)

        moved = self.add_node(pos=(700.0, 700.0))
        moved_item = self.view._nodes[moved.id]
        size = moved_item.size()
        moved_item.setSelected(True)
        self.view.begin_move_tracking()
        moved_item.setPos(midpoint.x() - size.width() / 2.0, midpoint.y() - size.height() / 2.0)
        self.assertIsNotNone(self.view._pending_move_target)
        # 拖动到连线上时连线高亮，释放后清理。
        self.assertIsNotNone(self.view._highlighted_connection)
        self.assertTrue(self.view._highlighted_connection.highlighted)
        self.view.finish_move_tracking()
        self.assertIsNone(self.view._highlighted_connection)

        self.assertNotIn(original.id, self.graph.connections)
        self.assertEqual(len(self.graph.connections), 2)
        upstream = self.graph.input_connection(moved.id, "image")
        self.assertIsNotNone(upstream)
        self.assertEqual(upstream.src_node, first.id)
        self.assertEqual(self.graph.successors(first.id), [moved.id])
        self.assertEqual(self.graph.predecessors(second.id), [moved.id])
        self.assertEqual(self.graph.outgoing(moved.id)[0].dst_node, second.id)
        moved_pos = self.graph.nodes[moved.id].pos

        # 移动 + 断线 + 两条新连线同属一个 undo 宏。
        self.assertEqual(self.undo.count(), 1)
        self.undo.undo()
        self.assertIn(original.id, self.graph.connections)
        self.assertEqual(len(self.graph.connections), 1)
        self.assertEqual(self.graph.nodes[moved.id].pos, (700.0, 700.0))
        self.undo.redo()
        self.assertNotIn(original.id, self.graph.connections)
        self.assertEqual(self.graph.nodes[moved.id].pos, moved_pos)

    def test_library_drag_move_highlights_port_and_leave_clears(self) -> None:
        from PySide6.QtGui import QDragEnterEvent, QDragLeaveEvent, QDragMoveEvent

        first = self.add_node(pos=(100.0, 100.0))
        output = self.view._nodes[first.id].output_ports["image"]
        mime = QMimeData()
        mime.setData(MIME_OPERATOR, json.dumps({"id": "test.blur"}).encode("utf-8"))
        point = self.view.mapFromScene(output.center_scene_pos())
        self.view.dragEnterEvent(
            QDragEnterEvent(
                point,
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        event = QDragMoveEvent(
            point,
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.view.dragMoveEvent(event)
        self.assertTrue(event.isAccepted())
        self.assertEqual(self.view._highlighted_ports, [output])

        self.view.dragLeaveEvent(QDragLeaveEvent())
        self.assertEqual(self.view._highlighted_ports, [])

    def test_library_drag_over_edge_highlights_and_inserts_in_one_undo(self) -> None:
        from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent

        first = self.add_node(pos=(0.0, 0.0))
        second = self.add_node(pos=(0.0, 420.0))
        original = self.connect(first, second)
        self.view._update_scene_rect()
        midpoint = self.view._connections[original.id].path().pointAtPercent(0.5)
        point = self.view.mapFromScene(midpoint)
        mime = QMimeData()
        mime.setData(MIME_OPERATOR, json.dumps({"id": "test.merge"}).encode("utf-8"))

        self.view.dragEnterEvent(
            QDragEnterEvent(
                point,
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        move_event = QDragMoveEvent(
            point,
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.view.dragMoveEvent(move_event)
        self.assertTrue(move_event.isAccepted())
        self.assertIsNotNone(self.view._highlighted_connection)

        drop = QDropEvent(
            QPointF(point),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.view.dropEvent(drop)
        self.assertTrue(drop.isAccepted())
        self.assertIsNone(self.view._highlighted_connection)
        self.assertNotIn(original.id, self.graph.connections)
        self.assertEqual(len(self.graph.nodes), 3)
        self.assertEqual(len(self.graph.connections), 2)
        # 插入 + 自动布局同属一步撤销。
        self.assertEqual(self.undo.count(), 1)
        self.undo.undo()
        self.assertEqual(len(self.graph.nodes), 2)
        self.assertIn(original.id, self.graph.connections)

    def test_library_drag_invalid_operator_does_not_highlight_edge(self) -> None:
        from PySide6.QtGui import QDragEnterEvent, QDragMoveEvent

        first = self.add_node(pos=(0.0, 0.0))
        second = self.add_node(pos=(0.0, 420.0))
        self.connect(first, second)
        self.view._update_scene_rect()
        midpoint = next(iter(self.view._connections.values())).path().pointAtPercent(0.5)
        point = self.view.mapFromScene(midpoint)
        mime = QMimeData()
        mime.setData(MIME_OPERATOR, json.dumps({"id": "not.exists"}).encode("utf-8"))

        self.view.dragEnterEvent(
            QDragEnterEvent(
                point,
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        move_event = QDragMoveEvent(
            point,
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.view.dragMoveEvent(move_event)
        # 没有注册的算子无法插入连线，不应高亮误导。
        self.assertIsNone(self.view._highlighted_connection)

    def test_image_drop_emits_paths(self) -> None:
        from PySide6.QtCore import QUrl

        mime = QMimeData()
        mime.setUrls([QUrl.fromLocalFile("C:/tmp/示例.png")])
        event = QDropEvent(
            QPointF(50.0, 50.0),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        captured: list[list[str]] = []
        self.view.images_dropped.connect(captured.append)
        self.view.dropEvent(event)
        self.assertTrue(event.isAccepted())
        self.assertEqual(len(captured), 1)
        self.assertTrue(captured[0][0].endswith("示例.png"))
        self.assertFalse(self.graph.nodes)

    # ------------------------------------------------------------ 结果与状态
    def test_notify_result_thumbnail_and_error(self) -> None:
        node = self.add_node(pos=(30.0, 30.0))
        item = self.view._nodes[node.id]
        self.assertIsNone(item.thumbnail_pixmap)
        base_height = item.size().height()

        image = ImageData(np.zeros((24, 36, 3), dtype=np.uint8), "结果")
        result = NodeResult(node.id, {"image": image}, 12.5, node.revision)
        self.view.notify_result(node.id, result)
        self.assertAlmostEqual(item.elapsed_ms, 12.5)
        self.assertIsNotNone(item.thumbnail_pixmap)
        self.assertFalse(item.thumbnail_pixmap.isNull())
        self.assertGreater(item.size().height(), base_height)

        self.view.notify_error(node.id, "算子执行失败")
        self.assertEqual(item.error_message, "算子执行失败")
        self.assertIn("算子执行失败", item.toolTip())

        self.view.notify_error(node.id, None)
        self.assertIsNone(item.error_message)

        self.view.set_thumbnails_enabled(False)
        self.assertIsNone(item.thumbnail_pixmap)
        self.assertLess(item.size().height(), base_height + 1.0)

        self.view.set_thumbnails_enabled(True)
        self.assertIsNotNone(item.thumbnail_pixmap)

    def test_note_card_shows_text_instead_of_result_data(self) -> None:
        node = self.add_node(
            "note.sticky",
            params={"text": "第一行便签\n第二行"},
            pos=(20.0, 20.0),
        )
        item = self.item(node)
        self.assertTrue(item.is_note_card)
        self.assertEqual(item.canvas_text, "第一行便签\n第二行")
        self.assertIsNone(item.thumbnail_pixmap)
        self.assertAlmostEqual(item.size().width(), NodeItem.NOTE_WIDTH)
        height = item.size().height()

        # 便签不显示结果数据：即使输出一张图像也不生成缩略图，高度只由文字决定。
        image = ImageData(np.zeros((24, 36, 3), dtype=np.uint8))
        self.view.notify_result(
            node.id, NodeResult(node.id, {"data": image}, 5.0, node.revision)
        )
        self.assertIsNone(item.thumbnail_pixmap)
        self.assertEqual(item.size().height(), height)
        self.assertIn("第一行便签", item.toolTip())

        # 参数变化会同步重排卡片：行数增加后卡片变高。
        longer = "\n".join(["便签内容"] * 10)
        self.graph.set_node_params(node.id, {"text": longer})
        self.assertEqual(item.canvas_text, longer)
        self.assertGreater(item.size().height(), height)

        # 长文字与占位文字都能正常绘制。
        canvas = QImage(320, 320, QImage.Format.Format_ARGB32)
        painter = QPainter(canvas)
        try:
            item.paint(painter, None, None)
        finally:
            painter.end()

    def test_note_card_placeholder_and_rename(self) -> None:
        node = self.add_node("note.sticky", pos=(0.0, 0.0))
        item = self.item(node)
        self.assertEqual(item.canvas_text, "")
        self.assertIn("空便签", item.toolTip())
        self.assertGreaterEqual(item.size().height(), NodeItem.NOTE_MIN_TEXT_H)

        self.graph.rename_node(node.id, "流程说明")
        self.assertIn("流程说明", item.toolTip())
        self.assertFalse(item.is_stale)

    def test_refresh_all_rebuilds_items(self) -> None:
        first = self.add_node("test.blur", pos=(0.0, 0.0))
        second = self.add_node("test.blur", pos=(0.0, 200.0))
        connection = self.connect(first, second)
        self.view.set_selected_node(first.id)

        self.view.refresh_all()
        self.assertEqual(len(self.view._nodes), 2)
        self.assertEqual(len(self.view._connections), 1)
        self.assertEqual(self.view.selected_node_id(), first.id)

    def test_set_selected_node(self) -> None:
        first = self.add_node(pos=(0.0, 0.0))
        second = self.add_node(pos=(0.0, 200.0))
        events: list[str] = []
        self.view.selection_changed.connect(events.append)

        self.view.set_selected_node(second.id)
        self.assertEqual(self.view.selected_node_id(), second.id)
        self.assertEqual(events, [])
        self.view.set_selected_node(None)
        self.assertIsNone(self.view.selected_node_id())
        self.assertEqual(events, [])

    def test_node_changed_position_keeps_item_instance(self) -> None:
        node = self.add_node(pos=(10.0, 10.0))
        item = self.view._nodes[node.id]
        self.graph.set_node_pos(node.id, (77.0, 88.0))
        self.assertIs(self.view._nodes[node.id], item)
        self.assertEqual(item.pos(), QPointF(77.0, 88.0))

    def test_set_registry_without_operator_still_renders(self) -> None:
        node = self.add_node("test.blur", pos=(0.0, 0.0))
        self.view.set_registry(FakeRegistry())
        self.assertIn(node.id, self.view._nodes)
        item = self.view._nodes[node.id]
        self.assertIsNone(item.spec)
        self.assertEqual(set(item.input_ports), {"image"})
        self.assertEqual(set(item.output_ports), {"image"})
        self.assertGreater(item.size().height(), 0.0)
        item.update()  # 不应崩溃

    # -------------------------------------------------------------- 画布导航
    def _prepare_navigation_canvas(self) -> NodeModel:
        """单个节点 + 适应视图：内容放得下时，左键拖动空白处仍应能平移画布。"""

        node = self.add_node(pos=(0.0, 0.0))
        self.view.resize(1000, 700)
        self.view.show()
        if _APP is not None:
            _APP.processEvents()
        self.view.centerOn(self.view._nodes[node.id].sceneBoundingRect().center())
        if _APP is not None:
            _APP.processEvents()
        return node

    def _send_wheel(
        self,
        angle: QPoint,
        modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
        pixel: QPoint | None = None,
    ) -> None:
        position = QPointF(self.view.viewport().rect().center())
        event = QWheelEvent(
            position,
            self.view.viewport().mapToGlobal(position.toPoint()),
            QPoint(0, 0) if pixel is None else pixel,
            angle,
            Qt.MouseButton.NoButton,
            modifiers,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(self.view.viewport(), event)
        if _APP is not None:
            _APP.processEvents()

    def test_canvas_scrollbars_are_hidden(self) -> None:
        self._prepare_navigation_canvas()
        self.assertFalse(self.view.horizontalScrollBar().isVisible())
        self.assertFalse(self.view.verticalScrollBar().isVisible())

    def test_shift_wheel_scrolls_canvas_horizontally(self) -> None:
        self._prepare_navigation_canvas()
        hbar = self.view.horizontalScrollBar()
        vbar = self.view.verticalScrollBar()
        h_before, v_before = hbar.value(), vbar.value()

        self._send_wheel(QPoint(0, 120), Qt.KeyboardModifier.ShiftModifier)
        self.assertLess(hbar.value(), h_before, "Shift+滚轮向下应向左滚动")
        self.assertEqual(vbar.value(), v_before, "Shift+滚轮不应再纵向滚动")

        self._send_wheel(QPoint(0, -120), Qt.KeyboardModifier.ShiftModifier)
        self.assertEqual(hbar.value(), h_before)
        self.assertEqual(vbar.value(), v_before)

    def test_shift_wheel_uses_pixel_delta(self) -> None:
        self._prepare_navigation_canvas()
        hbar = self.view.horizontalScrollBar()
        h_before = hbar.value()
        self._send_wheel(
            QPoint(0, 0),
            Qt.KeyboardModifier.ShiftModifier,
            QPoint(0, 40),
        )
        self.assertEqual(hbar.value(), h_before - 40)

    def test_plain_wheel_still_scrolls_canvas_vertically(self) -> None:
        self._prepare_navigation_canvas()
        hbar = self.view.horizontalScrollBar()
        vbar = self.view.verticalScrollBar()
        h_before, v_before = hbar.value(), vbar.value()

        self._send_wheel(QPoint(0, 120))
        self.assertLess(vbar.value(), v_before)
        self.assertEqual(hbar.value(), h_before)

    def test_ctrl_wheel_still_zooms_canvas(self) -> None:
        self._prepare_navigation_canvas()
        before = self.view.transform().m11()
        self._send_wheel(QPoint(0, 120), Qt.KeyboardModifier.ControlModifier)
        self.assertGreater(self.view.transform().m11(), before)

    def test_left_drag_blank_pans_canvas(self) -> None:
        node = self._prepare_navigation_canvas()
        hbar = self.view.horizontalScrollBar()
        vbar = self.view.verticalScrollBar()
        h_before, v_before = hbar.value(), vbar.value()
        node_pos = self.graph.nodes[node.id].pos
        start = QPoint(40, 40)
        end = QPoint(160, 120)
        viewport = self.view.viewport()

        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        self.assertTrue(self.view._panning)
        QTest.mouseMove(viewport, end)
        QTest.mouseRelease(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
        if _APP is not None:
            _APP.processEvents()

        self.assertFalse(self.view._panning)
        self.assertEqual(hbar.value(), h_before - 120)
        self.assertEqual(vbar.value(), v_before - 80)
        # 拖动画布不应移动节点。
        self.assertEqual(self.graph.nodes[node.id].pos, node_pos)

    def test_blank_click_clears_selection_without_panning(self) -> None:
        node = self._prepare_navigation_canvas()
        self.view.set_selected_node(node.id)
        hbar = self.view.horizontalScrollBar()
        vbar = self.view.verticalScrollBar()
        h_before, v_before = hbar.value(), vbar.value()

        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(20, 20),
        )
        if _APP is not None:
            _APP.processEvents()

        self.assertIsNone(self.view.selected_node_id())
        self.assertEqual((hbar.value(), vbar.value()), (h_before, v_before))

    def test_ctrl_drag_blank_keeps_rubber_band_selection(self) -> None:
        node = self._prepare_navigation_canvas()
        item = self.view._nodes[node.id]
        center = self.view.mapFromScene(item.sceneBoundingRect().center())
        start = QPoint(center.x() - 180, center.y() - 140)
        end = QPoint(center.x() + 180, center.y() + 140)
        viewport = self.view.viewport()
        self.view._scene.clearSelection()

        QTest.mousePress(viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, start)
        self.assertFalse(self.view._panning)
        QTest.mouseMove(viewport, end)
        QTest.mouseRelease(
            viewport, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.ControlModifier, end
        )
        if _APP is not None:
            _APP.processEvents()

        self.assertEqual(self.view.selected_node_id(), node.id)

    # ------------------------------------------------------------------ 点击
    def _show_and_focus_node(self, node: NodeModel):
        if _APP is not None:
            _APP.processEvents()
        self.view.fit_graph()
        self.view.show()
        self.view.setFocus()
        if _APP is not None:
            _APP.processEvents()
        item = self.view._nodes[node.id]
        return self.view.mapFromScene(item.sceneBoundingRect().center())

    def test_single_click_emits_selection_after_delay(self) -> None:
        node = self.add_node(pos=(0.0, 0.0))
        self.view.set_selected_node(None)
        point = self._show_and_focus_node(node)

        events: list[str] = []
        self.view.selection_changed.connect(events.append)
        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            point,
        )
        self.assertEqual(events, [])
        QTest.qWait(SINGLE_CLICK_DELAY_MS + 150)
        self.assertEqual(events, [node.id])

    def test_click_already_selected_node_emits_again(self) -> None:
        node = self.add_node(pos=(0.0, 0.0))
        point = self._show_and_focus_node(node)
        self.view.set_selected_node(node.id)

        events: list[str] = []
        self.view.selection_changed.connect(events.append)
        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            point,
        )
        self.assertEqual(events, [])
        QTest.qWait(SINGLE_CLICK_DELAY_MS + 150)
        self.assertEqual(events, [node.id])

    def test_double_click_cancels_selection_and_activates(self) -> None:
        node = self.add_node(pos=(0.0, 0.0))
        self.view.set_selected_node(None)
        point = self._show_and_focus_node(node)

        selection_events: list[str] = []
        activation_events: list[str] = []
        self.view.selection_changed.connect(selection_events.append)
        self.view.node_activated.connect(activation_events.append)

        QTest.mouseDClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            point,
        )
        QTest.qWait(SINGLE_CLICK_DELAY_MS + 150)
        self.assertEqual(activation_events, [node.id])
        self.assertEqual(selection_events, [])


    # ------------------------------------------------------------ 算子组
    def add_group(
        self,
        pos: tuple[float, float] = (400.0, 300.0),
        parent_group: str | None = None,
        title: str = "算子组",
    ) -> str:
        return self.view.add_group_at(QPointF(*pos), parent_group=parent_group, title=title)

    def add_child(
        self,
        group_id: str,
        op_id: str = "test.blur",
        pos: tuple[float, float] = (0.0, 0.0),
        params: dict | None = None,
    ) -> NodeModel:
        spec = self.registry.spec(op_id)
        assert spec is not None
        node = NodeModel(
            id=new_id("n"),
            op_id=op_id,
            title=spec.name,
            pos=pos,
            params=dict(param_defaults(spec)) if params is None else dict(params),
            parent_group=group_id,
        )
        self.graph.add_node(node)
        return node

    def add_missing_node(
        self,
        pos: tuple[float, float] = (0.0, 0.0),
        title: str = "失效算子",
    ) -> NodeModel:
        node = NodeModel(id=new_id("n"), op_id="missing.op", title=title, pos=pos)
        self.graph.add_node(node)
        return node

    def drag_port_to(self, source, target_scene_pos: QPointF) -> None:
        """从端口拖到场景坐标并松手，用于连线交互测试。"""

        self.view.fit_graph()
        self.view.show()
        self.view.setFocus()
        if _APP is not None:
            _APP.processEvents()
        start = self.view.mapFromScene(source.center_scene_pos())
        end = self.view.mapFromScene(target_scene_pos)
        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            start,
        )
        for step in range(1, 6):
            point = QPoint(
                int(start.x() + (end.x() - start.x()) * step / 5),
                int(start.y() + (end.y() - start.y()) * step / 5),
            )
            QTest.mouseMove(self.view.viewport(), point)
            if _APP is not None:
                _APP.processEvents()
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            end,
        )
        if _APP is not None:
            _APP.processEvents()

    def drag_node_to(self, node_id: str, target_scene_pos: QPointF) -> None:
        """拖动一个节点的卡片到场景坐标并松手（触发所属算子组变化）。"""

        self.view.fit_graph()
        self.view.show()
        self.view.setFocus()
        if _APP is not None:
            _APP.processEvents()
        item = self.view._nodes[node_id]
        start = self.view.mapFromScene(item.sceneBoundingRect().center())
        end = self.view.mapFromScene(target_scene_pos)
        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            start,
        )
        for step in range(1, 7):
            point = QPoint(
                int(start.x() + (end.x() - start.x()) * step / 6),
                int(start.y() + (end.y() - start.y()) * step / 6),
            )
            QTest.mouseMove(self.view.viewport(), point)
            if _APP is not None:
                _APP.processEvents()
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            end,
        )
        if _APP is not None:
            _APP.processEvents()

    def test_add_group_at_creates_default_ports_and_geometry(self) -> None:
        group_id = self.add_group(pos=(400.0, 300.0))
        node = self.graph.nodes[group_id]
        self.assertTrue(node.is_group)
        self.assertIsInstance(node.group, GroupModel)
        self.assertEqual(len(node.group.inputs), 1)
        self.assertEqual(len(node.group.outputs), 1)
        self.assertIsNone(node.group.inputs[0].data_type)
        self.assertIsNone(node.group.outputs[0].data_type)

        item = self.view._nodes[group_id]
        self.assertIsInstance(item, GroupItem)
        self.assertIsNone(item._header_chip())  # 健康状态不再显示「待定」徽标
        self.assertAlmostEqual(item.size().width(), DEFAULT_GROUP_WIDTH, places=1)
        self.assertAlmostEqual(item.size().height(), DEFAULT_GROUP_HEIGHT, places=1)
        self.assertEqual(set(item.outer_input_ports), {"input_1"})
        self.assertEqual(set(item.outer_output_ports), {"output_1"})
        self.assertEqual(set(item.inner_output_ports), {"input_1"})
        self.assertEqual(set(item.inner_input_ports), {"output_1"})
        self.assertTrue(item.outer_input_ports["input_1"].pending)

        outer_in = item.port_item("input_1", True)
        inner_out = item.port_item("input_1", False)
        outer_out = item.port_item("output_1", False)
        inner_in = item.port_item("output_1", True)
        self.assertIsNotNone(outer_in)
        self.assertIsNotNone(inner_out)
        self.assertAlmostEqual(
            inner_out.scenePos().y() - outer_in.scenePos().y(), INNER_OFFSET, places=1
        )
        self.assertAlmostEqual(
            outer_out.scenePos().y() - inner_in.scenePos().y(), INNER_OFFSET, places=1
        )

    def test_blank_context_menu_has_add_group_action(self) -> None:
        with patch("app.graph.view.QMenu") as menu_cls:
            self.view.contextMenuEvent(_FakeContextMenuEvent(QPoint(10, 10)))
        actions = [call.args[0] for call in menu_cls.return_value.addAction.call_args_list if call.args]
        self.assertIn("添加算子组", actions)
        self.assertIn("粘贴节点", actions)

    def test_group_input_connection_binds_type_and_grows_pending_port(self) -> None:
        group_id = self.add_group()
        source = self.add_source("源", pos=(0.0, -400.0))
        self.view.connect_ports(source.id, "image", group_id, "input_1")

        group = self.graph.nodes[group_id].group
        self.assertEqual(group.inputs[0].data_type, DataType.IMAGE)
        self.assertEqual(len(group.inputs), 2)
        self.assertIsNone(group.inputs[1].data_type)
        self.assertEqual(len(self.graph.connections), 1)
        connection = next(iter(self.graph.connections.values()))
        self.assertIn(connection.id, self.view._connections)

        item = self.view._nodes[group_id]
        self.assertIn("input_2", item.outer_input_ports)
        self.assertTrue(item.outer_input_ports["input_2"].pending)
        self.assertFalse(item.outer_input_ports["input_1"].pending)

        self.undo.undo()
        group = self.graph.nodes[group_id].group
        self.assertEqual(len(group.inputs), 1)
        self.assertIsNone(group.inputs[0].data_type)
        self.assertFalse(self.graph.connections)

        self.undo.redo()
        group = self.graph.nodes[group_id].group
        self.assertEqual(group.inputs[0].data_type, DataType.IMAGE)
        self.assertEqual(len(group.inputs), 2)

    def test_group_output_inner_connection_binds_and_grows_pending(self) -> None:
        group_id = self.add_group(pos=(400.0, 300.0))
        group_pos = self.graph.nodes[group_id].pos
        child = self.add_child(group_id, pos=(group_pos[0] + 60.0, group_pos[1] + 100.0))
        self.view.connect_ports(child.id, "image", group_id, "output_1")

        group = self.graph.nodes[group_id].group
        self.assertEqual(group.outputs[0].data_type, DataType.IMAGE)
        self.assertEqual(len(group.outputs), 2)
        self.assertIsNone(group.outputs[1].data_type)
        item = self.view._nodes[group_id]
        self.assertIn("output_2", item.outer_output_ports)
        self.assertTrue(item.outer_output_ports["output_2"].pending)
        self.assertEqual(item.outer_output_ports["output_1"].data_type, DataType.IMAGE)

        sink = self.add_node(pos=(0.0, 900.0))
        self.view.connect_ports(group_id, "output_1", sink.id, "image")
        self.assertEqual(len(self.graph.connections), 2)
        self.assertEqual(self.graph.input_connection(sink.id, "image").src_node, group_id)

    def test_drag_from_group_input_port_connects_inner_node(self) -> None:
        group_id = self.add_group(pos=(400.0, 300.0))
        group_pos = self.graph.nodes[group_id].pos
        child = self.add_child(group_id, pos=(group_pos[0] + 60.0, group_pos[1] + 100.0))
        outer_in = self.view._nodes[group_id].outer_input_ports["input_1"]
        child_in = self.view._nodes[child.id].input_ports["image"]

        # 从组输入端口（外框上的圆点）直接拖到组内节点：应以内侧输出端口为起点。
        self.drag_port_to(outer_in, child_in.center_scene_pos())

        self.assertEqual(len(self.graph.connections), 1)
        connection = next(iter(self.graph.connections.values()))
        self.assertEqual((connection.src_node, connection.src_port), (group_id, "input_1"))
        self.assertEqual((connection.dst_node, connection.dst_port), (child.id, "image"))

    def test_drop_on_incompatible_input_reports_reason(self) -> None:
        group_id = self.add_group()
        group_pos = self.graph.nodes[group_id].pos
        child = self.add_child(group_id, pos=(group_pos[0] + 40.0, group_pos[1] + 80.0))
        top = self.add_node(pos=(1000.0, 1000.0))
        messages: list[str] = []
        self.view.status_message.connect(messages.append)

        self.drag_port_to(
            self.view._nodes[top.id].output_ports["image"],
            self.view._nodes[child.id].input_ports["image"].center_scene_pos(),
        )

        self.assertFalse(self.graph.connections)
        self.assertTrue(any("层级" in message or "算子组" in message for message in messages))

    def test_drop_connection_on_node_body_reports_hint(self) -> None:
        source = self.add_node(pos=(0.0, 0.0))
        target = self.add_node(pos=(0.0, 500.0))
        messages: list[str] = []
        self.view.status_message.connect(messages.append)

        self.drag_port_to(
            self.view._nodes[source.id].output_ports["image"],
            self.view._nodes[target.id].sceneBoundingRect().center(),
        )

        self.assertFalse(self.graph.connections)
        self.assertTrue(any("输入端口" in message for message in messages))

    def test_click_group_input_port_does_not_emit_hint(self) -> None:
        group_id = self.add_group()
        outer_in = self.view._nodes[group_id].outer_input_ports["input_1"]
        messages: list[str] = []
        self.view.status_message.connect(messages.append)

        self.view.fit_graph()
        self.view.show()
        if _APP is not None:
            _APP.processEvents()
        point = self.view.mapFromScene(outer_in.center_scene_pos())
        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            point,
        )
        if _APP is not None:
            _APP.processEvents()

        self.assertFalse(any("输入端口" in message for message in messages))

    def _chain_nodes(self) -> tuple[NodeModel, NodeModel, NodeModel]:
        source = self.add_node(pos=(0.0, 0.0))
        middle = self.add_node(pos=(0.0, 300.0))
        sink = self.add_node(pos=(0.0, 600.0))
        self.connect(source, middle)
        self.connect(middle, sink)
        return source, middle, sink

    def _edge_set(self) -> set[tuple[str, str, str, str]]:
        return {
            (connection.src_node, connection.src_port, connection.dst_node, connection.dst_port)
            for connection in self.graph.connections.values()
        }

    def test_drag_node_into_group_reconnects_through_ports(self) -> None:
        source, middle, sink = self._chain_nodes()
        group_id = self.add_group(pos=(800.0, 300.0))
        before = self._edge_set()

        self.drag_node_to(middle.id, self.view._nodes[group_id].sceneBoundingRect().center())

        self.assertEqual(self.graph.nodes[middle.id].parent_group, group_id)
        edges = self._edge_set()
        self.assertEqual(len(edges), 4)
        input_key = next(
            dst_port
            for src_node, _src_port, dst_node, dst_port in edges
            if src_node == source.id and dst_node == group_id
        )
        output_key = next(
            dst_port
            for src_node, _src_port, dst_node, dst_port in edges
            if src_node == middle.id and dst_node == group_id
        )
        self.assertIn((source.id, "image", group_id, input_key), edges)
        self.assertIn((group_id, input_key, middle.id, "image"), edges)
        self.assertIn((middle.id, "image", group_id, output_key), edges)
        self.assertIn((group_id, output_key, sink.id, "image"), edges)

        self.undo.undo()
        self.assertIsNone(self.graph.nodes[middle.id].parent_group)
        self.assertEqual(self._edge_set(), before)

    def test_drag_node_into_empty_group_reuses_pending_ports(self) -> None:
        source, middle, sink = self._chain_nodes()
        group_id = self.add_group(pos=(800.0, 300.0))

        self.drag_node_to(middle.id, self.view._nodes[group_id].sceneBoundingRect().center())

        group = self.graph.nodes[group_id].group
        self.assertEqual([port.key for port in group.inputs], ["input_1", "input_2"])
        self.assertEqual(group.inputs[0].data_type, DataType.IMAGE)
        self.assertTrue(group.inputs[1].pending)
        self.assertEqual([port.key for port in group.outputs], ["output_1", "output_2"])
        self.assertEqual(group.outputs[0].data_type, DataType.IMAGE)
        self.assertTrue(group.outputs[1].pending)
        # 第一个待定端口必须被使用，不能跳过它再追加新端口。
        self.assertEqual(
            self._edge_set(),
            {
                (source.id, "image", group_id, "input_1"),
                (group_id, "input_1", middle.id, "image"),
                (middle.id, "image", group_id, "output_1"),
                (group_id, "output_1", sink.id, "image"),
            },
        )
        item = self.view._nodes[group_id]
        self.assertEqual(list(item.outer_input_ports), ["input_1", "input_2"])
        self.assertFalse(item.outer_input_ports["input_1"].pending)
        self.assertTrue(item.outer_input_ports["input_2"].pending)
        self.assertEqual(list(item.outer_output_ports), ["output_1", "output_2"])
        self.assertTrue(item.outer_output_ports["output_2"].pending)

        # 撤销拖入后端口恢复为初始的单个待定端口；重做再次复用它们。
        self.undo.undo()
        group = self.graph.nodes[group_id].group
        self.assertEqual([port.key for port in group.inputs], ["input_1"])
        self.assertTrue(group.inputs[0].pending)
        self.assertEqual([port.key for port in group.outputs], ["output_1"])
        self.assertTrue(group.outputs[0].pending)
        self.undo.redo()
        group = self.graph.nodes[group_id].group
        self.assertEqual([port.key for port in group.inputs], ["input_1", "input_2"])
        self.assertEqual(group.inputs[0].data_type, DataType.IMAGE)
        self.assertTrue(group.inputs[1].pending)

    def test_drag_node_into_group_reuses_legacy_free_typed_port(self) -> None:
        """旧版本遗留的“已定类型但已无连线”端口也应被第一条跨层连线复用。"""

        source, middle, sink = self._chain_nodes()
        group_id = self.add_group(pos=(800.0, 300.0))
        group = self.graph.nodes[group_id].group
        group.bind("input_1", True, DataType.IMAGE)
        group.bind("output_1", False, DataType.IMAGE)
        self.view._nodes[group_id].refresh_from_model()
        self.assertEqual([port.key for port in group.inputs], ["input_1", "input_2"])
        self.assertEqual([port.key for port in group.outputs], ["output_1", "output_2"])

        self.drag_node_to(middle.id, self.view._nodes[group_id].sceneBoundingRect().center())

        group = self.graph.nodes[group_id].group
        self.assertEqual([port.key for port in group.inputs], ["input_1", "input_2"])
        self.assertEqual(group.inputs[0].data_type, DataType.IMAGE)
        self.assertTrue(group.inputs[1].pending)
        self.assertEqual([port.key for port in group.outputs], ["output_1", "output_2"])
        self.assertEqual(group.outputs[0].data_type, DataType.IMAGE)
        self.assertTrue(group.outputs[1].pending)
        self.assertEqual(
            self._edge_set(),
            {
                (source.id, "image", group_id, "input_1"),
                (group_id, "input_1", middle.id, "image"),
                (middle.id, "image", group_id, "output_1"),
                (group_id, "output_1", sink.id, "image"),
            },
        )

    def test_drag_multi_input_node_into_group_orders_ports_by_inputs(self) -> None:
        first = self.add_node(pos=(0.0, -500.0))
        second = self.add_node(pos=(300.0, -500.0))
        merge = self.add_node("test.merge", pos=(150.0, 0.0))
        # 故意先连接输入 b、再连接输入 a：组端口仍按算子声明的 a、b 顺序分配，
        # 这样组内连线不会交叉。
        self.connect(second, merge, dst_port="b")
        self.connect(first, merge, dst_port="a")
        group_id = self.add_group(pos=(800.0, 300.0))

        self.drag_node_to(merge.id, self.view._nodes[group_id].sceneBoundingRect().center())

        edges = self._edge_set()
        self.assertIn((first.id, "image", group_id, "input_1"), edges)
        self.assertIn((group_id, "input_1", merge.id, "a"), edges)
        self.assertIn((second.id, "image", group_id, "input_2"), edges)
        self.assertIn((group_id, "input_2", merge.id, "b"), edges)
        group = self.graph.nodes[group_id].group
        self.assertEqual(
            [port.key for port in group.inputs], ["input_1", "input_2", "input_3"]
        )
        self.assertTrue(group.inputs[2].pending)

    def test_remove_group_input_connection_reclaims_port_and_reuses_pending(self) -> None:
        group_id = self.add_group(pos=(800.0, 300.0))
        source = self.add_source("源", pos=(0.0, -400.0))
        self.view.connect_ports(source.id, "image", group_id, "input_1")
        group = self.graph.nodes[group_id].group
        self.assertEqual([port.key for port in group.inputs], ["input_1", "input_2"])
        self.assertEqual(group.inputs[0].data_type, DataType.IMAGE)

        connection = self.graph.input_connection(group_id, "input_1")
        self.assertIsNotNone(connection)
        self.undo.push(RemoveConnectionCommand(self.graph, connection))

        # 连接删除后，已定类型端口也被删除；只剩一个待定端口等待下一条连线。
        group = self.graph.nodes[group_id].group
        self.assertEqual([port.key for port in group.inputs], ["input_1"])
        self.assertTrue(group.inputs[0].pending)
        item = self.view._nodes[group_id]
        self.assertEqual(list(item.outer_input_ports), ["input_1"])
        self.assertTrue(item.outer_input_ports["input_1"].pending)

        self.undo.undo()
        group = self.graph.nodes[group_id].group
        self.assertEqual([port.key for port in group.inputs], ["input_1", "input_2"])
        self.assertEqual(group.inputs[0].data_type, DataType.IMAGE)
        self.assertIsNotNone(self.graph.input_connection(group_id, "input_1"))

        self.undo.redo()
        group = self.graph.nodes[group_id].group
        self.assertEqual([port.key for port in group.inputs], ["input_1"])
        self.assertTrue(group.inputs[0].pending)

        # 下次连接重新落到待定端点上（而不是继续往后追加端口）。
        self.view.connect_ports(source.id, "image", group_id, "input_1")
        group = self.graph.nodes[group_id].group
        self.assertEqual([port.key for port in group.inputs], ["input_1", "input_2"])
        self.assertEqual(group.inputs[0].data_type, DataType.IMAGE)
        self.assertTrue(group.inputs[1].pending)

    def test_remove_group_output_connection_reclaims_port(self) -> None:
        group_id = self.add_group(pos=(700.0, 300.0))
        group_pos = self.graph.nodes[group_id].pos
        child = self.add_child(group_id, pos=(group_pos[0] + 60.0, group_pos[1] + 110.0))
        self.view.connect_ports(child.id, "image", group_id, "output_1")
        group = self.graph.nodes[group_id].group
        self.assertEqual([port.key for port in group.outputs], ["output_1", "output_2"])
        self.assertEqual(group.outputs[0].data_type, DataType.IMAGE)

        connection = self.graph.input_connection(group_id, "output_1")
        self.assertIsNotNone(connection)
        self.undo.push(RemoveConnectionCommand(self.graph, connection))
        group = self.graph.nodes[group_id].group
        self.assertEqual([port.key for port in group.outputs], ["output_1"])
        self.assertTrue(group.outputs[0].pending)

        self.undo.undo()
        group = self.graph.nodes[group_id].group
        self.assertEqual([port.key for port in group.outputs], ["output_1", "output_2"])
        self.assertEqual(group.outputs[0].data_type, DataType.IMAGE)
        self.assertIsNotNone(self.graph.input_connection(group_id, "output_1"))

    def test_drag_node_out_of_group_restores_direct_connections(self) -> None:
        source, middle, sink = self._chain_nodes()
        group_id = self.add_group(pos=(800.0, 300.0))
        self.drag_node_to(middle.id, self.view._nodes[group_id].sceneBoundingRect().center())
        self.assertEqual(self.graph.nodes[middle.id].parent_group, group_id)
        boundary_edges = self._edge_set()

        self.drag_node_to(middle.id, QPointF(0.0, 1300.0))

        self.assertIsNone(self.graph.nodes[middle.id].parent_group)
        self.assertEqual(
            self._edge_set(),
            {
                (source.id, "image", middle.id, "image"),
                (middle.id, "image", sink.id, "image"),
            },
        )

        self.undo.undo()
        self.assertEqual(self.graph.nodes[middle.id].parent_group, group_id)
        self.assertEqual(self._edge_set(), boundary_edges)

    def test_dissolve_group_keeps_children_and_reconnects(self) -> None:
        group_id = self.add_group(pos=(700.0, 300.0))
        group_pos = self.graph.nodes[group_id].pos
        child = self.add_child(group_id, pos=(group_pos[0] + 60.0, group_pos[1] + 110.0))
        source = self.add_node(pos=(0.0, 0.0))
        sink = self.add_node(pos=(0.0, 900.0))
        self.view.connect_ports(source.id, "image", group_id, "input_1")
        self.view.connect_ports(group_id, "input_1", child.id, "image")
        self.view.connect_ports(child.id, "image", group_id, "output_1")
        self.view.connect_ports(group_id, "output_1", sink.id, "image")
        child_pos = self.graph.nodes[child.id].pos

        self.view.dissolve_group(group_id)

        self.assertNotIn(group_id, self.graph.nodes)
        self.assertIsNone(self.graph.nodes[child.id].parent_group)
        self.assertEqual(self.graph.nodes[child.id].pos, child_pos)
        self.assertEqual(
            self._edge_set(),
            {
                (source.id, "image", child.id, "image"),
                (child.id, "image", sink.id, "image"),
            },
        )

        self.undo.undo()
        self.assertIn(group_id, self.graph.nodes)
        self.assertEqual(self.graph.nodes[child.id].parent_group, group_id)
        self.assertEqual(len(self.graph.connections), 4)

    def test_drag_group_moves_children_with_mouse(self) -> None:
        group_id = self.add_group(pos=(400.0, 300.0))
        group_pos = self.graph.nodes[group_id].pos
        child = self.add_child(group_id, pos=(group_pos[0] + 70.0, group_pos[1] + 110.0))
        self.view._sync_group_geometries()
        group_item = self.view._nodes[group_id]
        before_group = self.graph.nodes[group_id].pos
        before_child = self.graph.nodes[child.id].pos

        self.view.fit_graph()
        self.view.show()
        if _APP is not None:
            _APP.processEvents()
        start = self.view.mapFromScene(
            QPointF(group_item.pos().x() + 80.0, group_item.pos().y() + 10.0)
        )
        end = QPoint(start.x() + 80, start.y() + 60)
        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            start,
        )
        for step in range(1, 6):
            point = QPoint(
                start.x() + (end.x() - start.x()) * step // 5,
                start.y() + (end.y() - start.y()) * step // 5,
            )
            QTest.mouseMove(self.view.viewport(), point)
            if _APP is not None:
                _APP.processEvents()
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            end,
        )
        if _APP is not None:
            _APP.processEvents()

        group_dx = self.graph.nodes[group_id].pos[0] - before_group[0]
        group_dy = self.graph.nodes[group_id].pos[1] - before_group[1]
        self.assertGreater(abs(group_dx) + abs(group_dy), 1.0)
        self.assertAlmostEqual(
            self.graph.nodes[child.id].pos[0] - before_child[0], group_dx, places=1
        )
        self.assertAlmostEqual(
            self.graph.nodes[child.id].pos[1] - before_child[1], group_dy, places=1
        )

    def test_library_drop_inside_group_sets_parent_group(self) -> None:
        group_id = self.add_group(pos=(500.0, 400.0))
        group_item = self.view._nodes[group_id]
        center = group_item.sceneBoundingRect().center()
        before = set(self.graph.nodes)

        new_id = self.view.add_operator_at("test.blur", center)
        self.assertIsNotNone(new_id)
        created = [node_id for node_id in self.graph.nodes if node_id not in before]
        self.assertEqual(created, [new_id])
        node = self.graph.nodes[new_id]
        self.assertEqual(node.parent_group, group_id)
        self.assertIn(node.id, [item.id for item in self.graph.direct_children(group_id)])

        group_item = self.view._nodes[group_id]
        group_rect = QRectF(group_item.pos(), group_item.size())
        child_item = self.view._nodes[new_id]
        self.assertTrue(group_rect.contains(QRectF(child_item.pos(), child_item.size())))

        self.undo.undo()
        self.assertNotIn(new_id, self.graph.nodes)
        self.assertFalse(self.graph.direct_children(group_id))

    def test_drag_node_into_group_updates_parent_and_undo(self) -> None:
        group_id = self.add_group(pos=(400.0, 300.0))
        group_item = self.view._nodes[group_id]
        node = self.add_node(pos=(1200.0, 1200.0))
        item = self.view._nodes[node.id]
        self.view._scene.clearSelection()
        item.setSelected(True)

        self.view.begin_move_tracking()
        center = group_item.sceneBoundingRect().center()
        item.setPos(
            center.x() - item.size().width() / 2.0,
            center.y() - item.size().height() / 2.0,
        )
        self.view.finish_move_tracking()

        self.assertEqual(self.graph.nodes[node.id].parent_group, group_id)
        self.assertEqual(self.undo.count(), 2)

        self.undo.undo()
        self.assertIsNone(self.graph.nodes[node.id].parent_group)
        self.assertEqual(self.graph.nodes[node.id].pos, (1200.0, 1200.0))

    def test_drag_group_moves_all_descendants_single_undo(self) -> None:
        group_id = self.add_group(pos=(400.0, 300.0))
        group_pos = self.graph.nodes[group_id].pos
        child = self.add_child(group_id, pos=(group_pos[0] + 40.0, group_pos[1] + 80.0))
        self.view._sync_group_geometries()
        group_pos = self.graph.nodes[group_id].pos
        child_pos = self.graph.nodes[child.id].pos
        group_item = self.view._nodes[group_id]

        self.view._scene.clearSelection()
        group_item.setSelected(True)
        self.view.begin_move_tracking()
        delta = QPointF(140.0, 100.0)
        group_item.setPos(group_item.pos() + delta)
        self.view.finish_move_tracking()

        self.assertAlmostEqual(self.graph.nodes[child.id].pos[0], child_pos[0] + 140.0, places=1)
        self.assertAlmostEqual(self.graph.nodes[child.id].pos[1], child_pos[1] + 100.0, places=1)
        self.assertAlmostEqual(self.graph.nodes[group_id].pos[0], group_pos[0] + 140.0, places=1)
        self.assertAlmostEqual(self.graph.nodes[group_id].pos[1], group_pos[1] + 100.0, places=1)
        self.assertAlmostEqual(
            self.view._nodes[child.id].pos().x(), child_pos[0] + 140.0, places=1
        )
        self.assertEqual(self.undo.count(), 2)

        self.undo.undo()
        self.assertAlmostEqual(self.graph.nodes[child.id].pos[0], child_pos[0], places=1)
        self.assertAlmostEqual(self.graph.nodes[child.id].pos[1], child_pos[1], places=1)
        self.assertAlmostEqual(self.graph.nodes[group_id].pos[0], group_pos[0], places=1)
        self.assertAlmostEqual(self.graph.nodes[group_id].pos[1], group_pos[1], places=1)

    def test_delete_group_subtree_is_single_undo(self) -> None:
        group_id = self.add_group(pos=(400.0, 300.0))
        group_pos = self.graph.nodes[group_id].pos
        child = self.add_child(group_id, pos=(group_pos[0] + 40.0, group_pos[1] + 80.0))
        inner = self.graph.add_connection(child.id, "image", group_id, "output_1")
        sink = self.add_node(pos=(0.0, 1000.0))
        outer = self.graph.add_connection(group_id, "output_1", sink.id, "image")

        self.view._nodes[group_id].setSelected(True)
        self.view.delete_selected()
        self.assertNotIn(group_id, self.graph.nodes)
        self.assertNotIn(child.id, self.graph.nodes)
        self.assertFalse(self.graph.connections)
        self.assertFalse(self.view._connections)

        self.undo.undo()
        self.assertIn(group_id, self.graph.nodes)
        self.assertIn(child.id, self.graph.nodes)
        self.assertEqual(set(self.graph.connections), {inner.id, outer.id})
        self.assertIsInstance(self.view._nodes[group_id], GroupItem)
        self.assertIsInstance(self.view._nodes[child.id], NodeItem)

    def test_copy_paste_group_subtree_remaps_parent_group(self) -> None:
        group_id = self.add_group(pos=(400.0, 300.0))
        group_pos = self.graph.nodes[group_id].pos
        child = self.add_child(group_id, pos=(group_pos[0] + 40.0, group_pos[1] + 80.0))
        inner = self.graph.add_connection(child.id, "image", group_id, "output_1")

        self.view._nodes[group_id].setSelected(True)
        self.assertTrue(self.view.copy_selected())
        new_ids = self.view.paste_nodes()
        self.assertEqual(len(new_ids), 2)
        new_groups = [node_id for node_id in new_ids if self.graph.nodes[node_id].is_group]
        self.assertEqual(len(new_groups), 1)
        new_group_id = new_groups[0]
        new_children = [n for n in self.graph.direct_children(new_group_id)]
        self.assertEqual(len(new_children), 1)
        self.assertEqual(new_children[0].parent_group, new_group_id)
        self.assertIn(new_children[0].id, new_ids)

        pasted_connections = [
            connection for connection in self.graph.connections.values() if connection.id != inner.id
        ]
        self.assertEqual(len(pasted_connections), 1)
        self.assertEqual(pasted_connections[0].src_node, new_children[0].id)
        self.assertEqual(pasted_connections[0].dst_node, new_group_id)
        self.assertIn(inner.id, self.graph.connections)

        self.undo.undo()
        self.assertNotIn(new_group_id, self.graph.nodes)
        self.assertEqual(set(self.graph.nodes), {group_id, child.id})
        self.assertEqual(set(self.graph.connections), {inner.id})

    def test_save_group_menu_uses_name_and_emits_signal(self) -> None:
        group_id = self.add_group()
        group_pos = self.graph.nodes[group_id].pos
        self.add_child(group_id, pos=(group_pos[0] + 40.0, group_pos[1] + 80.0))
        statuses: list[str] = []
        changed: list[bool] = []
        self.view.status_message.connect(statuses.append)
        self.view.operators_changed.connect(lambda: changed.append(True))
        saved = SimpleNamespace(name="我的流程", op_id="group.demo")

        with patch(
            "app.graph.view.QInputDialog.getText", return_value=("我的流程", True)
        ) as dialog, patch(
            "app.graph.view.save_group_operator", return_value=saved
        ) as save:
            self.view.save_group_as_operator(group_id)

        dialog.assert_called_once()
        save.assert_called_once()
        args = save.call_args.args
        self.assertIs(args[0], self.graph)
        self.assertEqual(args[1], group_id)
        self.assertEqual(args[2], "我的流程")
        self.assertIs(save.call_args.kwargs.get("registry"), self.registry)
        self.assertEqual(changed, [True])
        self.assertTrue(any("已保存算子" in message for message in statuses))

    def test_save_group_rejects_source_image_nodes(self) -> None:
        group_id = self.add_group()
        group_pos = self.graph.nodes[group_id].pos
        source = NodeModel(
            id=new_id("n"),
            op_id=SOURCE_OP_ID,
            title="组内源图像",
            pos=(group_pos[0] + 30.0, group_pos[1] + 80.0),
            parent_group=group_id,
        )
        self.graph.add_node(source)
        statuses: list[str] = []
        self.view.status_message.connect(statuses.append)

        with patch("app.graph.view.QInputDialog.getText") as dialog, patch(
            "app.graph.view.save_group_operator"
        ) as save:
            self.view.save_group_as_operator(group_id)

        dialog.assert_not_called()
        save.assert_not_called()
        self.assertTrue(any("原始图像" in message for message in statuses))

    def test_replace_stale_node_via_dialog_keeps_connection_and_undo(self) -> None:
        source = self.add_source("源", pos=(0.0, 0.0))
        stale = self.add_missing_node(pos=(0.0, 300.0))
        connection = self.graph.add_connection(source.id, "image", stale.id, "image")
        old_item = self.view._nodes[stale.id]
        self.assertTrue(old_item.is_stale)

        class _FakeReplaceDialog:
            def __init__(self, registry, *, current_op_id=None, parent=None) -> None:
                self.current_op_id = current_op_id

            def exec(self) -> int:
                return 1

            def selected_op_id(self) -> str:
                return "test.blur"

        with patch("app.graph.view.ReplaceOperatorDialog", _FakeReplaceDialog):
            self.view._prompt_replace_operator(stale.id)

        node = self.graph.nodes[stale.id]
        self.assertEqual(node.op_id, "test.blur")
        self.assertEqual(node.title, "模糊")
        self.assertEqual(node.params.get("ksize"), 5)
        self.assertIsNot(old_item, self.view._nodes[stale.id])
        self.assertFalse(self.view._nodes[stale.id].is_stale)
        upstream = self.graph.input_connection(stale.id, "image")
        self.assertIsNotNone(upstream)
        self.assertEqual(upstream.src_node, source.id)

        self.undo.undo()
        self.assertEqual(self.graph.nodes[stale.id].op_id, "missing.op")
        self.assertTrue(self.view._nodes[stale.id].is_stale)
        self.assertIn(connection.id, self.graph.connections)

    def test_group_ports_compatible_respects_scope(self) -> None:
        group_id = self.add_group()
        group_pos = self.graph.nodes[group_id].pos
        child = self.add_child(group_id, pos=(group_pos[0] + 40.0, group_pos[1] + 80.0))
        top = self.add_node(pos=(1000.0, 1000.0))

        self.assertFalse(self.view.ports_compatible(top.id, "image", child.id, "image"))
        self.assertFalse(self.view.ports_compatible(child.id, "image", top.id, "image"))
        self.assertTrue(
            self.view.ports_compatible(child.id, "image", group_id, "output_1")
        )
        self.assertTrue(
            self.view.ports_compatible(group_id, "input_1", child.id, "image")
        )

        messages: list[str] = []
        self.view.status_message.connect(messages.append)
        self.view.connect_ports(top.id, "image", child.id, "image")
        self.assertFalse(self.graph.connections)
        self.assertTrue(any("层级" in message or "算子组" in message for message in messages))

        scene = self.view._scene
        top_output = self.view._nodes[top.id].output_ports["image"]
        child_input = self.view._nodes[child.id].input_ports["image"]
        scene._drag_source = top_output
        try:
            self.assertIsNone(scene._target_input_at(child_input.center_scene_pos()))
        finally:
            scene._drag_source = None

    def test_drop_group_operator_materializes_with_single_undo(self) -> None:
        self.view.set_registry(make_group_registry())
        group_id = self.view.add_operator_at("group.demo", QPointF(500.0, 400.0))
        self.assertIsNotNone(group_id)
        node = self.graph.nodes[group_id]
        self.assertTrue(node.is_group)
        self.assertEqual(node.group.operator_id, "group.demo")
        children = self.graph.direct_children(group_id)
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].op_id, "test.blur")
        self.assertEqual(children[0].params.get("ksize"), 5)
        self.assertEqual(len(self.graph.connections), 2)
        self.assertEqual(len(self.view._nodes), 2)
        self.assertEqual(len(self.view._connections), 2)
        self.assertEqual(self.undo.count(), 1)

        self.undo.undo()
        self.assertFalse(self.graph.nodes)
        self.assertFalse(self.graph.connections)
        self.assertFalse(self.view._nodes)

        self.undo.redo()
        self.assertEqual(len(self.graph.nodes), 2)
        self.assertEqual(len(self.graph.connections), 2)

    def test_auto_layout_lays_out_group_children(self) -> None:
        group_id = self.add_group(pos=(400.0, 300.0))
        group_pos = self.graph.nodes[group_id].pos
        first = self.add_child(group_id, pos=(group_pos[0] + 30.0, group_pos[1] + 90.0))
        second = self.add_child(group_id, pos=(group_pos[0] + 30.0, group_pos[1] + 240.0))
        self.graph.add_connection(first.id, "image", second.id, "image")
        top = self.add_node(pos=(-900.0, 700.0))
        self.view._sync_group_geometries()
        before = {node_id: self.graph.nodes[node_id].pos for node_id in (first.id, second.id)}

        changes = self.view._layout_changes()
        changed_ids = {change[0] for change in changes}
        self.assertIn(first.id, changed_ids)
        self.assertIn(second.id, changed_ids)
        self.assertIn(group_id, changed_ids)
        self.assertIn(top.id, changed_ids)

        self.view.auto_layout()

        first_pos = self.graph.nodes[first.id].pos
        second_pos = self.graph.nodes[second.id].pos
        group_pos = self.graph.nodes[group_id].pos
        # 组内节点按一条普通连线布局：下游排在下一层。
        self.assertGreater(second_pos[1], first_pos[1])
        # 组内布局从算子组内容区左上角开始。
        self.assertAlmostEqual(
            min(first_pos[0], second_pos[0]) - group_pos[0], GROUP_PADDING_X, places=1
        )
        group_item = self.view._nodes[group_id]
        self.assertTrue(
            group_item.sceneBoundingRect().contains(self.view._nodes[first.id].sceneBoundingRect())
        )
        self.assertTrue(
            group_item.sceneBoundingRect().contains(self.view._nodes[second.id].sceneBoundingRect())
        )

        self.undo.undo()
        self.assertEqual(
            {node_id: self.graph.nodes[node_id].pos for node_id in (first.id, second.id)},
            before,
        )

    def test_group_context_menu_has_save_and_replace_actions(self) -> None:
        group_id = self.add_group()
        group_node = self.graph.nodes[group_id]
        point = self._show_and_focus_node(group_node)
        with patch("app.graph.view.QMenu") as menu_cls:
            self.view.contextMenuEvent(_FakeContextMenuEvent(point))
        actions = [call.args[0] for call in menu_cls.return_value.addAction.call_args_list if call.args]
        self.assertIn("保存为新算子…", actions)
        self.assertIn("解散算子组", actions)

        missing = self.add_missing_node(pos=(1500.0, 1500.0))
        point = self._show_and_focus_node(missing)
        with patch("app.graph.view.QMenu") as menu_cls:
            self.view.contextMenuEvent(_FakeContextMenuEvent(point))
        actions = [call.args[0] for call in menu_cls.return_value.addAction.call_args_list if call.args]
        self.assertIn("替换算子…", actions)

    def test_move_out_of_group_command_and_undo(self) -> None:
        group_id = self.add_group()
        group_pos = self.graph.nodes[group_id].pos
        child = self.add_child(group_id, pos=(group_pos[0] + 30.0, group_pos[1] + 80.0))
        self.view.move_out_of_group(child.id)
        self.assertIsNone(self.graph.nodes[child.id].parent_group)
        # add_group_at 占一条撤销记录；移出算子组本身是一步。
        self.assertEqual(self.undo.count(), 2)

        self.undo.undo()
        self.assertEqual(self.graph.nodes[child.id].parent_group, group_id)


if __name__ == "__main__":
    unittest.main()
