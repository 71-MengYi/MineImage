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

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QMimeData, QPoint, QPointF, Qt
from PySide6.QtGui import QDropEvent, QUndoStack
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.config import AppSettings
from app.core.constants import MIME_OPERATOR, SINGLE_CLICK_DELAY_MS, SOURCE_OP_ID
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
    SetNodeParamsCommand,
)
from app.graph.layout import compute_layout
from app.graph.view import GraphView
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
    return FakeRegistry(BLUR_SPEC, MERGE_SPEC, SPLIT_SPEC)


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
        # 父节点尽量居中于子节点
        left_center = items[left.id].sceneBoundingRect().center().x()
        right_center = items[right.id].sceneBoundingRect().center().x()
        merge_center = items[merge.id].sceneBoundingRect().center().x()
        self.assertAlmostEqual(merge_center, (left_center + right_center) / 2.0, delta=2.0)

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


if __name__ == "__main__":
    unittest.main()
