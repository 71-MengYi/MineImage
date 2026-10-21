"""端口上下方向与纵向连线的几何测试（离屏 Qt）。

覆盖：
* 输入端口位于节点上边缘、输出端口位于下边缘；
* 多端口水平均分且互不重叠，``NodeItem.port_x`` 与实例端口位置一致；
* 连线的起点 / 终点贴合对应端口，曲线从输出端口向下出发、向上进入输入端口；
* 节点包围盒覆盖上下边缘的端口圆点；
* 端口标签、标题、缩略图与底部信息所在区域互不重叠；
* 连线 ``set_highlight`` 使用蓝色高亮。

运行方式（项目根目录）::

    $env:QT_QPA_PLATFORM = "offscreen"
    $env:PYTHONPATH = "."
    python -m unittest tests.test_port_geometry -v
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QImage, QPainter, QUndoStack
from PySide6.QtWidgets import QApplication, QStyleOptionGraphicsItem

from app.config import AppSettings
from app.core.model import NodeModel, WorkspaceModel, new_id
from app.core.types import ImageData
from app.engine.engine import NodeResult
from app.graph.connection_item import ConnectionItem
from app.graph.node_item import NodeItem
from app.graph.port_item import PortItem
from app.graph.view import GraphView
from app.operators.spec import OperatorSpec, PortSpec, param_defaults

_APP: QApplication | None = None

BLUR_SPEC = OperatorSpec(
    id="test.blur",
    name="模糊",
    category=("滤波与平滑", "平滑"),
    description="测试用模糊算子。",
    inputs=(PortSpec("image", "图像"),),
    outputs=(PortSpec("image", "图像"),),
)

MERGE_SPEC = OperatorSpec(
    id="test.merge",
    name="合并",
    category=("其他",),
    description="测试用多输入算子。",
    inputs=(
        PortSpec("a", "图像 A"),
        PortSpec("b", "图像 B"),
        PortSpec("c", "图像 C"),
    ),
    outputs=(PortSpec("image", "图像"),),
)

SPLIT_SPEC = OperatorSpec(
    id="test.split",
    name="分离",
    category=("通道与色彩",),
    description="测试用多输出算子。",
    inputs=(PortSpec("image", "图像"),),
    outputs=(
        PortSpec("left", "左通道"),
        PortSpec("right", "右通道"),
        PortSpec("alpha", "透明度"),
    ),
)


class _FakeRegistry:
    """仅实现 GraphView 需要的算子查询接口。"""

    def __init__(self, *specs: OperatorSpec) -> None:
        self._specs = {spec.id: spec for spec in specs}

    def spec(self, op_id: str) -> OperatorSpec | None:
        return self._specs.get(op_id)

    def get(self, op_id: str):
        return None

    def entry(self, op_id: str):
        return None


def setUpModule() -> None:
    global _APP
    _APP = QApplication.instance() or QApplication([])


def tearDownModule() -> None:
    global _APP
    if _APP is not None:
        _APP.processEvents()
        _APP = None


class PortGeometryTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory(prefix="mineimage-port-geom-")
        self.settings = AppSettings(Path(self._temp_dir.name) / "test.ini")
        self.workspace = WorkspaceModel("端口几何测试")
        self.graph = self.workspace.graph
        self.undo = QUndoStack()
        self.registry = _FakeRegistry(BLUR_SPEC, MERGE_SPEC, SPLIT_SPEC)
        self.view = GraphView(self.workspace, self.registry, self.undo, self.settings)
        self.view.resize(1000, 700)

    def tearDown(self) -> None:
        self.view.close()
        self.view.deleteLater()
        if _APP is not None:
            _APP.processEvents()
        try:
            self._temp_dir.cleanup()
        except OSError:
            pass

    # ------------------------------------------------------------------ 工具
    def add_node(
        self,
        op_id: str = "test.blur",
        pos: tuple[float, float] = (0.0, 0.0),
    ) -> NodeItem:
        spec = self.registry.spec(op_id)
        assert spec is not None
        node = NodeModel(
            id=new_id("n"),
            op_id=op_id,
            title=spec.name,
            pos=pos,
            params=param_defaults(spec),
        )
        self.graph.add_node(node)
        return self.view._nodes[node.id]

    @staticmethod
    def _bottom_y(item: NodeItem) -> float:
        return item.pos().y() + item.size().height()

    # ------------------------------------------------------------ 端口上下方向
    def test_single_port_is_top_and_bottom_centered(self) -> None:
        item = self.add_node()
        width = item.size().width()
        port_x = item.pos().x() + width / 2.0

        in_port = item.input_ports["image"]
        out_port = item.output_ports["image"]
        self.assertAlmostEqual(in_port.center_scene_pos().y(), item.pos().y(), delta=0.5)
        self.assertAlmostEqual(out_port.center_scene_pos().y(), self._bottom_y(item), delta=0.5)
        self.assertAlmostEqual(in_port.center_scene_pos().x(), port_x, delta=0.5)
        self.assertAlmostEqual(out_port.center_scene_pos().x(), port_x, delta=0.5)

    def test_multi_ports_are_spread_and_not_overlapping(self) -> None:
        merge = self.add_node("test.merge")
        input_positions = [port.center_scene_pos() for port in merge.input_ports.values()]
        self.assertEqual(len(input_positions), 3)
        left = merge.pos().x()
        right = left + merge.size().width()
        for position in input_positions:
            self.assertAlmostEqual(position.y(), merge.pos().y(), delta=0.5)
            self.assertGreater(position.x(), left)
            self.assertLess(position.x(), right)
        for first, second in zip(input_positions, input_positions[1:]):
            self.assertGreaterEqual(second.x() - first.x(), PortItem.HIT_RADIUS * 2.0)

        split = self.add_node("test.split", pos=(0.0, 400.0))
        output_positions = [port.center_scene_pos() for port in split.output_ports.values()]
        self.assertEqual(len(output_positions), 3)
        bottom = self._bottom_y(split)
        for position in output_positions:
            self.assertAlmostEqual(position.y(), bottom, delta=0.5)
        for first, second in zip(output_positions, output_positions[1:]):
            self.assertGreaterEqual(second.x() - first.x(), PortItem.HIT_RADIUS * 2.0)

    def test_port_x_helper_matches_layout(self) -> None:
        width = NodeItem.WIDTH
        self.assertAlmostEqual(NodeItem.port_x(0, 1), width / 2.0, delta=0.001)
        self.assertAlmostEqual(NodeItem.port_x(-3, 2), NodeItem.port_x(0, 2), delta=0.001)
        self.assertAlmostEqual(NodeItem.port_x(99, 2), NodeItem.port_x(1, 2), delta=0.001)

        positions = [NodeItem.port_x(index, 4) for index in range(4)]
        gaps = [second - first for first, second in zip(positions, positions[1:])]
        for gap in gaps:
            self.assertAlmostEqual(gap, gaps[0], delta=0.001)
        self.assertGreater(positions[0], 0.0)
        self.assertLess(positions[-1], width)

        item = self.add_node("test.merge")
        for index, port in enumerate(item.input_ports.values()):
            expected = item.pos().x() + NodeItem.port_x(index, 3)
            self.assertAlmostEqual(port.center_scene_pos().x(), expected, delta=0.5)

    # ------------------------------------------------------------ 连线几何
    def test_connection_endpoints_touch_ports(self) -> None:
        source = self.add_node(pos=(40.0, 20.0))
        target = self.add_node(pos=(140.0, 420.0))
        connection = self.graph.add_connection(
            source.node.id, "image", target.node.id, "image"
        )
        item = self.view._connections[connection.id]
        self.assertIsInstance(item, ConnectionItem)

        path = item.path()
        self.assertFalse(path.isEmpty())
        start = path.pointAtPercent(0.0)
        end = path.pointAtPercent(1.0)
        out_port = source.output_ports["image"].center_scene_pos()
        in_port = target.input_ports["image"].center_scene_pos()

        self.assertAlmostEqual(start.x(), out_port.x(), delta=0.5)
        self.assertAlmostEqual(start.y(), out_port.y(), delta=0.5)
        self.assertAlmostEqual(end.x(), in_port.x(), delta=0.5)
        self.assertAlmostEqual(end.y(), in_port.y(), delta=0.5)
        # 起点贴着源节点下边缘，终点贴着目标节点上边缘。
        self.assertAlmostEqual(start.y(), self._bottom_y(source), delta=0.5)
        self.assertAlmostEqual(end.y(), target.pos().y(), delta=0.5)

    def test_connection_curve_is_vertical_at_ports(self) -> None:
        source = self.add_node(pos=(30.0, 10.0))
        target = self.add_node(pos=(220.0, 400.0))
        connection = self.graph.add_connection(
            source.node.id, "image", target.node.id, "image"
        )
        path = self.view._connections[connection.id].path()

        start = path.pointAtPercent(0.0)
        near_start = path.pointAtPercent(0.01)
        # 从输出端口向下出发。
        self.assertGreater(near_start.y(), start.y())
        self.assertLess(abs(near_start.x() - start.x()), 1.0)

        end = path.pointAtPercent(1.0)
        near_end = path.pointAtPercent(0.99)
        # 从上方进入输入端口。
        self.assertLess(near_end.y(), end.y())
        self.assertLess(abs(near_end.x() - end.x()), 1.0)

        # 曲线整体是纵向 S 形：中点位于两端之间。
        middle = path.pointAtPercent(0.5)
        self.assertGreater(middle.y(), start.y())
        self.assertLess(middle.y(), end.y())

    # ------------------------------------------------------------ 包围盒
    def test_bounding_rect_covers_edge_port_dots(self) -> None:
        item = self.add_node(pos=(25.0, 35.0))
        rect = item.boundingRect()
        margin = PortItem.HIT_RADIUS + 3.0
        for port in list(item.input_ports.values()) + list(item.output_ports.values()):
            local = port.pos()
            self.assertTrue(rect.contains(local))
            # 圆点最外侧（含 hover 命中半径与描边）也在包围盒内。
            self.assertTrue(rect.contains(QPointF(local.x(), local.y() - margin + 0.01)))
            self.assertTrue(rect.contains(QPointF(local.x(), local.y() + margin - 0.01)))

    # ------------------------------------------------------------ 布局不重叠
    def test_labels_header_thumbnail_and_footer_do_not_overlap(self) -> None:
        item = self.add_node(pos=(0.0, 0.0))
        base_height = item.size().height()

        image = ImageData(np.zeros((48, 64, 3), dtype=np.uint8), "结果")
        result = NodeResult(item.node.id, {"image": image}, 7.5, item.node.revision)
        item.set_result(result)
        self.assertGreater(item.size().height(), base_height)

        height = item.size().height()
        # 输入标签在节点顶边与标题区之间。
        self.assertEqual(len(item._input_label_rects), 1)
        self.assertGreaterEqual(item._input_label_rects[0].top(), 0.0)
        self.assertLessEqual(item._input_label_rects[0].bottom(), item._header_rect.top())
        # 标题区、缩略图、底部信息、输出标签从上到下依次排列。
        self.assertGreaterEqual(item._thumb_rect.top(), item._header_rect.bottom())
        self.assertLessEqual(item._thumb_rect.bottom(), item._footer_rect.top())
        self.assertEqual(len(item._output_label_rects), 1)
        self.assertGreaterEqual(item._output_label_rects[0].top(), item._footer_rect.bottom())
        self.assertLessEqual(item._output_label_rects[0].bottom(), height)
        # 输出端口仍贴在新的底边。
        out_port = item.output_ports["image"]
        self.assertAlmostEqual(out_port.center_scene_pos().y(), self._bottom_y(item), delta=0.5)

        # 多端口标签槽位互不重叠。
        merge = self.add_node("test.merge", pos=(0.0, 500.0))
        rects = merge._input_label_rects
        self.assertEqual(len(rects), 3)
        for first, second in zip(rects, rects[1:]):
            self.assertLessEqual(first.right(), second.left())

    def test_connection_follows_output_port_when_thumbnail_toggles(self) -> None:
        source = self.add_node(pos=(10.0, 10.0))
        target = self.add_node(pos=(10.0, 360.0))
        connection = self.graph.add_connection(
            source.node.id, "image", target.node.id, "image"
        )
        item = self.view._connections[connection.id]
        base_height = source.size().height()

        image = ImageData(np.zeros((48, 64, 3), dtype=np.uint8), "结果")
        result = NodeResult(source.node.id, {"image": image}, 3.0, source.node.revision)
        self.view.notify_result(source.node.id, result)
        self.assertGreater(source.size().height(), base_height)
        start = item.path().pointAtPercent(0.0)
        self.assertAlmostEqual(
            start.y(), source.pos().y() + source.size().height(), delta=0.5
        )

        # 通过视图接口关闭缩略图：节点高度回落，连线起点必须同步回到底边端口。
        self.view.set_thumbnails_enabled(False)
        self.assertLess(source.size().height(), base_height + 1.0)
        start = item.path().pointAtPercent(0.0)
        self.assertAlmostEqual(
            start.y(), source.pos().y() + source.size().height(), delta=0.5
        )

    # ------------------------------------------------------------ 连线高亮
    def test_connection_highlight_renders_blue(self) -> None:
        source = self.add_node(pos=(20.0, 20.0))
        target = self.add_node(pos=(120.0, 380.0))
        connection = self.graph.add_connection(
            source.node.id, "image", target.node.id, "image"
        )
        item = self.view._connections[connection.id]
        self.assertFalse(item.highlighted)
        item.set_highlight(True)
        self.assertTrue(item.highlighted)

        if _APP is not None:
            _APP.processEvents()

        # 把连线绘制到透明画布上，检查中点附近出现蓝色像素。
        bounds = item.boundingRect()
        padding = 8
        width = int(bounds.width()) + padding * 2
        height = int(bounds.height()) + padding * 2
        image = QImage(width, height, QImage.Format.Format_ARGB32)
        image.fill(0)
        painter = QPainter(image)
        try:
            painter.translate(padding - bounds.left(), padding - bounds.top())
            item.paint(painter, QStyleOptionGraphicsItem(), None)
        finally:
            painter.end()

        middle = item.path().pointAtPercent(0.5)
        center = QPointF(
            middle.x() + (padding - bounds.left()),
            middle.y() + (padding - bounds.top()),
        )
        found_blue = False
        for dy in range(-4, 5):
            for dx in range(-4, 5):
                px = int(round(center.x())) + dx
                py = int(round(center.y())) + dy
                if px < 0 or py < 0 or px >= width or py >= height:
                    continue
                color = image.pixelColor(px, py)
                if color.blue() > 150 and color.blue() - color.red() > 60:
                    found_blue = True
                    break
            if found_blue:
                break
        self.assertTrue(found_blue, "高亮连线未绘制出蓝色像素")

        item.set_highlight(False)
        self.assertFalse(item.highlighted)


if __name__ == "__main__":
    unittest.main()
