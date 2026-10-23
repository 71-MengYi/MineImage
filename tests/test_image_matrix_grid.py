"""图像矩阵节点网格：内容识别、方形单元格布局与点选单图交互。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, QRectF, Qt
from PySide6.QtGui import QMouseEvent, QUndoStack
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.config import AppSettings
from app.core.constants import SINGLE_CLICK_DELAY_MS
from app.core.model import WorkspaceModel
from app.core.types import ImageData, ImageList
from app.engine.engine import NodeResult
from app.graph.view import GraphView
from app.operators.spec import OperatorSpec, PortSpec

_APP: QApplication | None = None

BLUR_SPEC = OperatorSpec(
    id="test.blur",
    name="模糊",
    category=("滤波与平滑", "平滑"),
    description="测试用模糊算子。",
    inputs=(PortSpec("image", "图像"),),
    outputs=(PortSpec("image", "图像"),),
)


class FakeRegistry:
    """仅实现 GraphView 需要的 spec 查询接口。"""

    def __init__(self, *specs: OperatorSpec) -> None:
        self._specs = {spec.id: spec for spec in specs}

    def spec(self, op_id: str) -> OperatorSpec | None:
        return self._specs.get(op_id)

    def get(self, op_id: str):
        return None

    def entry(self, op_id: str):
        return None


def _image(width: int = 32, height: int = 24, value: int = 100) -> ImageData:
    array = np.full((height, width, 3), value, dtype=np.uint8)
    return ImageData(array, f"img_{width}x{height}")


def _matrix(count: int = 3) -> ImageList:
    return ImageList(
        tuple(_image(24 + index * 4, 18 + index * 2, 30 + index * 25) for index in range(count))
    )


def setUpModule() -> None:  # noqa: N802
    global _APP
    _APP = QApplication.instance() or QApplication([])


def tearDownModule() -> None:  # noqa: N802
    global _APP
    if _APP is not None:
        _APP.processEvents()
        _APP = None


class ImageMatrixGridTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory(prefix="mineimage-image-grid-")
        self.settings = AppSettings(Path(self._temp_dir.name) / "test.ini")
        self.workspace = WorkspaceModel("图像矩阵测试")
        self.graph = self.workspace.graph
        self.undo = QUndoStack()
        self.view = GraphView(
            self.workspace, FakeRegistry(BLUR_SPEC), self.undo, self.settings
        )
        self.view.resize(1000, 700)
        self.view.show()
        self._process()
        self.node = self.graph.create_operator_node(
            "test.blur", "模糊", pos=(120.0, 90.0)
        )
        self.item = self.view._nodes[self.node.id]
        self._focus()

    def tearDown(self) -> None:
        self.view.close()
        self.view.deleteLater()
        self._process()
        try:
            self._temp_dir.cleanup()
        except OSError:
            # QSettings 在 Windows 上可能仍持有 ini 文件句柄；清理失败不影响测试。
            pass

    # ------------------------------------------------------------------ 工具
    def _process(self) -> None:
        if _APP is not None:
            _APP.processEvents()

    def _focus(self) -> None:
        self.view.fit_graph()
        self._process()

    def _set_images(self, value: object, result_revision: int | None = None) -> NodeResult:
        revision = self.node.revision if result_revision is None else result_revision
        result = NodeResult(self.node.id, {"image": value}, 4.0, revision)
        self.view.notify_result(self.node.id, result)
        self._process()
        self._focus()
        return result

    def _local_view_pos(self, local: QPointF) -> QPoint:
        return self.view.mapFromScene(self.item.mapToScene(QPointF(local)))

    def _tile_view_pos(self, index: int) -> QPoint:
        return self._local_view_pos(self.item.image_tile_rects()[index].center())

    def _click_local(self, local: QPointF) -> None:
        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            self._local_view_pos(local),
        )
        self._process()

    def _node_rect(self) -> QRectF:
        return QRectF(0.0, 0.0, self.item.size().width(), self.item.size().height())

    # ------------------------------------------------------------------ 布局
    def test_three_image_matrix_builds_grid(self) -> None:
        self._set_images(_image(40, 28, 80))
        self.assertEqual(self.item._content_mode, "thumb")
        single_height = self.item.size().height()

        matrix = _matrix(3)
        self._set_images(matrix)
        self._process()

        self.assertEqual(self.item._content_mode, "grid")
        self.assertIs(self.item.image_at(1), matrix[1])
        self.assertIsNone(self.item.image_at(3))
        self.assertIsNone(self.item.image_at(-9))
        self.assertTrue(self.item.has_image_grid)
        self.assertEqual(self.item.image_count, 3)
        self.assertEqual(self.item.matrix_image_count, 3)
        self.assertEqual(self.item.selected_image_index, 0)
        self.assertIsNone(self.item.thumbnail_pixmap)
        self.assertEqual(self.item._thumb_rect, QRectF())

        tiles = self.item.image_tile_rects()
        self.assertEqual(len(tiles), 3)
        node_rect = self._node_rect()
        for tile in tiles:
            self.assertTrue(node_rect.contains(tile), f"{tile} 超出节点 {node_rect}")
            self.assertGreaterEqual(tile.top(), self.item._header_rect.bottom())
            self.assertLessEqual(tile.bottom(), self.item._footer_rect.top())
            self.assertAlmostEqual(tile.width(), tile.height(), delta=0.01)
        for first, second in zip(tiles, tiles[1:]):
            self.assertFalse(first.intersects(second))
        # 3 张 -> 2 列：前两格同一行、间距约 6px，第三格在第二行。
        self.assertAlmostEqual(tiles[0].top(), tiles[1].top(), delta=0.01)
        self.assertAlmostEqual(tiles[1].left() - tiles[0].right(), 6.0, delta=0.01)
        self.assertGreater(tiles[2].top(), tiles[0].top())
        # 节点高度随行数增长，明显高于单图。
        self.assertGreater(self.item.size().height(), single_height)
        self.assertGreaterEqual(
            self.item._grid_rect.height() + 2.0 * self.item.THUMB_MARGIN,
            self.item.MIN_CONTENT_H,
        )
        # 命中测试与序号对应。
        self.assertEqual(self.item.image_index_at(tiles[1].center()), 1)
        self.assertIsNone(self.item.image_index_at(QPointF(2.0, 2.0)))
        self.assertEqual(len(self.item.image_tile_rects()), 3)

    def test_grid_columns_and_row_growth(self) -> None:
        self.assertEqual(self.item.grid_columns(1), 1)
        self.assertEqual(self.item.grid_columns(2), 2)
        self.assertEqual(self.item.grid_columns(4), 2)
        self.assertEqual(self.item.grid_columns(5), 3)
        self.assertEqual(self.item.grid_columns(9), 3)

        self._set_images(_matrix(2))
        height_two = self.item.size().height()
        self.assertEqual(len(self.item.image_tile_rects()), 2)
        self.assertAlmostEqual(
            self.item.image_tile_rects()[0].top(),
            self.item.image_tile_rects()[1].top(),
            delta=0.01,
        )

        self._set_images(_matrix(3))
        self.assertGreater(self.item.size().height(), height_two)

        self._set_images(_matrix(5))
        tiles = self.item.image_tile_rects()
        self.assertEqual(len(tiles), 5)
        self.assertEqual(self.item.image_count, 5)
        # 5 张 -> 3 列 2 行。
        self.assertAlmostEqual(tiles[0].top(), tiles[1].top(), delta=0.01)
        self.assertAlmostEqual(tiles[1].top(), tiles[2].top(), delta=0.01)
        self.assertGreater(tiles[3].top(), tiles[0].top())
        for tile in tiles:
            self.assertTrue(self._node_rect().contains(tile))

    # ------------------------------------------------------------ 选择状态
    def test_selected_image_index_clamps_and_survives_shrink(self) -> None:
        self._set_images(_matrix(4))
        self.item.set_selected_image_index(99)
        self.assertEqual(self.item.selected_image_index, 3)
        self.item.set_selected_image_index(-7)
        self.assertEqual(self.item.selected_image_index, 0)
        self.item.set_selected_image_index("2")  # type: ignore[arg-type]
        self.assertEqual(self.item.selected_image_index, 2)

        # 张数减少时保持尽可能相同的索引，且不崩溃。
        self._set_images(_matrix(2))
        self.assertEqual(self.item.image_count, 2)
        self.assertEqual(self.item.selected_image_index, 1)
        self.assertEqual(len(self.item.image_tile_rects()), 2)

        self._set_images(_matrix(1))
        self.assertEqual(self.item._content_mode, "thumb")
        self.assertFalse(self.item.has_image_grid)
        self.assertEqual(self.item.image_count, 0)
        self.assertEqual(self.item.matrix_image_count, 1)
        self.assertEqual(self.item.selected_image_index, 0)
        self.assertIsNotNone(self.item.thumbnail_pixmap)

        self._set_images(_matrix(3))
        self.assertTrue(self.item.has_image_grid)

    def test_set_selected_image_index_without_grid_is_safe(self) -> None:
        self.item.set_selected_image_index(3)
        self.assertEqual(self.item.selected_image_index, 0)
        self._set_images(_image(20, 16, 60))
        self.item.set_selected_image_index(9)
        self.assertEqual(self.item.selected_image_index, 0)

    # ---------------------------------------------------------------- 交互
    def test_click_thumbnail_selects_image_without_selecting_node(self) -> None:
        self._set_images(_matrix(3))
        self.view.set_selected_node(None)
        self.assertFalse(self.item.isSelected())

        emitted: list[tuple[str, int]] = []
        selection_events: list[str] = []
        self.view.node_image_selected.connect(
            lambda node_id, index: emitted.append((node_id, index))
        )
        self.view.selection_changed.connect(selection_events.append)
        start_pos = QPointF(self.item.pos())

        tile_center = self.item.image_tile_rects()[1].center()
        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            self._tile_view_pos(1),
        )
        self._process()

        self.assertEqual(self.item.selected_image_index, 1)
        self.assertEqual(emitted, [(self.node.id, 1)])
        self.assertFalse(self.item.isSelected())
        self.assertEqual(self.item.pos(), start_pos)
        self.assertTrue(self.item.image_tile_rects()[1].contains(tile_center))

        # 点击缩略图不应让画布误发“节点选中”信号（否则工作区预览会被清空）。
        QTest.qWait(SINGLE_CLICK_DELAY_MS + 120)
        self.assertEqual(selection_events, [])
        self.assertFalse(self.item.isSelected())

    def test_drag_from_thumbnail_does_not_move_node(self) -> None:
        self._set_images(_matrix(3))
        self.view.set_selected_node(None)
        start_pos = QPointF(self.item.pos())
        start = self._tile_view_pos(0)
        target = start + QPoint(26, 18)

        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            start,
        )
        move = QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(target),
            QPointF(self.view.viewport().mapToGlobal(target)),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(self.view.viewport(), move)
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            target,
        )
        self._process()

        self.assertEqual(self.item.pos(), start_pos)
        self.assertFalse(self.item.isSelected())
        self.assertEqual(self.item.selected_image_index, 0)

    def test_click_node_other_areas_selects_node(self) -> None:
        self._set_images(_matrix(3))
        self.view.set_selected_node(None)

        header_center = self.item._header_rect.center()
        self._click_local(header_center)
        self.assertTrue(self.item.isSelected())
        self.assertEqual(self.item.selected_image_index, 0)

        # footer 同样属于节点其它区域，点击后按普通节点选中。
        self.view.set_selected_node(None)
        footer_center = self.item._footer_rect.center()
        self._click_local(footer_center)
        self.assertTrue(self.item.isSelected())
        self.assertEqual(self.item.selected_image_index, 0)

    def test_click_other_tile_then_header_keeps_index_and_selects_node(self) -> None:
        self._set_images(_matrix(3))
        self.view.set_selected_node(None)
        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            self._tile_view_pos(2),
        )
        self._process()
        self.assertEqual(self.item.selected_image_index, 2)
        self.assertFalse(self.item.isSelected())

        self._click_local(self.item._header_rect.center())
        self.assertTrue(self.item.isSelected())
        self.assertEqual(self.item.selected_image_index, 2)

    def test_single_image_keeps_thumb_mode_and_click_selects_node(self) -> None:
        self._set_images(_image(36, 24, 90))
        self.assertEqual(self.item._content_mode, "thumb")
        self.assertFalse(self.item.has_image_grid)
        self.assertEqual(self.item.image_count, 0)
        self.assertEqual(self.item.matrix_image_count, 1)
        self.assertIsNotNone(self.item.thumbnail_pixmap)
        self.assertFalse(self.item.thumbnail_pixmap.isNull())
        self.assertEqual(self.item.image_tile_rects(), [])

        self.view.set_selected_node(None)
        self._click_local(self.item._thumb_rect.center())
        self.assertTrue(self.item.isSelected())
        self.assertEqual(self.item.selected_image_index, 0)

    def test_thumbnails_disabled_shows_matrix_text_without_grid(self) -> None:
        self._set_images(_matrix(3))
        self.item.set_selected_image_index(2)
        grid_height = self.item.size().height()

        self.view.set_thumbnails_enabled(False)
        self._process()
        self.assertFalse(self.item.has_image_grid)
        self.assertEqual(self.item.image_count, 0)
        self.assertEqual(self.item._content_mode, "matrix")
        self.assertIn("图像矩阵", self.item._content_label)
        self.assertIn("3", self.item._content_label)
        self.assertEqual(self.item.image_tile_rects(), [])
        self.assertLess(self.item.size().height(), grid_height)
        # 关闭缩略图后点击中部文本区域按普通节点选中。
        self.view.set_selected_node(None)
        self._click_local(self.item._content_rect.center())
        self.assertTrue(self.item.isSelected())

        self.view.set_thumbnails_enabled(True)
        self._process()
        self.assertTrue(self.item.has_image_grid)
        self.assertEqual(self.item.image_count, 3)
        self.assertEqual(self.item.selected_image_index, 2)

    def test_double_click_thumbnail_does_not_open_parameters(self) -> None:
        self._set_images(_matrix(3))
        self.view.set_selected_node(None)
        activated: list[str] = []
        selection_events: list[str] = []
        self.view.node_activated.connect(activated.append)
        self.view.selection_changed.connect(selection_events.append)

        QTest.mouseDClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            self._tile_view_pos(1),
        )
        self._process()
        QTest.qWait(SINGLE_CLICK_DELAY_MS + 120)

        self.assertEqual(activated, [])
        self.assertEqual(selection_events, [])
        self.assertEqual(self.item.selected_image_index, 1)
        self.assertFalse(self.item.isSelected())

    # ------------------------------------------------------- 视图 / 粘贴辅助
    def test_recreate_node_item_keeps_selected_image_index(self) -> None:
        self._set_images(_matrix(4))
        self.item.set_selected_image_index(2)
        self.view._recreate_node_item(self.node.id)
        self._process()
        item = self.view._nodes[self.node.id]
        self.assertIsNot(item, self.item)
        self.assertEqual(item.image_count, 4)
        self.assertEqual(item.selected_image_index, 2)
        self.assertEqual(self.view.selected_image_index(self.node.id), 2)

    def test_paste_source_matrix_requires_all_images(self) -> None:
        complete = self.workspace.add_source_images(
            [_image(10, 10, 10), _image(12, 12, 20)],
            ["one.png", "two.png"],
            pos=(0.0, 300.0),
        )
        missing = self.workspace.add_source_images(
            [_image(14, 14, 30)], ["three.png"], pos=(0.0, 460.0)
        )
        self._process()
        self.view.set_selected_node(None)
        self.view._nodes[complete.id].setSelected(True)
        self.view._nodes[missing.id].setSelected(True)
        self.assertTrue(self.view.copy_selected())

        # 其中一个源节点的图片从工作区移除后，整张节点不允许粘贴。
        self.workspace.images.pop(missing.source_ids[0])
        messages: list[str] = []
        self.view.status_message.connect(messages.append)
        new_ids = self.view.paste_nodes()

        self.assertEqual(len(new_ids), 1)
        pasted = self.graph.nodes[new_ids[0]]
        self.assertTrue(pasted.is_source)
        self.assertEqual(pasted.source_ids, complete.source_ids)
        self.assertTrue(any("跳过 1" in message for message in messages), messages)


if __name__ == "__main__":
    unittest.main(verbosity=2)
