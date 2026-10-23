"""图像矩阵的端到端界面测试：多图打开、节点网格点选、批量计算与预览。"""

from __future__ import annotations

import os
import time
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.config import AppSettings
from app.core.imageio import load_image, save_image
from app.core.types import ImageData, ImageList
from app.operators.loader import OperatorRegistry
from app.ui.main_window import MainWindow
from app.ui.theme import apply_theme

_TMP = Path(__file__).resolve().parent.parent / ".test_tmp"
_TMP.mkdir(exist_ok=True)
_APP = QApplication.instance() or QApplication([])
apply_theme(_APP)


def _write_image(path: Path, width: int, height: int, color: tuple[int, int, int]) -> Path:
    array = np.zeros((height, width, 3), dtype=np.uint8)
    array[:, :] = color
    save_image(path, ImageData(array, path.name))
    return path


class ImageMatrixUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = OperatorRegistry.build()
        cls.first_path = _write_image(_TMP / "matrix_first.png", 80, 60, (10, 20, 30))
        cls.second_path = _write_image(_TMP / "matrix_second.png", 40, 30, (200, 150, 100))

    def setUp(self) -> None:  # noqa: N802
        self.win = MainWindow(AppSettings(_TMP / f"matrix_ui_{uuid.uuid4().hex[:8]}.ini"))
        self.win.resize(1480, 900)
        self.win.show()
        _APP.processEvents()

    def tearDown(self) -> None:  # noqa: N802
        for widget in list(self.win._workspaces):
            try:
                widget.mark_clean()
            except Exception:  # noqa: BLE001
                pass
        self.win.close()
        self.win.deleteLater()
        _APP.processEvents()

    def _wait_idle(self, widget, timeout: float = 12.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            _APP.processEvents()
            if widget.engine.busy:
                QTest.qWait(15)
                continue
            _APP.processEvents()
            QTest.qWait(60)
            if not widget.engine.busy:
                return
        self.fail("后台计算超时")

    def _open_matrix(self):
        self.win.open_image_paths([str(self.first_path), str(self.second_path)])
        widget = self.win._active_workspace()
        self.assertIsNotNone(widget)
        self._wait_idle(widget)
        return widget

    @staticmethod
    def _tile_center(item, index: int) -> QPointF:
        return item.image_tile_rects()[index].center()

    def _click_item_point(self, widget, item, local: QPointF) -> None:
        scene_point = item.mapToScene(local)
        view_point = widget.graph_view.mapFromScene(scene_point)
        QTest.mouseClick(
            widget.graph_view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            QPoint(view_point.x(), view_point.y()),
        )
        _APP.processEvents()

    def test_open_multiple_images_creates_one_matrix_source(self) -> None:
        widget = self._open_matrix()
        graph = widget.workspace.graph
        sources = graph.source_nodes()
        self.assertEqual(self.win.tabs.count(), 1)
        self.assertEqual(len(sources), 1)
        source = sources[0]
        self.assertEqual(len(source.source_ids), 2)
        self.assertEqual(len(widget.workspace.images), 2)
        self.assertEqual(
            [image.width for image in widget.workspace.images_for_node(source)], [80, 40]
        )

        result = widget.engine.result(source.id)
        self.assertIsNotNone(result)
        self.assertTrue(result.has_image_matrix)
        self.assertEqual(result.image_count, 2)
        self.assertIsInstance(result.outputs["image"], ImageList)
        self.assertEqual([image.width for image in result.primary_images], [80, 40])

        item = widget.graph_view._nodes[source.id]
        self.assertEqual(item._content_mode, "grid")
        self.assertEqual(item.image_count, 2)
        self.assertEqual(len(item.image_tile_rects()), 2)
        self.assertIsNotNone(widget.preview.current_image())
        self.assertEqual(widget.preview.current_image().width, 80)

    def test_clicking_tile_selects_image_without_selecting_node(self) -> None:
        widget = self._open_matrix()
        source = widget.workspace.graph.source_nodes()[0]
        item = widget.graph_view._nodes[source.id]
        widget.graph_view.fit_graph()
        _APP.processEvents()
        # 打开图像时源节点默认选中；点缩略图应只切换单图并清掉节点选中框。
        self.assertTrue(item.isSelected())
        self.assertEqual(widget._current_node_id, source.id)

        self._click_item_point(widget, item, self._tile_center(item, 1))
        QTest.qWait(60)
        _APP.processEvents()
        self.assertEqual(item.selected_image_index, 1)
        self.assertFalse(item.isSelected())
        self.assertEqual(widget._current_node_id, source.id)
        self.assertIsNotNone(widget.preview.current_image())
        self.assertEqual(widget.preview.current_image().width, 40)
        self.assertEqual(widget.preview.current_image().height, 30)
        self.assertIn("图像矩阵 2 张", self.win._status_labels["node"].text())
        self.assertIn("第 2/2 张", self.win._status_labels["node"].text())
        self.assertIn("40×30", self.win._status_labels["size"].text())

        # 点击节点标题区域才选中节点，且保留刚才选中的第 2 张。
        widget.graph_view.set_selected_node(None)
        _APP.processEvents()
        header_point = QPointF(item.node_size().width() / 2.0, item._header_rect.center().y())
        self._click_item_point(widget, item, header_point)
        QTest.qWait(320)
        _APP.processEvents()
        self.assertTrue(item.isSelected())
        self.assertEqual(item.selected_image_index, 1)
        self.assertEqual(widget.preview.current_image().width, 40)

    def test_batch_operator_result_grid_and_compare_original(self) -> None:
        widget = self._open_matrix()
        graph = widget.workspace.graph
        source = graph.source_nodes()[0]
        gray_id = widget.graph_view.add_operator_at("basic.grayscale")
        self.assertIsNotNone(gray_id)
        assert gray_id is not None
        graph.add_connection(source.id, "image", gray_id, "image")
        widget._on_node_selected(gray_id)
        self._wait_idle(widget)

        result = widget.engine.result(gray_id)
        self.assertIsNotNone(result)
        self.assertTrue(result.has_image_matrix)
        self.assertEqual(result.image_count, 2)
        self.assertEqual([image.width for image in result.primary_images], [80, 40])
        self.assertTrue(
            all(image.color_space == "L" for image in result.primary_images),
            [image.color_space for image in result.primary_images],
        )

        item = widget.graph_view._nodes[gray_id]
        self.assertEqual(item._content_mode, "grid")
        widget.graph_view.fit_graph()
        _APP.processEvents()
        widget.graph_view.set_selected_node(None)
        self._click_item_point(widget, item, self._tile_center(item, 1))
        QTest.qWait(60)
        _APP.processEvents()
        self.assertEqual(item.selected_image_index, 1)
        self.assertFalse(item.isSelected())
        image = widget.preview.current_image()
        self.assertIsNotNone(image)
        self.assertEqual((image.width, image.height), (40, 30))
        # 对比模式的原始图像跟随相同序号。
        self.assertIsNotNone(widget.preview._original)
        self.assertEqual(widget.preview._original.width, 40)

        # 导出当前选中的第 2 张。
        out = _TMP / f"matrix_export_{uuid.uuid4().hex[:6]}.png"
        self.addCleanup(lambda: out.unlink(missing_ok=True))
        self.assertTrue(widget.export_current_image(out))
        exported = load_image(out)
        self.assertEqual((exported.width, exported.height), (40, 30))
        # 灰度化输出是单通道 (H, W)，导出再读回仍应保持相同形状。
        self.assertEqual(exported.array.shape, result.primary_images[1].array.shape)

    def test_project_round_trip_keeps_matrix(self) -> None:
        widget = self._open_matrix()
        project = _TMP / f"matrix_ui_{uuid.uuid4().hex[:8]}.mi"
        self.addCleanup(lambda: project.unlink(missing_ok=True))
        widget.save_to(project)
        self.win.open_project_path(project)
        loaded = self.win._active_workspace()
        self._wait_idle(loaded)
        sources = loaded.workspace.graph.source_nodes()
        self.assertEqual(len(sources), 1)
        self.assertEqual(len(sources[0].source_ids), 2)
        self.assertEqual(len(loaded.workspace.images_for_node(sources[0])), 2)
        result = loaded.engine.result(sources[0].id)
        self.assertIsNotNone(result)
        self.assertEqual(result.image_count, 2)
        item = loaded.graph_view._nodes[sources[0].id]
        self.assertEqual(item._content_mode, "grid")
        self.assertEqual(item.image_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
