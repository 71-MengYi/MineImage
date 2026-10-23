"""图像拖出 / 拖入合并的端到端界面测试。"""

from __future__ import annotations

import os
import time
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.config import AppSettings
from app.core.imageio import save_image
from app.core.types import ImageData
from app.graph.image_drag import ImageDragPayload, clear as clear_drag_payloads
from app.operators.loader import OperatorRegistry
from app.ui.main_window import MainWindow
from app.ui.theme import apply_theme

_TMP = Path(__file__).resolve().parent.parent / ".test_tmp"
_TMP.mkdir(exist_ok=True)
_APP = QApplication.instance() or QApplication([])
apply_theme(_APP)


def _write_image(path: Path, width: int, height: int, shade: int) -> Path:
    array = np.full((height, width, 3), shade, dtype=np.uint8)
    save_image(path, ImageData(array, path.name))
    return path


class ImageDragUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = OperatorRegistry.build()
        cls.paths = [
            _write_image(_TMP / "drag_a.png", 64, 48, 30),
            _write_image(_TMP / "drag_b.png", 32, 40, 120),
            _write_image(_TMP / "drag_c.png", 48, 24, 200),
        ]

    def setUp(self) -> None:  # noqa: N802
        clear_drag_payloads()
        self.win = MainWindow(AppSettings(_TMP / f"drag_ui_{uuid.uuid4().hex[:8]}.ini"))
        self.win.resize(1480, 900)
        self.win.show()
        _APP.processEvents()

    def tearDown(self) -> None:  # noqa: N802
        clear_drag_payloads()
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
        self.win.open_image_paths([str(path) for path in self.paths])
        widget = self.win._active_workspace()
        self.assertIsNotNone(widget)
        self._wait_idle(widget)
        return widget

    @staticmethod
    def _payload_from_result(result, index: int, source_node_id: str, title: str):
        image = result.primary_images[index]
        return ImageDragPayload(
            image=image.copy(),
            name=image.name or title,
            path=None,
            source_node_id=source_node_id,
            source_index=index,
            source_title=title,
        )

    def test_drag_result_image_into_source_matrix_recomputes_downstream(self) -> None:
        widget = self._open_matrix()
        graph = widget.workspace.graph
        source = graph.source_nodes()[0]
        gray_id = widget.graph_view.add_operator_at("basic.grayscale")
        self.assertIsNotNone(gray_id)
        assert gray_id is not None
        graph.add_connection(source.id, "image", gray_id, "image")
        widget._on_node_selected(gray_id)
        self._wait_idle(widget)

        gray_result = widget.engine.result(gray_id)
        self.assertIsNotNone(gray_result)
        self.assertEqual(gray_result.image_count, 3)
        source_result = widget.engine.result(source.id)
        self.assertEqual(source_result.image_count, 3)

        payload = self._payload_from_result(
            gray_result, 1, gray_id, graph.nodes[gray_id].display_title()
        )
        # 自动布局把源节点与灰度节点分开，避免落点被上层节点遮挡。
        widget.graph_view.auto_layout()
        _APP.processEvents()
        target_item = widget.graph_view._nodes[source.id]
        self.assertTrue(
            widget.graph_view.handle_image_drop(
                target_item.sceneBoundingRect().center(), payload
            )
        )
        self._wait_idle(widget)

        self.assertEqual(widget.engine.result(source.id).image_count, 4)
        self.assertEqual(widget.engine.result(gray_id).image_count, 4)
        self.assertEqual(widget._current_node_id, source.id)
        self.assertIsNotNone(widget.preview.current_image())
        self.assertEqual(len(widget.workspace.images_for_node(source)), 4)
        self.assertIn("图像矩阵 4 张", widget.preview.info_text())

        # 撤销后源节点与下游都回到 3 张，并重新计算。
        widget.undo_stack.undo()
        self._wait_idle(widget)
        self.assertEqual(widget.engine.result(source.id).image_count, 3)
        self.assertEqual(widget.engine.result(gray_id).image_count, 3)
        widget.undo_stack.redo()
        self._wait_idle(widget)
        self.assertEqual(widget.engine.result(source.id).image_count, 4)

    def test_drag_image_to_blank_creates_source_and_undo(self) -> None:
        widget = self._open_matrix()
        graph = widget.workspace.graph
        source = graph.source_nodes()[0]
        result = widget.engine.result(source.id)
        payload = self._payload_from_result(
            result, 2, source.id, graph.nodes[source.id].display_title()
        )
        before = set(graph.nodes)
        self.assertTrue(
            widget.graph_view.handle_image_drop(
                widget.graph_view.mapToScene(650, 420), payload
            )
        )
        self._wait_idle(widget)
        new_ids = [node_id for node_id in graph.nodes if node_id not in before]
        self.assertEqual(len(new_ids), 1)
        new_node = graph.nodes[new_ids[0]]
        self.assertTrue(new_node.is_source)
        self.assertEqual(widget._current_node_id, new_node.id)
        self.assertEqual(widget.engine.result(new_node.id).image_count, 1)
        self.assertIsNotNone(widget.preview.current_image())
        self.assertEqual(widget.preview.current_image().width, 48)

        widget.undo_stack.undo()
        self._wait_idle(widget)
        self.assertNotIn(new_node.id, graph.nodes)

    def test_delete_selected_subimage_recomputes_source_and_downstream(self) -> None:
        widget = self._open_matrix()
        graph = widget.workspace.graph
        source = graph.source_nodes()[0]
        gray_id = widget.graph_view.add_operator_at("basic.grayscale")
        assert gray_id is not None
        graph.add_connection(source.id, "image", gray_id, "image")
        widget._on_node_selected(source.id)
        self._wait_idle(widget)
        self.assertEqual(widget.engine.result(source.id).image_count, 3)
        self.assertEqual(widget.engine.result(gray_id).image_count, 3)

        widget.graph_view.auto_layout()
        widget.graph_view.fit_graph()
        _APP.processEvents()
        QTest.qWait(50)
        _APP.processEvents()
        item = widget.graph_view._nodes[source.id]
        self.assertTrue(item.has_image_grid)
        tile = item.image_tile_rects()[1].center()
        view_pos = widget.graph_view.mapFromScene(item.mapToScene(tile))
        QTest.mouseClick(
            widget.graph_view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            view_pos,
        )
        _APP.processEvents()
        self.assertEqual(widget.graph_view.selected_subimage(), (source.id, 1))

        widget.graph_view.setFocus()
        QTest.keyClick(widget.graph_view, Qt.Key.Key_Delete)
        self._wait_idle(widget)
        self.assertEqual(len(source.source_ids), 2)
        self.assertEqual(widget.engine.result(source.id).image_count, 2)
        self.assertEqual(widget.engine.result(gray_id).image_count, 2)
        self.assertIn("图像矩阵 2 张", widget.preview.info_text())
        self.assertEqual(widget._current_node_id, source.id)

        widget.undo_stack.undo()
        self._wait_idle(widget)
        self.assertEqual(len(source.source_ids), 3)
        self.assertEqual(widget.engine.result(source.id).image_count, 3)
        self.assertEqual(widget.engine.result(gray_id).image_count, 3)

    def test_merge_two_source_nodes_and_recompute_connected_branch(self) -> None:
        widget = self._open_matrix()
        graph = widget.workspace.graph
        target = graph.source_nodes()[0]
        gray_id = widget.graph_view.add_operator_at("basic.grayscale")
        assert gray_id is not None
        graph.add_connection(target.id, "image", gray_id, "image")
        extra = widget.workspace.add_source_images(
            [ImageData(np.full((20, 20, 3), 250, dtype=np.uint8), "extra.png")],
            [None],
            pos=(0.0, 420.0),
        )
        widget._on_node_selected(gray_id)
        self._wait_idle(widget)
        self.assertEqual(widget.engine.result(gray_id).image_count, 3)
        target_count = len(target.source_ids)

        self.assertTrue(widget.graph_view.merge_source_nodes(extra.id, target.id))
        self._wait_idle(widget)
        self.assertNotIn(extra.id, graph.nodes)
        self.assertEqual(len(target.source_ids), target_count + 1)
        self.assertEqual(widget.engine.result(gray_id).image_count, 4)
        self.assertEqual(widget._current_node_id, target.id)

        widget.undo_stack.undo()
        self._wait_idle(widget)
        self.assertIn(extra.id, graph.nodes)
        self.assertEqual(len(target.source_ids), target_count)
        self.assertEqual(widget.engine.result(gray_id).image_count, 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
