"""选中图像矩阵中的子图后按 Delete / Backspace 删除。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QUndoStack
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.config import AppSettings
from app.core.model import WorkspaceModel
from app.core.types import ImageData, ImageList
from app.engine.engine import NodeResult
from app.graph.view import GraphView

_APP: QApplication | None = None


class _Registry:
    def spec(self, op_id):
        return None

    def get(self, op_id):
        return None

    def entry(self, op_id):
        return None


def _image(value: int, width: int = 20, height: int = 16) -> ImageData:
    return ImageData(np.full((height, width, 3), value, dtype=np.uint8), f"img_{value}")


def setUpModule() -> None:  # noqa: N802
    global _APP
    _APP = QApplication.instance() or QApplication([])


def tearDownModule() -> None:  # noqa: N802
    global _APP
    if _APP is not None:
        _APP.processEvents()
        _APP = None


class DeleteSubImageTests(unittest.TestCase):
    def setUp(self) -> None:
        self._temp_dir = tempfile.TemporaryDirectory(prefix="mineimage-delete-subimage-")
        self.settings = AppSettings(Path(self._temp_dir.name) / "test.ini")
        self.workspace = WorkspaceModel("删除子图测试")
        self.graph = self.workspace.graph
        self.undo = QUndoStack()
        self.view = GraphView(self.workspace, _Registry(), self.undo, self.settings)
        self.view.resize(1000, 700)
        self.view.show()
        self._process()

    def tearDown(self) -> None:
        self.view.close()
        self.view.deleteLater()
        self._process()
        try:
            self._temp_dir.cleanup()
        except OSError:
            pass

    def _process(self) -> None:
        if _APP is not None:
            _APP.processEvents()

    def _add_matrix_source(self, count: int = 3, pos=(0.0, 0.0)):
        images = [_image(30 + index * 30, 20 + index * 4, 16 + index * 2) for index in range(count)]
        node = self.workspace.add_source_images(images, None, pos=pos)
        self.view.notify_result(
            node.id,
            NodeResult(
                node.id,
                {"image": ImageList(tuple(images))},
                1.0,
                node.revision,
            ),
        )
        self.view.fit_graph()
        self._process()
        return node

    def _select_tile(self, node, index: int) -> None:
        item = self.view._nodes[node.id]
        self.assertIsNotNone(item.image_index_at(item.image_tile_rects()[index].center()))
        # 与 NodeItem 的真实点击顺序一致：先更新卡片选中序号，再通知视图。
        item.set_selected_image_index(index)
        self.view.on_node_image_selected(item, index)
        self._process()

    def test_delete_middle_subimage_keeps_node_and_undoes(self) -> None:
        node = self._add_matrix_source(3)
        original_ids = list(node.source_ids)
        item = self.view._nodes[node.id]
        self._select_tile(node, 1)
        self.assertEqual(self.view.selected_subimage(), (node.id, 1))
        signals: list[str] = []
        self.view.source_images_changed.connect(signals.append)
        self.view.setFocus()
        QTest.keyClick(self.view, Qt.Key.Key_Delete)
        self._process()

        self.assertIn(node.id, self.graph.nodes)
        self.assertEqual(node.source_ids, [original_ids[0], original_ids[2]])
        self.assertEqual(item.selected_image_index, 1)
        self.assertEqual(self.view.selected_subimage(), (node.id, 1))
        self.assertIn(node.id, signals)
        self.assertEqual(self.undo.count(), 1)

        self.undo.undo()
        self._process()
        self.assertEqual(node.source_ids, original_ids)
        self.assertEqual(len(self.workspace.images), 3)
        self.undo.redo()
        self._process()
        self.assertEqual(node.source_ids, [original_ids[0], original_ids[2]])

    def test_backspace_deletes_subimage(self) -> None:
        node = self._add_matrix_source(2)
        original_ids = list(node.source_ids)
        self._select_tile(node, 0)
        self.view.setFocus()
        QTest.keyClick(self.view, Qt.Key.Key_Backspace)
        self._process()
        self.assertEqual(node.source_ids, [original_ids[1]])
        self.assertIn(node.id, self.graph.nodes)
        # 只剩一张后退回单图模式，不再保留子图选中状态。
        self.assertIsNone(self.view.selected_subimage())
        self.assertTrue(self.view._nodes[node.id].isSelected())

    def test_delete_subimage_on_result_node_is_ignored(self) -> None:
        node = self.graph.create_operator_node("test.op", "测试算子")
        images = (_image(10), _image(20), _image(30))
        self.view.notify_result(
            node.id,
            NodeResult(node.id, {"image": ImageList(images)}, 1.0, node.revision),
        )
        self.view.fit_graph()
        self._process()
        self._select_tile(node, 1)
        messages: list[str] = []
        self.view.status_message.connect(messages.append)
        self.view.setFocus()
        QTest.keyClick(self.view, Qt.Key.Key_Delete)
        self._process()
        self.assertIn(node.id, self.graph.nodes)
        self.assertEqual(self.undo.count(), 0)
        self.assertTrue(any("不能直接删除" in message for message in messages), messages)

    def test_delete_with_node_selected_still_deletes_node(self) -> None:
        node = self._add_matrix_source(3)
        item = self.view._nodes[node.id]
        self.view.set_selected_node(node.id)
        self._process()
        self.assertTrue(item.isSelected())
        self.view.setFocus()
        QTest.keyClick(self.view, Qt.Key.Key_Delete)
        self._process()
        self.assertNotIn(node.id, self.graph.nodes)
        self.assertEqual(self.undo.count(), 1)

    def test_escape_and_blank_click_clear_subimage_target(self) -> None:
        node = self._add_matrix_source(3)
        self._select_tile(node, 1)
        self.view.setFocus()
        QTest.keyClick(self.view, Qt.Key.Key_Escape)
        self._process()
        self.assertIsNone(self.view.selected_subimage())

        self._select_tile(node, 1)
        # 模拟点击空白处清空选择。
        self.view._clear_selected_subimage()
        self.view._scene.clearSelection()
        self._process()
        self.assertIsNone(self.view.selected_subimage())


if __name__ == "__main__":
    unittest.main(verbosity=2)
