"""图像拖动合并的数据层：工作区图像登记、源节点 id 管理与撤销命令。"""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import QApplication

from app.core.model import GraphError, WorkspaceModel
from app.core.types import ImageData
from app.graph.commands import AppendSourceImagesCommand, RemoveSourceImageCommand

_APP: QApplication | None = None


def _image(value: int, name: str = "img") -> ImageData:
    return ImageData(np.full((6, 8, 3), value, dtype=np.uint8), name)


def setUpModule() -> None:  # noqa: N802
    global _APP
    _APP = QApplication.instance() or QApplication([])


def tearDownModule() -> None:  # noqa: N802
    global _APP
    if _APP is not None:
        _APP.processEvents()
        _APP = None


class WorkspaceSourceImageTests(unittest.TestCase):
    def test_store_image_registers_copy_metadata(self) -> None:
        workspace = WorkspaceModel("拖图")
        image = _image(10, "原图")
        workspace.image_paths  # metadata dict exists
        image_id = workspace.store_image(image, "C:/images/a.png")
        self.assertIn(image_id, workspace.images)
        self.assertIs(workspace.images[image_id], image)
        self.assertEqual(workspace.image_paths[image_id], "C:/images/a.png")
        self.assertEqual(image.name, "a.png")

    def test_set_source_image_ids_updates_node_and_revision(self) -> None:
        workspace = WorkspaceModel("拖图")
        node = workspace.add_source_images([_image(1), _image(2)])
        extra = workspace.store_image(_image(3), None)
        before = node.revision
        changed: list[str] = []
        workspace.graph.node_changed.connect(changed.append)

        workspace.set_source_image_ids(node.id, [node.source_ids[0], extra])

        self.assertEqual(node.source_ids, [node.source_ids[0], extra])
        self.assertEqual(node.source_image_id, node.source_ids[0])
        self.assertEqual(node.revision, before + 1)
        self.assertEqual(changed, [node.id])

    def test_set_source_image_ids_validates_node_and_images(self) -> None:
        workspace = WorkspaceModel("拖图")
        node = workspace.add_source_images([_image(1)])
        with self.assertRaises(GraphError):
            workspace.set_source_image_ids("missing", [])
        with self.assertRaises(GraphError):
            workspace.set_source_image_ids(node.id, ["missing-image"])
        with self.assertRaises(GraphError):
            workspace.set_source_image_ids(
                workspace.graph.create_operator_node("basic.grayscale", "灰度").id, []
            )

    def test_append_deduplicates_and_keeps_order(self) -> None:
        workspace = WorkspaceModel("拖图")
        node = workspace.add_source_images([_image(1)])
        a = workspace.store_image(_image(2))
        b = workspace.store_image(_image(3))
        workspace.append_source_image_ids(node.id, [a, b, a])
        self.assertEqual(node.source_ids, [node.source_ids[0], a, b])


class AppendSourceImagesCommandTests(unittest.TestCase):
    def _make(self):
        workspace = WorkspaceModel("拖图")
        node = workspace.add_source_images([_image(1)])
        stack = QUndoStack()
        return workspace, node, stack

    def test_command_with_new_images_undo_and_redo(self) -> None:
        workspace, node, stack = self._make()
        old_ids = list(node.source_ids)
        source_image = _image(7, "拖入图")
        command = AppendSourceImagesCommand(
            workspace,
            workspace.graph,
            node.id,
            images=[source_image],
            paths=["C:/images/drag.png"],
        )
        stack.push(command)
        self.assertEqual(len(node.source_ids), 2)
        self.assertEqual(len(workspace.images), 2)
        appended_id = node.source_ids[1]
        self.assertEqual(workspace.image_paths[appended_id], "C:/images/drag.png")
        self.assertIsNot(workspace.images[appended_id], source_image)
        self.assertTrue(np.array_equal(workspace.images[appended_id].array, source_image.array))

        stack.undo()
        self.assertEqual(node.source_ids, old_ids)
        self.assertEqual(len(workspace.images), 2)  # 图像条目保留，保存时再清理

        stack.redo()
        self.assertEqual(node.source_ids, old_ids + [appended_id])
        self.assertEqual(len(workspace.images), 2)

    def test_redo_after_unused_image_pruned_restores_copy(self) -> None:
        workspace, node, stack = self._make()
        old_ids = list(node.source_ids)
        command = AppendSourceImagesCommand(
            workspace,
            workspace.graph,
            node.id,
            images=[_image(9, "临时图")],
            paths=[None],
        )
        stack.push(command)
        appended_id = node.source_ids[1]
        self.assertEqual(len(workspace.images), 2)

        stack.undo()
        workspace.prune_unused_images()  # 撤销后保存工程会清理未被引用的副本
        self.assertNotIn(appended_id, workspace.images)

        stack.redo()
        self.assertEqual(node.source_ids[: len(old_ids)], old_ids)
        self.assertEqual(len(node.source_ids), 2)
        restored_id = node.source_ids[1]
        self.assertIn(restored_id, workspace.images)
        self.assertTrue(np.array_equal(workspace.images[restored_id].array, _image(9).array))

    def test_remove_source_image_command_undo_redo(self) -> None:
        workspace, node, stack = self._make()
        workspace.append_source_image_ids(
            node.id, [workspace.store_image(_image(20)), workspace.store_image(_image(30))]
        )
        original_ids = list(node.source_ids)
        removed_id = original_ids[1]
        command = RemoveSourceImageCommand(
            workspace, workspace.graph, node.id, removed_id
        )
        stack.push(command)
        self.assertEqual(node.source_ids, [original_ids[0], original_ids[2]])
        self.assertEqual(command.image_id, removed_id)
        self.assertIn(removed_id, workspace.images)

        stack.undo()
        self.assertEqual(node.source_ids, original_ids)
        stack.redo()
        self.assertEqual(node.source_ids, [original_ids[0], original_ids[2]])

    def test_command_with_existing_image_ids(self) -> None:
        workspace, target, stack = self._make()
        source = workspace.add_source_images([_image(4), _image(5)])
        old_target = list(target.source_ids)
        command = AppendSourceImagesCommand(
            workspace,
            workspace.graph,
            target.id,
            image_ids=source.source_ids,
        )
        stack.push(command)
        self.assertEqual(target.source_ids, old_target + source.source_ids)
        self.assertEqual(len(workspace.images), 3)

        stack.undo()
        self.assertEqual(target.source_ids, old_target)
        stack.redo()
        self.assertEqual(target.source_ids, old_target + source.source_ids)


if __name__ == "__main__":
    unittest.main(verbosity=2)
