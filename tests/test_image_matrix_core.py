"""图像矩阵数据层：ImageList、多图源节点与工程往返。"""

from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

from app.core.data_types import DataType, NumberValue, data_type_of, summarize_value
from app.core.model import NodeModel, WorkspaceModel
from app.core.project import load_project, save_project
from app.core.types import ImageData, ImageList, first_image, image_values
from app.operators.spec import PortSpec, coerce_output_value, validate_inputs

_APP: QApplication | None = None


def _image(width: int, height: int, value: int = 10, color_space: str = "BGR") -> ImageData:
    if color_space == "L":
        array = np.full((height, width), value, dtype=np.uint8)
    else:
        array = np.full((height, width, 3), value, dtype=np.uint8)
    return ImageData(array, f"img_{width}x{height}", color_space)


def setUpModule() -> None:  # noqa: N802
    global _APP
    _APP = QApplication.instance() or QApplication([])


def tearDownModule() -> None:  # noqa: N802
    global _APP
    if _APP is not None:
        _APP.processEvents()
        _APP = None


class ImageListTests(unittest.TestCase):
    def test_container_and_type(self) -> None:
        images = (_image(20, 10, 1), _image(12, 8, 2, "L"))
        matrix = ImageList(images)
        self.assertEqual(matrix.count, 2)
        self.assertEqual(len(matrix), 2)
        self.assertEqual(list(matrix), list(images))
        self.assertIs(matrix[1], images[1])
        self.assertIs(matrix.first, images[0])
        self.assertEqual(matrix.width, 20)
        self.assertEqual(matrix.height, 10)
        self.assertIn("2 张", matrix.summary())
        self.assertIs(data_type_of(matrix), DataType.IMAGE)
        self.assertIn("2 张", summarize_value(matrix))
        self.assertEqual(image_values(matrix), images)
        self.assertIs(first_image(matrix), images[0])
        self.assertEqual(image_values(images[0]), (images[0],))

    def test_accepts_numpy_and_rejects_unknown(self) -> None:
        matrix = ImageList((np.zeros((4, 5, 3), dtype=np.uint8),))
        self.assertEqual(matrix.count, 1)
        with self.assertRaises(ValueError):
            ImageList((object(),))  # type: ignore[arg-type]

    def test_image_at_and_copy(self) -> None:
        matrix = ImageList((_image(4, 3, 1), _image(6, 5, 2)))
        self.assertIs(matrix.image_at(1), matrix[1])
        self.assertIsNone(matrix.image_at(9))
        copied = matrix.copy()
        self.assertIsNot(copied[0], matrix[0])
        self.assertEqual(copied[0].array.tolist(), matrix[0].array.tolist())


class ImageMatrixPortTests(unittest.TestCase):
    def test_validate_inputs_accepts_matrix(self) -> None:
        from app.operators.spec import OperatorSpec

        spec = OperatorSpec(
            id="test.matrix",
            name="矩阵输入",
            description="",
            inputs=(PortSpec("image", "图像"),),
            outputs=(PortSpec("image", "输出"),),
        )
        matrix = ImageList((_image(10, 10), _image(20, 20, color_space="L")))
        validate_inputs(spec, {"image": matrix})

    def test_color_space_checked_for_every_image(self) -> None:
        from app.operators.spec import OperatorError, OperatorSpec

        spec = OperatorSpec(
            id="test.gray_only",
            name="仅灰度",
            inputs=(PortSpec("image", "图像", accepts=("L",)),),
            outputs=(PortSpec("image", "输出"),),
        )
        with self.assertRaises(OperatorError):
            validate_inputs(
                spec,
                {"image": ImageList((_image(10, 10, color_space="L"), _image(10, 10)))},
            )

    def test_empty_matrix_rejected_as_input(self) -> None:
        from app.operators.spec import OperatorError, OperatorSpec

        spec = OperatorSpec(
            id="test.empty",
            name="空矩阵",
            inputs=(PortSpec("image", "图像"),),
            outputs=(PortSpec("image", "输出"),),
        )
        with self.assertRaises(OperatorError):
            validate_inputs(spec, {"image": ImageList(())})

    def test_coerce_output_keeps_matrix_and_wraps_image_list(self) -> None:
        port = PortSpec("image", "输出")
        matrix = ImageList((_image(4, 4), _image(5, 5)))
        self.assertIs(coerce_output_value(port, matrix), matrix)
        wrapped = coerce_output_value(port, [_image(4, 4), _image(5, 5)])
        self.assertIsInstance(wrapped, ImageList)
        self.assertEqual(wrapped.count, 2)
        self.assertIsInstance(coerce_output_value(port, _image(4, 4)), ImageData)

    def test_batch_number_list_type(self) -> None:
        values = [NumberValue(1, "均值"), NumberValue(2, "均值")]
        self.assertIs(data_type_of(values), DataType.NUMBER)
        self.assertIn("数值 ×2", summarize_value(values))


class SourceNodeModelTests(unittest.TestCase):
    def test_legacy_single_image_node(self) -> None:
        data = {
            "id": "n1",
            "op_id": "__source__",
            "title": "旧图",
            "source_image_id": "img_old",
            "pos": [0.0, 0.0],
            "params": {},
        }
        node = NodeModel.from_dict(data)
        self.assertEqual(node.source_ids, ["img_old"])
        self.assertEqual(node.source_image_id, "img_old")
        self.assertEqual(node.to_dict()["source_image_ids"], ["img_old"])

    def test_workspace_add_source_images(self) -> None:
        workspace = WorkspaceModel("多图")
        images = [_image(10, 10, 1), _image(20, 8, 2), _image(6, 6, 3, "L")]
        node = workspace.add_source_images(images, ["a.png", "b.png", "c.png"])
        self.assertEqual(len(workspace.images), 3)
        self.assertEqual(node.source_ids, list(workspace.images.keys()))
        self.assertEqual(node.source_image_id, node.source_ids[0])
        self.assertEqual([item.name for item in workspace.images_for_node(node)], ["a.png", "b.png", "c.png"])
        self.assertEqual([item.width for item in workspace.images_for_node(node)], [10, 20, 6])
        self.assertIs(workspace.image_for_node(node), workspace.images[node.source_ids[0]])
        self.assertIsNone(workspace.image_for_node(NodeModel("x", "op", "node")))

    def test_prune_keeps_all_images_of_matrix_node(self) -> None:
        workspace = WorkspaceModel("多图")
        node = workspace.add_source_images([_image(4, 4), _image(4, 4)])
        workspace.images["img_orphan"] = _image(4, 4)
        workspace.image_paths["img_orphan"] = None
        workspace.prune_unused_images()
        self.assertEqual(len(workspace.images), 2)
        self.assertEqual(len(workspace.images_for_node(node)), 2)

    def test_graph_create_source_node_accepts_ids(self) -> None:
        workspace = WorkspaceModel("多图")
        node = workspace.graph.create_source_node(
            "first", "多图", None, (0.0, 0.0), image_ids=["first", "second"]
        )
        self.assertEqual(node.source_ids, ["first", "second"])
        self.assertEqual(node.source_image_id, "first")


class ProjectMatrixTests(unittest.TestCase):
    def test_project_round_trip_preserves_image_order(self) -> None:
        workspace = WorkspaceModel("矩阵工程")
        images = [
            _image(14, 10, 1),
            _image(20, 7, 2, "L"),
            _image(9, 9, 3),
        ]
        node = workspace.add_source_images(images, ["one.png", "two.png", "three.png"])
        # 使用仓库内的 .test_tmp：Windows 平台临时目录可能被安全软件拖慢/阻塞。
        temp_dir = Path(__file__).resolve().parent.parent / ".test_tmp"
        temp_dir.mkdir(exist_ok=True)
        path = temp_dir / "image_matrix_project.mi"
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        save_project(path, workspace)
        loaded = load_project(path)
        loaded_nodes = loaded.graph.source_nodes()
        self.assertEqual(len(loaded_nodes), 1)
        self.assertEqual(loaded_nodes[0].source_ids, node.source_ids)
        loaded_images = loaded.images_for_node(loaded_nodes[0])
        self.assertEqual(len(loaded_images), 3)
        self.assertEqual([image.width for image in loaded_images], [14, 20, 9])
        self.assertEqual(
            [image.color_space for image in loaded_images], ["BGR", "L", "BGR"]
        )
        self.assertEqual(
            [image.name for image in loaded_images], ["one.png", "two.png", "three.png"]
        )
        for before, after in zip(images, loaded_images):
            self.assertIsNot(after.array, before.array)
            self.assertTrue(np.array_equal(before.array, after.array))


if __name__ == "__main__":
    unittest.main(verbosity=2)
