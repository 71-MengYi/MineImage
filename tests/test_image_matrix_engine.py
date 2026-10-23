"""图像矩阵（ImageList）的引擎逐张执行与数组输出测试。

运行方式（仓库根目录）::

    python -m unittest discover -s tests -t tests -p "test_image_matrix_engine.py" -v

测试通过真实异步引擎执行，只创建 QApplication 并手动 processEvents，
不启动 GUI 事件循环。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 复用引擎测试基类（隔离工作区、引擎、事件收集与 _wait 语义），
# 通过模块引用避免 unittest 重复收集 test_engine 里的测试类。
import tests.test_engine as engine_tests  # noqa: E402
from app.core.data_types import DataType, NumberValue, data_type_of  # noqa: E402
from app.core.types import ImageData, ImageList  # noqa: E402
from app.engine.engine import NodeResult  # noqa: E402
from app.operators.spec import (  # noqa: E402
    normalize_params,
    output_to_dict,
    validate_inputs,
)


def _array(
    height: int = 24,
    width: int = 32,
    channels: int = 3,
    seed: int = 1,
    dtype=np.uint8,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (height, width) if channels == 1 else (height, width, channels)
    array = rng.integers(0, 256, shape, dtype=np.uint8)
    if dtype == np.uint16:
        return (array.astype(np.uint16) * 257).astype(np.uint16)
    return array


class ImageMatrixEngineTestCase(engine_tests.EngineTestCase):
    """在通用引擎测试基类上补充图像矩阵构造与算子直调工具。"""

    def add_matrix_source(self, arrays, title: str = "图像矩阵"):
        images = [
            ImageData(np.asarray(array), f"矩阵图{index + 1}")
            for index, array in enumerate(arrays)
        ]
        return self.workspace.add_source_images(images, title=title)

    def run_directly(self, op_id: str, inputs: dict, params: dict | None = None) -> dict:
        """绕过引擎直接调用算子，作为逐张结果的对照。"""

        entry = self.registry.entry(op_id)
        self.assertIsNotNone(entry, f"注册表中缺少算子 {op_id}")
        normalized = normalize_params(entry.spec, params or {})
        validate_inputs(entry.spec, inputs)
        entry.operator.validate(inputs, normalized)
        call_kwargs = dict(inputs)
        call_kwargs.update(normalized)
        return output_to_dict(entry.spec, entry.operator.process(**call_kwargs))

    def seed_source_result(self, source, value) -> None:
        """给源节点写入一份快照结果，用于构造单张 / 空图像矩阵输入。"""

        self.engine._results[source.id] = NodeResult(
            source.id, {"image": value}, 0.0, source.revision
        )


class SourceMatrixTests(ImageMatrixEngineTestCase):
    def test_multi_source_returns_image_list_in_order(self) -> None:
        arrays = [
            _array(24, 32, 3, seed=1),
            _array(18, 20, 1, seed=2),
            _array(10, 12, 4, seed=3),
        ]
        source = self.add_matrix_source(arrays, title="三张图")

        snapshot = self.engine._capture_graph()
        self.assertEqual(
            snapshot.nodes[source.id].source_image_ids, list(source.source_ids)
        )

        self.engine.preview_node(source.id)
        self.assertTrue(self.wait_chain(source.id))
        result = self.engine.result(source.id)
        self.assertIsNotNone(result)

        value = result.outputs["image"]
        self.assertIsInstance(value, ImageList)
        self.assertEqual(value.name, "三张图")
        self.assertEqual(value.count, 3)
        self.assertTrue(result.has_image_matrix)
        self.assertEqual(result.image_count, 3)
        self.assertEqual(result.primary_images, value.images)
        self.assertIs(result.primary, value.images[0])
        self.assertIs(result.preview_image, value.images[0])
        self.assertIs(result.image_at(2), value.images[2])
        self.assertIsNone(result.image_at(9))
        self.assertIsNone(result.image_at("not-an-index"))

        # 顺序、尺寸、通道 / 颜色空间与 dtype 全部保持原始图像。
        self.assertEqual(
            [image.array.shape for image in value],
            [array.shape for array in arrays],
        )
        self.assertEqual(
            [image.color_space for image in value], ["BGR", "L", "BGRA"]
        )
        for actual, expected in zip(value, arrays):
            np.testing.assert_array_equal(actual.array, expected)
            self.assertEqual(actual.array.dtype, expected.dtype)
        self.assertEqual(
            [image.name for image in value], ["矩阵图1", "矩阵图2", "矩阵图3"]
        )

    def test_single_source_remains_image_data(self) -> None:
        array = _array(24, 32, 3, seed=5)
        source = self.add_source(array, name="单图")

        self.engine.preview_node(source.id)
        self.assertTrue(self.wait_chain(source.id))
        result = self.engine.result(source.id)
        self.assertIsNotNone(result)

        value = result.outputs["image"]
        self.assertIsInstance(value, ImageData)
        self.assertNotIsInstance(value, ImageList)
        self.assertFalse(result.has_image_matrix)
        self.assertEqual(result.image_count, 1)
        self.assertEqual(result.primary_images, (value,))
        self.assertIs(result.primary, value)
        self.assertIs(result.preview_image, value)
        self.assertIs(result.image_at(0), value)
        self.assertIsNone(result.image_at(1))
        np.testing.assert_array_equal(value.array, array)


class BatchOperatorTests(ImageMatrixEngineTestCase):
    def test_grayscale_processes_every_image(self) -> None:
        arrays = [
            _array(24, 32, 3, seed=11),
            _array(18, 20, 3, seed=12),
            _array(10, 12, 3, seed=13),
        ]
        source = self.add_matrix_source(arrays)
        gray = self.add_op("basic.grayscale", "灰度化", {"mode": "weighted"})
        self.connect(source, gray)

        self.engine.preview_node(gray.id)
        self.assertTrue(self.wait_chain(gray.id))
        result = self.engine.result(gray.id)
        self.assertIsNotNone(result)

        value = result.outputs["image"]
        self.assertIsInstance(value, ImageList)
        self.assertEqual(value.count, 3)
        self.assertTrue(result.has_image_matrix)
        self.assertEqual(result.image_count, 3)

        entry = self.registry.entry("basic.grayscale")
        params = normalize_params(entry.spec, {"mode": "weighted"})
        for index, array in enumerate(arrays):
            expected = entry.operator.process(image=ImageData(array), **params)
            np.testing.assert_array_equal(value.images[index].array, np.asarray(expected))
            self.assertEqual(value.images[index].array.shape[:2], array.shape[:2])
            self.assertEqual(value.images[index].color_space, "L")

    def test_grayscale_then_invert_keeps_all_images(self) -> None:
        arrays = [
            _array(22, 30, 3, seed=21),
            _array(16, 18, 3, seed=22),
            _array(12, 14, 3, seed=23),
        ]
        source = self.add_matrix_source(arrays)
        gray = self.add_op("basic.grayscale", "灰度化", {"mode": "average"})
        invert = self.add_op("basic.invert", "反相")
        self.connect(source, gray)
        self.connect(gray, invert)

        self.engine.preview_node(invert.id)
        self.assertTrue(self.wait_chain(invert.id))

        gray_result = self.engine.result(gray.id)
        invert_result = self.engine.result(invert.id)
        self.assertIsNotNone(gray_result)
        self.assertIsNotNone(invert_result)
        gray_value = gray_result.outputs["image"]
        invert_value = invert_result.outputs["image"]
        self.assertIsInstance(gray_value, ImageList)
        self.assertIsInstance(invert_value, ImageList)
        self.assertEqual(gray_value.count, 3)
        self.assertEqual(invert_value.count, 3)
        self.assertEqual(invert_result.image_count, 3)
        for index in range(3):
            np.testing.assert_array_equal(
                invert_value.images[index].array,
                255 - gray_value.images[index].array,
            )

    def test_output_display_passes_image_matrix_through(self) -> None:
        arrays = [
            _array(20, 26, 3, seed=31),
            _array(14, 16, 3, seed=32),
            _array(8, 10, 3, seed=33),
        ]
        source = self.add_matrix_source(arrays)
        display = self.add_op("output.display", "输出预览")
        self.connect(source, display, "image", "data")

        self.engine.preview_node(display.id)
        self.assertTrue(self.wait_chain(display.id))
        result = self.engine.result(display.id)
        self.assertIsNotNone(result)

        value = result.outputs["data"]
        self.assertIsInstance(value, ImageList)
        self.assertEqual(value.count, 3)
        self.assertIs(data_type_of(value), DataType.IMAGE)
        self.assertTrue(result.has_image_matrix)
        source_value = self.engine.result(source.id).outputs["image"]
        for index, array in enumerate(arrays):
            np.testing.assert_array_equal(value.images[index].array, array)
            self.assertIs(value.images[index], source_value.images[index])

    def test_numeric_outputs_become_batch_lists(self) -> None:
        arrays = [
            _array(12, 16, 3, seed=41),
            _array(10, 14, 3, seed=42),
            _array(9, 11, 1, seed=43),
        ]
        source = self.add_matrix_source(arrays)
        stats = self.add_op("analyze.image_stats", "图像统计")
        self.connect(source, stats)

        self.engine.preview_node(stats.id)
        self.assertTrue(self.wait_chain(stats.id))
        result = self.engine.result(stats.id)
        self.assertIsNotNone(result)
        self.assertIsNone(result.primary)
        self.assertEqual(result.image_count, 0)
        self.assertFalse(result.has_image_matrix)

        for key in ("mean", "std", "min", "max"):
            values = result.outputs[key]
            self.assertIsInstance(values, list)
            self.assertEqual(len(values), 3)
            self.assertTrue(all(isinstance(value, NumberValue) for value in values))
            self.assertIs(data_type_of(values), DataType.NUMBER)

        entry = self.registry.entry("analyze.image_stats")
        params = normalize_params(entry.spec, {})
        for index, array in enumerate(arrays):
            expected = output_to_dict(
                entry.spec, entry.operator.process(image=ImageData(array), **params)
            )
            for key in ("mean", "std", "min", "max"):
                self.assertAlmostEqual(
                    result.outputs[key][index].value, expected[key].value
                )
        self.assertIn("×3", result.summary_text())

    def test_single_image_operator_regression(self) -> None:
        array = _array(20, 24, 3, seed=51)
        source = self.add_source(array, name="单图")
        gray = self.add_op("basic.grayscale", "灰度化", {"mode": "weighted"})
        self.connect(source, gray)

        self.engine.preview_node(gray.id)
        self.assertTrue(self.wait_chain(gray.id))
        result = self.engine.result(gray.id)
        value = result.outputs["image"]
        self.assertIsInstance(value, ImageData)
        self.assertNotIsInstance(value, ImageList)
        self.assertEqual(result.image_count, 1)
        self.assertFalse(result.has_image_matrix)
        expected = self.run_directly(
            "basic.grayscale", {"image": ImageData(array)}, {"mode": "weighted"}
        )
        np.testing.assert_array_equal(value.array, expected["image"].array)

    def test_single_image_input_is_broadcast_over_matrix(self) -> None:
        single = _array(16, 20, 3, seed=61)
        matrix_arrays = [
            _array(16, 20, 3, seed=62),
            _array(16, 20, 3, seed=63),
            _array(16, 20, 3, seed=64),
        ]
        source_single = self.add_source(single, name="单图")
        source_matrix = self.add_matrix_source(matrix_arrays)
        add = self.add_op("basic.add", "加法")
        self.connect(source_single, add, "image", "image")
        self.connect(source_matrix, add, "image", "image2")

        self.engine.preview_node(add.id)
        self.assertTrue(self.wait_chain(add.id))
        result = self.engine.result(add.id)
        self.assertIsNotNone(result)
        value = result.outputs["image"]
        self.assertIsInstance(value, ImageList)
        self.assertEqual(value.count, 3)

        for index, array in enumerate(matrix_arrays):
            expected = self.run_directly(
                "basic.add",
                {"image": ImageData(single), "image2": ImageData(array)},
            )
            np.testing.assert_array_equal(
                value.images[index].array, expected["image"].array
            )


class BatchFailureTests(ImageMatrixEngineTestCase):
    def test_mismatched_matrix_lengths_fail(self) -> None:
        source_a = self.add_matrix_source(
            [_array(14, 18, 3, seed=71 + index) for index in range(3)]
        )
        source_b = self.add_matrix_source(
            [_array(14, 18, 3, seed=81 + index) for index in range(2)]
        )
        add = self.add_op("basic.add", "加法")
        self.connect(source_a, add, "image", "image")
        self.connect(source_b, add, "image", "image2")

        self.engine.preview_node(add.id)
        self.assertFalse(self.wait_chain(add.id))
        self.assertIsNone(self.engine.result(add.id))
        error = self.engine.error(add.id)
        self.assertIsNotNone(error)
        self.assertIn("输入图像矩阵数量不一致", error.message)
        self.assertIn("3", error.message)
        self.assertIn("2", error.message)

    def test_one_failed_image_fails_whole_node_with_index(self) -> None:
        arrays = [
            _array(14, 18, 3, seed=91),
            _array(14, 18, 3, seed=92, dtype=np.uint16),
            _array(14, 18, 3, seed=93),
        ]
        source = self.add_matrix_source(arrays)
        median = self.add_op("filter.median_blur", "中值模糊", {"ksize": 7})
        self.connect(source, median)

        self.engine.preview_node(median.id)
        self.assertFalse(self.wait_chain(median.id))
        error = self.engine.error(median.id)
        self.assertIsNotNone(error)
        self.assertIn("第 2 张", error.message)
        self.assertIn("仅支持 uint8", error.message)

    def test_empty_image_matrix_fails_cleanly(self) -> None:
        source = self.add_source(_array(10, 12, 3, seed=101), name="源图")
        gray = self.add_op("basic.grayscale", "灰度化")
        self.connect(source, gray)
        self.seed_source_result(source, ImageList(()))

        self.engine.preview_node(gray.id)
        self.assertFalse(self.wait_chain(gray.id))
        error = self.engine.error(gray.id)
        self.assertIsNotNone(error)
        self.assertIn("图像矩阵为空", error.message)

    def test_single_image_matrix_is_unwrapped(self) -> None:
        array = _array(10, 12, 3, seed=111)
        source = self.add_source(array, name="源图")
        gray = self.add_op("basic.grayscale", "灰度化", {"mode": "weighted"})
        self.connect(source, gray)
        self.seed_source_result(source, ImageList((ImageData(array, "单张矩阵"),)))

        self.engine.preview_node(gray.id)
        self.assertTrue(self.wait_chain(gray.id))
        result = self.engine.result(gray.id)
        self.assertIsNotNone(result)
        value = result.outputs["image"]
        # 只有 1 张的 ImageList 按单张解包，输出仍是 ImageData。
        self.assertIsInstance(value, ImageData)
        self.assertNotIsInstance(value, ImageList)
        self.assertFalse(result.has_image_matrix)
        expected = self.run_directly(
            "basic.grayscale", {"image": ImageData(array)}, {"mode": "weighted"}
        )
        np.testing.assert_array_equal(value.array, expected["image"].array)


if __name__ == "__main__":
    unittest.main(verbosity=2)
