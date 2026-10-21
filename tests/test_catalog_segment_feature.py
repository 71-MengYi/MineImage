"""OpenCV 目录算子 B（阈值 / 分割 / 边缘 / 关键点）的单元测试。

覆盖文件：
* ``threshold_advanced.py`` / ``segment.py`` / ``edge_advanced.py`` / ``keypoint.py``。

测试内容：
* 注册表里本组算子齐全、分类合法、第一个输出是图像；
* 每个算子用默认参数在 7 种样例数组（灰度/BGR/BGRA × uint8/uint16/float32）
  上执行，断言有输出且输入数组未被修改；
* 若干行为正确性断言（Otsu 二值化、自动 Canny、SIFT 关键点、距离变换峰值、
  GrabCut / 漫水填充的中心缺省行为）。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.data_types import (  # noqa: E402
    ComponentsData,
    ContoursData,
    DataType,
    EdgesData,
    NumberValue,
    is_data_value,
)
from app.core.types import ImageData  # noqa: E402
from app.operators.loader import OperatorRegistry  # noqa: E402
from app.operators.spec import (  # noqa: E402
    is_param_enabled,
    is_param_visible,
    normalize_params,
    output_to_dict,
    validate_inputs,
)

GROUP_FILES = (
    "threshold_advanced.py",
    "segment.py",
    "edge_advanced.py",
    "keypoint.py",
)

GROUP_IDS = (
    # threshold_advanced.py
    "threshold.otsu",
    "threshold.triangle",
    "threshold.trunc",
    "threshold.tozero",
    "threshold.range",
    "threshold.adaptive_advanced",
    # segment.py
    "segment.watershed",
    "segment.grabcut",
    "segment.floodfill",
    "segment.distance",
    "segment.components",
    "segment.kmeans",
    "segment.contours",
    "segment.convex_hull",
    "segment.approx_poly",
    "segment.min_area_rect",
    "segment.fit_ellipse",
    # edge_advanced.py
    "edge.scharr",
    "edge.laplacian",
    "edge.canny_auto",
    "edge.hough_lines",
    "edge.hough_lines_p",
    "edge.hough_circles",
    "edge.lsd",
    "edge.gradient_mag",
    # keypoint.py
    "keypoint.corner_harris",
    "keypoint.corner_min_eigen",
    "keypoint.good_features",
    "keypoint.fast",
    "keypoint.sift",
    "keypoint.orb",
    "keypoint.akaze",
    "keypoint.brisk",
    "keypoint.kaze",
    "keypoint.mser",
    "keypoint.blob",
    "keypoint.draw",
)

ALLOWED_TABS = ("阈值与分割", "边缘检测", "特征与分析")

CJK_START = "\u4e00"
CJK_END = "\u9fff"


def contains_cjk(text: str) -> bool:
    return any(CJK_START <= char <= CJK_END for char in str(text))


def _base_color(height: int = 48, width: int = 64) -> np.ndarray:
    rng = np.random.default_rng(20240607)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def bgr8(height: int = 48, width: int = 64) -> np.ndarray:
    return _base_color(height, width)


def gray8(height: int = 48, width: int = 64) -> np.ndarray:
    return bgr8(height, width)[:, :, 1].copy()


def bgra8(height: int = 48, width: int = 64) -> np.ndarray:
    color = bgr8(height, width)
    return np.dstack([color, np.full(color.shape[:2], 180, dtype=np.uint8)])


def gray16(height: int = 48, width: int = 64) -> np.ndarray:
    return (gray8(height, width).astype(np.uint16) * 257).astype(np.uint16)


def bgr16(height: int = 48, width: int = 64) -> np.ndarray:
    return (bgr8(height, width).astype(np.uint16) * 257).astype(np.uint16)


def grayf(height: int = 48, width: int = 64) -> np.ndarray:
    return (gray8(height, width).astype(np.float32) / 255.0).astype(np.float32)


def bgrf(height: int = 48, width: int = 64) -> np.ndarray:
    return (bgr8(height, width).astype(np.float32) / 255.0).astype(np.float32)


def sample_arrays() -> dict[str, np.ndarray]:
    return {
        "gray8": gray8(),
        "bgr8": bgr8(),
        "bgra8": bgra8(),
        "gray16": gray16(),
        "bgr16": bgr16(),
        "grayf": grayf(),
        "bgrf": bgrf(),
    }


class CatalogSegmentFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = OperatorRegistry.build()

    # ------------------------------------------------------------------ 工具
    def run_operator(
        self,
        op_id: str,
        inputs: dict | None = None,
        params: dict | None = None,
    ) -> dict:
        entry = self.registry.entry(op_id)
        self.assertIsNotNone(entry, f"缺少算子 {op_id}")
        normalized = normalize_params(entry.spec, params or {})
        resolved: dict = {}
        for port in entry.spec.inputs:
            raw = (inputs or {}).get(port.key)
            resolved[port.key] = (
                raw if raw is None or is_data_value(raw) else ImageData(raw)
            )
        validate_inputs(entry.spec, resolved)
        entry.operator.validate(resolved, normalized)
        kwargs = dict(resolved)
        kwargs.update(normalized)
        return output_to_dict(entry.spec, entry.operator.process(**kwargs))

    def group_ids(self) -> set[str]:
        return {
            entry.id
            for entry in self.registry.entries.values()
            if entry.path.name in GROUP_FILES
        }

    # ------------------------------------------------------------------ 注册表
    def test_group_registered_with_valid_specs(self) -> None:
        entries = [
            entry
            for entry in self.registry.entries.values()
            if entry.path.name in GROUP_FILES
        ]
        self.assertEqual(set(GROUP_IDS), {entry.id for entry in entries})
        self.assertEqual(self.group_ids(), set(GROUP_IDS))

        group_errors = [
            issue for issue in self.registry.errors if issue.path.name in GROUP_FILES
        ]
        self.assertEqual(group_errors, [])

        for entry in entries:
            spec = entry.spec
            with self.subTest(op_id=entry.id):
                self.assertTrue(entry.builtin)
                self.assertIn(spec.category[0], ALLOWED_TABS)
                self.assertTrue(contains_cjk(spec.name), f"{entry.id} 名称应为中文")
                self.assertTrue(contains_cjk(spec.description), f"{entry.id} 描述应为中文")
                self.assertEqual(spec.inputs[0].key, "image")
                self.assertEqual(spec.inputs[0].data_type, DataType.IMAGE)
                for port in spec.inputs[1:]:
                    self.assertTrue(port.optional, f"{entry.id} 的第二输入必须 optional")
                first_output = spec.outputs[0]
                self.assertEqual(
                    first_output.data_type,
                    DataType.IMAGE,
                    f"{entry.id} 的第一个输出必须是图像",
                )
                for param in spec.params:
                    self.assertTrue(param.label, f"{entry.id}.{param.key} 缺少标签")
                    for expression in (param.visible_when, param.enabled_when):
                        if expression:
                            self.assertNotIn(
                                "params.",
                                expression,
                                f"{entry.id}.{param.key} 使用了属性访问",
                            )
                defaults = normalize_params(spec, None)
                for param in spec.params:
                    self.assertIsInstance(is_param_visible(param, defaults), bool)
                    self.assertIsInstance(is_param_enabled(param, defaults), bool)

    def test_default_params_run_on_all_sample_arrays(self) -> None:
        arrays = sample_arrays()
        for op_id in GROUP_IDS:
            entry = self.registry.entry(op_id)
            self.assertIsNotNone(entry, f"缺少算子 {op_id}")
            first_key = entry.spec.outputs[0].key
            for name, array in arrays.items():
                with self.subTest(op_id=op_id, sample=name):
                    before = array.copy()
                    outputs = self.run_operator(op_id, {"image": array})
                    self.assertTrue(outputs, f"{op_id}/{name} 没有输出")
                    self.assertIn(first_key, outputs, f"{op_id}/{name} 缺少首个输出")
                    first = outputs[first_key]
                    self.assertIsInstance(first, ImageData)
                    self.assertIn(first.array.ndim, (2, 3))
                    self.assertGreater(first.array.size, 0)
                    self.assertIn(first.color_space, ("L", "BGR", "BGRA"))
                    self.assertTrue(
                        np.array_equal(array, before),
                        f"{op_id}/{name} 修改了输入图像",
                    )
                    for output in outputs.values():
                        if isinstance(output, ImageData):
                            self.assertIn(output.color_space, ("L", "BGR", "BGRA"))
                        elif not isinstance(
                            output, (NumberValue, EdgesData, ContoursData, ComponentsData)
                        ):
                            self.fail(f"{op_id}/{name} 输出类型异常：{type(output)}")

    # ------------------------------------------------------------------ 行为
    def test_otsu_binarizes_two_level_image(self) -> None:
        image = np.zeros((32, 32), dtype=np.uint8)
        image[:, 16:] = 200
        outputs = self.run_operator("threshold.otsu", {"image": image})
        binary = outputs["image"].array
        self.assertEqual(binary.dtype, np.uint8)
        self.assertTrue(set(np.unique(binary)).issubset({0, 255}))
        self.assertLess(float(binary[:, :16].mean()), 1.0)
        self.assertGreater(float(binary[:, 16:].mean()), 254.0)
        threshold_value = outputs["threshold"]
        self.assertIsInstance(threshold_value, NumberValue)
        self.assertGreaterEqual(threshold_value.value, 0.0)
        self.assertLessEqual(threshold_value.value, 200.0)

    def test_canny_auto_finds_vertical_edge(self) -> None:
        image = np.zeros((48, 48), dtype=np.uint8)
        image[:, 24:] = 255
        outputs = self.run_operator("edge.canny_auto", {"image": image})
        edges = outputs["image"].array
        self.assertEqual(edges.dtype, np.uint8)
        self.assertTrue(set(np.unique(edges)).issubset({0, 255}))
        self.assertGreater(int((edges > 0).sum()), 0)
        low, high = outputs["low"].value, outputs["high"].value
        self.assertGreaterEqual(low, 0.0)
        self.assertLessEqual(high, 255.0)
        self.assertLessEqual(low, high)

    def test_sift_detects_features_on_synthetic_corners(self) -> None:
        image = np.zeros((96, 96), dtype=np.uint8)
        cv2.rectangle(image, (10, 10), (35, 35), 255, -1)
        cv2.rectangle(image, (60, 60), (85, 85), 255, -1)
        outputs = self.run_operator("keypoint.sift", {"image": image})
        self.assertIsInstance(outputs["points"], EdgesData)
        self.assertIsInstance(outputs["count"], NumberValue)
        self.assertGreater(outputs["count"].value, 0.0)
        self.assertEqual(outputs["points"].count, int(outputs["count"].value))
        self.assertFalse(np.array_equal(outputs["image"].array, image))

    def test_distance_transform_peaks_at_square_center(self) -> None:
        image = np.zeros((64, 64), dtype=np.uint8)
        image[16:48, 16:48] = 255
        outputs = self.run_operator("segment.distance", {"image": image})
        self.assertIsInstance(outputs["max_distance"], NumberValue)
        self.assertAlmostEqual(outputs["max_distance"].value, 16.0, delta=1.5)
        distance_map = outputs["image"].array
        row, column = np.unravel_index(int(np.argmax(distance_map)), distance_map.shape)
        self.assertLessEqual(abs(int(row) - 32), 2)
        self.assertLessEqual(abs(int(column) - 32), 2)

    def test_grabcut_and_floodfill_use_center_defaults(self) -> None:
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        image[:, :] = (30, 30, 30)
        cv2.rectangle(image, (20, 20), (44, 44), (200, 200, 200), -1)
        grabcut = self.run_operator("segment.grabcut", {"image": image})
        self.assertIsInstance(grabcut["mask"], ImageData)
        self.assertEqual(grabcut["mask"].array.dtype, np.uint8)
        self.assertTrue(set(np.unique(grabcut["mask"].array)).issubset({0, 255}))
        ratio = grabcut["foreground_ratio"]
        self.assertIsInstance(ratio, NumberValue)
        self.assertGreater(ratio.value, 0.0)
        self.assertLess(ratio.value, 100.0)

        # 纯色图 + 中心种子 = 整幅图被填充，验证 BGRA 输入也能跑通默认参数。
        flat = bgra8(height=24, width=32)
        flat[:, :, :3] = 120
        floodfill = self.run_operator("segment.floodfill", {"image": flat})
        self.assertIsInstance(floodfill["mask"], ImageData)
        self.assertEqual(floodfill["mask"].array.shape, (24, 32))
        filled_area = int((floodfill["mask"].array > 0).sum())
        self.assertEqual(filled_area, 24 * 32)
        self.assertEqual(int(floodfill["area"].value), filled_area)
        self.assertFalse(np.array_equal(floodfill["image"].array, flat))


if __name__ == "__main__":
    unittest.main(verbosity=2)
