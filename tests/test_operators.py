"""内置算子库的单元测试（不启动 GUI 事件循环）。"""

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

from app.core.constants import CATEGORY_ORDER  # noqa: E402
from app.core.types import ImageData  # noqa: E402
from app.operators.loader import OperatorRegistry  # noqa: E402
from app.operators.spec import (  # noqa: E402
    OperatorError,
    ParamValidationError,
    is_param_enabled,
    is_param_visible,
    normalize_params,
    output_to_dict,
    validate_inputs,
)

CJK = ("\u4e00", "\u9fff")


def contains_cjk(text: str) -> bool:
    return any(CJK[0] <= char <= CJK[1] for char in text)


def bgr8(height: int = 24, width: int = 32) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def gray8(height: int = 24, width: int = 32) -> np.ndarray:
    return bgr8(height, width)[:, :, 1].copy()


def bgra8() -> np.ndarray:
    array = bgr8()
    return np.dstack([array, np.full(array.shape[:2], 180, dtype=np.uint8)])


def bgr16() -> np.ndarray:
    return (bgr8().astype(np.uint16) * 257).astype(np.uint16)


def gray16() -> np.ndarray:
    return (gray8().astype(np.uint16) * 257).astype(np.uint16)


def bgrf() -> np.ndarray:
    return (bgr8().astype(np.float32) / 255.0).astype(np.float32)


def grayf() -> np.ndarray:
    return (gray8().astype(np.float32) / 255.0).astype(np.float32)


#: 算子 id -> 期望的一级分类。
EXPECTED_CATEGORIES = {
    "basic.grayscale": "基础调整",
    "basic.brightness_contrast": "基础调整",
    "basic.invert": "基础调整",
    "basic.lut": "基础调整",
    "filter.gaussian_blur": "滤波与平滑",
    "filter.mean_blur": "滤波与平滑",
    "filter.median_blur": "滤波与平滑",
    "threshold.fixed": "阈值与分割",
    "threshold.adaptive": "阈值与分割",
    "edge.sobel": "边缘检测",
    "edge.canny": "边缘检测",
    "morph.erode": "形态学",
    "morph.dilate": "形态学",
    "morph.open": "形态学",
    "morph.close": "形态学",
    "geom.resize": "几何变换",
    "geom.rotate": "几何变换",
    "geom.flip": "几何变换",
    "channel.split": "通道与色彩",
    "channel.merge": "通道与色彩",
    "interactive.roi_crop": "交互处理",
    "interactive.draw_line": "交互处理",
}

_PARAMS: dict[str, dict] = {
    "basic.brightness_contrast": {"brightness": 20, "contrast": 1.15},
    "basic.lut": {"mode": "gamma", "gamma": 1.4, "black": 8, "white": 245},
    "filter.gaussian_blur": {"ksize": 5, "sigma": 0.0},
    "filter.mean_blur": {"ksize": 5},
    "filter.median_blur": {"ksize": 3},
    "threshold.fixed": {"value": 100, "mode": "binary"},
    "threshold.adaptive": {"block_size": 11, "constant": 2.0},
    "edge.sobel": {"dx": 1, "dy": 0, "ksize": 3},
    "edge.canny": {"low": 50, "high": 150},
    "morph.erode": {"shape": "ellipse", "ksize": 3, "iterations": 1},
    "morph.dilate": {"shape": "ellipse", "ksize": 3, "iterations": 1},
    "morph.open": {"ksize": 3},
    "morph.close": {"ksize": 3},
    "geom.resize": {"mode": "scale", "scale": 0.5},
    "geom.rotate": {"angle": 30.0, "expand": True},
    "geom.flip": {"axis": "horizontal"},
    "interactive.roi_crop": {"roi": [2.0, 3.0, 10.0, 8.0]},
    "interactive.draw_line": {"line": [1.0, 1.0, 20.0, 15.0], "color": "#ff3b30", "thickness": 2},
}


class OperatorLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = OperatorRegistry.build()

    # ------------------------------------------------------------------ 工具
    def run_operator(self, op_id: str, inputs: dict | None = None, params: dict | None = None):
        entry = self.registry.entry(op_id)
        self.assertIsNotNone(entry, f"缺少算子 {op_id}")
        merged_params = dict(params) if params is not None else dict(_PARAMS.get(op_id, {}))
        normalized = normalize_params(entry.spec, merged_params)
        resolved = {}
        for port in entry.spec.inputs:
            raw = (inputs or {}).get(port.key)
            resolved[port.key] = raw if isinstance(raw, ImageData) or raw is None else ImageData(raw)
        validate_inputs(entry.spec, resolved)
        entry.operator.validate(resolved, normalized)
        kwargs = dict(resolved)
        kwargs.update(normalized)
        return output_to_dict(entry.spec, entry.operator.process(**kwargs))

    def single(self, op_id: str, array: np.ndarray, params: dict | None = None) -> np.ndarray:
        outputs = self.run_operator(op_id, {"image": array}, params)
        self.assertTrue(outputs)
        return outputs[self.registry.spec(op_id).outputs[0].key].array

    # ------------------------------------------------------------------ 注册表
    def test_registry_contains_expected_builtin_operators(self) -> None:
        self.assertGreaterEqual(len(self.registry), len(EXPECTED_CATEGORIES))
        missing = set(EXPECTED_CATEGORIES) - set(self.registry.entries)
        self.assertEqual(missing, set(), f"缺少内置算子：{missing}")
        self.assertEqual(self.registry.errors, [])
        for op_id, category in EXPECTED_CATEGORIES.items():
            entry = self.registry.entry(op_id)
            self.assertTrue(entry.builtin)
            self.assertEqual(entry.spec.category[0], category)
            self.assertTrue(contains_cjk(entry.spec.name), f"{op_id} 名称应为中文")
            self.assertTrue(contains_cjk(entry.spec.description), f"{op_id} 描述应为中文")
            for param in entry.spec.params:
                self.assertTrue(param.label, f"{op_id}.{param.key} 缺少标签")

    def test_registry_categories_are_ordered_by_constants(self) -> None:
        tabs = list(self.registry.tabs())
        self.assertTrue(set(tabs).issubset(set(CATEGORY_ORDER)))
        self.assertEqual(tabs, [category for category in CATEGORY_ORDER if category in tabs])
        for entries in self.registry.tabs().values():
            self.assertTrue(all(entry.spec.category for entry in entries))

    def test_all_operators_run_and_do_not_mutate_inputs(self) -> None:
        arrays = {
            "bgr8": bgr8(),
            "bgra8": bgra8(),
            "gray8": gray8(),
            "bgr16": bgr16(),
            "gray16": gray16(),
            "bgrf": bgrf(),
            "grayf": grayf(),
        }
        for op_id in EXPECTED_CATEGORIES:
            if op_id == "channel.merge":
                continue
            spec = self.registry.spec(op_id)
            for name, array in arrays.items():
                if spec.inputs and spec.inputs[0].accepts:
                    color_space = ImageData(array).color_space
                    if color_space not in spec.inputs[0].accepts:
                        continue
                before = array.copy()
                outputs = self.run_operator(op_id, {"image": array})
                self.assertTrue(outputs, f"{op_id}/{name} 没有输出")
                if not np.array_equal(array, before):
                    self.fail(f"{op_id}/{name} 修改了输入图像")
                for output in outputs.values():
                    self.assertIn(output.array.ndim, (2, 3))
                    self.assertGreater(output.array.size, 0)
                    self.assertIn(output.color_space, ("L", "BGR", "BGRA"))

    # ------------------------------------------------------------------ 基础调整
    def test_grayscale_weighted_matches_opencv(self) -> None:
        array = bgr8()
        result = self.single("basic.grayscale", array)
        np.testing.assert_array_equal(result, cv2.cvtColor(array, cv2.COLOR_BGR2GRAY))

    def test_grayscale_modes(self) -> None:
        array = bgr8()
        average = self.single("basic.grayscale", array, {"mode": "average"})
        expected = np.rint(array.astype(np.float32).mean(axis=2)).astype(np.uint8)
        np.testing.assert_array_equal(average, expected)
        maximum = self.single("basic.grayscale", array, {"mode": "max"})
        np.testing.assert_array_equal(maximum, array.max(axis=2))
        # BGRA 忽略 alpha。
        bgra = bgra8()
        np.testing.assert_array_equal(
            self.single("basic.grayscale", bgra),
            cv2.cvtColor(bgra[:, :, :3], cv2.COLOR_BGR2GRAY),
        )

    def test_brightness_contrast_defaults_are_identity(self) -> None:
        for array in (bgr8(), gray8(), bgr16(), gray16(), bgrf(), grayf()):
            result = self.single("basic.brightness_contrast", array, {"brightness": 0, "contrast": 1.0})
            np.testing.assert_array_equal(result, array)

    def test_invert_is_dtype_aware_and_involutive(self) -> None:
        array8 = gray8()
        np.testing.assert_array_equal(self.single("basic.invert", array8), 255 - array8)
        self.assertFalse(np.array_equal(self.single("basic.invert", array8), array8))
        array16 = gray16()
        inverted16 = self.single("basic.invert", array16)
        self.assertEqual(inverted16.dtype, np.uint16)
        np.testing.assert_array_equal(inverted16, np.iinfo(np.uint16).max - array16)
        arrayf = grayf()
        invertedf = self.single("basic.invert", arrayf)
        self.assertEqual(invertedf.dtype, np.float32)
        np.testing.assert_allclose(invertedf, 1.0 - arrayf, rtol=0, atol=1e-6)
        for array in (array8, array16, arrayf):
            twice = self.single("basic.invert", self.single("basic.invert", array))
            np.testing.assert_allclose(twice, array, rtol=0, atol=1e-6)

    def test_invert_keeps_alpha_unless_requested(self) -> None:
        array = bgra8()
        kept = self.single("basic.invert", array)
        np.testing.assert_array_equal(kept[:, :, 3], array[:, :, 3])
        inverted_alpha = self.single("basic.invert", array, {"invert_alpha": True})
        np.testing.assert_array_equal(inverted_alpha[:, :, 3], 255 - array[:, :, 3])

    def test_lut_identity_and_gamma(self) -> None:
        linear = {"mode": "linear", "black": 0, "white": 255}
        array8 = gray8()
        identity = self.single("basic.lut", array8, linear)
        np.testing.assert_array_equal(identity, array8)
        array16 = gray16()
        identity16 = self.single("basic.lut", array16, linear)
        self.assertEqual(identity16.dtype, np.uint16)
        np.testing.assert_array_equal(identity16, array16)
        brighter = self.single("basic.lut", array8, {"mode": "gamma", "gamma": 2.0})
        self.assertGreater(float(brighter.mean()), float(array8.mean()))
        darker = self.single("basic.lut", array8, {"mode": "gamma", "gamma": 0.5})
        self.assertLess(float(darker.mean()), float(array8.mean()))
        inverted = self.single("basic.lut", array8, {**linear, "invert": True})
        np.testing.assert_array_equal(inverted, 255 - array8)

    def test_lut_rejects_invalid_black_white(self) -> None:
        with self.assertRaises(OperatorError):
            self.run_operator(
                "basic.lut",
                {"image": gray8()},
                {"black": 200, "white": 100},
            )

    # ------------------------------------------------------------------ 滤波
    def test_blur_operators_preserve_dtype_and_shape(self) -> None:
        for array in (bgr8(), bgra8(), bgr16(), bgrf()):
            for op_id in ("filter.gaussian_blur", "filter.mean_blur", "filter.median_blur"):
                result = self.single(op_id, array)
                self.assertEqual(result.dtype, array.dtype, op_id)
                self.assertEqual(result.shape, array.shape, op_id)

    def test_gaussian_blur_smooths_impulse(self) -> None:
        impulse = np.zeros((15, 15), dtype=np.uint8)
        impulse[7, 7] = 255
        blurred = self.single("filter.gaussian_blur", impulse, {"ksize": 5, "sigma": 0.0})
        self.assertLess(int(blurred[7, 7]), 255)
        self.assertGreater(int(blurred[7, 7]), 0)

    def test_median_blur_removes_salt_noise(self) -> None:
        salt = np.zeros((15, 15), dtype=np.uint8)
        salt[3, 3] = 255
        salt[11, 12] = 255
        median = self.single("filter.median_blur", salt, {"ksize": 3})
        self.assertEqual(int(median.max()), 0)

    def test_median_blur_large_kernel_rejects_non_uint8(self) -> None:
        with self.assertRaises(OperatorError):
            self.single("filter.median_blur", gray16(), {"ksize": 7})
        # uint8 大核可用。
        self.assertEqual(self.single("filter.median_blur", gray8(), {"ksize": 7}).dtype, np.uint8)

    def test_gaussian_ksize_must_be_odd(self) -> None:
        with self.assertRaises(ParamValidationError):
            normalize_params(self.registry.spec("filter.gaussian_blur"), {"ksize": 4})

    # ------------------------------------------------------------------ 阈值
    def test_fixed_threshold_dtype_aware(self) -> None:
        result8 = self.single("threshold.fixed", gray8(), {"value": 128})
        self.assertEqual(result8.dtype, np.uint8)
        self.assertTrue(set(np.unique(result8)).issubset({0, 255}))
        result16 = self.single("threshold.fixed", gray16(), {"value": 128})
        self.assertEqual(result16.dtype, np.uint16)
        self.assertTrue(set(np.unique(result16)).issubset({0, 65535}))
        resultf = self.single("threshold.fixed", grayf(), {"value": 128})
        self.assertEqual(resultf.dtype, np.float32)
        self.assertTrue(set(np.unique(resultf)).issubset({0.0, 1.0}))
        inverted = self.single("threshold.fixed", gray8(), {"mode": "binary_inv"})
        forward = self.single("threshold.fixed", gray8(), {"mode": "binary"})
        np.testing.assert_array_equal(inverted, 255 - forward)

    def test_fixed_threshold_otsu(self) -> None:
        image = np.zeros((20, 20), dtype=np.uint8)
        image[:, 10:] = 200
        result = self.single("threshold.fixed", image, {"otsu": True})
        self.assertTrue(set(np.unique(result)).issubset({0, 255}))
        with self.assertRaises(OperatorError):
            self.single("threshold.fixed", grayf(), {"otsu": True})

    def test_adaptive_threshold_handles_16bit_and_float(self) -> None:
        for array in (gray8(), gray16(), grayf(), bgr8()):
            result = self.single("threshold.adaptive", array)
            self.assertEqual(result.dtype, np.uint8)
            self.assertEqual(result.shape, array.shape[:2])
        with self.assertRaises(ParamValidationError):
            normalize_params(self.registry.spec("threshold.adaptive"), {"block_size": 4})

    # ------------------------------------------------------------------ 边缘
    def test_sobel_outputs_float32_gradients(self) -> None:
        result = self.single("edge.sobel", gray8())
        self.assertEqual(result.dtype, np.float32)
        flat = np.full((12, 12), 100, dtype=np.uint8)
        self.assertEqual(float(np.abs(self.single("edge.sobel", flat)).max()), 0.0)
        edge = np.zeros((12, 12), dtype=np.uint8)
        edge[:, 6:] = 255
        self.assertGreater(float(self.single("edge.sobel", edge).max()), 0.0)
        with self.assertRaises(OperatorError):
            self.single("edge.sobel", gray8(), {"dx": 0, "dy": 0})

    def test_canny_returns_binary_edge_map(self) -> None:
        edge = np.zeros((24, 24), dtype=np.uint8)
        edge[:, 12:] = 255
        result = self.single("edge.canny", edge, {"low": 50, "high": 150})
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(set(np.unique(result)).issubset({0, 255}))
        self.assertGreater(int(result.sum()), 0)
        with self.assertRaises(OperatorError):
            self.single("edge.canny", edge, {"low": 200, "high": 100})

    # ------------------------------------------------------------------ 形态学
    def test_morphology_grows_and_shrinks_bright_regions(self) -> None:
        image = np.zeros((21, 21), dtype=np.uint8)
        image[10, 10] = 255
        dilated = self.single("morph.dilate", image, {"ksize": 3})
        self.assertGreater(int(dilated.sum()), int(image.sum()))
        eroded = self.single("morph.erode", image, {"ksize": 3})
        self.assertEqual(int(eroded.max()), 0)
        block = np.zeros((21, 21), dtype=np.uint8)
        block[8:13, 8:13] = 255
        for op_id in ("morph.open", "morph.close"):
            result = self.single(op_id, block, {"ksize": 3})
            self.assertEqual(result.shape, block.shape)
            self.assertEqual(result.dtype, block.dtype)
            self.assertGreater(int(result.sum()), 0)
        # 迭代次数必须影响结果。
        one = self.single("morph.dilate", image, {"ksize": 3, "iterations": 1})
        two = self.single("morph.dilate", image, {"ksize": 3, "iterations": 2})
        self.assertGreater(int(two.sum()), int(one.sum()))

    # ------------------------------------------------------------------ 几何
    def test_resize_by_scale_and_size(self) -> None:
        array = bgr8(height=20, width=40)
        half = self.single("geom.resize", array, {"mode": "scale", "scale": 0.5})
        self.assertEqual(half.shape, (10, 20, 3))
        fitted = self.single(
            "geom.resize",
            array,
            {"mode": "size", "width": 20, "height": 20, "keep_aspect": True},
        )
        self.assertEqual(fitted.shape, (10, 20, 3))
        stretched = self.single(
            "geom.resize",
            array,
            {"mode": "size", "width": 20, "height": 30, "keep_aspect": False},
        )
        self.assertEqual(stretched.shape, (30, 20, 3))

    def test_resize_param_visibility(self) -> None:
        spec = self.registry.spec("geom.resize")
        size_params = normalize_params(spec, {"mode": "size"})
        scale_params = normalize_params(spec, {"mode": "scale"})
        self.assertTrue(is_param_visible(spec.param("width"), size_params))
        self.assertFalse(is_param_visible(spec.param("width"), scale_params))
        self.assertTrue(is_param_enabled(spec.param("scale"), scale_params))
        self.assertFalse(is_param_enabled(spec.param("scale"), size_params))

    def test_rotate_expand_and_identity(self) -> None:
        array = bgr8(height=20, width=40)
        identity = self.single("geom.rotate", array, {"angle": 0.0, "expand": False})
        np.testing.assert_array_equal(identity, array)
        rotated = self.single("geom.rotate", array, {"angle": 90.0, "expand": True})
        self.assertEqual(rotated.shape[0], 40)
        self.assertEqual(rotated.shape[1], 20)
        twice = self.single(
            "geom.rotate",
            self.single("geom.rotate", array, {"angle": 180.0, "expand": True}),
            {"angle": 180.0, "expand": True},
        )
        np.testing.assert_array_equal(twice, array)

    def test_flip_axes(self) -> None:
        array = bgr8()
        np.testing.assert_array_equal(
            self.single("geom.flip", array, {"axis": "horizontal"}), array[:, ::-1]
        )
        np.testing.assert_array_equal(
            self.single("geom.flip", array, {"axis": "vertical"}), array[::-1, :]
        )
        both = self.single("geom.flip", array, {"axis": "both"})
        np.testing.assert_array_equal(both, array[::-1, ::-1])

    # ------------------------------------------------------------------ 通道
    def test_channel_split_outputs_three_ports(self) -> None:
        array = bgr8()
        outputs = self.run_operator("channel.split", {"image": array})
        self.assertEqual(set(outputs), {"b", "g", "r"})
        np.testing.assert_array_equal(outputs["b"].array, array[:, :, 0])
        np.testing.assert_array_equal(outputs["g"].array, array[:, :, 1])
        np.testing.assert_array_equal(outputs["r"].array, array[:, :, 2])
        bgra = bgra8()
        bgra_outputs = self.run_operator("channel.split", {"image": bgra})
        np.testing.assert_array_equal(bgra_outputs["b"].array, bgra[:, :, 0])

    def test_channel_split_rejects_gray(self) -> None:
        with self.assertRaises(OperatorError):
            self.run_operator("channel.split", {"image": gray8()})

    def test_channel_merge_roundtrip_and_dtype(self) -> None:
        array = bgr8()
        merged = self.run_operator(
            "channel.merge",
            {"b": array[:, :, 0], "g": array[:, :, 1], "r": array[:, :, 2]},
        )["image"].array
        np.testing.assert_array_equal(merged, array)
        mixed = self.run_operator(
            "channel.merge",
            {"b": gray8(), "g": gray16(), "r": gray8()},
        )["image"]
        self.assertEqual(mixed.array.dtype, np.uint16)

    def test_channel_merge_rejects_bad_shape_and_color_input(self) -> None:
        with self.assertRaises(OperatorError):
            self.run_operator(
                "channel.merge",
                {"b": gray8(10, 10), "g": gray8(11, 10), "r": gray8(10, 10)},
            )
        with self.assertRaises(OperatorError):
            self.run_operator(
                "channel.merge",
                {"b": bgr8(), "g": gray8(), "r": gray8()},
            )

    # ------------------------------------------------------------------ 交互
    def test_roi_crop(self) -> None:
        array = bgr8(height=20, width=30)
        cropped = self.single("interactive.roi_crop", array, {"roi": [2.0, 3.0, 10.0, 8.0]})
        np.testing.assert_array_equal(cropped, array[3:11, 2:12])
        clipped = self.single("interactive.roi_crop", array, {"roi": [25.0, 15.0, 20.0, 20.0]})
        np.testing.assert_array_equal(clipped, array[15:20, 25:30])
        with self.assertRaises(OperatorError):
            self.single("interactive.roi_crop", array, {"roi": None})
        with self.assertRaises(OperatorError):
            self.single("interactive.roi_crop", array, {"roi": [100.0, 100.0, 5.0, 5.0]})
        with self.assertRaises(ParamValidationError):
            normalize_params(self.registry.spec("interactive.roi_crop"), {"roi": [1, 2, 0, 5]})

    def test_draw_line(self) -> None:
        array = bgr8(height=20, width=20)
        before = array.copy()
        drawn = self.single("interactive.draw_line", array, {"line": [0, 0, 19, 19]})
        np.testing.assert_array_equal(array, before)
        self.assertFalse(np.array_equal(drawn, array))
        # 线宽 3 时中点附近必然被覆盖。
        thick = self.single(
            "interactive.draw_line",
            array,
            {"line": [0, 10, 19, 10], "color": "#00ff00", "thickness": 3},
        )
        np.testing.assert_array_equal(thick[10, 10], np.array([0, 255, 0], dtype=np.uint8))
        # BGRA 保留 alpha。
        bgra = bgra8()
        drawn_bgra = self.single("interactive.draw_line", bgra, {"line": [0, 0, 10, 10]})
        np.testing.assert_array_equal(drawn_bgra[:, :, 3], bgra[:, :, 3])
        # 16 位与 float 也能绘制。
        for array in (gray16(), grayf()):
            result = self.single("interactive.draw_line", array, {"line": [1, 1, 5, 5]})
            self.assertEqual(result.dtype, array.dtype)
            self.assertFalse(np.array_equal(result, array))
        with self.assertRaises(OperatorError):
            self.single("interactive.draw_line", array, {"line": None})
        with self.assertRaises(OperatorError):
            self.single("interactive.draw_line", array, {"line": [5, 5, 5, 5]})

    # ------------------------------------------------------------------ 参数
    def test_unknown_parameter_is_rejected(self) -> None:
        with self.assertRaises(ParamValidationError):
            normalize_params(self.registry.spec("basic.grayscale"), {"unknown": 1})

    def test_lut_gamma_visibility(self) -> None:
        spec = self.registry.spec("basic.lut")
        linear = normalize_params(spec, {"mode": "linear"})
        gamma = normalize_params(spec, {"mode": "gamma"})
        self.assertFalse(is_param_visible(spec.param("gamma"), linear))
        self.assertTrue(is_param_visible(spec.param("gamma"), gamma))
        self.assertFalse(is_param_enabled(spec.param("gamma"), linear))
        self.assertTrue(is_param_enabled(spec.param("gamma"), gamma))

    def test_threshold_value_hidden_when_otsu(self) -> None:
        spec = self.registry.spec("threshold.fixed")
        manual = normalize_params(spec, {"otsu": False})
        otsu = normalize_params(spec, {"otsu": True})
        self.assertTrue(is_param_visible(spec.param("value"), manual))
        self.assertTrue(is_param_enabled(spec.param("value"), manual))
        self.assertFalse(is_param_visible(spec.param("value"), otsu))
        self.assertFalse(is_param_enabled(spec.param("value"), otsu))

    def test_condition_expressions_avoid_attribute_access(self) -> None:
        """eval_condition 的 params 是 dict；``params.xxx`` 会被吞成 True。"""
        for entry in self.registry.all_entries():
            for param in entry.spec.params:
                for expression in (param.visible_when, param.enabled_when):
                    if expression:
                        self.assertNotIn("params.", expression, f"{entry.id}/{param.key} 使用了属性访问")

    def test_all_specs_have_valid_defaults(self) -> None:
        for entry in self.registry.all_entries():
            defaults = normalize_params(entry.spec, None)
            self.assertEqual(set(defaults), {param.key for param in entry.spec.params})
            for param in entry.spec.params:
                if param.visible_when or param.enabled_when:
                    # 联动表达式必须可解析且返回布尔值。
                    self.assertIsInstance(is_param_visible(param, defaults), bool)
                    self.assertIsInstance(is_param_enabled(param, defaults), bool)


if __name__ == "__main__":
    unittest.main(verbosity=2)
