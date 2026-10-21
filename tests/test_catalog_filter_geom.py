"""OpenCV 目录算子 A（滤波 / 形态学 / 几何 / 去噪）的单元测试。

覆盖本组 34 个算子在 7 种样例数组上的默认参数执行、输入不可变性，
以及若干行为正确性断言。测试只依赖本文件内的简短 helper。
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

from app.core.data_types import is_data_value  # noqa: E402
from app.core.types import ImageData  # noqa: E402
from app.operators.loader import OperatorRegistry  # noqa: E402
from app.operators.spec import (  # noqa: E402
    normalize_params,
    output_to_dict,
    validate_inputs,
)

CJK = ("\u4e00", "\u9fff")

FILES = (
    "filter_advanced.py",
    "morph_advanced.py",
    "geom_advanced.py",
    "enhance_denoise.py",
)

FILTER_IDS = (
    "filter.bilateral",
    "filter.box",
    "filter.sqr_box",
    "filter.filter2d",
    "filter.sep_filter",
    "filter.sharpen",
    "filter.unsharp",
    "filter.laplacian",
    "filter.pyrdown",
    "filter.pyrup",
    "filter.pyr_mean_shift",
)
MORPH_IDS = (
    "morph.gradient",
    "morph.tophat",
    "morph.blackhat",
    "morph.hitmiss",
    "morph.ex",
    "morph.kernel",
)
GEOM_IDS = (
    "geom.transpose",
    "geom.warp_affine",
    "geom.warp_perspective",
    "geom.remap",
    "geom.polar",
    "geom.pad",
    "geom.shear",
    "geom.rotate90",
    "geom.crop",
    "geom.tile",
)
ENHANCE_IDS = (
    "enhance.denoise",
    "enhance.denoise_color",
    "enhance.denoise_tvl1",
    "enhance.edge_preserving",
    "enhance.detail",
    "enhance.pencil_sketch",
    "enhance.stylization",
)

EXPECTED_CATEGORY = {
    **{op_id: "滤波与平滑" for op_id in FILTER_IDS},
    **{op_id: "形态学" for op_id in MORPH_IDS},
    **{op_id: "几何变换" for op_id in GEOM_IDS},
    **{op_id: "图像增强" for op_id in ENHANCE_IDS},
}

ALL_IDS = tuple(EXPECTED_CATEGORY)


def contains_cjk(text: str) -> bool:
    return any(CJK[0] <= char <= CJK[1] for char in str(text))


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


class CatalogFilterGeomTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = OperatorRegistry.build()

    # ------------------------------------------------------------------ 工具
    def run_operator(self, op_id: str, inputs: dict | None = None, params: dict | None = None):
        entry = self.registry.entry(op_id)
        self.assertIsNotNone(entry, f"缺少算子 {op_id}")
        spec = entry.spec
        normalized = normalize_params(spec, params or {})
        resolved: dict = {}
        for port in spec.inputs:
            raw = (inputs or {}).get(port.key)
            resolved[port.key] = raw if raw is None or is_data_value(raw) else ImageData(raw)
        validate_inputs(spec, resolved)
        entry.operator.validate(resolved, normalized)
        kwargs = dict(resolved)
        kwargs.update(normalized)
        return output_to_dict(spec, entry.operator.process(**kwargs))

    def single(self, op_id: str, array: np.ndarray, params: dict | None = None) -> np.ndarray:
        outputs = self.run_operator(op_id, {"image": array}, params)
        self.assertTrue(outputs, f"{op_id} 没有输出")
        port = self.registry.spec(op_id).outputs[0].key
        self.assertIn(port, outputs, f"{op_id} 缺少主输出端口 {port}")
        value = outputs[port]
        self.assertIsInstance(value, ImageData, f"{op_id} 主输出不是图像")
        return value.array

    def group_errors(self):
        return [issue for issue in self.registry.errors if issue.path.name in FILES]

    # ------------------------------------------------------------------ 注册表
    def test_registry_contains_group_operators(self) -> None:
        missing = set(ALL_IDS) - set(self.registry.entries)
        self.assertEqual(missing, set(), f"缺少本组算子：{missing}")
        self.assertEqual(self.group_errors(), [], f"本组算子加载错误：{self.group_errors()}")
        self.assertEqual(len(ALL_IDS), 34)
        for op_id in ALL_IDS:
            entry = self.registry.entry(op_id)
            self.assertTrue(entry.builtin)
            self.assertEqual(entry.spec.category[0], EXPECTED_CATEGORY[op_id])
            self.assertTrue(contains_cjk(entry.spec.name), f"{op_id} 名称应为中文")
            self.assertTrue(contains_cjk(entry.spec.description), f"{op_id} 描述应为中文")
            self.assertTrue(entry.spec.keywords, f"{op_id} 应提供关键词")
            first = entry.spec.inputs[0]
            self.assertEqual(first.key, "image")
            self.assertEqual(first.label, "输入图像")
            self.assertEqual(entry.spec.outputs[0].data_type.name, "IMAGE")

    def test_specs_use_chinese_labels_and_safe_conditions(self) -> None:
        for op_id in ALL_IDS:
            spec = self.registry.spec(op_id)
            for param in spec.params:
                self.assertTrue(param.label, f"{op_id}.{param.key} 缺少标签")
                self.assertTrue(
                    contains_cjk(param.label), f"{op_id}.{param.key} 标签应为中文：{param.label}"
                )
                for expression in (param.visible_when, param.enabled_when):
                    if expression:
                        self.assertNotIn("params.", expression, f"{op_id}.{param.key} 用了属性访问")
            for port in spec.inputs[1:]:
                self.assertTrue(port.optional, f"{op_id} 的第二输入 {port.key} 必须 optional")

    def test_defaults_run_for_all_sample_arrays(self) -> None:
        arrays = sample_arrays()
        for op_id in ALL_IDS:
            spec = self.registry.spec(op_id)
            for name, array in arrays.items():
                if spec.inputs[0].accepts:
                    color_space = ImageData(array).color_space
                    if color_space not in spec.inputs[0].accepts:
                        continue
                before = array.copy()
                outputs = self.run_operator(op_id, {"image": array})
                self.assertTrue(outputs, f"{op_id}/{name} 没有输出")
                image = outputs[spec.outputs[0].key]
                self.assertIsInstance(image, ImageData, f"{op_id}/{name} 主输出不是图像")
                self.assertGreater(image.array.size, 0, f"{op_id}/{name} 输出为空")
                self.assertIn(image.array.ndim, (2, 3), f"{op_id}/{name} 输出维度异常")
                self.assertIn(image.color_space, ("L", "BGR", "BGRA"))
                if not np.array_equal(array, before):
                    self.fail(f"{op_id}/{name} 修改了输入图像")

    # ------------------------------------------------------------------ 行为正确性
    def test_bilateral_smooths_noise(self) -> None:
        rng = np.random.default_rng(11)
        base = np.full((32, 32), 120.0, dtype=np.float32)
        noisy = np.clip(base + rng.integers(-40, 41, base.shape), 0, 255).astype(np.uint8)
        result = self.single("filter.bilateral", noisy)
        input_std = float(noisy.astype(np.float32).std())
        output_std = float(result.astype(np.float32).std())
        self.assertGreater(input_std, 10.0)
        self.assertLess(output_std, input_std * 0.7, "双边滤波应明显降低噪声标准差")

    def test_transpose_matches_numpy(self) -> None:
        array = np.random.default_rng(3).integers(0, 256, (17, 23, 3), dtype=np.uint8)
        result = self.single("geom.transpose", array)
        np.testing.assert_array_equal(result, np.transpose(array, (1, 0, 2)))

    def test_warp_affine_translation(self) -> None:
        array = np.random.default_rng(5).integers(0, 256, (32, 40), dtype=np.uint8)
        result = self.single(
            "geom.warp_affine",
            array,
            {
                "angle": 0.0,
                "scale": 1.0,
                "translate_x": 3,
                "translate_y": 2,
                "interpolation": "nearest",
                "border": "constant",
            },
        )
        np.testing.assert_array_equal(result[2:, 3:], array[:-2, :-3])

    def test_morph_gradient_non_negative(self) -> None:
        array = bgr8()
        result = self.single("morph.gradient", array)
        self.assertGreaterEqual(int(result.min()), 0)
        self.assertGreater(int(result.max()), 0)

    def test_crop_returns_requested_roi(self) -> None:
        array = gray8()
        result = self.single("geom.crop", array, {"roi": [4, 5, 10, 8]})
        self.assertEqual(result.shape, (8, 10))
        np.testing.assert_array_equal(result, array[5:13, 4:14])


if __name__ == "__main__":
    unittest.main()
