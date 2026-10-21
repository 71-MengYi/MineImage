"""算术 / 颜色 / 直方图目录算子的单元测试（不启动 GUI）。"""

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

from app.core.data_types import NumberValue  # noqa: E402
from app.core.types import ImageData  # noqa: E402
from app.operators.loader import OperatorRegistry  # noqa: E402
from app.operators.spec import (  # noqa: E402
    normalize_params,
    output_to_dict,
    validate_inputs,
)

#: 本组测试覆盖的算子 id（算术 / 颜色 / 直方图）。
OPERATOR_IDS = (
    # arithmetic.py
    "basic.add",
    "basic.subtract",
    "basic.multiply",
    "basic.divide",
    "basic.absdiff",
    "basic.add_weighted",
    "basic.bitwise_and",
    "basic.bitwise_or",
    "basic.bitwise_xor",
    "basic.bitwise_not",
    "basic.compare",
    "basic.normalize",
    "basic.convert_scale",
    "basic.gamma",
    # color.py
    "color.cvt",
    "color.swap",
    "color.hsv_adjust",
    "color.mix",
    "color.palette",
    "color.white_balance",
    "color.temperature",
    "color.alpha_composite",
    "color.extract_alpha",
    "color.set_alpha",
    "color.lut_curve",
    # histogram.py
    "analysis.histogram",
    "analysis.equalize",
    "analysis.clahe",
    "analysis.hist_compare",
    "analysis.backproject",
    "analysis.match_hist",
    "analysis.hist_2d",
)

GROUP_FILES = ("arithmetic.py", "color.py", "histogram.py")

CATEGORY_BY_PREFIX = {
    "basic.": "基础调整",
    "color.": "通道与色彩",
    "analysis.": "特征与分析",
}


def _rng() -> np.random.Generator:
    return np.random.default_rng(20240607)


def _samples() -> dict[str, np.ndarray]:
    """7 种样例数组：灰度 / BGR / BGRA × uint8 / uint16 / float32。"""
    generator = _rng()
    bgr8 = generator.integers(0, 256, (24, 32, 3), dtype=np.uint8)
    gray8 = bgr8[:, :, 1].copy()
    bgra8 = np.dstack([bgr8, np.full(bgr8.shape[:2], 180, dtype=np.uint8)])
    bgr16 = (bgr8.astype(np.uint16) * 257).astype(np.uint16)
    gray16 = (gray8.astype(np.uint16) * 257).astype(np.uint16)
    bgrf = (bgr8.astype(np.float32) / 255.0).astype(np.float32)
    grayf = (gray8.astype(np.float32) / 255.0).astype(np.float32)
    return {
        "gray8": gray8,
        "bgr8": bgr8,
        "bgra8": bgra8,
        "gray16": gray16,
        "bgr16": bgr16,
        "grayf": grayf,
        "bgrf": bgrf,
    }


def _contains_cjk(text: str) -> bool:
    return any("\u4e00" <= char <= "\u9fff" for char in str(text))


class CatalogArithmeticColorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = OperatorRegistry.build()

    # ------------------------------------------------------------------ 工具
    def run_group_operator(
        self,
        op_id: str,
        array: np.ndarray,
        params: dict | None = None,
        second: np.ndarray | None = None,
    ) -> dict:
        """按注册表声明执行一个算子，返回规范化后的多输出字典。"""
        entry = self.registry.entry(op_id)
        self.assertIsNotNone(entry, f"缺少算子 {op_id}")
        spec = entry.spec
        resolved: dict = {}
        for port in spec.inputs:
            if port.key == "image":
                resolved[port.key] = ImageData(array)
            elif port.key == "image2":
                resolved[port.key] = ImageData(second) if second is not None else None
            else:  # 本组没有其它输入端口
                resolved[port.key] = None
        normalized = normalize_params(spec, params or {})
        validate_inputs(spec, resolved)
        entry.operator.validate(resolved, normalized)
        kwargs = dict(resolved)
        kwargs.update(normalized)
        return output_to_dict(spec, entry.operator.process(**kwargs))

    def single(self, op_id: str, array: np.ndarray, params: dict | None = None) -> np.ndarray:
        outputs = self.run_group_operator(op_id, array, params)
        spec = self.registry.entry(op_id).spec
        first = spec.outputs[0].key
        self.assertIn(first, outputs)
        first_output = outputs[first]
        self.assertIsInstance(first_output, ImageData)
        return first_output.array

    # ------------------------------------------------------------------ 注册表
    def test_group_operators_are_registered_with_expected_metadata(self) -> None:
        missing = set(OPERATOR_IDS) - set(self.registry.entries)
        self.assertEqual(missing, set(), f"缺少算子：{sorted(missing)}")
        group_errors = [
            issue
            for issue in self.registry.errors
            if issue.path.name in GROUP_FILES
        ]
        self.assertEqual(group_errors, [], [issue.display() for issue in group_errors])
        for op_id in OPERATOR_IDS:
            entry = self.registry.entry(op_id)
            self.assertEqual(entry.path.name in GROUP_FILES, True, op_id)
            prefix = op_id.split(".", 1)[0] + "."
            self.assertEqual(entry.spec.category[0], CATEGORY_BY_PREFIX[prefix], op_id)
            self.assertTrue(_contains_cjk(entry.spec.name), f"{op_id} 名称应为中文")
            self.assertTrue(_contains_cjk(entry.spec.description), f"{op_id} 描述应为中文")

    # ------------------------------------------------------------------ 通用执行
    def test_all_group_operators_run_with_defaults_and_keep_inputs(self) -> None:
        samples = _samples()
        for op_id in OPERATOR_IDS:
            with self.subTest(op_id=op_id):
                spec = self.registry.entry(op_id).spec
                for name, array in samples.items():
                    before = array.copy()
                    outputs = self.run_group_operator(op_id, array)
                    self.assertTrue(outputs, f"{op_id}/{name} 没有输出")
                    first = spec.outputs[0].key
                    self.assertIn(first, outputs, f"{op_id}/{name} 缺少第一个输出")
                    self.assertIsInstance(outputs[first], ImageData)
                    self.assertGreater(outputs[first].array.size, 0)
                    self.assertTrue(
                        np.array_equal(array, before),
                        f"{op_id}/{name} 修改了输入图像",
                    )

    # ------------------------------------------------------------------ 算术行为
    def test_add_defaults_brighten_and_are_dtype_aware(self) -> None:
        samples = _samples()
        gray8 = samples["gray8"]
        result8 = self.single("basic.add", gray8)
        self.assertEqual(result8.dtype, np.uint8)
        expected = np.clip(gray8.astype(np.int32) + 30, 0, 255).astype(np.uint8)
        np.testing.assert_array_equal(result8, expected)
        for name in ("gray16", "bgr16", "grayf", "bgrf"):
            array = samples[name]
            result = self.single("basic.add", array)
            self.assertEqual(result.dtype, array.dtype, name)
            self.assertGreater(float(result.mean()), float(array.mean()), name)

    def test_bitwise_not_inverts_and_is_involutive(self) -> None:
        samples = _samples()
        gray8 = samples["gray8"]
        inverted = self.single("basic.bitwise_not", gray8)
        np.testing.assert_array_equal(inverted, 255 - gray8)
        np.testing.assert_array_equal(
            self.single("basic.bitwise_not", inverted), gray8
        )
        gray16 = samples["gray16"]
        inverted16 = self.single("basic.bitwise_not", gray16)
        self.assertEqual(inverted16.dtype, np.uint16)
        np.testing.assert_array_equal(
            inverted16, np.iinfo(np.uint16).max - gray16
        )

    # ------------------------------------------------------------------ 颜色行为
    def test_hsv_adjust_neutral_is_identity_and_saturation_grays_out(self) -> None:
        color = _samples()["bgr8"]
        neutral = self.single("color.hsv_adjust", color)
        np.testing.assert_array_equal(neutral, color)
        hue_shifted = self.single("color.hsv_adjust", color, {"hue_shift": 60})
        self.assertFalse(np.array_equal(hue_shifted, color))
        grayed = self.single("color.hsv_adjust", color, {"saturation": 0.0})
        grayed_int = grayed.astype(np.int32)
        spread = grayed_int.max(axis=2) - grayed_int.min(axis=2)
        self.assertLessEqual(int(spread.max()), 2)

    # ------------------------------------------------------------------ 直方图行为
    def test_equalize_spreads_narrow_histogram_and_keeps_dtype(self) -> None:
        image = np.full((32, 32), 120, dtype=np.uint8)
        image[16:, :] = 140
        equalized = self.single("analysis.equalize", image)
        self.assertEqual(equalized.dtype, np.uint8)
        self.assertGreater(float(equalized.std()), float(image.std()))
        self.assertGreaterEqual(len(np.unique(equalized)), 2)
        for name in ("gray16", "bgr16", "grayf", "bgrf", "bgra8"):
            array = _samples()[name]
            result = self.single("analysis.equalize", array)
            self.assertEqual(result.dtype, array.dtype, name)
            self.assertEqual(result.shape, array.shape, name)

    def test_hist_compare_matches_identical_and_separates_disjoint(self) -> None:
        source = np.full((24, 32), 100, dtype=np.uint8)
        reference = np.full((24, 32), 200, dtype=np.uint8)
        identical = self.run_group_operator(
            "analysis.hist_compare", source, {"method": "intersection"}
        )
        different = self.run_group_operator(
            "analysis.hist_compare",
            source,
            {"method": "intersection"},
            second=reference,
        )
        for outputs in (identical, different):
            self.assertEqual(set(outputs), {"image", "similarity"})
            self.assertIsInstance(outputs["similarity"], NumberValue)
            self.assertIsInstance(outputs["image"], ImageData)
        self.assertAlmostEqual(identical["similarity"].value, 1.0, places=6)
        self.assertLess(different["similarity"].value, 0.1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
