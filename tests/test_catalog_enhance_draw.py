"""OpenCV 目录算子 D（图像增强 / 绘图 / 布局）的单元测试。

测试只依赖本文件内的简短 helper，不导入 ``tests`` 下的其它模块：

* 遍历本组全部算子 × 7 种样例数组，使用默认参数执行，断言有图像输出、
  输出尺寸有效、输入数组未被原地修改；
* 另外用 5 个行为断言检查绘制保 alpha、修复保尺寸、拼接尺寸、差值叠加与
  边框 / 四宫格几何。
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

from app.core.types import ImageData  # noqa: E402
from app.operators.loader import OperatorRegistry  # noqa: E402
from app.operators.spec import (  # noqa: E402
    normalize_params,
    output_to_dict,
    validate_inputs,
)

#: 本组负责的全部算子 id。
GROUP_IDS = (
    # 图像增强（photo 组）
    "enhance.inpaint",
    "enhance.seamless_clone",
    "enhance.color_change",
    "enhance.illumination",
    "enhance.texture_flatten",
    "enhance.hdr_merge",
    "enhance.tonemap",
    "enhance.retinex",
    "enhance.local_contrast",
    "enhance.dehaze",
    # 交互处理（绘图组）
    "draw.rectangle",
    "draw.circle",
    "draw.ellipse",
    "draw.polygon",
    "draw.text",
    "draw.arrow",
    "draw.marker",
    "draw.crosshair",
    "draw.grid",
    "draw.scale_bar",
    "draw.watermark",
    "draw.roi_highlight",
    # 布局
    "layout.montage",
    "layout.overlay",
    "layout.contact_sheet",
    "layout.border",
    "layout.split",
)

#: 本组合法的一级分类。
GROUP_CATEGORIES = ("图像增强", "交互处理")

_HEIGHT, _WIDTH = 24, 32


# ---------------------------------------------------------------------------
# 本地样例数据 helper
# ---------------------------------------------------------------------------
def _bgr8(height: int = _HEIGHT, width: int = _WIDTH) -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def _samples() -> dict[str, np.ndarray]:
    """7 种样例数组：L/BGR/BGRA × uint8/uint16/float32 的代表组合。"""
    bgr = _bgr8()
    gray = bgr[:, :, 1].copy()
    bgra = np.dstack([bgr, np.full(bgr.shape[:2], 180, dtype=np.uint8)])
    return {
        "gray8": gray,
        "bgr8": bgr,
        "bgra8": bgra,
        "gray16": (gray.astype(np.uint16) * 257).astype(np.uint16),
        "bgr16": (bgr.astype(np.uint16) * 257).astype(np.uint16),
        "grayf": (gray.astype(np.float32) / 255.0).astype(np.float32),
        "bgrf": (bgr.astype(np.float32) / 255.0).astype(np.float32),
    }


class CatalogEnhanceDrawTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = OperatorRegistry.build()

    # ------------------------------------------------------------------ 工具
    def run_operator(
        self,
        op_id: str,
        array: np.ndarray | None = None,
        params: dict | None = None,
    ):
        entry = self.registry.entry(op_id)
        self.assertIsNotNone(entry, f"缺少算子 {op_id}")
        normalized = normalize_params(entry.spec, dict(params or {}))
        inputs: dict[str, ImageData | None] = {}
        for port in entry.spec.inputs:
            if port.key == "image":
                self.assertIsNotNone(array, f"{op_id} 需要 image 输入")
                inputs[port.key] = ImageData(array)
            else:
                inputs[port.key] = None
        validate_inputs(entry.spec, inputs)
        kwargs = dict(inputs)
        kwargs.update(normalized)
        return output_to_dict(entry.spec, entry.operator.process(**kwargs))

    def first_image(self, op_id: str, array: np.ndarray, params: dict | None = None) -> np.ndarray:
        outputs = self.run_operator(op_id, array, params)
        self.assertTrue(outputs, f"{op_id} 没有输出")
        first_key = self.registry.spec(op_id).outputs[0].key
        output = outputs[first_key]
        self.assertIsInstance(output, ImageData, f"{op_id} 的第一个输出必须是图像")
        return output.array

    # ------------------------------------------------------------------ 注册表
    def test_group_is_registered_and_categorized(self) -> None:
        entries = self.registry.entries
        missing = [op_id for op_id in GROUP_IDS if op_id not in entries]
        self.assertEqual(missing, [], f"缺少本组算子：{missing}")
        own_errors = [
            issue
            for issue in self.registry.errors
            if issue.path.name in ("photo_enhance.py", "draw.py", "layout.py")
        ]
        self.assertEqual(own_errors, [], "本组算子文件存在加载错误")
        for op_id in GROUP_IDS:
            spec = self.registry.spec(op_id)
            self.assertIn(spec.category[0], GROUP_CATEGORIES, f"{op_id} 一级分类非法")
            self.assertTrue(spec.name.strip(), f"{op_id} 缺少中文名称")
            self.assertTrue(spec.description.strip(), f"{op_id} 缺少中文描述")
            self.assertEqual(spec.inputs[0].key, "image", f"{op_id} 首选输入应为 image")
            self.assertEqual(spec.inputs[0].label, "输入图像", f"{op_id} 输入标签应为“输入图像”")
            self.assertIn("L", spec.inputs[0].accepts, f"{op_id} 应支持灰度输入")
            self.assertIn("BGR", spec.inputs[0].accepts, f"{op_id} 应支持 BGR 输入")
            self.assertIn("BGRA", spec.inputs[0].accepts, f"{op_id} 应支持 BGRA 输入")
            for param in spec.params:
                self.assertTrue(param.label.strip(), f"{op_id}.{param.key} 缺少中文标签")

    # ------------------------------------------------------------------ 全量默认参数
    def test_all_group_operators_run_with_defaults(self) -> None:
        samples = _samples()
        for op_id in GROUP_IDS:
            for name, array in samples.items():
                before = array.copy()
                outputs = self.run_operator(op_id, array)
                self.assertTrue(outputs, f"{op_id}/{name} 没有输出")
                first_key = self.registry.spec(op_id).outputs[0].key
                output = outputs[first_key]
                self.assertIsInstance(output, ImageData, f"{op_id}/{name} 第一个输出不是图像")
                self.assertGreater(output.array.size, 0, f"{op_id}/{name} 输出为空")
                self.assertIn(output.array.ndim, (2, 3), f"{op_id}/{name} 输出维度异常")
                self.assertEqual(
                    output.array.dtype, array.dtype, f"{op_id}/{name} 未保持 dtype"
                )
                self.assertEqual(
                    output.color_space,
                    ImageData(array).color_space,
                    f"{op_id}/{name} 未保持通道数",
                )
                if not np.array_equal(array, before):
                    self.fail(f"{op_id}/{name} 修改了输入图像")

    # ------------------------------------------------------------------ 行为断言
    def test_draw_rectangle_changes_pixels_and_keeps_alpha(self) -> None:
        bgra = np.dstack(
            [_bgr8(), np.full((_HEIGHT, _WIDTH), 180, dtype=np.uint8)]
        )
        drawn = self.first_image("draw.rectangle", bgra)
        self.assertEqual(drawn.shape, bgra.shape)
        self.assertEqual(drawn.dtype, np.uint8)
        self.assertFalse(np.array_equal(drawn, bgra), "矩形没有改变像素")
        np.testing.assert_array_equal(drawn[:, :, 3], bgra[:, :, 3])

        # 16 位 / 浮点图像绘制后 dtype 不变，且中心区域确实被覆盖。
        for array in (
            (_bgr8().astype(np.uint16) * 257).astype(np.uint16),
            (_bgr8().astype(np.float32) / 255.0).astype(np.float32),
        ):
            result = self.first_image("draw.rectangle", array)
            self.assertEqual(result.dtype, array.dtype)
            self.assertFalse(np.array_equal(result, array))

    def test_inpaint_preserves_size_and_dtype(self) -> None:
        for array in (_bgr8(), _bgr8()[:, :, 1].copy().astype(np.uint16) * 257):
            result = self.first_image("enhance.inpaint", array)
            self.assertEqual(result.shape, array.shape)
            self.assertEqual(result.dtype, array.dtype)

    def test_montage_dimensions(self) -> None:
        bgr = _bgr8()
        horizontal = self.first_image("layout.montage", bgr, {"gap": 0})
        self.assertEqual(horizontal.shape, (_HEIGHT, _WIDTH * 2, 3))
        vertical = self.first_image(
            "layout.montage", bgr, {"gap": 0, "direction": "vertical"}
        )
        self.assertEqual(vertical.shape, (_HEIGHT * 2, _WIDTH, 3))
        # 默认间隔会额外增加像素。
        default = self.first_image("layout.montage", bgr)
        self.assertEqual(default.shape, (_HEIGHT, _WIDTH * 2 + 8, 3))

    def test_overlay_difference_of_identical_images_is_black(self) -> None:
        bgr = _bgr8()
        result = self.first_image(
            "layout.overlay", bgr, {"mode": "difference", "opacity": 1.0}
        )
        self.assertEqual(result.shape, bgr.shape)
        self.assertEqual(int(result.max()), 0)

    def test_border_and_split_geometry(self) -> None:
        bgr = _bgr8()
        bordered = self.first_image("layout.border", bgr, {"width": 6})
        self.assertEqual(bordered.shape, (_HEIGHT + 12, _WIDTH + 12, 3))
        np.testing.assert_array_equal(bordered[6:30, 6:38], bgr)
        split = self.first_image("layout.split", bgr, {"gap": 4})
        self.assertEqual(split.shape, (_HEIGHT + 4, _WIDTH + 4, 3))
        zero_gap = self.first_image("layout.split", bgr, {"gap": 0})
        self.assertEqual(zero_gap.shape, bgr.shape)


if __name__ == "__main__":
    unittest.main(verbosity=2)
