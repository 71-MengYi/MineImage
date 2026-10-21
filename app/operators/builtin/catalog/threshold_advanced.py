"""OpenCV 目录算子：高级阈值（自动阈值 / 截断 / 置零 / 区间 / 局部）。

所有算子都通过 :mod:`app.operators.builtin.catalog._cv` 读取输入、换算 dtpye
满量程，保证灰度 / BGR / BGRA 以及 uint8 / uint16 / float32 都能直接运行。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.core.data_types import DataType, NumberValue
from app.operators.builtin.catalog import _cv
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec

_INPUT = PortSpec(
    "image",
    "输入图像",
    accepts=("L", "BGR", "BGRA"),
    description="灰度、BGR 或 BGRA 图像；彩色图像会先转为灰度。",
)


def _binary_flag(mode: str) -> int:
    if mode == "binary":
        return cv2.THRESH_BINARY
    if mode == "binary_inv":
        return cv2.THRESH_BINARY_INV
    raise OperatorError(f"不支持的二值模式：{mode}")


class OtsuThreshold(Operator):
    """Otsu 大津法自动阈值。"""

    spec = OperatorSpec(
        id="threshold.otsu",
        name="Otsu 自动阈值",
        category=("阈值与分割", "阈值"),
        description=(
            "用 Otsu 大津法按灰度直方图自动求最佳阈值并二值化。内部统一转成 8 位"
            "灰度图，因此 uint16 / float32 输入也能直接使用；同时输出自动求得的阈值。"
        ),
        keywords=("otsu", "大津", "自动阈值", "threshold", "二值", "分割"),
        inputs=(_INPUT,),
        outputs=(
            PortSpec("image", "二值图", accepts=("L",)),
            PortSpec("threshold", "自动阈值", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "max_value",
                "前景值",
                "int",
                default=255,
                min=0,
                max=255,
                step=1,
                tooltip="二值化后前景像素的取值（0-255 通用刻度）。",
            ),
            ParamSpec(
                "mode",
                "二值模式",
                "enum",
                choices=(("binary", "二值"), ("binary_inv", "反二值")),
                default="binary",
                tooltip="二值：亮于阈值置为前景值；反二值：暗于阈值置为前景值。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        max_value: int = 255,
        mode: str = "binary",
        **_: Any,
    ):
        gray = _cv.source_gray8(image, "输入图像")
        value, binary = cv2.threshold(
            gray, 0, int(max_value), _binary_flag(mode) | cv2.THRESH_OTSU
        )
        return {
            "image": binary,
            "threshold": NumberValue(value=float(value), label="Otsu 阈值"),
        }


class TriangleThreshold(Operator):
    """Triangle 三角法自动阈值。"""

    spec = OperatorSpec(
        id="threshold.triangle",
        name="Triangle 自动阈值",
        category=("阈值与分割", "阈值"),
        description=(
            "用 Triangle（三角）法自动求阈值，适合直方图只有一个明显峰值的偏态图像。"
            "内部统一转成 8 位灰度图，同时输出自动求得的阈值。"
        ),
        keywords=("triangle", "三角", "自动阈值", "threshold", "二值", "分割"),
        inputs=(_INPUT,),
        outputs=(
            PortSpec("image", "二值图", accepts=("L",)),
            PortSpec("threshold", "自动阈值", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "max_value",
                "前景值",
                "int",
                default=255,
                min=0,
                max=255,
                step=1,
                tooltip="二值化后前景像素的取值（0-255 通用刻度）。",
            ),
            ParamSpec(
                "mode",
                "二值模式",
                "enum",
                choices=(("binary", "二值"), ("binary_inv", "反二值")),
                default="binary",
            ),
        ),
    )

    def process(
        self,
        image=None,
        max_value: int = 255,
        mode: str = "binary",
        **_: Any,
    ):
        gray = _cv.source_gray8(image, "输入图像")
        value, binary = cv2.threshold(
            gray, 0, int(max_value), _binary_flag(mode) | cv2.THRESH_TRIANGLE
        )
        return {
            "image": binary,
            "threshold": NumberValue(value=float(value), label="Triangle 阈值"),
        }


class TruncThreshold(Operator):
    """截断阈值：高于阈值的像素被压到阈值。"""

    spec = OperatorSpec(
        id="threshold.trunc",
        name="截断阈值",
        category=("阈值与分割", "阈值"),
        description=(
            "把高于阈值的像素直接截断到阈值，低于阈值的像素保持不变。"
            "阈值使用 0-255 通用刻度并按图像 dtype 自动换算；彩色图像会先转灰度，"
            "输出灰度图并保持输入 dtype。"
        ),
        keywords=("trunc", "截断", "threshold", "阈值"),
        inputs=(_INPUT,),
        outputs=(PortSpec("image", "截断结果", accepts=("L",)),),
        params=(
            ParamSpec(
                "value",
                "阈值",
                "int",
                default=128,
                min=0,
                max=255,
                step=1,
                tooltip="0-255 通用刻度；uint16 会换算为 ×257，float 按满量程换算。",
            ),
        ),
    )

    def process(self, image=None, value: int = 128, **_: Any):
        gray = _cv.source_gray(image, "输入图像")
        native = _cv.scale_value(gray, int(value))
        _result_value, result = cv2.threshold(gray, native, native, cv2.THRESH_TRUNC)
        return result


class ToZeroThreshold(Operator):
    """置零阈值：低于（或高于）阈值的像素被置零。"""

    spec = OperatorSpec(
        id="threshold.tozero",
        name="置零阈值",
        category=("阈值与分割", "阈值"),
        description=(
            "低于阈值的像素置零（或反向：高于阈值的像素置零），其余像素保持原值。"
            "阈值使用 0-255 通用刻度并按 dtype 自动换算；彩色图像会先转灰度，"
            "输出灰度图并保持输入 dtype。"
        ),
        keywords=("tozero", "置零", "threshold", "阈值"),
        inputs=(_INPUT,),
        outputs=(PortSpec("image", "置零结果", accepts=("L",)),),
        params=(
            ParamSpec(
                "value",
                "阈值",
                "int",
                default=128,
                min=0,
                max=255,
                step=1,
                tooltip="0-255 通用刻度，按图像 dtype 自动换算。",
            ),
            ParamSpec(
                "mode",
                "置零方式",
                "enum",
                choices=(("tozero", "低于阈值置零"), ("tozero_inv", "高于阈值置零")),
                default="tozero",
            ),
        ),
    )

    def process(self, image=None, value: int = 128, mode: str = "tozero", **_: Any):
        gray = _cv.source_gray(image, "输入图像")
        flag = cv2.THRESH_TOZERO if mode == "tozero" else cv2.THRESH_TOZERO_INV
        _result_value, result = cv2.threshold(
            gray,
            _cv.scale_value(gray, int(value)),
            _cv.scale_value(gray, 255.0),
            flag,
        )
        return result


class RangeThreshold(Operator):
    """区间阈值：保留灰度落在 [下限, 上限] 内的像素。"""

    spec = OperatorSpec(
        id="threshold.range",
        name="区间阈值",
        category=("阈值与分割", "阈值"),
        description=(
            "用 inRange 保留灰度落在下限与上限之间的像素，输出 0/255 掩膜与覆盖率。"
            "可连接可选的限制蒙版，只在蒙版范围内统计；下限大于上限时会自动交换。"
        ),
        keywords=("inrange", "区间", "范围阈值", "threshold", "掩膜", "分割"),
        inputs=(
            _INPUT,
            PortSpec(
                "mask",
                "限制蒙版（可选）",
                accepts=("L",),
                optional=True,
                description="非零像素参与统计；未连接时对整幅图生效。",
            ),
        ),
        outputs=(
            PortSpec("image", "区间蒙版", accepts=("L",)),
            PortSpec("ratio", "覆盖率", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec("low", "下限", "int", default=100, min=0, max=255, step=1),
            ParamSpec("high", "上限", "int", default=200, min=0, max=255, step=1),
            ParamSpec(
                "invert",
                "反选",
                "bool",
                default=False,
                tooltip="开启后保留区间之外的像素。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        mask=None,
        low: int = 100,
        high: int = 200,
        invert: bool = False,
        **_: Any,
    ):
        gray = _cv.source_gray(image, "输入图像")
        low_value, high_value = int(low), int(high)
        if low_value > high_value:
            low_value, high_value = high_value, low_value
        result = cv2.inRange(
            gray,
            _cv.scale_value(gray, float(low_value)),
            _cv.scale_value(gray, float(high_value)),
        )
        if invert:
            result = cv2.bitwise_not(result)
        if mask is not None:
            mask_gray = _cv.source_gray8(mask, "限制蒙版")
            _cv.same_size(gray, mask_gray, "限制蒙版")
            result = cv2.bitwise_and(result, ((mask_gray > 0).astype(np.uint8) * 255))
        ratio = float((result > 0).mean() * 100.0) if result.size else 0.0
        return {
            "image": result,
            "ratio": NumberValue(value=ratio, label="区间覆盖率", unit="%"),
        }


class AdaptiveThresholdAdvanced(Operator):
    """局部（自适应）阈值高级版。"""

    spec = OperatorSpec(
        id="threshold.adaptive_advanced",
        name="局部阈值（高级）",
        category=("阈值与分割", "阈值"),
        description=(
            "用自适应阈值处理光照不均的图像：均值或高斯加权计算每个像素的局部阈值。"
            "内部统一转成 8 位灰度，输出 0/255 二值图与前景占比。"
        ),
        keywords=("adaptive", "自适应", "局部阈值", "threshold", "二值", "分割"),
        inputs=(_INPUT,),
        outputs=(
            PortSpec("image", "二值图", accepts=("L",)),
            PortSpec("foreground_ratio", "前景占比", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "block_size",
                "邻域大小",
                "int",
                default=11,
                min=3,
                max=99,
                step=2,
                odd=True,
                tooltip="用于计算局部阈值的邻域边长，必须为奇数；图像较小时会自动缩小。",
            ),
            ParamSpec(
                "constant",
                "偏移量",
                "float",
                default=2.0,
                min=-50.0,
                max=50.0,
                step=0.5,
                tooltip="从局部均值/加权均值中减去的常数，用于微调二值化松紧。",
            ),
            ParamSpec(
                "method",
                "加权方式",
                "enum",
                choices=(("mean", "均值"), ("gaussian", "高斯")),
                default="gaussian",
            ),
            ParamSpec(
                "mode",
                "二值模式",
                "enum",
                choices=(("binary", "二值"), ("binary_inv", "反二值")),
                default="binary",
            ),
        ),
    )

    def process(
        self,
        image=None,
        block_size: int = 11,
        constant: float = 2.0,
        method: str = "gaussian",
        mode: str = "binary",
        **_: Any,
    ):
        gray = _cv.source_gray8(image, "输入图像")
        height, width = gray.shape[:2]
        limit = min(height, width)
        if limit < 3:
            raise OperatorError(
                f"图像尺寸 {width}×{height} 太小，自适应阈值至少需要 3×3 的图像。"
            )
        block = int(block_size)
        if block % 2 == 0:
            block += 1
        block = min(block, limit if limit % 2 == 1 else limit - 1)
        block = max(block, 3)
        method_flag = (
            cv2.ADAPTIVE_THRESH_MEAN_C
            if method == "mean"
            else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        )
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            method_flag,
            _binary_flag(mode),
            block,
            float(constant),
        )
        ratio = float((binary > 0).mean() * 100.0) if binary.size else 0.0
        return {
            "image": binary,
            "foreground_ratio": NumberValue(value=ratio, label="前景占比", unit="%"),
        }


OPERATORS = (
    OtsuThreshold,
    TriangleThreshold,
    TruncThreshold,
    ToZeroThreshold,
    RangeThreshold,
    AdaptiveThresholdAdvanced,
)
