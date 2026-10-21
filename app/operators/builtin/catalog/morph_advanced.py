"""高级形态学算子：梯度、顶帽/黑帽、击中击不中、通用形态学与结构元素预览。

所有算子都通过 :mod:`app.operators.builtin.catalog._cv` 读取输入，遵循
“不原地修改输入、异常用中文提示”的约定。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.operators.builtin.catalog import _cv
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec

_SHAPES = (
    ("rect", "矩形"),
    ("ellipse", "椭圆"),
    ("cross", "十字"),
)

def _kernel(shape: str, ksize: int) -> np.ndarray:
    return _cv.made_kernel(shape, int(ksize))


class MorphGradient(Operator):
    """形态学梯度（膨胀 - 腐蚀）。"""

    spec = OperatorSpec(
        id="morph.gradient",
        name="形态学梯度",
        category=("形态学",),
        description=(
            "用膨胀图减腐蚀图得到形态学梯度，突出亮度变化剧烈的边缘。"
            "结果始终非负；支持 L/BGR/BGRA 与 uint8/uint16/float32。"
        ),
        keywords=("形态学", "梯度", "gradient", "边缘", "morphology"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec("shape", "结构元素", "enum", choices=_SHAPES, default="rect"),
            ParamSpec("ksize", "核大小", "int", default=3, min=1, max=99, step=2, odd=True, unit="px"),
            ParamSpec("iterations", "迭代次数", "int", default=1, min=1, max=20, step=1),
        ),
    )

    def process(self, image=None, shape: str = "rect", ksize: int = 3, iterations: int = 1, **_: Any):
        array = _cv.source(image)
        return cv2.morphologyEx(
            array, cv2.MORPH_GRADIENT, _kernel(shape, ksize), iterations=int(iterations)
        )


class MorphTopHat(Operator):
    """顶帽变换（原图 - 开运算）。"""

    spec = OperatorSpec(
        id="morph.tophat",
        name="顶帽变换",
        category=("形态学",),
        description=(
            "原图减去开运算结果，提取比结构元素更小的亮细节，例如小亮点与细亮线。"
            "对彩色图像按通道独立处理。"
        ),
        keywords=("顶帽", "tophat", "亮细节", "形态学", "morphology"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec("shape", "结构元素", "enum", choices=_SHAPES, default="rect"),
            ParamSpec("ksize", "核大小", "int", default=9, min=1, max=99, step=2, odd=True, unit="px"),
            ParamSpec("iterations", "迭代次数", "int", default=1, min=1, max=20, step=1),
        ),
    )

    def process(self, image=None, shape: str = "rect", ksize: int = 9, iterations: int = 1, **_: Any):
        array = _cv.source(image)
        return cv2.morphologyEx(
            array, cv2.MORPH_TOPHAT, _kernel(shape, ksize), iterations=int(iterations)
        )


class MorphBlackHat(Operator):
    """黑帽变换（闭运算 - 原图）。"""

    spec = OperatorSpec(
        id="morph.blackhat",
        name="黑帽变换",
        category=("形态学",),
        description=(
            "闭运算结果减去原图，提取比结构元素更小的暗细节，例如小暗点与细暗线。"
            "对彩色图像按通道独立处理。"
        ),
        keywords=("黑帽", "blackhat", "暗细节", "形态学", "morphology"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec("shape", "结构元素", "enum", choices=_SHAPES, default="rect"),
            ParamSpec("ksize", "核大小", "int", default=9, min=1, max=99, step=2, odd=True, unit="px"),
            ParamSpec("iterations", "迭代次数", "int", default=1, min=1, max=20, step=1),
        ),
    )

    def process(self, image=None, shape: str = "rect", ksize: int = 9, iterations: int = 1, **_: Any):
        array = _cv.source(image)
        return cv2.morphologyEx(
            array, cv2.MORPH_BLACKHAT, _kernel(shape, ksize), iterations=int(iterations)
        )


class MorphHitMiss(Operator):
    """击中击不中变换（二值模板匹配）。"""

    spec = OperatorSpec(
        id="morph.hitmiss",
        name="击中击不中",
        category=("形态学",),
        description=(
            "把输入转为 8 位二值图，用环形结构元素做击中击不中变换，"
            "检测孤立的前景亮点（或按需检测孤立暗点）。输出 0/255 二值图。"
            "该 OpenCV 接口只支持 8 位单通道，内部会先做灰度化与阈值化。"
        ),
        keywords=("击中击不中", "hitmiss", "hit or miss", "二值", "孤立点", "模板"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像", accepts=("L",)),),
        params=(
            ParamSpec(
                "ksize",
                "模板大小",
                "int",
                default=3,
                min=3,
                max=15,
                step=2,
                odd=True,
                unit="px",
                tooltip="环形模板边长，3×3 检测孤立像素。",
            ),
            ParamSpec(
                "threshold",
                "二值化阈值",
                "int",
                default=0,
                min=0,
                max=255,
                step=1,
                tooltip="0 表示使用 Otsu 自动阈值。",
            ),
            ParamSpec(
                "polarity",
                "检测目标",
                "enum",
                choices=(("bright", "孤立亮点"), ("dark", "孤立暗点")),
                default="bright",
            ),
        ),
    )

    def process(
        self,
        image=None,
        ksize: int = 3,
        threshold: int = 0,
        polarity: str = "bright",
        **_: Any,
    ):
        gray = _cv.source_gray8(image)
        if int(threshold) <= 0:
            _value, binary = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            _value, binary = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)

        size = int(ksize)
        kernel = np.zeros((size, size), dtype=np.int8)
        kernel[0, :] = -1
        kernel[-1, :] = -1
        kernel[:, 0] = -1
        kernel[:, -1] = -1
        kernel[size // 2, size // 2] = 1
        if polarity == "dark":
            kernel = -kernel
        elif polarity != "bright":
            raise OperatorError(f"不支持的检测目标：{polarity}")

        return cv2.morphologyEx(binary, cv2.MORPH_HITMISS, kernel)


class MorphGeneral(Operator):
    """通用形态学（可选用蒙版限制作用区域）。"""

    spec = OperatorSpec(
        id="morph.ex",
        name="通用形态学",
        category=("形态学",),
        description=(
            "在一个算子内选择腐蚀、膨胀、开/闭运算、梯度、顶帽或黑帽。"
            "可选蒙版输入：蒙版为 0 的区域保持原图不变；未连接蒙版时对整幅图处理。"
        ),
        keywords=("形态学", "morphology", "腐蚀", "膨胀", "开运算", "闭运算", "gradient"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec("mask", "蒙版（可选）", accepts=("L",), optional=True, description="白色区域参与形态学运算。"),
        ),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec(
                "op",
                "运算类型",
                "enum",
                choices=(
                    ("erode", "腐蚀"),
                    ("dilate", "膨胀"),
                    ("open", "开运算"),
                    ("close", "闭运算"),
                    ("gradient", "形态学梯度"),
                    ("tophat", "顶帽"),
                    ("blackhat", "黑帽"),
                ),
                default="open",
            ),
            ParamSpec("shape", "结构元素", "enum", choices=_SHAPES, default="ellipse"),
            ParamSpec("ksize", "核大小", "int", default=5, min=1, max=99, step=2, odd=True, unit="px"),
            ParamSpec("iterations", "迭代次数", "int", default=1, min=1, max=20, step=1),
        ),
    )

    _OPS = {
        "erode": cv2.MORPH_ERODE,
        "dilate": cv2.MORPH_DILATE,
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "gradient": cv2.MORPH_GRADIENT,
        "tophat": cv2.MORPH_TOPHAT,
        "blackhat": cv2.MORPH_BLACKHAT,
    }

    def process(
        self,
        image=None,
        mask=None,
        op: str = "open",
        shape: str = "ellipse",
        ksize: int = 5,
        iterations: int = 1,
        **_: Any,
    ):
        array = _cv.source(image)
        morph_op = self._OPS.get(str(op))
        if morph_op is None:
            raise OperatorError(f"不支持的形态学运算：{op}")

        result = cv2.morphologyEx(
            array, morph_op, _kernel(shape, ksize), iterations=int(iterations)
        )
        if mask is None:
            return result

        mask_array = _cv.source_gray(mask, "蒙版")
        _cv.same_size(array, mask_array, "蒙版")
        keep = mask_array > 0
        if result.ndim == 3:
            keep = keep[:, :, None]
        return np.where(keep, result, array)


class MorphKernel(Operator):
    """结构元素预览（生成图像）。"""

    spec = OperatorSpec(
        id="morph.kernel",
        name="结构元素预览",
        category=("形态学",),
        description=(
            "把矩形 / 椭圆 / 十字结构元素放大成像素块并生成预览图，"
            "右侧标注核形状、尺寸、前景像素数与输入尺寸。输出为 8 位 BGR 图像。"
        ),
        keywords=("结构元素", "kernel", "structuring element", "预览", "形态学"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec("shape", "结构元素", "enum", choices=_SHAPES, default="ellipse"),
            ParamSpec("ksize", "核大小", "int", default=5, min=1, max=31, step=2, odd=True, unit="px"),
            ParamSpec(
                "cell",
                "单元格像素",
                "int",
                default=24,
                min=8,
                max=64,
                step=1,
                unit="px",
                tooltip="每个核元素放大后的像素边长。",
            ),
        ),
    )

    def process(self, image=None, shape: str = "ellipse", ksize: int = 5, cell: int = 24, **_: Any):
        array = _cv.source(image)
        kernel = _kernel(shape, ksize)
        size = int(ksize)
        step = int(cell)
        side = max(size * step, 1)

        height, width = array.shape[:2]
        canvas_height = max(side + 16, 140)
        panel_width = 220
        canvas = np.zeros((canvas_height, side + panel_width, 3), dtype=np.uint8)
        for row in range(size):
            for col in range(size):
                if kernel[row, col] <= 0:
                    continue
                y0 = row * step + 2
                x0 = col * step + 2
                canvas[y0 : y0 + step - 4, x0 : x0 + step - 4] = (230, 230, 230)
        cv2.rectangle(canvas, (0, 0), (side - 1, canvas_height - 1), (110, 110, 110), 1)

        lines = [
            f"Kernel: {shape}",
            f"Size: {size} x {size}",
            f"Active: {int(kernel.sum())}",
            f"Input: {width} x {height}",
        ]
        canvas[:, side : side + panel_width] = _cv.text_panel(panel_width, canvas_height, lines)
        return canvas


OPERATORS = (
    MorphGradient,
    MorphTopHat,
    MorphBlackHat,
    MorphHitMiss,
    MorphGeneral,
    MorphKernel,
)
