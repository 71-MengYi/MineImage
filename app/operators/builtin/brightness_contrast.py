"""亮度 / 对比度算子。"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.operators.builtin._utils import as_array, cast_like, full_scale
from app.operators.spec import Operator, OperatorSpec, ParamSpec


class BrightnessContrast(Operator):
    """线性亮度 / 对比度调整。"""

    spec = OperatorSpec(
        id="basic.brightness_contrast",
        name="亮度/对比度",
        category=("基础调整",),
        description="以指定基准点做线性亮度与对比度调整，保持输入 dtype 与通道数。",
        keywords=("亮度", "对比度", "brightness", "contrast", "exposure"),
        params=(
            ParamSpec(
                "brightness",
                "亮度",
                "int",
                default=0,
                min=-100,
                max=100,
                step=1,
                unit="级",
                tooltip="向基准点整体加减亮度，正数变亮。",
            ),
            ParamSpec(
                "contrast",
                "对比度",
                "float",
                default=1.0,
                min=0.1,
                max=3.0,
                step=0.05,
                tooltip="1.0 表示不变，大于 1 增强对比度，小于 1 降低对比度。",
            ),
            ParamSpec(
                "pivot",
                "基准点",
                "enum",
                choices=(
                    ("mid", "中灰（满量程一半）"),
                    ("mean", "图像均值"),
                    ("zero", "零点"),
                ),
                default="mid",
                tooltip="对比度围绕该数值拉伸。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        brightness: int = 0,
        contrast: float = 1.0,
        pivot: str = "mid",
        **_: Any,
    ):
        array = as_array(image)
        if float(brightness) == 0.0 and float(contrast) == 1.0:
            # 默认参数保持逐位不变（float 图像避免无意义的舍入误差）。
            return array.copy()
        work = array.astype(np.float32)
        scale = full_scale(array)
        if pivot == "mean":
            center = float(work.mean())
        elif pivot == "zero":
            center = 0.0
        else:
            center = scale / 2.0
        offset = scale * float(brightness) / 255.0
        result = (work - center) * float(contrast) + center + offset
        if array.dtype == np.float32:
            return result.astype(np.float32)
        return cast_like(result, array.dtype)


OPERATOR = BrightnessContrast
