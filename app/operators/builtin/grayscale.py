"""灰度化算子。"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.operators.builtin._utils import as_array, cast_like, gray_view, native_array
from app.operators.spec import Operator, OperatorSpec, ParamSpec


class Grayscale(Operator):
    """彩色转灰度。"""

    spec = OperatorSpec(
        id="basic.grayscale",
        name="灰度化",
        category=("基础调整",),
        description=(
            "把彩色图像转换为单通道灰度图。加权平均使用 OpenCV 的 BT.601 系数，"
            "算术平均对 B/G/R 直接求均值，最大值取三通道中最亮的像素。"
        ),
        keywords=("灰度", "gray", "grayscale", "去色", "desaturate"),
        params=(
            ParamSpec(
                "mode",
                "转换方式",
                "enum",
                choices=(
                    ("weighted", "加权平均（BT.601）"),
                    ("average", "算术平均"),
                    ("max", "最大值"),
                ),
                default="weighted",
                tooltip="加权平均最接近人眼感受；算术平均更中性；最大值保留亮部。",
            ),
        ),
    )

    def process(self, image=None, mode: str = "weighted", **_: Any):
        array = as_array(image)
        if array.ndim == 2:
            return array
        work = native_array(array, "输入图像")
        if mode == "weighted":
            return gray_view(array)
        color = work[:, :, :3] if work.ndim == 3 and work.shape[2] == 4 else work
        if mode == "average":
            result = np.mean(color, axis=2, dtype=np.float32)
        else:
            result = np.max(color, axis=2)
        if array.dtype in (np.uint8, np.uint16, np.float32):
            return cast_like(result, array.dtype)
        return result.astype(np.float32)


OPERATOR = Grayscale
