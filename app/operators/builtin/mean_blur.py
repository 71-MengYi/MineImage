"""均值模糊算子。"""

from __future__ import annotations

from typing import Any

import cv2

from app.operators.builtin._utils import as_array, border_mode, native_array
from app.operators.spec import Operator, OperatorSpec, ParamSpec


class MeanBlur(Operator):
    """均值（盒式）模糊。"""

    spec = OperatorSpec(
        id="filter.mean_blur",
        name="均值模糊",
        category=("滤波与平滑",),
        description="用方框核求局部平均，计算简单、速度快；支持 uint8/uint16/float32。",
        keywords=("均值", "mean", "box", "模糊", "blur", "平滑"),
        params=(
            ParamSpec(
                "ksize",
                "核大小",
                "int",
                default=5,
                min=1,
                max=99,
                step=2,
                odd=True,
                unit="px",
                tooltip="必须为奇数，越大越模糊。",
            ),
            ParamSpec(
                "border",
                "边界模式",
                "enum",
                choices=(
                    ("replicate", "复制边缘"),
                    ("reflect", "镜像"),
                    ("reflect101", "镜像（不含边缘）"),
                    ("constant", "常量填充（黑）"),
                ),
                default="replicate",
            ),
        ),
    )

    def process(self, image=None, ksize: int = 5, border: str = "replicate", **_: Any):
        array = native_array(as_array(image), "输入图像")
        return cv2.blur(array, (int(ksize), int(ksize)), borderType=border_mode(border))


OPERATOR = MeanBlur
