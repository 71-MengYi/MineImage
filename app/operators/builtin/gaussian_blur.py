"""高斯模糊算子。"""

from __future__ import annotations

from typing import Any

import cv2

from app.operators.builtin._utils import as_array, border_mode, native_array
from app.operators.spec import Operator, OperatorSpec, ParamSpec


class GaussianBlur(Operator):
    """高斯模糊。"""

    spec = OperatorSpec(
        id="filter.gaussian_blur",
        name="高斯模糊",
        category=("滤波与平滑",),
        description="使用高斯核对图像做平滑，适合去噪与柔化边缘；支持 uint8/uint16/float32。",
        keywords=("高斯", "gaussian", "模糊", "blur", "平滑"),
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
                "sigma",
                "标准差 σ",
                "float",
                default=0.0,
                min=0.0,
                max=50.0,
                step=0.1,
                tooltip="0 表示根据核大小自动计算。",
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

    def process(self, image=None, ksize: int = 5, sigma: float = 0.0, border: str = "replicate", **_: Any):
        array = native_array(as_array(image), "输入图像")
        result = cv2.GaussianBlur(
            array,
            (int(ksize), int(ksize)),
            float(sigma),
            borderType=border_mode(border),
        )
        return result


OPERATOR = GaussianBlur
