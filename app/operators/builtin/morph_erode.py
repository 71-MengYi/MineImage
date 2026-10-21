"""腐蚀算子。"""

from __future__ import annotations

from typing import Any

import cv2

from app.operators.builtin._utils import as_array, native_array, struct_kernel
from app.operators.spec import Operator, OperatorSpec, ParamSpec


class Erode(Operator):
    """形态学腐蚀。"""

    spec = OperatorSpec(
        id="morph.erode",
        name="腐蚀",
        category=("形态学",),
        description="用结构元素取邻域最小值，使亮区收缩、暗区扩张，可去除小亮噪点。",
        keywords=("腐蚀", "erode", "形态学", "morphology"),
        params=(
            ParamSpec(
                "shape",
                "结构元素",
                "enum",
                choices=(("rect", "矩形"), ("ellipse", "椭圆"), ("cross", "十字")),
                default="rect",
            ),
            ParamSpec(
                "ksize",
                "核大小",
                "int",
                default=3,
                min=1,
                max=99,
                step=2,
                odd=True,
                unit="px",
            ),
            ParamSpec(
                "iterations",
                "迭代次数",
                "int",
                default=1,
                min=1,
                max=20,
                step=1,
                tooltip="重复腐蚀的次数，越多效果越强。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        shape: str = "rect",
        ksize: int = 3,
        iterations: int = 1,
        **_: Any,
    ):
        array = native_array(as_array(image), "输入图像")
        kernel = struct_kernel(shape, int(ksize))
        return cv2.erode(array, kernel, iterations=int(iterations))


OPERATOR = Erode
