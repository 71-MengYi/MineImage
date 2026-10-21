"""闭运算算子。"""

from __future__ import annotations

from typing import Any

import cv2

from app.operators.builtin._utils import as_array, native_array, struct_kernel
from app.operators.spec import Operator, OperatorSpec, ParamSpec


class Close(Operator):
    """形态学闭运算（先膨胀后腐蚀）。"""

    spec = OperatorSpec(
        id="morph.close",
        name="闭运算",
        category=("形态学",),
        description="先膨胀后腐蚀，可填充小暗孔、连接邻近亮区，同时基本保持大区域面积。",
        keywords=("闭运算", "close", "形态学", "morphology", "填孔"),
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
        return cv2.morphologyEx(array, cv2.MORPH_CLOSE, kernel, iterations=int(iterations))


OPERATOR = Close
