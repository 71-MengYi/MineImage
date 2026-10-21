"""Sobel 边缘检测算子。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.operators.builtin._utils import as_array, native_array
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec


class Sobel(Operator):
    """Sobel 一阶梯度。"""

    spec = OperatorSpec(
        id="edge.sobel",
        name="Sobel 边缘",
        category=("边缘检测",),
        description=(
            "计算图像在 X/Y 方向的一阶梯度（CV_32F）。dx、dy 可分别取 0-2，"
            "默认输出梯度绝对值；灰度/BGR/BGRA 均可直接处理。"
        ),
        keywords=("sobel", "边缘", "edge", "梯度", "gradient"),
        params=(
            ParamSpec(
                "dx",
                "X 方向阶数",
                "int",
                default=1,
                min=0,
                max=2,
                step=1,
                tooltip="0 表示不计算该方向；dx 与 dy 不能同时为 0。",
            ),
            ParamSpec(
                "dy",
                "Y 方向阶数",
                "int",
                default=0,
                min=0,
                max=2,
                step=1,
                tooltip="0 表示不计算该方向。",
            ),
            ParamSpec(
                "ksize",
                "核大小",
                "int",
                default=3,
                min=1,
                max=31,
                step=2,
                odd=True,
                unit="px",
            ),
            ParamSpec(
                "scale",
                "缩放系数",
                "float",
                default=1.0,
                min=0.1,
                max=10.0,
                step=0.1,
                tooltip="梯度值乘以该系数后再输出。",
            ),
            ParamSpec(
                "absolute",
                "取绝对值",
                "bool",
                default=True,
                tooltip="关闭时保留正负梯度（float32）。",
            ),
        ),
    )

    def validate(self, inputs, params) -> None:
        if int(params["dx"]) == 0 and int(params["dy"]) == 0:
            raise OperatorError("Sobel 的 X 方向阶数与 Y 方向阶数不能同时为 0。")

    def process(
        self,
        image=None,
        dx: int = 1,
        dy: int = 0,
        ksize: int = 3,
        scale: float = 1.0,
        absolute: bool = True,
        **_: Any,
    ):
        if int(dx) == 0 and int(dy) == 0:
            raise OperatorError("Sobel 的 X 方向阶数与 Y 方向阶数不能同时为 0。")
        array = native_array(as_array(image), "输入图像")
        result = cv2.Sobel(
            array,
            cv2.CV_32F,
            int(dx),
            int(dy),
            ksize=int(ksize),
            scale=float(scale),
        )
        if absolute:
            result = np.abs(result)
        return np.asarray(result, dtype=np.float32)


OPERATOR = Sobel
