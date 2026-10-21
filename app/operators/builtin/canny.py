"""Canny 边缘检测算子。"""

from __future__ import annotations

from typing import Any

import cv2

from app.operators.builtin._utils import as_array, gray_for_8bit
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec


class Canny(Operator):
    """Canny 边缘检测。"""

    spec = OperatorSpec(
        id="edge.canny",
        name="Canny 边缘",
        category=("边缘检测",),
        description=(
            "先统一为 8 位灰度图，再用 Canny 双阈值检测边缘，输出 0/255 边缘图。"
            "uint16/float 输入会先做满量程归一化。"
        ),
        keywords=("canny", "边缘", "edge", "双阈值"),
        params=(
            ParamSpec(
                "low",
                "低阈值",
                "int",
                default=100,
                min=0,
                max=255,
                step=1,
                tooltip="低于低阈值的梯度会被丢弃。",
            ),
            ParamSpec(
                "high",
                "高阈值",
                "int",
                default=200,
                min=0,
                max=255,
                step=1,
                tooltip="高于高阈值的梯度直接视为边缘，必须不小于低阈值。",
            ),
            ParamSpec(
                "aperture",
                "Sobel 孔径",
                "enum",
                choices=((3, "3×3"), (5, "5×5"), (7, "7×7")),
                default=3,
            ),
            ParamSpec(
                "l2",
                "L2 梯度",
                "bool",
                default=False,
                tooltip="开启后使用更精确的 L2 范数计算梯度幅值。",
            ),
        ),
    )

    def validate(self, inputs, params) -> None:
        if int(params["high"]) < int(params["low"]):
            raise OperatorError("Canny 的高阈值必须大于或等于低阈值。")

    def process(
        self,
        image=None,
        low: int = 100,
        high: int = 200,
        aperture: int = 3,
        l2: bool = False,
        **_: Any,
    ):
        if int(high) < int(low):
            raise OperatorError("Canny 的高阈值必须大于或等于低阈值。")
        gray = gray_for_8bit(as_array(image), "输入图像")
        return cv2.Canny(
            gray,
            int(low),
            int(high),
            apertureSize=int(aperture),
            L2gradient=bool(l2),
        )


OPERATOR = Canny
