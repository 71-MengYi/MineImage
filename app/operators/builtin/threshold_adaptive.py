"""自适应阈值算子。"""

from __future__ import annotations

from typing import Any

import cv2

from app.operators.builtin._utils import as_array, gray_for_8bit
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec


class AdaptiveThreshold(Operator):
    """局部自适应阈值。"""

    spec = OperatorSpec(
        id="threshold.adaptive",
        name="自适应阈值",
        category=("阈值与分割",),
        description=(
            "按邻域均值或高斯加权均值计算局部阈值，适合光照不均的图像。"
            "OpenCV 仅支持 8 位单通道，灰度/uint16/float 输入会先转换为 uint8。"
        ),
        keywords=("自适应", "adaptive", "阈值", "threshold", "二值", "局部"),
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
                unit="px",
                tooltip="必须为奇数，越大越接近全局阈值。",
            ),
            ParamSpec(
                "constant",
                "常数 C",
                "float",
                default=2.0,
                min=-50.0,
                max=50.0,
                step=0.5,
                tooltip="从局部阈值中减去的常数，用于微调前景比例。",
            ),
            ParamSpec(
                "method",
                "计算方法",
                "enum",
                choices=(("mean", "邻域均值"), ("gaussian", "高斯加权")),
                default="gaussian",
            ),
            ParamSpec(
                "invert",
                "反相二值",
                "bool",
                default=False,
                tooltip="开启后前景与背景互换（THRESH_BINARY_INV）。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        block_size: int = 11,
        constant: float = 2.0,
        method: str = "gaussian",
        invert: bool = False,
        **_: Any,
    ):
        gray = gray_for_8bit(as_array(image), "输入图像")
        block = int(block_size)
        if block < 3 or block % 2 == 0:
            raise OperatorError("邻域大小必须是大于等于 3 的奇数。")
        method_flag = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if method == "gaussian"
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )
        threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        return cv2.adaptiveThreshold(
            gray,
            255,
            method_flag,
            threshold_type,
            block,
            float(constant),
        )


OPERATOR = AdaptiveThreshold
