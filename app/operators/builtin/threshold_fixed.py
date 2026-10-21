"""固定阈值（含 Otsu 自动阈值）算子。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.operators.builtin._utils import as_array, native_array, scale_value
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec

_MODES = {
    "binary": cv2.THRESH_BINARY,
    "binary_inv": cv2.THRESH_BINARY_INV,
    "trunc": cv2.THRESH_TRUNC,
    "tozero": cv2.THRESH_TOZERO,
    "tozero_inv": cv2.THRESH_TOZERO_INV,
}


class FixedThreshold(Operator):
    """固定阈值 / Otsu 自动阈值。"""

    spec = OperatorSpec(
        id="threshold.fixed",
        name="固定阈值",
        category=("阈值与分割",),
        description=(
            "按给定阈值把图像二值化或截断。阈值参数使用 0-255 的通用刻度，"
            "引擎会按图像 dtype 自动换算（uint16 为 ×257，float 按满量程）。"
            "勾选 Otsu 后由直方图自动确定阈值。"
        ),
        keywords=("阈值", "threshold", "二值", "binary", "otsu", "分割"),
        params=(
            ParamSpec(
                "value",
                "阈值",
                "int",
                default=128,
                min=0,
                max=255,
                step=1,
                visible_when="not params['otsu']",
                enabled_when="not params['otsu']",
                tooltip="0-255 通用刻度，按图像 dtype 自动换算。",
            ),
            ParamSpec(
                "max_value",
                "最大值",
                "int",
                default=255,
                min=0,
                max=255,
                step=1,
                tooltip="二值化时赋给前景的值，同样按 dtype 换算。",
            ),
            ParamSpec(
                "mode",
                "阈值类型",
                "enum",
                choices=(
                    ("binary", "二值"),
                    ("binary_inv", "反二值"),
                    ("trunc", "截断"),
                    ("tozero", "低于阈值置零"),
                    ("tozero_inv", "高于阈值置零"),
                ),
                default="binary",
            ),
            ParamSpec(
                "otsu",
                "Otsu 自动阈值",
                "bool",
                default=False,
                tooltip="仅支持 uint8/uint16 图像；开启后忽略手动阈值。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        value: int = 128,
        max_value: int = 255,
        mode: str = "binary",
        otsu: bool = False,
        **_: Any,
    ):
        array = as_array(image)
        flag = _MODES[mode]
        if otsu:
            if array.dtype not in (np.uint8, np.uint16):
                raise OperatorError(
                    f"Otsu 自动阈值仅支持 uint8/uint16 图像，当前为 {array.dtype}。"
                    "请先转换为 uint8/uint16，或关闭 Otsu 使用手动阈值。"
                )
            max_native = scale_value(array, max_value)
            _, result = cv2.threshold(array, 0, max_native, flag | cv2.THRESH_OTSU)
            return result
        work = native_array(array, "输入图像")
        _, result = cv2.threshold(
            work,
            scale_value(work, value),
            scale_value(work, max_value),
            flag,
        )
        return result


OPERATOR = FixedThreshold
