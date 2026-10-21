"""查找表（LUT）算子。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.operators.builtin._utils import as_array, cast_like, full_scale
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec


class LookupTable(Operator):
    """通过 256 级查找表做色调映射。"""

    spec = OperatorSpec(
        id="basic.lut",
        name="查找表（LUT）",
        category=("基础调整",),
        description=(
            "用黑场/白场和伽马生成 256 级查找表，再映射整幅图像。"
            "uint8 走 OpenCV LUT，其它 dtype 按满量程线性插值，保持原 dtype。"
        ),
        keywords=("lut", "查找表", "gamma", "伽马", "映射"),
        params=(
            ParamSpec(
                "mode",
                "映射方式",
                "enum",
                choices=(("linear", "线性拉伸"), ("gamma", "伽马校正")),
                default="linear",
            ),
            ParamSpec(
                "black",
                "黑场",
                "int",
                default=0,
                min=0,
                max=254,
                step=1,
                tooltip="小于等于该输入值的像素映射为黑。",
            ),
            ParamSpec(
                "white",
                "白场",
                "int",
                default=255,
                min=1,
                max=255,
                step=1,
                tooltip="大于等于该输入值的像素映射为白。",
            ),
            ParamSpec(
                "gamma",
                "伽马",
                "float",
                default=1.0,
                min=0.1,
                max=5.0,
                step=0.05,
                visible_when="params['mode'] == 'gamma'",
                enabled_when="params['mode'] == 'gamma'",
                tooltip="大于 1 提亮中间调，小于 1 压暗中间调。",
            ),
            ParamSpec(
                "invert",
                "反相输出",
                "bool",
                default=False,
            ),
        ),
    )

    def validate(self, inputs, params) -> None:
        if int(params["white"]) <= int(params["black"]):
            raise OperatorError("白场必须大于黑场，请调小黑场或调大白场。")

    def process(
        self,
        image=None,
        mode: str = "linear",
        black: int = 0,
        white: int = 255,
        gamma: float = 1.0,
        invert: bool = False,
        **_: Any,
    ):
        array = as_array(image)
        if white <= black:
            raise OperatorError("白场必须大于黑场，请调小黑场或调大白场。")
        x = np.arange(256, dtype=np.float32)
        normalized = np.clip((x - float(black)) / float(max(int(white) - int(black), 1)), 0.0, 1.0)
        if mode == "gamma":
            normalized = np.power(normalized, 1.0 / float(max(gamma, 1e-6)))
        table = np.clip(np.rint(normalized * 255.0), 0, 255).astype(np.uint8)
        if invert:
            table = 255 - table
        if array.dtype == np.uint8:
            return cv2.LUT(array, table)
        scale = full_scale(array)
        values = table.astype(np.float32) / 255.0 * scale
        domain = np.linspace(0.0, scale, 256, dtype=np.float32)
        work = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=scale, neginf=0.0)
        mapped = np.interp(work.reshape(-1), domain, values).reshape(array.shape)
        if array.dtype.kind == "f":
            return mapped.astype(array.dtype)
        return cast_like(mapped, array.dtype)


OPERATOR = LookupTable
