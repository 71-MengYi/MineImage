"""反相算子。"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.operators.builtin._utils import as_array, full_scale
from app.operators.spec import Operator, OperatorSpec, ParamSpec


class Invert(Operator):
    """按 dtype 满量程反相。"""

    spec = OperatorSpec(
        id="basic.invert",
        name="反相",
        category=("基础调整",),
        description=(
            "按数据类型满量程取反：uint8 用 255-x，uint16 用 65535-x，"
            "float 在 [0,1] 区间用 1-x。"
        ),
        keywords=("反相", "invert", "negative", "补色"),
        params=(
            ParamSpec(
                "invert_alpha",
                "同时反相透明通道",
                "bool",
                default=False,
                tooltip="仅对 BGRA 图像有效；关闭时保留 alpha 不变。",
            ),
        ),
    )

    def process(self, image=None, invert_alpha: bool = False, **_: Any):
        array = as_array(image)

        def invert(part: np.ndarray) -> np.ndarray:
            if part.dtype.kind == "b":
                return ~part
            if part.dtype.kind == "f":
                return (full_scale(part) - part).astype(part.dtype)
            if part.dtype.kind == "u":
                return np.iinfo(part.dtype).max - part
            # 有符号整数：取反后由显示层按 min/max 拉伸。
            return -part

        if array.ndim == 3 and array.shape[2] == 4 and not invert_alpha:
            result = array.copy()
            result[:, :, :3] = invert(array[:, :, :3])
            return result
        return invert(array)


OPERATOR = Invert
