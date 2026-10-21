"""翻转算子。"""

from __future__ import annotations

from typing import Any

import cv2

from app.operators.builtin._utils import as_array, native_array
from app.operators.spec import Operator, OperatorSpec, ParamSpec

_FLIP_CODES = {"horizontal": 1, "vertical": 0, "both": -1}


class Flip(Operator):
    """水平 / 垂直翻转。"""

    spec = OperatorSpec(
        id="geom.flip",
        name="翻转",
        category=("几何变换",),
        description="沿水平、垂直或同时两个方向翻转图像，支持灰度/BGR/BGRA。",
        keywords=("翻转", "flip", "镜像", "mirror"),
        params=(
            ParamSpec(
                "axis",
                "翻转方向",
                "enum",
                choices=(
                    ("horizontal", "水平翻转"),
                    ("vertical", "垂直翻转"),
                    ("both", "水平 + 垂直"),
                ),
                default="horizontal",
            ),
        ),
    )

    def process(self, image=None, axis: str = "horizontal", **_: Any):
        array = native_array(as_array(image), "输入图像")
        return cv2.flip(array, _FLIP_CODES[str(axis)])


OPERATOR = Flip
