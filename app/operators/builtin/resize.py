"""缩放算子。"""

from __future__ import annotations

from typing import Any

import cv2

from app.operators.builtin._utils import as_array, interpolation_flag, native_array
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec


class Resize(Operator):
    """按尺寸或比例缩放图像。"""

    spec = OperatorSpec(
        id="geom.resize",
        name="缩放",
        category=("几何变换",),
        description=(
            "按目标尺寸或比例缩放图像。按尺寸时可保持宽高比，把结果限制在目标框内；"
            "输出宽高至少为 1 像素。"
        ),
        keywords=("缩放", "resize", "尺寸", "scale", "放大", "缩小"),
        params=(
            ParamSpec(
                "mode",
                "缩放方式",
                "enum",
                choices=(("size", "按尺寸"), ("scale", "按比例")),
                default="size",
            ),
            ParamSpec(
                "width",
                "目标宽度",
                "int",
                default=640,
                min=1,
                max=16384,
                step=1,
                unit="px",
                visible_when="params['mode'] == 'size'",
                enabled_when="params['mode'] == 'size'",
            ),
            ParamSpec(
                "height",
                "目标高度",
                "int",
                default=480,
                min=1,
                max=16384,
                step=1,
                unit="px",
                visible_when="params['mode'] == 'size'",
                enabled_when="params['mode'] == 'size'",
            ),
            ParamSpec(
                "keep_aspect",
                "保持宽高比",
                "bool",
                default=True,
                visible_when="params['mode'] == 'size'",
                tooltip="开启时结果会完整放入目标框，并保持原始宽高比。",
            ),
            ParamSpec(
                "scale",
                "缩放比例",
                "float",
                default=0.5,
                min=0.05,
                max=8.0,
                step=0.05,
                visible_when="params['mode'] == 'scale'",
                enabled_when="params['mode'] == 'scale'",
                tooltip="0.5 表示缩小一半，2.0 表示放大两倍。",
            ),
            ParamSpec(
                "interpolation",
                "插值方式",
                "enum",
                choices=(
                    ("nearest", "最近邻"),
                    ("linear", "双线性"),
                    ("cubic", "双三次"),
                    ("area", "区域平均"),
                    ("lanczos", "Lanczos"),
                ),
                default="linear",
                tooltip="缩小时区域平均更平滑，放大时双线性/双三次更常用。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        mode: str = "size",
        width: int = 640,
        height: int = 480,
        keep_aspect: bool = True,
        scale: float = 0.5,
        interpolation: str = "linear",
        **_: Any,
    ):
        array = native_array(as_array(image), "输入图像")
        source_h, source_w = array.shape[:2]
        if mode == "scale":
            factor = float(scale)
            if factor <= 0:
                raise OperatorError("缩放比例必须大于 0。")
            target_w = max(1, int(round(source_w * factor)))
            target_h = max(1, int(round(source_h * factor)))
        else:
            target_w = int(width)
            target_h = int(height)
            if target_w < 1 or target_h < 1:
                raise OperatorError("目标宽度和高度必须大于等于 1 像素。")
            if keep_aspect:
                ratio = min(target_w / float(source_w), target_h / float(source_h))
                target_w = max(1, int(round(source_w * ratio)))
                target_h = max(1, int(round(source_h * ratio)))
        return cv2.resize(
            array,
            (target_w, target_h),
            interpolation=interpolation_flag(interpolation),
        )


OPERATOR = Resize
