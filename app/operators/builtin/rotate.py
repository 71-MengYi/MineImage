"""旋转算子。"""

from __future__ import annotations

from typing import Any

import cv2

from app.operators.builtin._utils import as_array, border_mode, interpolation_flag, native_array
from app.operators.spec import Operator, OperatorSpec, ParamSpec


class Rotate(Operator):
    """绕图像中心旋转。"""

    spec = OperatorSpec(
        id="geom.rotate",
        name="旋转",
        category=("几何变换",),
        description=(
            "绕图像中心旋转任意角度。开启“保持完整画面”时会自动扩大画布，"
            "避免内容被裁掉；边界可用常量、复制或镜像填充。"
        ),
        keywords=("旋转", "rotate", "角度", "angle"),
        params=(
            ParamSpec(
                "angle",
                "旋转角度",
                "float",
                default=0.0,
                min=-360.0,
                max=360.0,
                step=1.0,
                unit="°",
                tooltip="逆时针为正方向。",
            ),
            ParamSpec(
                "expand",
                "保持完整画面",
                "bool",
                default=True,
                tooltip="开启后自动扩大输出画布，内容不会被裁切。",
            ),
            ParamSpec(
                "interpolation",
                "插值方式",
                "enum",
                choices=(
                    ("nearest", "最近邻"),
                    ("linear", "双线性"),
                    ("cubic", "双三次"),
                ),
                default="linear",
            ),
            ParamSpec(
                "border",
                "边界填充",
                "enum",
                choices=(
                    ("constant", "常量（黑）"),
                    ("replicate", "复制边缘"),
                    ("reflect", "镜像"),
                ),
                default="constant",
            ),
        ),
    )

    def process(
        self,
        image=None,
        angle: float = 0.0,
        expand: bool = True,
        interpolation: str = "linear",
        border: str = "constant",
        **_: Any,
    ):
        array = native_array(as_array(image), "输入图像")
        height, width = array.shape[:2]
        center = ((width - 1) / 2.0, (height - 1) / 2.0)
        matrix = cv2.getRotationMatrix2D(center, float(angle), 1.0)
        target_w, target_h = width, height
        if expand and float(angle) % 360.0 != 0.0:
            cos = abs(matrix[0, 0])
            sin = abs(matrix[0, 1])
            target_w = max(1, int(round(height * sin + width * cos)))
            target_h = max(1, int(round(height * cos + width * sin)))
            # 以像素中心 (n-1)/2 对齐，避免 180° 等整数倍旋转产生半像素偏移。
            matrix[0, 2] += (target_w - 1) / 2.0 - center[0]
            matrix[1, 2] += (target_h - 1) / 2.0 - center[1]
        if float(angle) % 360.0 == 0.0 and (target_w, target_h) == (width, height):
            return array.copy()
        return cv2.warpAffine(
            array,
            matrix,
            (target_w, target_h),
            flags=interpolation_flag(interpolation),
            borderMode=border_mode(border),
            borderValue=0,
        )


OPERATOR = Rotate
