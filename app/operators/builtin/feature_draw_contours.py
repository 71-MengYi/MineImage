"""轮廓绘制算子：把轮廓特征叠加到图像上。"""

from __future__ import annotations

from typing import Any

import cv2

from app.core.data_types import ContoursData, DataType
from app.operators.builtin._utils import as_array, drawing_color, to_uint8
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec


class FeatureDrawContours(Operator):
    """把轮廓数据绘制到图像上，用于验证非图像类型的输入连接。"""

    spec = OperatorSpec(
        id="feature.draw_contours",
        name="轮廓绘制",
        category=("特征与分析", "轮廓"),
        description=(
            "接收一幅图像作为底图，以及“轮廓特征”算子输出的轮廓数据，"
            "把每条轮廓按指定颜色和线宽绘制到底图副本上并输出 BGR 结果。"
            "该算子同时用于验证非图像类型端口的连线：轮廓输入必须是轮廓特征数据，"
            "直接连接数值等其他类型会报错。"
        ),
        keywords=("轮廓", "绘制", "drawcontours", "叠加", "标注", "特征"),
        inputs=(
            PortSpec(
                "image",
                "底图",
                accepts=("L", "BGR", "BGRA"),
                description="灰度、BGR 或 BGRA 图像；输出统一为 BGR。",
            ),
            PortSpec(
                "contours",
                "轮廓",
                data_type=DataType.CONTOURS,
                description="由轮廓特征算子产生的轮廓数据。",
            ),
        ),
        outputs=(PortSpec("image", "绘制结果", accepts=("BGR",)),),
        params=(
            ParamSpec(
                "color",
                "轮廓颜色",
                "color",
                default="#00ff00",
                tooltip="轮廓颜色，格式为 #RRGGBB。",
            ),
            ParamSpec(
                "thickness",
                "线宽",
                "int",
                default=2,
                min=1,
                max=20,
                step=1,
                tooltip="轮廓线条宽度（像素）。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        contours=None,
        color: str = "#00ff00",
        thickness: int = 2,
        **_: Any,
    ):
        if not isinstance(contours, ContoursData):
            actual = type(contours).__name__ if contours is not None else "无"
            raise OperatorError(
                f"输入“轮廓”需要轮廓特征数据，当前为{actual}；请连接轮廓特征算子的输出。"
            )

        array = as_array(image, "底图")
        work = to_uint8(array, "底图")
        if work.ndim == 2:
            canvas = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
        elif work.shape[2] == 4:
            canvas = work[:, :, :3].copy()
        else:
            canvas = work.copy()

        color_bgr = drawing_color(canvas, color)
        polylines = [contour.reshape(-1, 1, 2) for contour in contours.contours]
        if polylines:
            cv2.drawContours(canvas, polylines, -1, color_bgr, int(thickness))
        return canvas


OPERATOR = FeatureDrawContours
