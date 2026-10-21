"""轮廓特征算子：二值化后提取外轮廓。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.core.data_types import ContoursData, DataType, NumberValue
from app.operators.builtin._utils import as_array, gray_for_8bit
from app.operators.spec import Operator, OperatorSpec, ParamSpec, PortSpec

_MODES = {
    "binary": cv2.THRESH_BINARY,
    "binary_inv": cv2.THRESH_BINARY_INV,
}


class FeatureContours(Operator):
    """二值化并提取外轮廓，输出轮廓掩膜、轮廓集合与数量。"""

    spec = OperatorSpec(
        id="feature.contours",
        name="轮廓特征",
        category=("特征与分析", "轮廓"),
        description=(
            "先把输入统一为 8 位灰度图并二值化，再提取最外层轮廓（RETR_EXTERNAL，"
            "CHAIN_APPROX_SIMPLE）；面积小于最小面积的轮廓会被过滤。"
            "输出包括绘制了保留轮廓的二值掩膜图、轮廓点集以及轮廓数量。"
        ),
        keywords=("轮廓", "contours", "查找轮廓", "findcontours", "特征", "二值"),
        inputs=(
            PortSpec(
                "image",
                "输入图像",
                accepts=("L", "BGR", "BGRA"),
                description="灰度、BGR 或 BGRA 图像；彩色图像会先转为灰度。",
            ),
        ),
        outputs=(
            PortSpec("mask", "轮廓图", accepts=("L",)),
            PortSpec("contours", "轮廓", data_type=DataType.CONTOURS),
            PortSpec("count", "轮廓数", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "threshold",
                "阈值",
                "int",
                default=128,
                min=0,
                max=255,
                step=1,
                tooltip="0-255 通用刻度；图像会先统一到 8 位灰度再按该阈值二值化。",
            ),
            ParamSpec(
                "mode",
                "模式",
                "enum",
                choices=(("binary", "二值"), ("binary_inv", "反二值")),
                default="binary",
                tooltip="二值：亮于阈值为前景；反二值：暗于阈值为前景。",
            ),
            ParamSpec(
                "min_area",
                "最小面积",
                "int",
                default=10,
                min=0,
                max=100000,
                step=1,
                tooltip="轮廓面积小于该值时被过滤。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        threshold: int = 128,
        mode: str = "binary",
        min_area: int = 10,
        **_: Any,
    ):
        gray = gray_for_8bit(as_array(image, "输入图像"), "输入图像")
        flag = _MODES[mode]
        _, binary = cv2.threshold(gray, int(threshold), 255, flag)

        contours, _hierarchy = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        filtered = [
            contour for contour in contours if cv2.contourArea(contour) >= int(min_area)
        ]

        mask = np.zeros_like(binary)
        cv2.drawContours(mask, filtered, -1, 255, 1)

        height, width = gray.shape[:2]
        points = tuple(contour.reshape(-1, 2) for contour in filtered)
        return {
            "mask": mask,
            "contours": ContoursData(contours=points, image_size=(width, height)),
            "count": NumberValue(value=len(filtered), label="轮廓数", unit="个"),
        }


OPERATOR = FeatureContours
