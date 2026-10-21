"""边缘特征算子：Canny 边缘图 + 概率霍夫线段。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.core.data_types import DataType, EdgesData, NumberValue
from app.operators.builtin._utils import as_array, gray_for_8bit
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec


class FeatureEdges(Operator):
    """检测 0/255 边缘图，并用概率霍夫变换提取线段。"""

    spec = OperatorSpec(
        id="feature.edges",
        name="边缘特征",
        category=("特征与分析", "边缘"),
        description=(
            "先把输入统一为 8 位灰度图，再用 Canny 双阈值检测边缘并输出 0/255 边缘图；"
            "随后在边缘图上执行概率霍夫变换（HoughLinesP）提取线段，"
            "同时输出线段集合与线段数量。uint16/float 输入会先做满量程归一化。"
        ),
        keywords=("边缘", "edges", "线段", "霍夫", "hough", "canny", "特征"),
        inputs=(
            PortSpec(
                "image",
                "输入图像",
                accepts=("L", "BGR", "BGRA"),
                description="灰度、BGR 或 BGRA 图像；彩色图像会先转为灰度。",
            ),
        ),
        outputs=(
            PortSpec("edges", "边缘图", accepts=("L",)),
            PortSpec("lines", "边缘线段", data_type=DataType.EDGES),
            PortSpec("count", "线段数", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "low",
                "低阈值",
                "int",
                default=100,
                min=0,
                max=255,
                step=1,
                tooltip="Canny 低阈值，低于该值的梯度会被丢弃。",
            ),
            ParamSpec(
                "high",
                "高阈值",
                "int",
                default=200,
                min=0,
                max=255,
                step=1,
                tooltip="Canny 高阈值，必须大于或等于低阈值。",
            ),
            ParamSpec(
                "threshold",
                "霍夫阈值",
                "int",
                default=60,
                min=1,
                max=500,
                step=1,
                tooltip="霍夫累加器阈值，越大检出的线段越少。",
            ),
            ParamSpec(
                "min_length",
                "最小线段长度",
                "int",
                default=20,
                min=1,
                max=500,
                step=1,
            ),
            ParamSpec(
                "max_gap",
                "最大间隙",
                "int",
                default=3,
                min=0,
                max=100,
                step=1,
                tooltip="共线线段之间允许的最大断裂间隙。",
            ),
        ),
    )

    def validate(self, inputs, params) -> None:
        if int(params["high"]) < int(params["low"]):
            raise OperatorError("边缘特征的高阈值必须大于或等于低阈值。")

    def process(
        self,
        image=None,
        low: int = 100,
        high: int = 200,
        threshold: int = 60,
        min_length: int = 20,
        max_gap: int = 3,
        **_: Any,
    ):
        if int(high) < int(low):
            raise OperatorError("边缘特征的高阈值必须大于或等于低阈值。")

        gray = gray_for_8bit(as_array(image, "输入图像"), "输入图像")
        edges = cv2.Canny(gray, int(low), int(high))
        lines = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=int(threshold),
            minLineLength=int(min_length),
            maxLineGap=int(max_gap),
        )

        segments: list[np.ndarray] = []
        if lines is not None:
            for line in lines:
                segments.append(np.asarray(line[0], dtype=np.int32).reshape(2, 2))

        height, width = gray.shape[:2]
        return {
            "edges": edges,
            "lines": EdgesData(segments=tuple(segments), image_size=(width, height)),
            "count": NumberValue(value=len(segments), label="线段数", unit="条"),
        }


OPERATOR = FeatureEdges
