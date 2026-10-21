"""图像统计算子（均值 / 标准差 / 最小值 / 最大值）。"""

from __future__ import annotations

from typing import Any

import numpy as np

from app.core.data_types import DataType, NumberValue
from app.operators.builtin._utils import as_array, gray_view
from app.operators.spec import Operator, OperatorSpec, PortSpec


class ImageStats(Operator):
    """统计整幅图像的灰度像素值。"""

    spec = OperatorSpec(
        id="analyze.image_stats",
        name="图像统计",
        category=("特征与分析", "度量"),
        description=(
            "先把输入图像统一为灰度图，再统计全部像素的均值、标准差、最小值和最大值。"
            "统计对象是灰度化后的原始像素值，不做 0-255 归一化：uint16 输入按 0-65535 统计，"
            "float 输入按自身数值统计，因此数值尺度与输入 dtype 一致。"
        ),
        keywords=("统计", "均值", "标准差", "最小值", "最大值", "stats", "mean", "std"),
        inputs=(
            PortSpec(
                "image",
                "输入图像",
                accepts=("L", "BGR", "BGRA"),
                description="灰度、BGR 或 BGRA 图像；彩色图像会先转为灰度。",
            ),
        ),
        outputs=(
            PortSpec("mean", "均值", data_type=DataType.NUMBER),
            PortSpec("std", "标准差", data_type=DataType.NUMBER),
            PortSpec("min", "最小值", data_type=DataType.NUMBER),
            PortSpec("max", "最大值", data_type=DataType.NUMBER),
        ),
    )

    def process(self, image=None, **_: Any):
        array = as_array(image, "输入图像")
        gray = gray_view(array, "输入图像")
        pixels = gray.astype(np.float64, copy=False)
        return {
            "mean": NumberValue(value=float(pixels.mean()), label="均值"),
            "std": NumberValue(value=float(pixels.std()), label="标准差"),
            "min": NumberValue(value=float(pixels.min()), label="最小值"),
            "max": NumberValue(value=float(pixels.max()), label="最大值"),
        }


OPERATOR = ImageStats
