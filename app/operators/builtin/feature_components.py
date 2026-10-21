"""连通域分析算子。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.core.data_types import ComponentsData, DataType, NumberValue
from app.operators.builtin._utils import as_array, gray_for_8bit
from app.operators.spec import Operator, OperatorSpec, ParamSpec, PortSpec

_MODES = {
    "binary": cv2.THRESH_BINARY,
    "binary_inv": cv2.THRESH_BINARY_INV,
}


class FeatureComponents(Operator):
    """标记连通域，输出彩色标签图、连通域统计与数量。"""

    spec = OperatorSpec(
        id="feature.components",
        name="连通域分析",
        category=("特征与分析", "连通域"),
        description=(
            "先把输入统一为 8 位灰度图并二值化，再用 connectedComponentsWithStats 标记连通域；"
            "面积小于最小面积的连通域会被删除，并通过重映射得到从 1 开始、背景为 0 的新标签图。"
            "输出彩色标签图、连通域数据（标签图 / 面积 / 质心）、数量以及最大面积。"
        ),
        keywords=(
            "连通域",
            "components",
            "connected",
            "标记",
            "label",
            "面积",
            "特征",
        ),
        inputs=(
            PortSpec(
                "image",
                "输入图像",
                accepts=("L", "BGR", "BGRA"),
                description="灰度、BGR 或 BGRA 图像；彩色图像会先转为灰度。",
            ),
        ),
        outputs=(
            PortSpec("labels", "标签图", accepts=("BGR",)),
            PortSpec("components", "连通域", data_type=DataType.COMPONENTS),
            PortSpec("count", "数量", data_type=DataType.NUMBER),
            PortSpec("largest_area", "最大面积", data_type=DataType.NUMBER),
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
                tooltip="面积小于该值的连通域会被删除。",
            ),
            ParamSpec(
                "connectivity",
                "连通方式",
                "enum",
                choices=((4, "4 连通"), (8, "8 连通")),
                default=8,
                tooltip="4 连通只看上下左右，8 连通还包含对角方向。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        threshold: int = 128,
        mode: str = "binary",
        min_area: int = 10,
        connectivity: int = 8,
        **_: Any,
    ):
        gray = gray_for_8bit(as_array(image, "输入图像"), "输入图像")
        flag = _MODES[mode]
        _, binary = cv2.threshold(gray, int(threshold), 255, flag)

        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=int(connectivity)
        )

        filtered = np.zeros_like(labels, dtype=np.int32)
        next_label = 1
        for index in range(1, num_labels):
            if int(stats[index, cv2.CC_STAT_AREA]) >= int(min_area):
                filtered[labels == index] = next_label
                next_label += 1

        components = ComponentsData.from_labels(filtered)
        colorized = components.render_preview().array
        return {
            "labels": colorized,
            "components": components,
            "count": NumberValue(
                value=components.count, label="连通域数量", unit="个"
            ),
            "largest_area": NumberValue(
                value=components.largest_area, label="最大面积", unit="像素"
            ),
        }


OPERATOR = FeatureComponents
