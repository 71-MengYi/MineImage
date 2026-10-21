"""中值模糊算子。"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.operators.builtin._utils import as_array, native_array
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec


class MedianBlur(Operator):
    """中值滤波，适合去除椒盐噪声。"""

    spec = OperatorSpec(
        id="filter.median_blur",
        name="中值模糊",
        category=("滤波与平滑",),
        description=(
            "用邻域中值替换中心像素，去除椒盐噪声的同时较好保留边缘。"
            "核大小大于 5 时 OpenCV 仅支持 uint8，其它 dtype 会给出提示。"
        ),
        keywords=("中值", "median", "去噪", "椒盐", "blur"),
        params=(
            ParamSpec(
                "ksize",
                "核大小",
                "int",
                default=5,
                min=3,
                max=31,
                step=2,
                odd=True,
                unit="px",
                tooltip="必须为奇数；大于 5 时非 uint8 图像请先转换数据类型。",
            ),
        ),
    )

    def process(self, image=None, ksize: int = 5, **_: Any):
        array = native_array(as_array(image), "输入图像")
        ksize = int(ksize)
        if ksize % 2 == 0 or ksize < 3:
            raise OperatorError("中值模糊的核大小必须是大于等于 3 的奇数。")
        if ksize > 5 and array.dtype != np.uint8:
            raise OperatorError(
                f"中值模糊在核大小大于 5 时仅支持 uint8 图像，当前为 {array.dtype}。"
                "请把核大小设为 3 或 5，或先在基础调整中转换为 uint8。"
            )
        try:
            return cv2.medianBlur(array, ksize)
        except cv2.error as exc:  # pragma: no cover - 预防平台差异
            raise OperatorError(
                f"中值模糊处理 {array.dtype} 图像失败，请尝试 uint8 图像或较小的核。"
            ) from exc


OPERATOR = MedianBlur
