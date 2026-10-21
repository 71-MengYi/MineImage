"""ROI 裁剪算子（交互式矩形参数）。"""

from __future__ import annotations

import math
from typing import Any

from app.operators.builtin._utils import as_array
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec


class RoiCrop(Operator):
    """按矩形 ROI 裁剪图像。"""

    spec = OperatorSpec(
        id="interactive.roi_crop",
        name="ROI 裁剪",
        category=("交互处理",),
        description=(
            "按预览中框选的矩形裁剪图像。ROI 以 [x, y, w, h] 表示；超出图像的部分会"
            "自动裁掉，完全位于图像外时提示重新框选。"
        ),
        keywords=("roi", "裁剪", "crop", "矩形", "区域"),
        params=(
            ParamSpec(
                "roi",
                "裁剪区域",
                "roi",
                default=None,
                placeholder="[x, y, w, h]",
                tooltip="在右侧预览中点击“框选”后拖拽出矩形区域。",
            ),
        ),
    )

    def process(self, image=None, roi=None, **_: Any):
        array = as_array(image)
        if not roi:
            raise OperatorError("请先在预览中框选一个 ROI 区域，再执行裁剪。")
        if len(roi) != 4:
            raise OperatorError("ROI 需要 4 个数值：[x, y, w, h]。")
        try:
            x, y, width, height = (float(value) for value in roi)
        except (TypeError, ValueError) as exc:
            raise OperatorError("ROI 的坐标和宽高必须是数字。") from exc
        if width <= 0 or height <= 0:
            raise OperatorError("ROI 的宽高必须大于 0，请重新框选。")
        image_h, image_w = array.shape[:2]
        x0 = max(0, int(math.floor(x)))
        y0 = max(0, int(math.floor(y)))
        x1 = min(image_w, int(math.ceil(x + width)))
        y1 = min(image_h, int(math.ceil(y + height)))
        if x1 <= x0 or y1 <= y0:
            raise OperatorError(
                f"ROI 区域完全位于图像之外（图像尺寸 {image_w}×{image_h}），请重新框选。"
            )
        # 返回副本，保证结果不与源图共享内存。
        return array[y0:y1, x0:x1].copy()


OPERATOR = RoiCrop
