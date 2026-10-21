"""画线标注算子（交互式线段参数）。"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from app.operators.builtin._utils import as_array, drawing_color, line_type_for, native_array
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec


class DrawLine(Operator):
    """在图像上叠加一条线段。"""

    spec = OperatorSpec(
        id="interactive.draw_line",
        name="画线标注",
        category=("交互处理",),
        description=(
            "在预览中拖拽一条线段，并在图像副本上绘制，用于长度标注或辅助线。"
            "线段以 [x1, y1, x2, y2] 表示，支持灰度/BGR/BGRA 与 uint8/uint16/float32。"
        ),
        keywords=("标注", "画线", "line", "draw", "线段"),
        params=(
            ParamSpec(
                "line",
                "线段",
                "line",
                default=None,
                placeholder="[x1, y1, x2, y2]",
                tooltip="在右侧预览中点击“画线”后拖拽确定起点与终点。",
            ),
            ParamSpec(
                "color",
                "线条颜色",
                "color",
                default="#ff3b30",
                tooltip="BGRA 图像会以不透明颜色绘制；灰度图使用亮度换算。",
            ),
            ParamSpec(
                "thickness",
                "线宽",
                "int",
                default=2,
                min=1,
                max=51,
                step=1,
                unit="px",
            ),
        ),
    )

    def process(
        self,
        image=None,
        line=None,
        color: str = "#ff3b30",
        thickness: int = 2,
        **_: Any,
    ):
        array = as_array(image)
        if not line:
            raise OperatorError("请先在预览中拖拽绘制一条线段。")
        if len(line) != 4:
            raise OperatorError("线段需要 4 个数值：[x1, y1, x2, y2]。")
        try:
            x1, y1, x2, y2 = (float(value) for value in line)
        except (TypeError, ValueError) as exc:
            raise OperatorError("线段的端点坐标必须是数字。") from exc
        if math.hypot(x2 - x1, y2 - y1) < 1e-6:
            raise OperatorError("线段的两个端点重合，请重新绘制。")
        work = native_array(array, "输入图像")
        if work.ndim == 3 and work.shape[2] == 4:
            # 透明通道往往承载业务含义（例如 alpha 蒙版），标注时保持原样。
            color_target = work[:, :, :3]
            canvas = color_target.copy()
        else:
            color_target = work
            canvas = work.copy()
        cv2.line(
            canvas,
            (int(round(x1)), int(round(y1))),
            (int(round(x2)), int(round(y2))),
            drawing_color(color_target, color),
            int(thickness),
            lineType=line_type_for(canvas),
        )
        if work.ndim == 3 and work.shape[2] == 4:
            return np.dstack([canvas, work[:, :, 3]])
        return canvas


OPERATOR = DrawLine
