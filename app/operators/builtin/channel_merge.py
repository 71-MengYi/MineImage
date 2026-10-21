"""通道合并算子（B/G/R 三路输入）。"""

from __future__ import annotations

from typing import Any

from app.operators.builtin._utils import as_array, merge_channels
from app.operators.spec import Operator, OperatorSpec, PortSpec


class ChannelMerge(Operator):
    """把三路单通道图像合成为 BGR 图像。"""

    spec = OperatorSpec(
        id="channel.merge",
        name="通道合并",
        category=("通道与色彩",),
        description=(
            "把蓝、绿、红三路单通道图像合成为 BGR 彩色图。三路尺寸必须一致；"
            "dtype 不同时会自动提升到公共类型（任一 float → float32，否则任一 uint16 → uint16）。"
        ),
        keywords=("通道", "channel", "合并", "merge", "bgr"),
        inputs=(
            PortSpec("b", "蓝色通道 (B)", accepts=("L",)),
            PortSpec("g", "绿色通道 (G)", accepts=("L",)),
            PortSpec("r", "红色通道 (R)", accepts=("L",)),
        ),
        outputs=(
            PortSpec("image", "合成图", accepts=("BGR",), description="按 B/G/R 顺序合成的 BGR 图像。"),
        ),
    )

    def process(self, b=None, g=None, r=None, **_: Any):
        blue = as_array(b, "蓝色通道")
        green = as_array(g, "绿色通道")
        red = as_array(r, "红色通道")
        return merge_channels(blue, green, red)


OPERATOR = ChannelMerge
