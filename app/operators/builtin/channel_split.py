"""通道拆分算子（B/G/R 三路输出）。"""

from __future__ import annotations

from typing import Any

from app.operators.builtin._utils import as_array, split_channels
from app.operators.spec import Operator, OperatorSpec, PortSpec


class ChannelSplit(Operator):
    """把 BGR/BGRA 图像拆分为三路灰度通道。"""

    spec = OperatorSpec(
        id="channel.split",
        name="通道拆分",
        category=("通道与色彩",),
        description=(
            "把彩色图像拆分为蓝、绿、红三路单通道输出；BGRA 输入的透明通道会被忽略。"
            "每路输出都可以单独连接到后续算子。"
        ),
        keywords=("通道", "channel", "拆分", "split", "bgr"),
        inputs=(
            PortSpec(
                "image",
                "彩色图像",
                accepts=("BGR", "BGRA"),
                description="需要 BGR 或 BGRA 图像，灰度图无法拆分。",
            ),
        ),
        outputs=(
            PortSpec("b", "蓝色通道 (B)", accepts=("L",)),
            PortSpec("g", "绿色通道 (G)", accepts=("L",)),
            PortSpec("r", "红色通道 (R)", accepts=("L",)),
        ),
    )

    def process(self, image=None, **_: Any):
        array = as_array(image, "彩色图像")
        blue, green, red = split_channels(array, "彩色图像")
        return {"b": blue, "g": green, "r": red}


OPERATOR = ChannelSplit
