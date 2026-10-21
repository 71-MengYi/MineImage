"""测试用合法用户算子。"""

import numpy as np

from app.operators.spec import Operator, OperatorSpec


class Double(Operator):
    spec = OperatorSpec(
        id="user.double",
        name="用户翻倍",
        category=("用户算子",),
        description="把像素值翻倍并饱和到 uint8。",
    )

    def process(self, image=None, **_):
        return np.clip(image.array.astype(np.int32) * 2, 0, 255).astype(np.uint8)


OPERATOR = Double
