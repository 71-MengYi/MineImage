"""测试用慢速算子：用于验证取消与过期结果丢弃。"""

import time

from app.operators.spec import Operator, OperatorSpec, ParamSpec


class Slow(Operator):
    spec = OperatorSpec(
        id="test.slow",
        name="慢速测试算子",
        category=("测试",),
        description="用于验证取消与过期结果丢弃。",
        params=(ParamSpec("delay", "延时", "float", default=0.2, min=0.0, max=5.0),),
    )

    def process(self, image=None, delay=0.2, **_):
        time.sleep(float(delay))
        return image.array.copy()


OPERATOR = Slow
