"""测试用异常算子：用于验证异常捕获与错误详情。"""

from app.operators.spec import Operator, OperatorSpec


class Boom(Operator):
    spec = OperatorSpec(
        id="test.boom",
        name="异常测试算子",
        category=("测试",),
        description="总是抛出异常。",
    )

    def process(self, image=None, **_):
        raise RuntimeError("boom-from-operator")


OPERATOR = Boom
