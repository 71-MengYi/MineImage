"""测试用覆盖内置 id 的用户算子。"""

from app.operators.spec import Operator, OperatorSpec


class CustomGray(Operator):
    spec = OperatorSpec(
        id="basic.grayscale",
        name="自定义灰度化",
        category=("用户算子",),
        description="覆盖内置灰度化，仅取第一个通道。",
    )

    def process(self, image=None, **_):
        array = image.array
        return array[:, :, 0] if array.ndim == 3 else array


OPERATOR = CustomGray
