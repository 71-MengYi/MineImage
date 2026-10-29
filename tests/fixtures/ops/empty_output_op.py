"""测试用算子：验证 ``OperatorSpec.allow_empty_output`` 的空输出门禁。

* ``test.empty_output``：默认不允许空输出，执行时应报“没有产生任何输出”；
* ``test.empty_output_allowed``：显式允许空输出，执行成功但 outputs 为空。
"""

from __future__ import annotations

from typing import Any

from app.operators.spec import Operator, OperatorSpec, PortSpec


class EmptyOutput(Operator):
    spec = OperatorSpec(
        id="test.empty_output",
        name="空输出",
        category=("其他", "测试"),
        inputs=(),
        outputs=(PortSpec("data", "任意数据", any_type=True),),
    )

    def process(self, **_: Any) -> Any:
        return {}


class EmptyOutputAllowed(EmptyOutput):
    spec = OperatorSpec(
        id="test.empty_output_allowed",
        name="允许空输出",
        category=("其他", "测试"),
        inputs=(),
        outputs=(PortSpec("data", "任意数据", any_type=True),),
        allow_empty_output=True,
    )


OPERATORS = (EmptyOutput, EmptyOutputAllowed)
