"""算子模板：复制本文件并去掉文件名开头的下划线即可被发现。

模块级 ``OPERATOR`` 必须指向算子类（或实例）。参数规范会自动生成界面控件，
框架会负责类型/范围/奇数等校验；process() 在后台线程执行，禁止弹窗和阻塞。
"""

from __future__ import annotations

import cv2

from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec


class MyOperator(Operator):
    spec = OperatorSpec(
        id="example.my_operator",              # 全局唯一，建议 分类.名称
        name="我的算子",
        category=("其他", "示例"),              # 一级分类 / 二级分类 / ...
        description="一句话说明这个算子做什么。",
        version="1.0.0",
        inputs=(
            PortSpec("image", "输入", accepts=("L", "BGR", "BGRA")),
        ),
        outputs=(PortSpec("image", "输出"),),
        params=(
            ParamSpec(
                "strength",
                "强度",
                "float",
                default=0.5,
                min=0.0,
                max=1.0,
                step=0.05,
                tooltip="0 表示不处理，1 表示最强。",
            ),
            ParamSpec(
                "channel",
                "通道",
                "enum",
                default="gray",
                choices=(("gray", "灰度"), ("color", "彩色")),
            ),
        ),
    )

    def process(self, image, strength=0.5, channel="gray", **_):
        array = image.array
        if channel == "gray" and array.ndim == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        if channel == "color" and array.ndim == 2:
            raise OperatorError("“彩色”模式需要彩色输入图像")
        # 不要原地修改输入数组
        result = cv2.addWeighted(array, 1.0 - strength, array, strength, 0)
        return result


OPERATOR = MyOperator
