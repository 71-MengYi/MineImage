"""算子模板：复制本文件并去掉文件名开头的下划线即可被发现。

模块级 ``OPERATOR`` 必须指向算子类（或实例）。参数规范会自动生成界面控件，
框架会负责类型/范围/奇数等校验；process() 在后台线程执行，禁止弹窗和阻塞。

端口默认是图像（``data_type=DataType.IMAGE``）；数值 / 边缘特征 / 轮廓特征 /
连通域端口需要显式声明 ``data_type``。多输出节点在画布底部按声明顺序排列端口，
返回值建议用 ``{端口key: 值}`` 字典，引擎会按端口类型规范化。
"""

from __future__ import annotations

import cv2

from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec

# ---------------------------------------------------------------------------
# 多输出 / 类型化输出写法示意（需要时把对应 PortSpec 写进 OperatorSpec.outputs）
# ---------------------------------------------------------------------------
# from app.core.data_types import DataType, EdgesData, NumberValue
#
#     outputs=(
#         PortSpec("image", "边缘图", accepts=("L",)),
#         PortSpec("edges", "边缘线段", data_type=DataType.EDGES),
#         PortSpec("count", "线段数", data_type=DataType.NUMBER),
#     )
#
#     def process(self, image=None, **_):
#         ...
#         return {
#             "image": edge_map,                                   # ndarray → ImageData
#             "edges": EdgesData(segments, image_size=(width, height)),
#             "count": NumberValue(len(segments), label="线段数"),  # 也可直接返回 int
#         }
#
# 返回值会按输出端口类型规范化：裸 int/float → NumberValue；(N,2) 点集 →
# EdgesData / ContoursData；二维 int32 标签图 → ComponentsData（0 为背景）。
# 非图像端口不参与颜色空间校验，写在其上的 accepts 会被忽略。
# ---------------------------------------------------------------------------


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
