"""输出节点：单一万能入口 / 出口，透传任意已知类型的数据。

节点位于「其他 → 可视化」，用于把任意一支链路的最终结果接到一个终点上：

* 输入端口声明为万能端口（`any_type=True`），图像、数值、边缘特征、轮廓特征、
  连通域都可以直接连入；
* 输出端口同样为万能端口，输入对象会**原样**透传给下游，不复制、不转换；
* 节点内不再生成任何汇总渲染图：图像显示缩略图（单击节点在右侧预览），单个数值
  直接显示在节点卡片内，边缘 / 轮廓 / 连通域等数组数据只在双击进入参数面板后的
  「输出数据」区展示；
* 参数面板的输出数据区可以查看数值、图像摘要，以及数组的预览与统计信息。
"""

from __future__ import annotations

from typing import Any

from app.operators.spec import Operator, OperatorSpec, PortSpec


class OutputDisplay(Operator):
    """输出预览：万能入口 / 出口，原样透传数据。"""

    spec = OperatorSpec(
        id="output.display",
        name="输出预览",
        category=("其他", "可视化"),
        description=(
            "一个万能入口接收图像、数值、边缘特征、轮廓特征或连通域，并从一个万能"
            "出口原样透传，可作为流程终点或链路检查点。图像在节点内显示缩略图并在"
            "预览窗口查看，单个数值直接显示在节点内，数组数据在参数面板中查看。"
        ),
        keywords=(
            "输出",
            "显示",
            "预览",
            "可视化",
            "透传",
            "检查点",
            "万能端口",
            "output",
            "display",
            "viewer",
            "sink",
            "debug",
            "any",
        ),
        notes=(
            "输入 / 输出端口都是任意类型端口：连线时不再做具体类型匹配。"
            "节点不在画布上生成汇总渲染图；数组详情请双击节点进入参数面板查看。"
        ),
        inputs=(
            PortSpec(
                "data",
                "任意数据",
                any_type=True,
                description="接收图像、数值、边缘特征、轮廓特征或连通域。",
            ),
        ),
        outputs=(PortSpec("data", "任意数据", any_type=True),),
    )

    def process(self, data: Any = None, **_: Any) -> Any:
        # input 为必填，正常执行时 data 一定是一个已知数据对象；原样返回即可。
        return data


OPERATOR = OutputDisplay
