"""便签节点：在画布上记录文字说明，同时原样透传任意类型的数据。

节点位于「其他 → 可视化」，用于给流程写备注：

* 输入端 / 输出端都是万能端口（``any_type=True``），图像、数值、边缘特征、
  轮廓特征、连通域都可以直接连入 / 连出；
* 节点卡片只显示参数 ``text`` 的内容，不显示缩略图、数值或数组提示；文字在
  参数面板（双击节点）中编辑，可换行；
* 输入可选：便签不接线也能放在画布上，此时不产生输出；接了输入则把数据原样
  透传给下游，方便把说明挂在某条链路中间。
"""

from __future__ import annotations

from typing import Any

from app.operators.spec import Operator, OperatorSpec, ParamSpec, PortSpec


class StickyNote(Operator):
    """便签：显示一段文字，并原样透传任意类型的输入数据。"""

    spec = OperatorSpec(
        id="note.sticky",
        name="便签",
        category=("其他", "可视化"),
        description=(
            "在画布上记录文字说明：节点卡片直接显示便签文字，双击节点可在参数面板中"
            "编辑（支持多行）。输入 / 输出都是万能端口，可接在任意类型的数据链路中间"
            "原样透传；不接线时也能单独作为注释使用，不会报错。"
        ),
        keywords=(
            "便签",
            "注释",
            "备注",
            "说明",
            "文字",
            "记录",
            "note",
            "sticky",
            "memo",
            "annotation",
            "comment",
            "any",
        ),
        notes=(
            "便签卡片只显示文字，不显示图像缩略图、数值或数组提示。"
            "输入端口是可选的：未接线时节点没有输出，但不会报错；"
            "接了输入后，输入对象会原样透传给下游，不复制、不转换。"
            "如果把便签输出接到下游节点，请同时接上便签输入，否则下游拿不到数据。"
        ),
        inputs=(
            PortSpec(
                "data",
                "任意数据",
                any_type=True,
                optional=True,
                description="可选：接收图像、数值、边缘特征、轮廓特征或连通域，并原样透传。",
            ),
        ),
        outputs=(PortSpec("data", "任意数据", any_type=True),),
        params=(
            ParamSpec(
                "text",
                "便签文字",
                "string",
                default="",
                multiline=True,
                placeholder="在此输入便签文字，可换行",
                tooltip="便签文字会显示在画布节点卡片内；支持多行，双击节点即可编辑。",
            ),
        ),
        canvas_text_param="text",
        allow_empty_output=True,
    )

    def process(self, data: Any = None, **_: Any) -> Any:
        # 未接输入时便签只作注释：返回空输出（spec.allow_empty_output 允许）。
        if data is None:
            return {}
        return {"data": data}


OPERATOR = StickyNote
