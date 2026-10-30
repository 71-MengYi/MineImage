"""界面「新建算子」使用的空白算子模板。

生成的文本可以被 :mod:`app.operators.loader` 直接扫描加载，并带有
``MINEIMAGE_UI_CREATED = True`` 标记：界面重命名文件时会据此判断是否要同步
``OperatorSpec.name``（见 :func:`app.core.operator_files.update_generated_operator_name`）。
"""

from __future__ import annotations

import keyword
import re
from typing import Iterable

#: 模块级标记名：表示该算子文件由界面自动生成。
UI_CREATED_MARKER = "MINEIMAGE_UI_CREATED"

_CATEGORY_SPLIT = re.compile(r"[/>]")

#: 未命名时使用的默认算子名称。
DEFAULT_OPERATOR_NAME = "新建算子"

#: 未指定分类时写入模板的分类（与 loader 对顶层文件的默认分类一致）。
DEFAULT_CATEGORY = ("其他",)


def _category_parts(category: str | Iterable[str] | None) -> tuple[str, ...]:
    """把分类统一成层级元组；字符串支持 ``"一级/二级"`` 写法。"""
    if category is None:
        return ()
    if isinstance(category, str):
        return tuple(part.strip() for part in _CATEGORY_SPLIT.split(category) if part.strip())
    return tuple(str(part).strip() for part in category if str(part).strip())


def _class_identifier(class_name: str) -> str:
    name = str(class_name or "").strip()
    if name.isidentifier() and not keyword.iskeyword(name):
        return name
    return "MyOperator"


def render_operator_template(
    *,
    operator_id: str,
    name: str,
    category: str | Iterable[str] | None = (),
    class_name: str = "MyOperator",
) -> str:
    """渲染一个可直接加载的空白算子模块源码。

    ``process()`` 只做“复制输入图像”的占位实现并标注 TODO；参数、分类、说明等
    都用注释留出修改位置，用户可以在任意编辑器里补全。
    """

    op_id = str(operator_id or "").strip()
    if not op_id:
        raise ValueError("算子 id 不能为空")
    display_name = str(name or "").strip() or DEFAULT_OPERATOR_NAME
    parts = _category_parts(category)
    category_literal = repr(parts)
    description = f"{display_name}（空白模板，待补充说明）"
    identifier = _class_identifier(class_name)
    return f'''"""由 MineImage 界面新建的空白算子。

本文件已能被算子库直接加载；请按下面的 TODO 逐步完善：
1. 修改 id / name / category / description；
2. 需要参数时在 params 中声明 ParamSpec（参数面板会自动生成控件）；
3. 在 process() 里实现真正的处理逻辑。

{UI_CREATED_MARKER} 标记表示文件由界面生成：在界面中重命名文件时，如果这里的
name 仍与旧文件名一致，会自动同步；手动改过 name 之后则不再覆盖。
"""

from __future__ import annotations

from app.operators.spec import Operator, OperatorSpec, PortSpec

#: 由 MineImage 界面新建（重命名时据此决定是否同步 name）
{UI_CREATED_MARKER} = True


class {identifier}(Operator):
    spec = OperatorSpec(
        # TODO: 算子 id 必须全局唯一，只能包含字母 / 数字 / 下划线 / 点 / 连字符
        id={op_id!r},
        # TODO: 显示名称（默认取文件名，可自行修改）
        name={display_name!r},
        # TODO: 一级分类 / 二级分类 / ...（决定在算子库中的位置）
        category={category_literal},
        # TODO: 一句话说明这个算子做什么
        description={description!r},
        # TODO: 需要多个输入 / 输出或非图像类型时，在这里修改端口声明
        inputs=(PortSpec("image", "输入", accepts=("L", "BGR", "BGRA")),),
        outputs=(PortSpec("image", "输出"),),
        params=(
            # TODO: 需要参数时先在文件顶部 import ParamSpec，再按下面的写法声明：
            # from app.operators.spec import ParamSpec
            # ParamSpec("strength", "强度", "float", default=0.5, min=0.0,
            #           max=1.0, step=0.05, tooltip="0 表示不处理，1 表示最强。"),
        ),
    )

    def process(self, image=None, **_):
        # TODO: 在这里实现真正的处理逻辑；process() 可能在后台线程被并发调用，
        # 不要原地修改输入数组，也不要弹窗或阻塞等待。
        if image is None:
            return None
        # 占位实现：原样复制输入图像，保证不修改调用方的数组。
        return image.array.copy()


OPERATOR = {identifier}
'''


__all__ = [
    "DEFAULT_CATEGORY",
    "DEFAULT_OPERATOR_NAME",
    "UI_CREATED_MARKER",
    "render_operator_template",
]
