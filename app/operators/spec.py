"""算子声明式规范：端口、参数、校验与元数据。

算子作者只需写一个普通 Python 文件，并在模块级定义 ``OPERATOR``（算子类或
实例）。参数类型、范围、联动条件等全部由本模块的 dataclass 描述，界面和引擎
会据此自动生成控件、做校验。

示例::

    from app.operators.spec import Operator, OperatorSpec, ParamSpec, PortSpec

    class GaussianBlur(Operator):
        spec = OperatorSpec(
            id="filter.gaussian_blur",
            name="高斯模糊",
            category=("滤波与平滑",),
            description="对图像做高斯模糊。",
            params=(ParamSpec("ksize", "核大小", "int", default=5, min=1, max=99, odd=True),),
        )

        def process(self, image, ksize=5, **_):
            ...
            return blurred

    OPERATOR = GaussianBlur
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Any, ClassVar, Iterable, Mapping, Sequence

from app.core.types import ImageData

PARAM_TYPES = (
    "int",
    "float",
    "bool",
    "enum",
    "string",
    "color",
    "path",
    "points",
    "roi",
    "line",
)

INTERACTIVE_TOOLS = {
    "points": "points",
    "roi": "rect",
    "line": "line",
}


class OperatorError(RuntimeError):
    """算子执行/参数错误，会显示在节点与参数面板上。"""


class ParamValidationError(OperatorError):
    def __init__(self, key: str, message: str) -> None:
        super().__init__(message)
        self.key = key
        self.message = message


@dataclass(frozen=True)
class PortSpec:
    key: str
    label: str
    accepts: tuple[str, ...] = ("L", "BGR", "BGRA")
    optional: bool = False
    description: str = ""

    def accepts_image(self, image: ImageData) -> bool:
        if not self.accepts:
            return True
        return image.color_space in self.accepts

    def accept_text(self) -> str:
        return "/".join(self.accepts) if self.accepts else "任意"


@dataclass(frozen=True)
class ParamSpec:
    key: str
    label: str
    type: str = "int"
    default: Any = None
    min: float | None = None
    max: float | None = None
    step: float | None = None
    choices: Sequence[Any] = ()
    unit: str = ""
    group: str = ""
    tooltip: str = ""
    visible_when: str | None = None
    enabled_when: str | None = None
    odd: bool = False
    count: int = 0
    interactive: str | None = None
    multiline: bool = False
    placeholder: str = ""

    def __post_init__(self) -> None:
        if not self.key or not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", self.key):
            raise ValueError(f"非法参数 key：{self.key!r}")
        if self.type not in PARAM_TYPES:
            raise ValueError(f"参数 {self.key} 的类型 {self.type!r} 不受支持")
        if not self.label:
            object.__setattr__(self, "label", self.key)
        if self.type == "enum" and not self.choices:
            raise ValueError(f"枚举参数 {self.key} 必须提供 choices")
        if self.type == "points" and self.count < 0:
            raise ValueError(f"参数 {self.key} 的 count 不能为负数")
        if self.type not in ("points", "roi", "line") and self.interactive:
            raise ValueError(f"只有 points/roi/line 参数可以指定 interactive")
        if self.count and self.type != "points":
            raise ValueError(f"只有 points 参数可以使用 count")

        normalized: list[tuple[Any, str]] = []
        for item in self.choices or ():
            if isinstance(item, Mapping):
                value = item.get("value", item.get("label"))
                label = str(item.get("label", value))
            elif isinstance(item, (tuple, list)) and len(item) >= 2:
                value, label = item[0], str(item[1])
            else:
                value, label = item, str(item)
            normalized.append((value, label))
        object.__setattr__(self, "choices", tuple(normalized))

    # ------------------------------------------------------------------ 工具
    @property
    def group_name(self) -> str:
        return self.group or "参数"

    def default_value(self) -> Any:
        if self.default is not None:
            return coerce_param_value(self, self.default)
        if self.type == "int":
            return int(self.min) if self.min is not None and self.min > 0 else 0
        if self.type == "float":
            return float(self.min) if self.min is not None and self.min > 0 else 0.0
        if self.type == "bool":
            return False
        if self.type == "enum":
            return self.choices[0][0] if self.choices else None
        if self.type == "color":
            return "#ffffff"
        if self.type == "string" or self.type == "path":
            return ""
        if self.type == "points":
            return []
        return None

    def choice_label(self, value: Any) -> str:
        for choice_value, label in self.choices:
            if choice_value == value:
                return label
        return str(value)

    def choice_values(self) -> list[Any]:
        return [value for value, _label in self.choices]

    def tool_name(self) -> str | None:
        if self.interactive:
            return self.interactive
        return INTERACTIVE_TOOLS.get(self.type)

    def ui_step(self) -> float:
        if self.step is not None:
            return float(self.step)
        if self.type == "float":
            return 0.1
        return 1.0

    def with_default(self, value: Any) -> "ParamSpec":
        return replace(self, default=value)


@dataclass(frozen=True)
class OperatorSpec:
    id: str
    name: str
    category: tuple[str, ...] | str = ()
    description: str = ""
    version: str = "1.0.0"
    author: str = ""
    inputs: tuple[PortSpec, ...] = (PortSpec("image", "输入"),)
    outputs: tuple[PortSpec, ...] = (PortSpec("image", "输出"),)
    params: tuple[ParamSpec, ...] = ()
    keywords: tuple[str, ...] = ()
    notes: str = ""

    def __post_init__(self) -> None:
        if not self.id or not re.match(r"^[A-Za-z0-9_.-]+$", self.id):
            raise ValueError(f"非法算子 id：{self.id!r}")
        if not self.name:
            raise ValueError("算子名称不能为空")
        category = self.category
        if isinstance(category, str):
            parts = tuple(part.strip() for part in re.split(r"[/>]", category) if part.strip())
        else:
            parts = tuple(str(part).strip() for part in category if str(part).strip())
        object.__setattr__(self, "category", parts)
        object.__setattr__(self, "inputs", tuple(self.inputs))
        object.__setattr__(self, "outputs", tuple(self.outputs))
        object.__setattr__(self, "params", tuple(self.params))
        if not self.outputs:
            raise ValueError("算子至少需要一个输出端口")
        for attr in ("inputs", "outputs"):
            keys = [port.key for port in getattr(self, attr)]
            if len(keys) != len(set(keys)):
                raise ValueError(f"{attr} 端口 key 重复")
            if any(not key for key in keys):
                raise ValueError(f"{attr} 端口 key 不能为空")
        input_keys = {port.key for port in self.inputs}
        param_keys = [param.key for param in self.params]
        if len(param_keys) != len(set(param_keys)):
            raise ValueError("参数 key 重复")
        if input_keys & set(param_keys):
            raise ValueError(f"参数 key 与输入端口同名：{input_keys & set(param_keys)}")
        if any(param.key in ("self", "kwargs", "params") for param in self.params):
            raise ValueError("参数 key 不能使用 self / kwargs / params")
        defaults = param_defaults(self)
        for param in self.params:
            try:
                coerce_param_value(param, defaults[param.key])
            except ParamValidationError as exc:
                raise ValueError(f"参数 {param.key} 的默认值非法：{exc.message}") from exc

    @property
    def tab(self) -> str:
        return self.category[0] if self.category else "其他"

    @property
    def groups(self) -> tuple[str, ...]:
        return self.category[1:] if len(self.category) > 1 else ()

    def param(self, key: str) -> ParamSpec:
        for param in self.params:
            if param.key == key:
                return param
        raise KeyError(key)

    def input(self, key: str) -> PortSpec:
        for port in self.inputs:
            if port.key == key:
                return port
        raise KeyError(key)

    def output(self, key: str) -> PortSpec:
        for port in self.outputs:
            if port.key == key:
                return port
        raise KeyError(key)

    def with_category(self, category: Iterable[str]) -> "OperatorSpec":
        return replace(self, category=tuple(category))


class Operator:
    """所有算子的基类。

    ``process`` 接收以端口名/参数名命名的关键字参数：

    * 输入图像：``ImageData`` 或 ``None``（可选且未连接时）；
    * 参数：按 :class:`ParamSpec` 校验、转换后的值。

    返回值可以是单个 ``np.ndarray``（对应唯一输出端口），或
    ``{output_port_key: ndarray}`` 的多输出字典。请勿原地修改输入数组。
    """

    spec: ClassVar[OperatorSpec]

    def process(self, **kwargs: Any) -> Any:  # pragma: no cover - 由子类实现
        raise NotImplementedError

    def validate(self, inputs: Mapping[str, ImageData], params: Mapping[str, Any]) -> None:
        """可选的算子级校验；抛出 :class:`OperatorError` 即可阻止执行。"""


# ---------------------------------------------------------------------------
# 参数校验与联动
# ---------------------------------------------------------------------------
def param_defaults(spec: OperatorSpec) -> dict[str, Any]:
    return {param.key: param.default_value() for param in spec.params}


def normalize_params(
    spec: OperatorSpec,
    values: Mapping[str, Any] | None,
    *,
    add_defaults: bool = True,
) -> dict[str, Any]:
    """按规范整理参数：补默认值、检查未知字段并做类型/范围校验。"""
    result: dict[str, Any] = {}
    if add_defaults:
        result.update(param_defaults(spec))
    unknown = set(values or {}) - {param.key for param in spec.params}
    if unknown:
        raise ParamValidationError(sorted(unknown)[0], f"未知参数：{', '.join(sorted(unknown))}")
    for key, value in (values or {}).items():
        result[key] = value
    for param in spec.params:
        if param.key not in result:
            result[param.key] = param.default_value()
        result[param.key] = coerce_param_value(param, result[param.key])
    return result


def collect_param_errors(spec: OperatorSpec, values: Mapping[str, Any]) -> dict[str, str]:
    errors: dict[str, str] = {}
    try:
        normalize_params(spec, values)
    except ParamValidationError as exc:
        errors[exc.key] = exc.message
    except OperatorError as exc:
        errors["__all__"] = str(exc)
    return errors


def coerce_param_value(param: ParamSpec, value: Any) -> Any:
    key = param.key
    type_ = param.type
    if type_ == "int":
        if isinstance(value, bool):
            raise ParamValidationError(key, f"{param.label} 必须是整数")
        try:
            number = int(value)
        except (TypeError, ValueError) as exc:
            raise ParamValidationError(key, f"{param.label} 必须是整数") from exc
        if isinstance(value, float) and not value.is_integer():
            raise ParamValidationError(key, f"{param.label} 必须是整数")
        if param.min is not None and number < param.min:
            raise ParamValidationError(key, f"{param.label} 不能小于 {int(param.min)}")
        if param.max is not None and number > param.max:
            raise ParamValidationError(key, f"{param.label} 不能大于 {int(param.max)}")
        if param.odd and number % 2 == 0:
            raise ParamValidationError(key, f"{param.label} 必须为奇数")
        return number
    if type_ == "float":
        if isinstance(value, bool):
            raise ParamValidationError(key, f"{param.label} 必须是数字")
        try:
            number = float(value)
        except (TypeError, ValueError) as exc:
            raise ParamValidationError(key, f"{param.label} 必须是数字") from exc
        if param.min is not None and number < param.min - 1e-12:
            raise ParamValidationError(key, f"{param.label} 不能小于 {param.min:g}")
        if param.max is not None and number > param.max + 1e-12:
            raise ParamValidationError(key, f"{param.label} 不能大于 {param.max:g}")
        return number
    if type_ == "bool":
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            low = value.strip().lower()
            if low in {"true", "1", "yes", "on"}:
                return True
            if low in {"false", "0", "no", "off", ""}:
                return False
        raise ParamValidationError(key, f"{param.label} 必须是布尔值")
    if type_ == "enum":
        for choice in param.choice_values():
            if value == choice:
                return choice
        raise ParamValidationError(key, f"{param.label} 的取值无效：{value!r}")
    if type_ == "color":
        if not isinstance(value, str) or not re.match(r"^#(?:[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$", value):
            raise ParamValidationError(key, f"{param.label} 必须是 #RRGGBB 颜色")
        return value.lower()
    if type_ in ("string", "path"):
        if value is None:
            return ""
        if not isinstance(value, str):
            value = str(value)
        if param.multiline:
            return value
        return value.strip()
    if type_ == "points":
        if value is None:
            value = []
        if not isinstance(value, (list, tuple)):
            raise ParamValidationError(key, f"{param.label} 必须是点列表")
        points: list[list[float]] = []
        for point in value:
            if not isinstance(point, (list, tuple)) or len(point) != 2:
                raise ParamValidationError(key, f"{param.label} 的点必须是 [x, y]")
            try:
                points.append([float(point[0]), float(point[1])])
            except (TypeError, ValueError) as exc:
                raise ParamValidationError(key, f"{param.label} 的点坐标必须是数字") from exc
        if param.count and len(points) != param.count:
            raise ParamValidationError(key, f"请选择 {param.count} 个点（当前 {len(points)} 个）")
        return points
    if type_ in ("roi", "line"):
        if value in (None, [], ()):
            return None
        if not isinstance(value, (list, tuple)) or len(value) != 4:
            raise ParamValidationError(key, f"{param.label} 需要 4 个数值")
        try:
            numbers = [float(item) for item in value]
        except (TypeError, ValueError) as exc:
            raise ParamValidationError(key, f"{param.label} 的数值无效") from exc
        if type_ == "roi":
            if numbers[2] <= 0 or numbers[3] <= 0:
                raise ParamValidationError(key, f"{param.label} 的宽高必须大于 0")
        return numbers
    raise ParamValidationError(key, f"不支持的参数类型：{type_}")


def validate_params(
    spec: OperatorSpec,
    values: Mapping[str, Any],
    *,
    add_defaults: bool = True,
) -> dict[str, Any]:
    """校验并返回规范化参数；失败时抛出带字段信息的异常。"""
    return normalize_params(spec, values, add_defaults=add_defaults)


def eval_condition(expression: str | None, params: Mapping[str, Any]) -> bool:
    """安全地执行参数联动表达式；异常时视为 True（保持可见）。"""
    if not expression:
        return True
    try:
        namespace = {"__builtins__": {}, "params": dict(params), "abs": abs, "len": len, "min": min, "max": max}
        return bool(eval(expression, namespace, dict(params)))  # noqa: S307 - 算子作者提供的可信表达式
    except Exception:  # noqa: BLE001
        return True


def is_param_visible(param: ParamSpec, params: Mapping[str, Any]) -> bool:
    return eval_condition(param.visible_when, params)


def is_param_enabled(param: ParamSpec, params: Mapping[str, Any]) -> bool:
    return eval_condition(param.enabled_when, params)


def params_by_group(spec: OperatorSpec) -> dict[str, list[ParamSpec]]:
    groups: dict[str, list[ParamSpec]] = {}
    for param in spec.params:
        groups.setdefault(param.group_name, []).append(param)
    return groups


def validate_inputs(spec: OperatorSpec, inputs: Mapping[str, ImageData | None]) -> None:
    """检查端口是否齐全、颜色空间是否匹配。"""
    missing = []
    for port in spec.inputs:
        image = inputs.get(port.key)
        if image is None:
            if not port.optional:
                missing.append(port.label)
            continue
        if not port.accepts_image(image):
            raise OperatorError(
                f"输入“{port.label}”需要 {port.accept_text()} 图像，当前为 {image.color_space}。"
            )
    if missing:
        raise OperatorError(f"缺少输入：{'、'.join(missing)}")


def output_to_dict(spec: OperatorSpec, output: Any) -> dict[str, ImageData]:
    """把算子返回值规范化为 ``{port: ImageData}``。"""
    if isinstance(output, ImageData):
        output = output.array
    if isinstance(output, dict):
        result: dict[str, ImageData] = {}
        for key, value in output.items():
            if key not in {port.key for port in spec.outputs}:
                raise OperatorError(f"算子返回了未声明的输出端口：{key}")
            if value is None:
                continue
            image = value if isinstance(value, ImageData) else ImageData(value)
            result[key] = image
        return result
    if output is None:
        raise OperatorError("算子没有返回图像")
    if isinstance(output, (list, tuple)):
        if len(output) != len(spec.outputs):
            raise OperatorError(f"算子返回了 {len(output)} 个输出，但声明了 {len(spec.outputs)} 个端口")
        return {
            port.key: (value if isinstance(value, ImageData) else ImageData(value))
            for port, value in zip(spec.outputs, output)
        }
    image = output if isinstance(output, ImageData) else ImageData(output)
    return {spec.outputs[0].key: image}
