"""OpenCV 目录算子：算术运算（基础调整）。

本模块在一个文件里导出多个算子，由 ``OPERATORS`` 一次性注册。所有算子都：

* 用 :mod:`app.operators.builtin.catalog._cv` 读取输入、校验尺寸并恢复 dtype；
* 逐像素结果在 ``0..1`` 单位域计算，再写回输入的 dtype（整型自动饱和）；
* 同时支持 L / BGR / BGRA 与 uint8 / uint16 / float32；
* 返回新数组，绝不原地修改输入；
* 默认参数即可直接运行（第二幅图未连接时使用参数值或第一幅图的派生结果）。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.operators.builtin.catalog import _cv
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec

#: 0..255 参数换算到 0..1 单位域时使用的满量程。
_U8 = 255.0


# ---------------------------------------------------------------------------
# 共享辅助
# ---------------------------------------------------------------------------
def _unit_like(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """把 ``other`` 映射到 0..1 并对齐到 ``reference`` 的通道数。"""
    unit = _cv.unit_array(other)
    ref_channels = _cv.channels(reference)
    other_channels = _cv.channels(unit)
    if ref_channels == other_channels:
        return unit
    if ref_channels == 1:
        if other_channels >= 3:
            # BGR / BGRA 按亮度合成单通道。
            return 0.299 * unit[..., 2] + 0.587 * unit[..., 1] + 0.114 * unit[..., 0]
        return unit
    if other_channels == 1:
        return np.repeat(unit[:, :, None], ref_channels, axis=2)
    if ref_channels == 3:
        return unit[:, :, :3]
    alpha = np.ones((unit.shape[0], unit.shape[1], 1), dtype=np.float32)
    return np.concatenate([unit[:, :, :3], alpha], axis=2)


def _u8_like(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """把 ``other`` 转成 uint8 并对齐到 ``reference`` 的通道数。"""
    work = _cv.to_uint8(other, "第二幅图像")
    ref_channels = _cv.channels(reference)
    work_channels = _cv.channels(work)
    if ref_channels == work_channels:
        return work
    if ref_channels == 1:
        if work_channels == 3:
            return cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        if work_channels == 4:
            return cv2.cvtColor(work, cv2.COLOR_BGRA2GRAY)
        return work
    if work_channels == 1:
        return np.repeat(work[:, :, None], ref_channels, axis=2)
    if ref_channels == 3:
        return work[:, :, :3]
    alpha = np.full((work.shape[0], work.shape[1], 1), 255, dtype=np.uint8)
    return np.concatenate([work[:, :, :3], alpha], axis=2)


def _second_unit(array: np.ndarray, image2, fallback: np.ndarray) -> np.ndarray:
    """取第二幅图的 0..1 数据；未连接时直接使用 ``fallback``（单位域数组）。

    ``fallback`` 允许出现负数或大于 1 的值（例如减法的负偏移、乘法的增益系数），
    因此不能再次经过 ``_cv.unit_array``（它会把结果裁剪到 0..1）。
    """
    if image2 is None:
        return fallback
    other = _cv.optional_second(image2, fallback, "第二幅图像")
    return _unit_like(array, other)


def _finish(unit: np.ndarray, array: np.ndarray) -> np.ndarray:
    """裁剪到 0..1 并恢复输入 dtype。"""
    return _cv.from_unit(np.clip(unit, 0.0, 1.0), array)


# ---------------------------------------------------------------------------
# 基础算术
# ---------------------------------------------------------------------------
class Add(Operator):
    """逐像素加法。"""

    spec = OperatorSpec(
        id="basic.add",
        name="加法",
        category=("基础调整", "算术"),
        description=(
            "把图像与第二幅图像逐像素相加；第二幅图未连接时改为加上参数常数值。"
            "结果按整型饱和裁剪，浮点保留小数，BGRA 的 alpha 一并参与运算。"
        ),
        keywords=("加法", "相加", "add", "plus", "算术", "arithmetic"),
        inputs=(
            PortSpec("image", "输入图像", description="参与相加的图像，支持 L / BGR / BGRA。"),
            PortSpec(
                "image2",
                "第二幅图像（可选）",
                optional=True,
                description="未连接时使用参数“常数值”。",
            ),
        ),
        outputs=(PortSpec("image", "相加结果"),),
        params=(
            ParamSpec(
                "value",
                "常数值",
                "int",
                default=30,
                min=-255,
                max=255,
                step=1,
                unit="级",
                tooltip="第二幅图像未连接时，把该值（0..255 满量程）加到每个像素，可为负。",
            ),
        ),
    )

    def process(self, image=None, image2=None, value: int = 30, **_):
        array = _cv.source(image)
        unit = _cv.unit_array(array)
        fallback = np.full_like(unit, float(value) / _U8)
        second = _second_unit(array, image2, fallback)
        return _finish(unit + second, array)


class Subtract(Operator):
    """逐像素减法。"""

    spec = OperatorSpec(
        id="basic.subtract",
        name="减法",
        category=("基础调整", "算术"),
        description=(
            "从图像中减去第二幅图像；第二幅图未连接时改为减去参数常数值。"
            "结果小于 0 的部分按整型饱和为 0。"
        ),
        keywords=("减法", "相减", "subtract", "minus", "算术"),
        inputs=(
            PortSpec("image", "输入图像", description="被减图像，支持 L / BGR / BGRA。"),
            PortSpec(
                "image2",
                "第二幅图像（可选）",
                optional=True,
                description="作为减数的图像；未连接时使用参数“常数值”。",
            ),
        ),
        outputs=(PortSpec("image", "相减结果"),),
        params=(
            ParamSpec(
                "value",
                "常数值",
                "int",
                default=30,
                min=-255,
                max=255,
                step=1,
                unit="级",
                tooltip="第二幅图像未连接时，从每个像素减去该值（0..255 满量程）。",
            ),
        ),
    )

    def process(self, image=None, image2=None, value: int = 30, **_):
        array = _cv.source(image)
        unit = _cv.unit_array(array)
        fallback = np.full_like(unit, float(value) / _U8)
        second = _second_unit(array, image2, fallback)
        return _finish(unit - second, array)


class Multiply(Operator):
    """逐像素乘法。"""

    spec = OperatorSpec(
        id="basic.multiply",
        name="乘法",
        category=("基础调整", "算术"),
        description=(
            "把图像与第二幅图像按 0..1 归一化后逐像素相乘；第二幅图未连接时改为乘以参数系数。"
            "适合做亮度增益或蒙版式压暗。"
        ),
        keywords=("乘法", "相乘", "multiply", "增益", "gain", "算术"),
        inputs=(
            PortSpec("image", "输入图像", description="参与相乘的图像，支持 L / BGR / BGRA。"),
            PortSpec(
                "image2",
                "第二幅图像（可选）",
                optional=True,
                description="作为乘数的图像；未连接时使用参数“系数”。",
            ),
        ),
        outputs=(PortSpec("image", "相乘结果"),),
        params=(
            ParamSpec(
                "factor",
                "系数",
                "float",
                default=1.5,
                min=0.05,
                max=4.0,
                step=0.05,
                unit="×",
                tooltip="第二幅图像未连接时，每个像素乘以该系数；1.0 表示不变。",
            ),
        ),
    )

    def process(self, image=None, image2=None, factor: float = 1.5, **_):
        array = _cv.source(image)
        unit = _cv.unit_array(array)
        fallback = np.full_like(unit, float(factor))
        second = _second_unit(array, image2, fallback)
        return _finish(unit * second, array)


class Divide(Operator):
    """逐像素除法。"""

    spec = OperatorSpec(
        id="basic.divide",
        name="除法",
        category=("基础调整", "算术"),
        description=(
            "把图像逐像素除以第二幅图像；第二幅图未连接时改为除以参数系数。"
            "分母会被钳制到极小值，避免除零。"
        ),
        keywords=("除法", "相除", "divide", "算术"),
        inputs=(
            PortSpec("image", "输入图像", description="被除图像，支持 L / BGR / BGRA。"),
            PortSpec(
                "image2",
                "第二幅图像（可选）",
                optional=True,
                description="作为除数的图像；未连接时使用参数“系数”。",
            ),
        ),
        outputs=(PortSpec("image", "相除结果"),),
        params=(
            ParamSpec(
                "factor",
                "系数",
                "float",
                default=1.5,
                min=0.05,
                max=4.0,
                step=0.05,
                unit="×",
                tooltip="第二幅图像未连接时，每个像素除以该系数。",
            ),
        ),
    )

    def process(self, image=None, image2=None, factor: float = 1.5, **_):
        array = _cv.source(image)
        unit = _cv.unit_array(array)
        fallback = np.full_like(unit, float(factor))
        second = _second_unit(array, image2, fallback)
        return _finish(unit / np.maximum(second, 1e-6), array)


class AbsDiff(Operator):
    """绝对差分。"""

    spec = OperatorSpec(
        id="basic.absdiff",
        name="绝对差分",
        category=("基础调整", "算术"),
        description=(
            "求图像与第二幅图像的逐像素绝对差；第二幅图未连接时求与参数常数值的绝对差。"
            "常用作运动检测或变化检测的差图。"
        ),
        keywords=("绝对差", "差分", "absdiff", "difference", "变化检测"),
        inputs=(
            PortSpec("image", "输入图像", description="第一幅图像，支持 L / BGR / BGRA。"),
            PortSpec(
                "image2",
                "第二幅图像（可选）",
                optional=True,
                description="第二幅图像；未连接时使用参数“常数值”。",
            ),
        ),
        outputs=(PortSpec("image", "差分结果"),),
        params=(
            ParamSpec(
                "value",
                "常数值",
                "int",
                default=30,
                min=0,
                max=255,
                step=1,
                unit="级",
                tooltip="第二幅图像未连接时，与每个像素求绝对差。",
            ),
        ),
    )

    def process(self, image=None, image2=None, value: int = 30, **_):
        array = _cv.source(image)
        unit = _cv.unit_array(array)
        fallback = np.full_like(unit, float(value) / _U8)
        second = _second_unit(array, image2, fallback)
        return _finish(np.abs(unit - second), array)


class AddWeighted(Operator):
    """加权叠加。"""

    spec = OperatorSpec(
        id="basic.add_weighted",
        name="加权叠加",
        category=("基础调整", "算术"),
        description=(
            "按 alpha / beta / gamma 对两幅图像做加权和；第二幅图未连接时等价于对第一幅图"
            "整体乘以 (alpha + beta) 再加 gamma。适合做过渡、淡入淡出与曝光混合。"
        ),
        keywords=("加权", "叠加", "add_weighted", "blend", "混合", "alpha"),
        inputs=(
            PortSpec("image", "输入图像", description="第一幅图像，支持 L / BGR / BGRA。"),
            PortSpec(
                "image2",
                "第二幅图像（可选）",
                optional=True,
                description="未连接时使用第一幅图自身。",
            ),
        ),
        outputs=(PortSpec("image", "叠加结果"),),
        params=(
            ParamSpec(
                "alpha",
                "第一幅权重",
                "float",
                default=0.5,
                min=0.0,
                max=2.0,
                step=0.05,
                tooltip="第一幅图像（或常量项）的权重。",
            ),
            ParamSpec(
                "beta",
                "第二幅权重",
                "float",
                default=0.5,
                min=0.0,
                max=2.0,
                step=0.05,
                tooltip="第二幅图像（或常量项）的权重。",
            ),
            ParamSpec(
                "gamma",
                "附加偏移",
                "int",
                default=0,
                min=-255,
                max=255,
                step=1,
                unit="级",
                tooltip="叠加后整体增加的偏移量，可为负。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        image2=None,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: int = 0,
        **_,
    ):
        array = _cv.source(image)
        unit = _cv.unit_array(array)
        second = _second_unit(array, image2, unit)
        result = float(alpha) * unit + float(beta) * second + float(gamma) / _U8
        return _finish(result, array)


# ---------------------------------------------------------------------------
# 按位运算
# ---------------------------------------------------------------------------
def _bitwise_second_u8(array: np.ndarray, image2, value: int) -> np.ndarray:
    if image2 is None:
        return np.full(array.shape, int(np.clip(round(float(value)), 0, 255)), dtype=np.uint8)
    other = _cv.source(image2, "第二幅图像")
    _cv.same_size(array, other, "第二幅图像")
    return _u8_like(array, other)


def _bitwise_second_native(array: np.ndarray, image2) -> np.ndarray:
    other = _cv.source(image2, "第二幅图像")
    _cv.same_size(array, other, "第二幅图像")
    return _cv.from_unit(_unit_like(array, other), array)


def _bitwise_result(array: np.ndarray, image2, value: int, operation) -> np.ndarray:
    """按位运算的统一入口：整型走原生位宽，float32 先转 8 位再还原。"""
    if array.dtype == np.uint8:
        second = _bitwise_second_u8(array, image2, value)
        return operation(array, second)
    if array.dtype == np.uint16:
        if image2 is None:
            scalar = int(round(float(value) / _U8 * np.iinfo(np.uint16).max))
            scalar = int(np.clip(scalar, 0, np.iinfo(np.uint16).max))
            return operation(array, scalar)
        return operation(array, _bitwise_second_native(array, image2))
    # float32：先映射到 8 位，运算后按满量程还原。
    first = _cv.to_uint8(array)
    if image2 is None:
        second = np.full(first.shape, int(np.clip(round(float(value)), 0, 255)), dtype=np.uint8)
    else:
        other = _cv.source(image2, "第二幅图像")
        _cv.same_size(array, other, "第二幅图像")
        second = _u8_like(array, other)
    work = operation(first, second)
    return _cv.from_unit(work.astype(np.float32) / _U8, array)


class BitwiseAnd(Operator):
    """按位与。"""

    spec = OperatorSpec(
        id="basic.bitwise_and",
        name="按位与",
        category=("基础调整", "算术"),
        description=(
            "对图像与第二幅图（或参数常量）做逐位与。整型按原生位宽运算，float32 先转 8 位"
            "再还原；常用作蒙版裁剪。"
        ),
        keywords=("按位与", "bitwise_and", "mask", "蒙版", "位运算"),
        inputs=(
            PortSpec("image", "输入图像", description="第一幅图像，支持 L / BGR / BGRA。"),
            PortSpec(
                "image2",
                "第二幅图像（可选）",
                optional=True,
                description="作为位掩码的图像；未连接时使用参数“掩码值”。",
            ),
        ),
        outputs=(PortSpec("image", "按位与结果"),),
        params=(
            ParamSpec(
                "value",
                "掩码值",
                "int",
                default=128,
                min=0,
                max=255,
                step=1,
                tooltip="第二幅图像未连接时，与每个像素做按位与的常量值。",
            ),
        ),
    )

    def process(self, image=None, image2=None, value: int = 128, **_):
        array = _cv.source(image)
        return _bitwise_result(array, image2, value, cv2.bitwise_and)


class BitwiseOr(Operator):
    """按位或。"""

    spec = OperatorSpec(
        id="basic.bitwise_or",
        name="按位或",
        category=("基础调整", "算术"),
        description=(
            "对图像与第二幅图（或参数常量）做逐位或。整型按原生位宽运算，float32 先转 8 位"
            "再还原；常用作叠加掩码。"
        ),
        keywords=("按位或", "bitwise_or", "位运算", "掩码"),
        inputs=(
            PortSpec("image", "输入图像", description="第一幅图像，支持 L / BGR / BGRA。"),
            PortSpec(
                "image2",
                "第二幅图像（可选）",
                optional=True,
                description="作为位掩码的图像；未连接时使用参数“掩码值”。",
            ),
        ),
        outputs=(PortSpec("image", "按位或结果"),),
        params=(
            ParamSpec(
                "value",
                "掩码值",
                "int",
                default=128,
                min=0,
                max=255,
                step=1,
                tooltip="第二幅图像未连接时，与每个像素做按位或的常量值。",
            ),
        ),
    )

    def process(self, image=None, image2=None, value: int = 128, **_):
        array = _cv.source(image)
        return _bitwise_result(array, image2, value, cv2.bitwise_or)


class BitwiseXor(Operator):
    """按位异或。"""

    spec = OperatorSpec(
        id="basic.bitwise_xor",
        name="按位异或",
        category=("基础调整", "算术"),
        description=(
            "对图像与第二幅图（或参数常量）做逐位异或。整型按原生位宽运算，float32 先转 8 位"
            "再还原；常用来找两幅图的差异位。"
        ),
        keywords=("按位异或", "bitwise_xor", "位运算", "差异"),
        inputs=(
            PortSpec("image", "输入图像", description="第一幅图像，支持 L / BGR / BGRA。"),
            PortSpec(
                "image2",
                "第二幅图像（可选）",
                optional=True,
                description="参与异或的图像；未连接时使用参数“掩码值”。",
            ),
        ),
        outputs=(PortSpec("image", "按位异或结果"),),
        params=(
            ParamSpec(
                "value",
                "掩码值",
                "int",
                default=128,
                min=0,
                max=255,
                step=1,
                tooltip="第二幅图像未连接时，与每个像素做按位异或的常量值。",
            ),
        ),
    )

    def process(self, image=None, image2=None, value: int = 128, **_):
        array = _cv.source(image)
        return _bitwise_result(array, image2, value, cv2.bitwise_xor)


class BitwiseNot(Operator):
    """按位取反。"""

    spec = OperatorSpec(
        id="basic.bitwise_not",
        name="按位取反",
        category=("基础调整", "算术"),
        description=(
            "对每个像素按位取反。uint8 等价于 255 - 像素，uint16 等价于 65535 - 像素；"
            "float32 先转 8 位取反再按满量程还原。"
        ),
        keywords=("按位取反", "反相", "bitwise_not", "invert", "位运算"),
        inputs=(PortSpec("image", "输入图像", description="要取反的图像，支持 L / BGR / BGRA。"),),
        outputs=(PortSpec("image", "取反结果"),),
    )

    def process(self, image=None, **_):
        array = _cv.source(image)
        if array.dtype in (np.uint8, np.uint16):
            return cv2.bitwise_not(array)
        work = _cv.to_uint8(array)
        inverted = cv2.bitwise_not(work)
        return _cv.from_unit(inverted.astype(np.float32) / _U8, array)


# ---------------------------------------------------------------------------
# 比较与数值映射
# ---------------------------------------------------------------------------
_COMPARE_OPERATIONS = {
    "gt": "大于",
    "ge": "大于等于",
    "lt": "小于",
    "le": "小于等于",
    "eq": "等于",
    "ne": "不等于",
}


class Compare(Operator):
    """逐像素比较并输出二值掩膜。"""

    spec = OperatorSpec(
        id="basic.compare",
        name="图像比较",
        category=("基础调整", "算术"),
        description=(
            "把图像与第二幅图（或参数常数值）逐像素比较，输出单通道二值掩膜（满足为 255）。"
            "彩色图要求所有通道都满足条件；支持 L / BGR / BGRA 与三种 dtype。"
        ),
        keywords=("比较", "compare", "阈值", "掩膜", "mask"),
        inputs=(
            PortSpec("image", "输入图像", description="第一幅图像，支持 L / BGR / BGRA。"),
            PortSpec(
                "image2",
                "第二幅图像（可选）",
                optional=True,
                description="未连接时与参数“常数值”比较。",
            ),
        ),
        outputs=(PortSpec("image", "比较掩膜", accepts=("L",)),),
        params=(
            ParamSpec(
                "method",
                "比较方式",
                "enum",
                choices=tuple((key, label) for key, label in _COMPARE_OPERATIONS.items()),
                default="gt",
                tooltip="满足条件（或全部通道满足）的像素输出 255。",
            ),
            ParamSpec(
                "value",
                "常数值",
                "int",
                default=128,
                min=0,
                max=255,
                step=1,
                unit="级",
                tooltip="第二幅图像未连接时，与该常量值比较。",
            ),
        ),
    )

    def process(self, image=None, image2=None, method: str = "gt", value: int = 128, **_):
        array = _cv.source(image)
        unit = _cv.unit_array(array)
        fallback = np.full_like(unit, float(value) / _U8)
        second = _second_unit(array, image2, fallback)
        if method == "gt":
            mask = unit > second
        elif method == "ge":
            mask = unit >= second - 0.5 / _U8
        elif method == "lt":
            mask = unit < second
        elif method == "le":
            mask = unit <= second + 0.5 / _U8
        elif method == "eq":
            mask = np.abs(unit - second) <= 0.5 / _U8
        elif method == "ne":
            mask = np.abs(unit - second) > 0.5 / _U8
        else:  # pragma: no cover - 参数枚举已限制
            raise OperatorError(f"不支持的比较方式：{method}")
        if mask.ndim == 3:
            mask = np.all(mask, axis=2)
        return (mask.astype(np.uint8) * 255).copy()


class Normalize(Operator):
    """最小-最大归一化。"""

    spec = OperatorSpec(
        id="basic.normalize",
        name="归一化",
        category=("基础调整", "算术"),
        description=(
            "把整幅图像的像素按最小 / 最大值拉伸到指定区间。常数图像输出该区间下限；"
            "输出 dtype 与输入一致，uint16 / float32 会按满量程换算。"
        ),
        keywords=("归一化", "normalize", "拉伸", "对比度", "minmax"),
        inputs=(PortSpec("image", "输入图像", description="需要拉伸的图像，支持 L / BGR / BGRA。"),),
        outputs=(PortSpec("image", "归一化结果"),),
        params=(
            ParamSpec(
                "low",
                "输出下限",
                "int",
                default=0,
                min=0,
                max=255,
                step=1,
                unit="级",
                tooltip="拉伸后的最小像素值（0..255 满量程）。",
            ),
            ParamSpec(
                "high",
                "输出上限",
                "int",
                default=255,
                min=0,
                max=255,
                step=1,
                unit="级",
                tooltip="拉伸后的最大像素值，必须大于下限。",
            ),
        ),
    )

    def process(self, image=None, low: int = 0, high: int = 255, **_):
        array = _cv.source(image)
        if int(high) <= int(low):
            raise OperatorError("归一化的输出上限必须大于下限。")
        unit = _cv.unit_array(array)
        low_value = float(low) / _U8
        high_value = float(high) / _U8
        low_pixel = float(unit.min())
        high_pixel = float(unit.max())
        span = high_pixel - low_pixel
        if span <= 1e-12:
            normalized = np.zeros_like(unit)
        else:
            normalized = (unit - low_pixel) / span
        result = low_value + normalized * (high_value - low_value)
        return _finish(result, array)


class ConvertScale(Operator):
    """线性缩放（缩放 + 偏移）。"""

    spec = OperatorSpec(
        id="basic.convert_scale",
        name="线性缩放",
        category=("基础调整", "算术"),
        description=(
            "按“像素 × alpha + beta”做线性映射，可选输出绝对值（类似 convertScaleAbs）。"
            "适合整体提亮、压暗或扩展动态范围。"
        ),
        keywords=("线性缩放", "convert_scale", "增益", "偏移", "alpha", "beta"),
        inputs=(PortSpec("image", "输入图像", description="需要线性映射的图像，支持 L / BGR / BGRA。"),),
        outputs=(PortSpec("image", "映射结果"),),
        params=(
            ParamSpec(
                "alpha",
                "缩放系数",
                "float",
                default=1.2,
                min=0.0,
                max=8.0,
                step=0.05,
                unit="×",
                tooltip="像素乘以该系数；1.0 表示不变。",
            ),
            ParamSpec(
                "beta",
                "偏移量",
                "float",
                default=0.0,
                min=-255.0,
                max=255.0,
                step=1.0,
                unit="级",
                tooltip="乘以系数后整体加上的偏移（0..255 满量程）。",
            ),
            ParamSpec(
                "absolute",
                "取绝对值",
                "bool",
                default=False,
                tooltip="勾选后输出 |像素 × alpha + beta|，负值不会被直接截断。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        alpha: float = 1.2,
        beta: float = 0.0,
        absolute: bool = False,
        **_,
    ):
        array = _cv.source(image)
        unit = _cv.unit_array(array)
        result = unit * float(alpha) + float(beta) / _U8
        if absolute:
            result = np.abs(result)
        return _finish(result, array)


class Gamma(Operator):
    """伽马校正。"""

    spec = OperatorSpec(
        id="basic.gamma",
        name="伽马校正",
        category=("基础调整", "算术"),
        description=(
            "按 out = in^(1/γ) 做逐像素伽马校正：γ 大于 1 提亮中间调，小于 1 压暗中间调。"
            "在 0..1 单位域计算后恢复输入 dtype。"
        ),
        keywords=("伽马", "gamma", "校正", "中间调", "亮度"),
        inputs=(PortSpec("image", "输入图像", description="需要做伽马校正的图像，支持 L / BGR / BGRA。"),),
        outputs=(PortSpec("image", "伽马校正结果"),),
        params=(
            ParamSpec(
                "gamma",
                "伽马值",
                "float",
                default=1.0,
                min=0.05,
                max=5.0,
                step=0.05,
                tooltip="大于 1 提亮，小于 1 压暗，1.0 表示不变。",
            ),
        ),
    )

    def process(self, image=None, gamma: float = 1.0, **_):
        array = _cv.source(image)
        if float(gamma) <= 0.0:
            raise OperatorError("伽马值必须大于 0。")
        unit = _cv.unit_array(array)
        result = np.power(unit, 1.0 / float(gamma))
        return _finish(result, array)


OPERATORS = (
    Add,
    Subtract,
    Multiply,
    Divide,
    AbsDiff,
    AddWeighted,
    BitwiseAnd,
    BitwiseOr,
    BitwiseXor,
    BitwiseNot,
    Compare,
    Normalize,
    ConvertScale,
    Gamma,
)
