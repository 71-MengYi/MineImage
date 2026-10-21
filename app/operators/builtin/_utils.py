"""内置算子共享的图像工具。

文件名以下划线开头，:class:`app.operators.loader.OperatorRegistry` 会跳过它，
因此这里可以放任意辅助函数，而不需要定义 ``OPERATOR``。

约定
----
* 所有函数都 **不修改** 传入数组；需要输出时返回新数组（或只读视图）。
* 内核支持的 dtype 为 ``uint8`` / ``uint16`` / ``float32``；其它 dtype 会先
  转成 float32 再处理，避免 OpenCV 对 int32 / float16 等直接报英文错误。
* 不支持的形状/通道组合统一抛 :class:`app.operators.spec.OperatorError`，
  消息为可直接展示给用户的中文。
"""

from __future__ import annotations

from typing import Any, Iterable

import cv2
import numpy as np

from app.core.types import ImageData
from app.operators.spec import OperatorError

#: 内置算子直接支持的 dtype（OpenCV 对这些类型的通用滤波/阈值支持最好）。
NATIVE_DTYPES = (np.dtype(np.uint8), np.dtype(np.uint16), np.dtype(np.float32))

_INTERPOLATIONS = {
    "nearest": cv2.INTER_NEAREST,
    "linear": cv2.INTER_LINEAR,
    "cubic": cv2.INTER_CUBIC,
    "area": cv2.INTER_AREA,
    "lanczos": cv2.INTER_LANCZOS4,
}

_BORDERS = {
    "constant": cv2.BORDER_CONSTANT,
    "replicate": cv2.BORDER_REPLICATE,
    "reflect": cv2.BORDER_REFLECT,
    "reflect101": cv2.BORDER_REFLECT_101,
    "wrap": cv2.BORDER_WRAP,
}

_MORPH_SHAPES = {
    "rect": cv2.MORPH_RECT,
    "ellipse": cv2.MORPH_ELLIPSE,
    "cross": cv2.MORPH_CROSS,
}


# ---------------------------------------------------------------------------
# 基础访问
# ---------------------------------------------------------------------------
def as_array(image: ImageData | np.ndarray | None, label: str = "输入图像") -> np.ndarray:
    """取出 ndarray，并做统一的形状/通道检查。"""
    if image is None:
        raise OperatorError(f"缺少输入图像：{label}")
    array = image.array if isinstance(image, ImageData) else image
    array = np.asarray(array)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim not in (2, 3):
        raise OperatorError(f"{label} 必须是二维灰度图或三维彩色图，当前形状为 {array.shape}。")
    if array.ndim == 3 and array.shape[2] not in (3, 4):
        raise OperatorError(
            f"{label} 仅支持灰度、BGR 或 BGRA 图像，当前有 {array.shape[2]} 个通道。"
        )
    if array.size == 0:
        raise OperatorError(f"{label} 为空图像，无法处理。")
    return array


def channel_count(array: np.ndarray) -> int:
    return 1 if array.ndim == 2 else int(array.shape[2])


def color_space(array: np.ndarray) -> str:
    channels = channel_count(array)
    if channels == 1:
        return "L"
    if channels == 3:
        return "BGR"
    if channels == 4:
        return "BGRA"
    return f"CH{channels}"


def native_array(array: np.ndarray, label: str = "输入图像") -> np.ndarray:
    """把不受支持的 dtype 转成 float32，保持 OpenCV 调用可控。"""
    if array.dtype in NATIVE_DTYPES:
        return array
    try:
        return array.astype(np.float32)
    except (TypeError, ValueError) as exc:  # pragma: no cover - ndarray 总能转换
        raise OperatorError(f"{label} 的数据类型 {array.dtype} 无法转换为 float32。") from exc


def require_native(array: np.ndarray, label: str = "输入图像") -> None:
    if array.dtype not in NATIVE_DTYPES:
        raise OperatorError(
            f"{label} 的数据类型为 {array.dtype}，该算子仅支持 uint8 / uint16 / float32。"
        )


# ---------------------------------------------------------------------------
# 数值范围 / dtype 转换
# ---------------------------------------------------------------------------
def full_scale(array: np.ndarray) -> float:
    """返回「255 级参数」映射到该图像 dtype 时的满量程。"""
    dtype = array.dtype
    if dtype.kind == "b":
        return 1.0
    if dtype.kind in "ui":
        return float(np.iinfo(dtype).max)
    if dtype.kind == "f":
        if array.size == 0:
            return 1.0
        finite = np.nan_to_num(array.astype(np.float32, copy=False), nan=0.0, posinf=0.0, neginf=0.0)
        low = float(finite.min())
        high = float(finite.max())
        if low >= 0.0 and high <= 1.0:
            return 1.0
        if low >= 0.0 and high <= 255.0:
            return 255.0
        return max(high, 1.0)
    return 255.0


def scale_value(array: np.ndarray, value_255: float) -> float:
    """把 0..255 的参数值换算为图像自身 dtype 的数值。"""
    return float(value_255) * full_scale(array) / 255.0


def cast_like(values: np.ndarray, dtype: np.dtype | type) -> np.ndarray:
    """按目标 dtype 做四舍五入与饱和，避免 wrap-around。"""
    dtype = np.dtype(dtype)
    if dtype.kind == "b":
        return np.asarray(values) > 0
    if dtype.kind in "ui":
        info = np.iinfo(dtype)
        work = np.nan_to_num(
            np.asarray(values, dtype=np.float64),
            nan=float(info.min),
            posinf=float(info.max),
            neginf=float(info.min),
        )
        return np.clip(np.rint(work), info.min, info.max).astype(dtype)
    return np.asarray(values).astype(dtype)


def to_uint8(array: np.ndarray, label: str = "输入图像") -> np.ndarray:
    """把任意支持 dtype 的图像映射到 uint8（供只支持 8 位的 OpenCV 接口使用）。"""
    if array.dtype == np.uint8:
        return array
    if array.dtype == np.uint16:
        return np.clip(np.rint(array.astype(np.float32) / 257.0), 0, 255).astype(np.uint8)
    if array.dtype.kind == "b":
        return (array.astype(np.uint8) * 255)

    work = np.nan_to_num(
        array.astype(np.float32, copy=False), nan=0.0, posinf=255.0, neginf=0.0
    )
    low = float(work.min()) if work.size else 0.0
    high = float(work.max()) if work.size else 0.0
    if low >= 0.0 and high <= 1.0:
        work = work * 255.0
    elif low < 0.0 or high > 255.0:
        span = high - low
        if span <= 1e-12:
            work = np.zeros_like(work) if low <= 0.0 else np.full_like(work, 255.0)
        else:
            work = (work - low) / span * 255.0
    return np.clip(np.rint(work), 0, 255).astype(np.uint8)


def gray_view(array: np.ndarray, label: str = "输入图像") -> np.ndarray:
    """把灰度/BGR/BGRA 统一成单通道；dtype 尽量保持不变。"""
    array = native_array(array, label)
    if array.ndim == 2:
        return array
    channels = array.shape[2]
    if channels == 3:
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if channels == 4:
        return cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
    raise OperatorError(f"{label} 仅支持灰度、BGR 或 BGRA 图像，当前有 {channels} 个通道。")


def gray_for_8bit(array: np.ndarray, label: str = "输入图像") -> np.ndarray:
    """8 位滤波/边缘检测（Canny、自适应阈值）使用的灰度图。"""
    return to_uint8(gray_view(array, label), label)


# ---------------------------------------------------------------------------
# 枚举 / 卷积核
# ---------------------------------------------------------------------------
def interpolation_flag(name: str) -> int:
    try:
        return _INTERPOLATIONS[str(name)]
    except KeyError as exc:
        raise OperatorError(f"不支持的插值方式：{name}") from exc


def border_mode(name: str) -> int:
    try:
        return _BORDERS[str(name)]
    except KeyError as exc:
        raise OperatorError(f"不支持的边界模式：{name}") from exc


def struct_kernel(shape: str, ksize: int) -> np.ndarray:
    if ksize < 1:
        raise OperatorError("核大小必须大于等于 1。")
    try:
        cv_shape = _MORPH_SHAPES[str(shape)]
    except KeyError as exc:
        raise OperatorError(f"不支持的核形状：{shape}") from exc
    return cv2.getStructuringElement(cv_shape, (int(ksize), int(ksize)))


# ---------------------------------------------------------------------------
# 颜色 / 绘制
# ---------------------------------------------------------------------------
def parse_hex_color(color: str) -> tuple[int, int, int, int]:
    text = str(color or "").strip()
    if not text.startswith("#") or len(text) not in (7, 9):
        raise OperatorError(f"颜色必须是 #RRGGBB 或 #RRGGBBAA 格式，当前为 {color!r}。")
    try:
        red = int(text[1:3], 16)
        green = int(text[3:5], 16)
        blue = int(text[5:7], 16)
        alpha = int(text[7:9], 16) if len(text) == 9 else 255
    except ValueError as exc:
        raise OperatorError(f"颜色值无效：{color!r}。") from exc
    return red, green, blue, alpha


def _channel_value(array: np.ndarray, value_255: float) -> Any:
    if array.dtype.kind == "f":
        return float(scale_value(array, value_255))
    value = int(round(scale_value(array, value_255)))
    info = np.iinfo(array.dtype)
    return int(min(max(value, info.min), info.max))


def drawing_color(array: np.ndarray, color: str) -> Any:
    """把 #RRGGBB(AA) 颜色转换为与图像 dtype/通道数匹配的绘制颜色。"""
    red, green, blue, alpha = parse_hex_color(color)
    channels = channel_count(array)
    if channels == 1:
        luma = 0.299 * red + 0.587 * green + 0.114 * blue
        return _channel_value(array, luma)
    if channels == 3:
        return tuple(_channel_value(array, value) for value in (blue, green, red))
    return tuple(_channel_value(array, value) for value in (blue, green, red, alpha))


def line_type_for(array: np.ndarray) -> int:
    # 抗锯齿只对 8 位图像稳定支持，其余 dtype 使用普通线型。
    return cv2.LINE_AA if array.dtype == np.uint8 else cv2.LINE_8


# ---------------------------------------------------------------------------
# 多通道辅助
# ---------------------------------------------------------------------------
def split_channels(array: np.ndarray, label: str = "输入图像") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """返回 B/G/R 三个通道；BGRA 忽略 alpha。"""
    if array.ndim == 2 or array.shape[2] == 1:
        raise OperatorError(
            f"{label} 是灰度图，无法拆分通道；请连接彩色图像（BGR 或 BGRA）。"
        )
    if array.shape[2] == 4:
        array = array[:, :, :3]
    if array.shape[2] != 3:
        raise OperatorError(f"{label} 需要 3 或 4 个通道，当前为 {array.shape[2]}。")
    blue, green, red = cv2.split(array)
    return blue, green, red


def merge_channels(
    blue: np.ndarray,
    green: np.ndarray,
    red: np.ndarray,
    labels: Iterable[str] = ("蓝色通道", "绿色通道", "红色通道"),
) -> np.ndarray:
    """把三路单通道图像合成为 BGR；自动提升到公共 dtype。"""
    parts = []
    for label, array in zip(labels, (blue, green, red)):
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        if array.ndim != 2:
            raise OperatorError(f"{label} 必须是单通道灰度图，当前形状为 {array.shape}。")
        parts.append(array)
    shapes = {part.shape for part in parts}
    if len(shapes) != 1:
        raise OperatorError(
            "通道合并要求三路输入的宽高完全一致，当前尺寸为："
            + "、".join(f"{part.shape[1]}×{part.shape[0]}" for part in parts)
            + "。"
        )
    dtypes = [part.dtype for part in parts]
    if len(set(dtypes)) == 1:
        target = dtypes[0]
    elif any(dtype.kind == "f" for dtype in dtypes):
        target = np.dtype(np.float32)
    elif any(dtype == np.uint16 for dtype in dtypes):
        target = np.dtype(np.uint16)
    else:
        target = np.dtype(np.uint8)
    if target not in NATIVE_DTYPES:
        target = np.dtype(np.float32)
    return cv2.merge([part.astype(target, copy=False) for part in parts])
