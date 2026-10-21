"""OpenCV 5.0 算子目录共享工具。

``app/operators/builtin/catalog/`` 下的算子文件很多，它们统一通过本模块做输入
校验、dtype 归一化和输出恢复，保证对灰度 / BGR / BGRA 以及
uint8 / uint16 / float32 的行为一致。

本文件名以下划线开头，算子扫描器会跳过它，可以安全存放共享代码。
"""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import cv2
import numpy as np

from app.operators.builtin._utils import (
    as_array,
    border_mode,
    cast_like,
    channel_count,
    color_space as color_space_of,
    drawing_color,
    full_scale,
    gray_for_8bit,
    gray_view,
    interpolation_flag,
    line_type_for,
    native_array,
    parse_hex_color,
    scale_value,
    struct_kernel,
    to_uint8,
)
from app.operators.spec import OperatorError

#: 支持全部颜色空间的端口声明（图像端口默认值）。
ALL_SPACES = ("L", "BGR", "BGRA")
#: 只接受彩色图像。
COLOR_SPACES = ("BGR", "BGRA")
#: 只接受灰度图像。
GRAY_SPACES = ("L",)

#: 0..255 参数在整型图像上使用的满量程。
U8_SCALE = 255.0


# ---------------------------------------------------------------------------
# 输入读取：所有算子都从这里拿数组
# ---------------------------------------------------------------------------
def source(image, label: str = "输入图像") -> np.ndarray:
    """读取输入并转换为 OpenCV 原生 dtype（uint8/uint16/float32）。"""
    return native_array(as_array(image, label), label)


def source_gray(image, label: str = "输入图像") -> np.ndarray:
    """读取输入并转为单通道，尽量保持 dtype。"""
    return gray_view(as_array(image, label), label)


def source_gray8(image, label: str = "输入图像") -> np.ndarray:
    """读取输入并转为 8 位灰度（只支持 8 位的 OpenCV 接口使用）。"""
    return gray_for_8bit(as_array(image, label), label)


def source_u8(image, label: str = "输入图像") -> np.ndarray:
    """读取输入并映射到 uint8，保持通道数。"""
    return to_uint8(as_array(image, label), label)


def source_bgr(image, label: str = "输入图像") -> np.ndarray:
    """读取输入并返回同 dtype 的 BGR 三通道数组（BGRA 去掉 alpha）。"""
    array = source(image, label)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.shape[2] == 4:
        return array[:, :, :3].copy()
    if array.shape[2] == 3:
        return array.copy()
    raise OperatorError(f"{label} 仅支持灰度、BGR 或 BGRA 图像，当前有 {array.shape[2]} 个通道。")


def source_bgr8(image, label: str = "输入图像") -> np.ndarray:
    """读取输入并返回 uint8 BGR，适合绘图与可视化。"""
    return to_uint8(source_bgr(image, label), label)


def source_alpha(image, label: str = "输入图像") -> np.ndarray | None:
    """取 BGRA 的 alpha 通道；没有 alpha 时返回 ``None``。"""
    array = source(image, label)
    if array.ndim == 3 and array.shape[2] == 4:
        return array[:, :, 3].copy()
    return None


def require_color(image, label: str = "输入图像") -> np.ndarray:
    """要求彩色图像，返回同 dtype 的 BGR 三通道数组。"""
    array = source(image, label)
    if array.ndim == 3 and array.shape[2] >= 3:
        return source_bgr(array, label)
    raise OperatorError(f"{label} 需要彩色图像（BGR/BGRA），当前为灰度图。")


def same_size(reference: np.ndarray, other: np.ndarray, label: str = "第二幅图像") -> None:
    """要求两幅图宽高一致，否则给出中文错误。"""
    if reference.shape[:2] != other.shape[:2]:
        raise OperatorError(
            f"{label}的尺寸为 {other.shape[1]}×{other.shape[0]}，"
            f"与参考图像 {reference.shape[1]}×{reference.shape[0]} 不一致。"
        )


def optional_second(image, fallback: np.ndarray, label: str = "第二幅图像") -> np.ndarray:
    """可选第二输入；未连接时使用 ``fallback``（通常是第一幅图的派生图）。"""
    if image is None:
        return fallback
    array = source(image, label)
    same_size(fallback, array, label)
    return array


# ---------------------------------------------------------------------------
# 输出恢复：逐像素运算结果写回输入 dtype
# ---------------------------------------------------------------------------
def restore(result: np.ndarray, template: np.ndarray, label: str = "结果") -> np.ndarray:
    """把同形状的数值结果转换为 ``template`` 的 dtype。

    * 整型：四舍五入 + 饱和裁剪，避免溢出回绕；
    * float32/float64：转换为 float32；
    * 形状不一致视为算子错误。
    """
    src = np.asarray(template)
    res = np.asarray(result)
    if res.shape != src.shape:
        raise OperatorError(
            f"{label}的形状 {res.shape} 与输入 {src.shape} 不一致。"
        )
    if res.dtype == src.dtype:
        return res
    if src.dtype.kind in "ui":
        return cast_like(res, src.dtype)
    if src.dtype.kind == "f":
        return res.astype(src.dtype, copy=False)
    if src.dtype.kind == "b":
        return res > 0
    return res.astype(src.dtype, copy=False)


def unit_array(array: np.ndarray) -> np.ndarray:
    """把图像归一化到 float32 ``0..1``，用于加权、混合等逐像素计算。"""
    work = np.asarray(array)
    if work.dtype.kind in "ui":
        scale = float(np.iinfo(work.dtype).max)
    elif work.dtype.kind == "b":
        scale = 1.0
    else:
        scale = full_scale(work)
    return np.clip(work.astype(np.float32) / max(scale, 1e-12), 0.0, 1.0)


def from_unit(values: np.ndarray, template: np.ndarray) -> np.ndarray:
    """把 ``0..1`` 的 float 结果写回 ``template`` 的 dtype。"""
    scale = full_scale(template)
    result = np.asarray(values, dtype=np.float32) * scale
    return restore(result, template)


def clip_uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 颜色、绘制辅助
# ---------------------------------------------------------------------------
def draw_color(array: np.ndarray, color: str) -> Any:
    """把 ``#RRGGBB`` 颜色转换为与数组 dtype / 通道数匹配的绘制值。"""
    return drawing_color(array, color)


def draw_type(array: np.ndarray) -> int:
    return line_type_for(array)


def hex_rgb(color: str) -> tuple[int, int, int]:
    red, green, blue, _alpha = parse_hex_color(color)
    return red, green, blue


def match_interpolation(name: str) -> int:
    return interpolation_flag(name)


def match_border(name: str) -> int:
    return border_mode(name)


def made_kernel(shape: str, ksize: int) -> np.ndarray:
    return struct_kernel(shape, ksize)


def channels(array: np.ndarray) -> int:
    return channel_count(array)


def space_of(array: np.ndarray) -> str:
    return color_space_of(array)


# ---------------------------------------------------------------------------
# 区域辅助：交互参数缺省时给出中心区域，保证默认参数可运行
# ---------------------------------------------------------------------------
def center_rect(width: int, height: int, fraction: float = 0.5) -> tuple[int, int, int, int]:
    """返回居中的 ``(x, y, w, h)``；``fraction`` 为占宽高的比例。"""
    fraction = min(max(float(fraction), 0.05), 1.0)
    rect_w = max(1, int(round(width * fraction)))
    rect_h = max(1, int(round(height * fraction)))
    x = max(0, (width - rect_w) // 2)
    y = max(0, (height - rect_h) // 2)
    return x, y, rect_w, rect_h


def resolve_rect(
    roi: Sequence[float] | None,
    width: int,
    height: int,
    *,
    fraction: float = 0.5,
    label: str = "矩形区域",
) -> tuple[int, int, int, int]:
    """把 ROI 参数解析为裁剪到图像范围内的整数矩形。"""
    if not roi or len(roi) != 4:
        return center_rect(width, height, fraction)
    x, y, rect_w, rect_h = (int(round(float(value))) for value in roi)
    if rect_w <= 0 or rect_h <= 0:
        raise OperatorError(f"{label}的宽高必须大于 0。")
    x0 = min(max(x, 0), max(width - 1, 0))
    y0 = min(max(y, 0), max(height - 1, 0))
    x1 = min(max(x + rect_w, x0 + 1), width)
    y1 = min(max(y + rect_h, y0 + 1), height)
    return x0, y0, x1 - x0, y1 - y0


def resolve_line(
    line: Sequence[float] | None,
    width: int,
    height: int,
) -> tuple[int, int, int, int]:
    """把直线参数解析为整数端点；缺省时使用图像对角线。"""
    if not line or len(line) != 4:
        return 0, 0, max(width - 1, 1), max(height - 1, 1)
    x1, y1, x2, y2 = (int(round(float(value))) for value in line)
    if (x1, y1) == (x2, y2):
        raise OperatorError("线段端点不能重合。")
    return x1, y1, x2, y2


def resolve_points(
    points: Sequence[Sequence[float]] | None,
    width: int,
    height: int,
    *,
    count: int = 0,
) -> list[tuple[int, int]]:
    """把点集参数解析为整数点；缺省时生成均匀分布的点。"""
    if not points:
        if count <= 0:
            return []
        margin_x = max(1, width // 8)
        margin_y = max(1, height // 8)
        return [
            (
                margin_x + int(round((width - 2 * margin_x) * index / max(count - 1, 1))),
                margin_y + int(round((height - 2 * margin_y) * index / max(count - 1, 1))),
            )
            for index in range(count)
        ]
    resolved = [(int(round(float(point[0]))), int(round(float(point[1])))) for point in points]
    if count and len(resolved) != count:
        raise OperatorError(f"需要 {count} 个点，当前为 {len(resolved)} 个。")
    return resolved


def text_panel(
    width: int,
    height: int,
    lines: Iterable[str],
    *,
    color: tuple[int, int, int] = (255, 255, 255),
    line_height: int = 22,
    font_scale: float = 0.55,
) -> np.ndarray:
    """生成深色背景的 BGR 文本面板，用于只输出数值 / 统计结果的算子预览。"""
    width = max(int(width), 120)
    height = max(int(height), 40)
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    y = 24
    for line in lines:
        if y > height - 4:
            break
        cv2.putText(
            canvas,
            str(line)[: max(width // 8, 8)],
            (8, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            float(font_scale),
            color,
            1,
            cv2.LINE_AA,
        )
        y += line_height
    return canvas


def stats_lines(values: Sequence[float], name: str = "数值") -> list[str]:
    """把一串数值整理成紧凑文本行（面板与调试用）。"""
    return [f"{name}：{float(value):.6g}" for value in values]


def safe_roi_for_mask(mask: np.ndarray | None, array: np.ndarray) -> np.ndarray:
    """取二值 mask；未提供时用 Otsu 从灰度图自动生成。"""
    if mask is not None:
        mask_array = source_gray(mask, "蒙版")
        same_size(array, mask_array, "蒙版")
        return (mask_array > 0).astype(np.uint8) * 255
    gray = source_gray8(array)
    _threshold, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary
