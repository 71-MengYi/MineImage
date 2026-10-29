"""图像数据容器、颜色约定与显示转换工具。

颜色约定
--------
内部图像统一采用 OpenCV 习惯：

* 灰度：``(H, W)`` 或 ``(H, W, 1)``
* 彩色：``(H, W, 3)``，通道顺序 **BGR**
* 带透明：``(H, W, 4)``，通道顺序 **BGRA**

显示层会转换为 RGB，状态栏则会以 R/G/B/A 的顺序展示原始数值。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np

_ACCEPTED_DTYPES = {
    np.dtype(np.uint8),
    np.dtype(np.uint16),
    np.dtype(np.int8),
    np.dtype(np.int16),
    np.dtype(np.int32),
    np.dtype(np.float16),
    np.dtype(np.float32),
    np.dtype(np.float64),
}


def infer_color_space(array: np.ndarray) -> str:
    """根据 ndim / 通道数推断颜色空间。"""
    if array.ndim == 2:
        return "L"
    if array.ndim == 3:
        channels = array.shape[2]
        if channels == 1:
            return "L"
        if channels == 3:
            return "BGR"
        if channels == 4:
            return "BGRA"
        return f"CH{channels}"
    raise ValueError(f"不支持的图像形状：{array.shape}")


def channel_names(color_space: str) -> tuple[str, ...]:
    if color_space == "L":
        return ("Gray",)
    if color_space == "BGR":
        return ("R", "G", "B")
    if color_space == "BGRA":
        return ("R", "G", "B", "A")
    if color_space.startswith("CH"):
        try:
            count = int(color_space[2:])
        except ValueError:
            count = 1
        return tuple(f"C{i}" for i in range(count))
    return ()


@dataclass
class ImageData:
    """不可变语义的图像容器（数组本身请不要被算子原地修改）。"""

    array: np.ndarray
    name: str = "image"
    color_space: str | None = None

    def __post_init__(self) -> None:
        array = np.asarray(self.array)
        if array.ndim not in (2, 3):
            raise ValueError(f"图像必须是 2D 或 3D 数组，收到 {array.shape}")
        if array.ndim == 3 and array.shape[2] == 1:
            array = array[:, :, 0]
        if array.dtype not in _ACCEPTED_DTYPES:
            # 未知类型统一提升为 float32，避免显示层崩溃。
            array = array.astype(np.float32)
        self.array = array
        if not self.color_space:
            self.color_space = infer_color_space(array)

    # ------------------------------------------------------------------ 属性
    @property
    def width(self) -> int:
        return int(self.array.shape[1])

    @property
    def height(self) -> int:
        return int(self.array.shape[0])

    @property
    def channels(self) -> int:
        if self.array.ndim == 2:
            return 1
        return int(self.array.shape[2])

    @property
    def dtype_name(self) -> str:
        return str(self.array.dtype)

    @property
    def is_gray(self) -> bool:
        return self.color_space == "L" or self.channels == 1

    def summary(self) -> str:
        return f"{self.width}×{self.height} · {self.color_space} · {self.dtype_name}"

    def copy(self) -> "ImageData":
        return ImageData(self.array.copy(), self.name, self.color_space)

    def with_array(self, array: np.ndarray, name: str | None = None) -> "ImageData":
        return ImageData(array, name if name is not None else self.name)


@dataclass(frozen=True)
class PixelSample:
    """状态栏使用的像素采样结果；数值为原始 dtype。"""

    row: int
    col: int
    values: tuple[float, ...]
    channels: tuple[str, ...]
    in_bounds: bool
    dtype_name: str = ""

    def value_text(self) -> str:
        if not self.in_bounds:
            return "--"
        parts = []
        for channel, value in zip(self.channels, self.values):
            if isinstance(value, float) and not value.is_integer():
                parts.append(f"{channel}={value:.4g}")
            else:
                parts.append(f"{channel}={int(value)}")
        return "  ".join(parts)


def _normalize_to_uint8(array: np.ndarray) -> np.ndarray:
    """把任意支持 dtype 的图像缩放到 uint8（保留空间布局）。"""
    if array.dtype == np.uint8:
        return array
    if array.dtype == np.bool_:
        return (array.astype(np.uint8) * 255)
    if array.dtype == np.uint16:
        return np.clip(array.astype(np.float32) / 257.0 + 0.5, 0, 255).astype(np.uint8)
    if array.dtype == np.int8:
        return np.clip(array.astype(np.int16) + 128, 0, 255).astype(np.uint8)

    work = np.nan_to_num(array.astype(np.float32), nan=0.0, posinf=255.0, neginf=0.0)
    if work.size == 0:
        return np.zeros(array.shape, dtype=np.uint8)
    if work.dtype.kind == "f" and work.min() >= 0.0 and work.max() <= 1.0:
        work = work * 255.0
    else:
        lo = float(work.min())
        hi = float(work.max())
        if hi <= lo:
            work = np.zeros_like(work) if lo <= 0 else np.full_like(work, 255.0)
        else:
            work = (work - lo) / (hi - lo) * 255.0
    return np.clip(work, 0, 255).astype(np.uint8)


def to_display_rgb(image: ImageData | np.ndarray, max_dim: int | None = None) -> np.ndarray:
    """生成用于屏幕显示的 RGB uint8 数组 ``(H, W, 3)``。

    * uint16 按 65535→255 线性映射；
    * float 若处于 [0,1] 则视为归一化图像，否则按数据 min/max 拉伸；
    * 带 alpha 的图像显示时忽略 alpha；
    * ``max_dim`` 只影响显示分辨率，不改变引擎缓存。
    """
    array = image.array if isinstance(image, ImageData) else np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    rgb = _normalize_to_uint8(array)

    if rgb.ndim == 3 and rgb.shape[2] == 1:
        rgb = rgb[:, :, 0]
    if rgb.ndim == 2:
        rgb = np.repeat(rgb[:, :, None], 3, axis=2)
    elif rgb.ndim == 3 and rgb.shape[2] == 3:
        rgb = rgb[:, :, ::-1]
    elif rgb.ndim == 3 and rgb.shape[2] >= 4:
        rgb = rgb[:, :, 2::-1]
    else:
        raise ValueError(f"无法显示形状为 {array.shape} 的图像")

    rgb = np.ascontiguousarray(rgb)
    if max_dim and max_dim > 0:
        height, width = rgb.shape[:2]
        longest = max(height, width)
        if longest > max_dim:
            scale = max_dim / float(longest)
            new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            rgb = cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(rgb)


#: imagesc 显示模式使用的色带；优先 MATLAB 默认的 parula。
_DISPLAY_COLORMAP = getattr(cv2, "COLORMAP_PARULA", cv2.COLORMAP_JET)


def image_intensity(image: ImageData | np.ndarray) -> np.ndarray:
    """返回单通道 ``float32`` 强度图，供 imagesc / mesh 显示模式使用。

    彩色按 BGR 亮度权重（0.114 B + 0.587 G + 0.299 R）折算；alpha 忽略；
    NaN / Inf 归零，避免归一化时污染整幅图。
    """

    array = image.array if isinstance(image, ImageData) else np.asarray(image)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim == 2:
        work = array.astype(np.float32)
    elif array.ndim == 3:
        if array.shape[2] >= 3:
            blue = array[:, :, 0].astype(np.float32)
            green = array[:, :, 1].astype(np.float32)
            red = array[:, :, 2].astype(np.float32)
            work = blue * 0.114 + green * 0.587 + red * 0.299
        else:
            work = array[:, :, 0].astype(np.float32)
    else:
        raise ValueError(f"无法解析形状为 {array.shape} 的图像强度")
    return np.nan_to_num(work, nan=0.0, posinf=0.0, neginf=0.0)


def display_colormap_rgb(values: np.ndarray) -> np.ndarray:
    """把 0–1 的强度数组映射为 RGB uint8（imagesc / mesh 共用色带）。"""

    normalized = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    scaled = np.clip(normalized * 255.0 + 0.5, 0, 255).astype(np.uint8)
    colored = cv2.applyColorMap(scaled, _DISPLAY_COLORMAP)
    return np.ascontiguousarray(colored[:, :, ::-1])


def intensity_grid(image: ImageData | np.ndarray, max_side: int = 64) -> np.ndarray:
    """把强度图按块平均降采样，最长边不超过 ``max_side``。

    mesh 模式用它生成高度场：块平均能压住单像素噪声 / 高光造成的尖刺，
    比直接抽点更接近 MATLAB ``mesh`` 在平滑数据上的观感。
    """

    work = image_intensity(image)
    height, width = work.shape[:2]
    if height <= 0 or width <= 0:
        return work
    limit = max(2, int(max_side))
    longest = max(height, width)
    if longest <= limit:
        return work
    scale = limit / float(longest)
    target = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(work, target, interpolation=cv2.INTER_AREA)



def to_scaled_rgb(image: ImageData | np.ndarray, max_dim: int | None = None) -> np.ndarray:
    """生成 imagesc 模式使用的伪彩色 RGB uint8 数组 ``(H, W, 3)``。

    与 :func:`to_display_rgb` 不同，这里把整幅图的强度按 **数据 min/max**
    拉伸到 0–255 后再套用色带（对应 MATLAB ``imagesc`` + colorbar 的语义），
    因此数值范围很窄的浮点图也能看清结构。``max_dim`` 只影响显示分辨率。
    """

    work = image_intensity(image)
    if work.size == 0:
        height = 0 if work.ndim < 1 else int(work.shape[0])
        width = 0 if work.ndim < 2 else int(work.shape[1])
        return np.zeros((max(0, height), max(0, width), 3), dtype=np.uint8)
    low = float(np.min(work))
    high = float(np.max(work))
    if high > low:
        work = (work - low) / (high - low)
    else:
        work = np.zeros_like(work)
    work = np.clip(work, 0.0, 1.0)
    if max_dim and max_dim > 0:
        height, width = work.shape[:2]
        longest = max(height, width)
        if longest > max_dim:
            scale = max_dim / float(longest)
            new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            work = cv2.resize(work, new_size, interpolation=cv2.INTER_AREA)
    return display_colormap_rgb(work)


def make_thumbnail(image: ImageData | np.ndarray, box: int = 72) -> np.ndarray:
    """生成缩略图 RGB uint8，保持比例，最长边不超过 ``box``。"""
    rgb = to_display_rgb(image)
    height, width = rgb.shape[:2]
    if height == 0 or width == 0:
        return rgb
    scale = min(box / width, box / height)
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def sample_pixel(image: ImageData, row: int, col: int) -> PixelSample:
    """读取原始像素值；彩色按 R/G/B/A 顺序返回。"""
    array = image.array
    height, width = array.shape[:2]
    if row < 0 or col < 0 or row >= height or col >= width:
        return PixelSample(row, col, (), (), False, image.dtype_name)

    if array.ndim == 2:
        values = (float(array[row, col]),)
        names = ("Gray",)
    else:
        raw = array[row, col]
        if array.shape[2] >= 4:
            values = (float(raw[2]), float(raw[1]), float(raw[0]), float(raw[3]))
            names = ("R", "G", "B", "A")
        elif array.shape[2] == 3:
            values = (float(raw[2]), float(raw[1]), float(raw[0]))
            names = ("R", "G", "B")
        else:
            values = tuple(float(v) for v in np.atleast_1d(raw))
            names = channel_names(infer_color_space(array))
    return PixelSample(row, col, values, names, True, image.dtype_name)


def dtype_full_scale(dtype: np.dtype) -> float:
    """返回整数 dtype 的最大值，用于取反等算子。"""
    dtype = np.dtype(dtype)
    if dtype.kind in "ui":
        return float(np.iinfo(dtype).max)
    if dtype.kind == "b":
        return 1.0
    return 1.0


def ensure_odd(value: int) -> int:
    value = int(value)
    return value if value % 2 == 1 else value + 1
