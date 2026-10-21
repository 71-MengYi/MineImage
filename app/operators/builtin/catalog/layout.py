"""OpenCV 布局目录算子（layout 组）。

本模块提供 5 个面向展示与对比的布局算子：并排 / 上下拼接、叠加、缩略图网格、
边框与四宫格预览。

约定
----
* 所有算子只返回图像，并保持第一幅输入的 dtype 与通道数；BGRA 的 alpha 会保留；
* 需要第二幅图像时使用 ``optional=True`` 端口，未连接时回退到第一幅图像，
  保证默认参数直接可运行；
* 拼接与缩略图在 uint8 上完成排版，再按原 dtype 还原量程；
* 所有错误使用 :class:`app.operators.spec.OperatorError` 的中文提示。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.operators.builtin.catalog import _cv
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec

SPACES = ("L", "BGR", "BGRA")

#: 拼接对齐方式。
_ALIGNS = (
    ("center", "居中"),
    ("start", "靠上/靠左"),
)


# ---------------------------------------------------------------------------
# 本地辅助：通道 / dtype 归一化
# ---------------------------------------------------------------------------
def _scale_u8(values: np.ndarray, template: np.ndarray) -> np.ndarray:
    """把 uint8 结果映射回 ``template`` 的 dtype 与数值量程。

    与 :func:`_cv.from_unit` 不同，这里只参考 ``template`` 的 dtype / 满量程，
    不要求形状一致——布局算子会改变输出尺寸。
    """
    values = np.ascontiguousarray(values)
    dtype = template.dtype
    if dtype == np.uint8:
        return values
    if dtype.kind in "ui":
        scale = float(np.iinfo(dtype).max) / 255.0
    elif dtype.kind == "f":
        scale = _cv.full_scale(template) / 255.0
    else:
        return values.astype(dtype, copy=False)
    return np.ascontiguousarray(_cv.cast_like(values.astype(np.float32) * scale, dtype))


def _to_u8_channels(array: np.ndarray, target_channels: int) -> np.ndarray:
    """把任意图像转换为 ``target_channels`` 通道的 uint8 数组。"""
    work = _cv.to_uint8(array)
    alpha: np.ndarray | None = None
    if work.ndim == 3 and work.shape[2] == 4:
        alpha = work[:, :, 3]
        work = work[:, :, :3]
    if target_channels == 1:
        if work.ndim == 3:
            work = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        return np.ascontiguousarray(work)
    if work.ndim == 2:
        work = cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
    work = np.ascontiguousarray(work)
    if target_channels == 3:
        return work
    if alpha is None:
        alpha = np.full(work.shape[:2], 255, dtype=np.uint8)
    return np.dstack([work, alpha])


def _match_channels(array: np.ndarray, target_channels: int) -> np.ndarray:
    """把数组归一化到目标通道数，尽量保持 dtype（用于逐像素叠加）。"""
    channels = _cv.channels(array)
    if channels == target_channels:
        return array
    if target_channels == 1:
        if channels == 4:
            return cv2.cvtColor(array, cv2.COLOR_BGRA2GRAY)
        return cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
    if channels == 1:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if channels == 4:
        return array[:, :, :3]
    if channels == 3:
        if array.dtype.kind in "ui":
            fill = np.iinfo(array.dtype).max
        elif array.dtype.kind == "f":
            fill = 1.0
        else:
            fill = 255
        alpha = np.full(array.shape[:2], fill, dtype=array.dtype)
        return np.dstack([array, alpha])
    raise OperatorError(f"不支持 {channels} 通道的图像。")  # pragma: no cover


def _blank_canvas(height: int, width: int, channels: int) -> np.ndarray:
    """按通道数创建 uint8 画布；单通道使用二维数组，与灰度工作图一致。"""
    shape = (height, width) if channels == 1 else (height, width, channels)
    return np.zeros(shape, dtype=np.uint8)


def _fill_alpha(canvas: np.ndarray) -> None:
    if canvas.ndim == 3 and canvas.shape[2] == 4:
        canvas[:, :, 3] = 255


def _finish_u8(template: np.ndarray, work: np.ndarray) -> np.ndarray:
    """把 uint8 排版结果写回第一幅输入的 dtype 与通道数。"""
    channels = _cv.channels(template)
    if channels == 1:
        gray = work if work.ndim == 2 else cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        return _scale_u8(gray, template)
    bgr = work if work.ndim == 3 else cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
    bgr = np.ascontiguousarray(bgr[:, :, :3])
    if channels == 3:
        return _scale_u8(bgr, template)
    color = _scale_u8(bgr, template[:, :, :3])
    alpha = _scale_u8(np.ascontiguousarray(work[:, :, 3]), template[:, :, 3])
    return np.dstack([color, alpha])


# ---------------------------------------------------------------------------
# 拼接 / 叠加
# ---------------------------------------------------------------------------
class Montage(Operator):
    """把两幅图像并排或上下拼接。"""

    spec = OperatorSpec(
        id="layout.montage",
        name="并排拼接",
        category=("交互处理", "布局与对比"),
        description=(
            "把两幅图像按水平（并排）或垂直（上下）方向拼接成一张对比图，可设置"
            "间隔与对齐方式。未连接第二幅图像时重复第一幅图像。"
        ),
        keywords=("拼接", "montage", "并排", "对比", "对比图", "layout"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec(
                "image2",
                "第二幅图像（可选）",
                optional=True,
                description="未连接时重复第一幅图像。",
            ),
        ),
        params=(
            ParamSpec(
                "direction",
                "拼接方向",
                "enum",
                choices=(("horizontal", "水平并排"), ("vertical", "垂直上下")),
                default="horizontal",
            ),
            ParamSpec(
                "gap",
                "间隔",
                "int",
                default=8,
                min=0,
                max=100,
                step=1,
                unit="px",
            ),
            ParamSpec(
                "align",
                "对齐方式",
                "enum",
                choices=_ALIGNS,
                default="center",
            ),
        ),
    )

    def process(self, image=None, image2=None, direction="horizontal", gap=8, align="center", **_: Any):
        template = _cv.source(image)
        channels = _cv.channels(template)
        first = _to_u8_channels(template, channels)
        second_input = _cv.optional_second(image2, template, "第二幅图像")
        second = _to_u8_channels(second_input, channels)
        first_h, first_w = first.shape[:2]
        second_h, second_w = second.shape[:2]
        gap = max(0, int(gap))
        if direction == "horizontal":
            out_h = max(first_h, second_h)
            out_w = first_w + second_w + gap
        else:
            out_h = first_h + second_h + gap
            out_w = max(first_w, second_w)
        canvas = _blank_canvas(out_h, out_w, channels)
        _fill_alpha(canvas)
        if direction == "horizontal":
            y1 = 0 if align == "start" else (out_h - first_h) // 2
            y2 = 0 if align == "start" else (out_h - second_h) // 2
            canvas[y1 : y1 + first_h, 0:first_w] = first
            canvas[y2 : y2 + second_h, first_w + gap : first_w + gap + second_w] = second
        else:
            x1 = 0 if align == "start" else (out_w - first_w) // 2
            x2 = 0 if align == "start" else (out_w - second_w) // 2
            canvas[0:first_h, x1 : x1 + first_w] = first
            canvas[first_h + gap : first_h + gap + second_h, x2 : x2 + second_w] = second
        return _finish_u8(template, canvas)


class Overlay(Operator):
    """按混合 / 差值 / 变亮等方式叠加两幅图像。"""

    spec = OperatorSpec(
        id="layout.overlay",
        name="图像叠加",
        category=("交互处理", "布局与对比"),
        description=(
            "把第二幅图像与第一幅按混合、差值、变亮、变暗或正片叠底方式叠加，"
            "用于变化检测或效果对比。未连接第二幅图像时重复第一幅图像。"
        ),
        keywords=("叠加", "overlay", "混合", "差值", "对比", "blend"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec(
                "image2",
                "第二幅图像（可选）",
                optional=True,
                description="未连接时重复第一幅图像。",
            ),
        ),
        params=(
            ParamSpec(
                "mode",
                "叠加方式",
                "enum",
                choices=(
                    ("blend", "混合"),
                    ("difference", "差值"),
                    ("lighten", "变亮"),
                    ("darken", "变暗"),
                    ("multiply", "正片叠底"),
                ),
                default="blend",
            ),
            ParamSpec(
                "opacity",
                "叠加不透明度",
                "float",
                default=0.5,
                min=0.0,
                max=1.0,
                step=0.05,
            ),
        ),
    )

    def process(self, image=None, image2=None, mode="blend", opacity=0.5, **_: Any):
        template = _cv.source(image)
        color_template = template[:, :, :3] if _cv.channels(template) == 4 else template
        alpha = template[:, :, 3] if _cv.channels(template) == 4 else None
        second_input = _cv.optional_second(image2, template, "第二幅图像")
        second = _match_channels(_cv.source(second_input, "第二幅图像"), _cv.channels(color_template))
        base_unit = _cv.unit_array(color_template).astype(np.float32)
        second_unit = _cv.unit_array(second).astype(np.float32)
        if mode == "blend":
            combined = second_unit
        elif mode == "difference":
            combined = np.abs(base_unit - second_unit)
        elif mode == "lighten":
            combined = np.maximum(base_unit, second_unit)
        elif mode == "darken":
            combined = np.minimum(base_unit, second_unit)
        else:
            combined = base_unit * second_unit
        result = base_unit * (1.0 - float(opacity)) + combined * float(opacity)
        result = np.clip(result, 0.0, 1.0)
        color = _cv.from_unit(result, color_template)
        return color if alpha is None else np.dstack([color, alpha])


# ---------------------------------------------------------------------------
# 缩略图网格 / 边框 / 四宫格
# ---------------------------------------------------------------------------
class ContactSheet(Operator):
    """把图像切成缩略图并排成网格。"""

    spec = OperatorSpec(
        id="layout.contact_sheet",
        name="缩略图网格",
        category=("交互处理", "布局与对比"),
        description=(
            "把图像切成 rows×cols 个局部块，缩放后统一排成缩略图网格，适合快速"
            "浏览局部细节。每块保持原始宽高比并居中放置。"
        ),
        keywords=("缩略图", "contact sheet", "网格", "预览", "分块"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec(
                "columns",
                "列数",
                "int",
                default=3,
                min=1,
                max=8,
                step=1,
            ),
            ParamSpec(
                "rows",
                "行数",
                "int",
                default=3,
                min=1,
                max=8,
                step=1,
            ),
            ParamSpec(
                "cell",
                "单元格尺寸",
                "int",
                default=128,
                min=32,
                max=512,
                step=8,
                unit="px",
            ),
            ParamSpec(
                "gap",
                "间隔",
                "int",
                default=8,
                min=0,
                max=64,
                step=1,
                unit="px",
            ),
            ParamSpec("background", "背景颜色", "color", default="#202020"),
        ),
    )

    def process(
        self,
        image=None,
        columns=3,
        rows=3,
        cell=128,
        gap=8,
        background="#202020",
        **_: Any,
    ):
        template = _cv.source(image)
        channels = _cv.channels(template)
        work = _to_u8_channels(template, channels)
        height, width = work.shape[:2]
        cols = max(1, int(columns))
        row_count = max(1, int(rows))
        cell_size = max(8, int(cell))
        gap = max(0, int(gap))
        out_w = cols * cell_size + (cols - 1) * gap
        out_h = row_count * cell_size + (row_count - 1) * gap
        canvas = _blank_canvas(out_h, out_w, channels)
        canvas[:] = _cv.draw_color(canvas, background)
        _fill_alpha(canvas)
        for row in range(row_count):
            y0 = row * height // row_count
            y1 = (row + 1) * height // row_count
            for col in range(cols):
                x0 = col * width // cols
                x1 = (col + 1) * width // cols
                tile = work[y0:y1, x0:x1]
                if tile.size == 0:
                    continue
                tile_h, tile_w = tile.shape[:2]
                scale = min(cell_size / max(tile_w, 1), cell_size / max(tile_h, 1))
                new_w = max(1, int(round(tile_w * scale)))
                new_h = max(1, int(round(tile_h * scale)))
                interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
                thumb = cv2.resize(tile, (new_w, new_h), interpolation=interpolation)
                origin_x = col * (cell_size + gap) + (cell_size - new_w) // 2
                origin_y = row * (cell_size + gap) + (cell_size - new_h) // 2
                canvas[origin_y : origin_y + new_h, origin_x : origin_x + new_w] = thumb
        return _finish_u8(template, canvas)


class Border(Operator):
    """给图像添加纯色边框。"""

    spec = OperatorSpec(
        id="layout.border",
        name="添加边框",
        category=("交互处理", "布局与对比"),
        description=(
            "在图像四周添加指定宽度与颜色的边框，可选用双层边框样式，适合排版"
            "拼图或做展示留白。"
        ),
        keywords=("边框", "border", "画框", "留白", "排版"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec(
                "width",
                "边框宽度",
                "int",
                default=12,
                min=0,
                max=200,
                step=1,
                unit="px",
            ),
            ParamSpec("color", "边框颜色", "color", default="#ffffff"),
            ParamSpec(
                "style",
                "边框样式",
                "enum",
                choices=(("solid", "单层"), ("double", "双层")),
                default="solid",
            ),
        ),
    )

    def process(self, image=None, width=12, color="#ffffff", style="solid", **_: Any):
        template = _cv.source(image)
        channels = _cv.channels(template)
        work = _to_u8_channels(template, channels)
        height, w = work.shape[:2]
        border = max(0, int(width))
        out_h = height + border * 2
        out_w = w + border * 2
        canvas = _blank_canvas(out_h, out_w, channels)
        canvas[:] = _cv.draw_color(canvas, color)
        _fill_alpha(canvas)
        canvas[border : border + height, border : border + w] = work
        if style == "double" and border >= 3:
            inner = max(1, border // 3)
            inset = max(0, border // 2)
            cv2.rectangle(
                canvas,
                (inset, inset),
                (out_w - 1 - inset, out_h - 1 - inset),
                _cv.draw_color(canvas, color),
                inner,
                _cv.draw_type(canvas),
            )
        return _finish_u8(template, canvas)


class Split(Operator):
    """把图像切成四块并留出间隔，形成四宫格预览。"""

    spec = OperatorSpec(
        id="layout.split",
        name="四宫格预览",
        category=("交互处理", "布局与对比"),
        description=(
            "把图像按 2×2 切成四块并留出间隔重新排布，形成四宫格预览，方便观察"
            "局部细节。间隔为 0 时输出尺寸与输入完全相同。"
        ),
        keywords=("四宫格", "split", "切块", "预览", "2x2"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec(
                "gap",
                "间隔",
                "int",
                default=6,
                min=0,
                max=64,
                step=1,
                unit="px",
            ),
            ParamSpec("color", "间隔颜色", "color", default="#ffffff"),
        ),
    )

    def process(self, image=None, gap=6, color="#ffffff", **_: Any):
        template = _cv.source(image)
        channels = _cv.channels(template)
        work = _to_u8_channels(template, channels)
        height, width = work.shape[:2]
        gap = max(0, int(gap))
        top = height // 2
        left = width // 2
        canvas = _blank_canvas(height + gap, width + gap, channels)
        canvas[:] = _cv.draw_color(canvas, color)
        _fill_alpha(canvas)
        canvas[0:top, 0:left] = work[0:top, 0:left]
        canvas[0:top, left + gap : width + gap] = work[0:top, left:width]
        canvas[top + gap : height + gap, 0:left] = work[top:height, 0:left]
        canvas[top + gap : height + gap, left + gap : width + gap] = work[top:height, left:width]
        return _finish_u8(template, canvas)


#: 模块导出的算子类；loader 支持一个模块注册多个算子。
OPERATORS = (
    Montage,
    Overlay,
    ContactSheet,
    Border,
    Split,
)
