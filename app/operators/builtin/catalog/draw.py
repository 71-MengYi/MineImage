"""OpenCV 绘图目录算子（draw 组）。

本模块提供 12 个标注 / 可视化算子：矩形、圆、椭圆、多边形、文字、箭头、标记点、
十字线、网格、比例尺、水印与 ROI 高亮。

约定
----
* 交互参数（roi / points / line）缺省时使用图像中心区域、中心点或对角线，
  保证不传参数也能直接运行；
* 输出保持输入的 dtype 与通道数，BGRA 图像保留原 alpha 通道；
* 统一在 uint8 BGR 工作图上绘制，再按原 dtype 还原量程（uint16 ×257、float 按
  满量程缩放），因此 L / BGR / BGRA 与 uint8 / uint16 / float32 都能使用；
* 所有错误使用 :class:`app.operators.spec.OperatorError` 的中文提示。
"""

from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from app.operators.builtin.catalog import _cv
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec

SPACES = ("L", "BGR", "BGRA")
FONT = cv2.FONT_HERSHEY_SIMPLEX

#: draw.marker 的标记类型。
_MARKERS = {
    "cross": cv2.MARKER_CROSS,
    "tilted_cross": cv2.MARKER_TILTED_CROSS,
    "star": cv2.MARKER_STAR,
    "diamond": cv2.MARKER_DIAMOND,
    "square": cv2.MARKER_SQUARE,
    "triangle_up": cv2.MARKER_TRIANGLE_UP,
    "triangle_down": cv2.MARKER_TRIANGLE_DOWN,
}

#: 水印 / 比例尺的放置位置。
_POSITIONS = (
    ("top_left", "左上角"),
    ("top_right", "右上角"),
    ("bottom_left", "左下角"),
    ("bottom_right", "右下角"),
)


# ---------------------------------------------------------------------------
# 本地辅助：uint8 BGR 工作图 <-> 原始 dtype / 通道数
# ---------------------------------------------------------------------------
def _prepare(image, label: str = "输入图像") -> tuple[np.ndarray, np.ndarray]:
    """返回 ``(原始模板, uint8 BGR 工作图)``；工作图始终是独立副本。"""
    template = _cv.source(image, label)
    work = np.ascontiguousarray(_cv.source_bgr8(template, label))
    return template, work


def _scale_u8(values: np.ndarray, template: np.ndarray) -> np.ndarray:
    if template.dtype == np.uint8:
        return np.ascontiguousarray(values)
    unit = np.asarray(values, dtype=np.float32) / 255.0
    return np.ascontiguousarray(_cv.from_unit(unit, template))


def _finish(template: np.ndarray, work: np.ndarray) -> np.ndarray:
    """把绘制后的 uint8 工作图写回原始 dtype 与通道数；BGRA 保留 alpha。"""
    channels = _cv.channels(template)
    if channels == 1:
        gray = work if work.ndim == 2 else cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
        return _scale_u8(gray, template)
    bgr = work if work.ndim == 3 else cv2.cvtColor(work, cv2.COLOR_GRAY2BGR)
    if channels == 3:
        return _scale_u8(bgr, template)
    color = _scale_u8(np.ascontiguousarray(bgr), template[:, :, :3])
    return np.dstack([color, template[:, :, 3]])


def _center_point(points, width: int, height: int) -> tuple[int, int]:
    """点集参数为空时回退到图像中心。"""
    resolved = _cv.resolve_points(points, width, height)
    if resolved:
        return resolved[0]
    return width // 2, height // 2


def _text_scale(width: int, height: int, thickness: int, text: str) -> float:
    """计算能放进画面的字号，保证默认水印 / 比例尺可见。"""
    if not text:
        return 0.5
    (text_w, text_h), _baseline = cv2.getTextSize(text, FONT, 1.0, max(1, thickness))
    scale_w = width * 0.75 / max(text_w, 1)
    scale_h = height * 0.35 / max(text_h, 1)
    return max(0.2, min(scale_w, scale_h, 4.0))


# ---------------------------------------------------------------------------
# 矩形 / 圆 / 椭圆 / 多边形
# ---------------------------------------------------------------------------
class Rectangle(Operator):
    """在图像上绘制矩形。"""

    spec = OperatorSpec(
        id="draw.rectangle",
        name="绘制矩形",
        category=("交互处理", "基础图形"),
        description=(
            "在图像上绘制矩形边框或实心矩形，用于框选目标或强调区域。未指定区域时"
            "使用图像中心 50% 的大小。"
        ),
        keywords=("矩形", "rectangle", "框选", "标注", "draw"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec(
                "roi",
                "矩形区域",
                "roi",
                default=None,
                placeholder="[x, y, w, h]",
                tooltip="在预览中框选矩形；缺省使用图像中心区域。",
            ),
            ParamSpec("color", "颜色", "color", default="#ff3b30"),
            ParamSpec(
                "thickness",
                "线宽",
                "int",
                default=2,
                min=1,
                max=51,
                step=1,
                unit="px",
                visible_when="not filled",
                enabled_when="not filled",
            ),
            ParamSpec("filled", "填充", "bool", default=False),
        ),
    )

    def process(self, image=None, roi=None, color="#ff3b30", thickness=2, filled=False, **_: Any):
        template, work = _prepare(image)
        height, width = work.shape[:2]
        x, y, rect_w, rect_h = _cv.resolve_rect(roi, width, height, fraction=0.5)
        cv2.rectangle(
            work,
            (x, y),
            (x + rect_w - 1, y + rect_h - 1),
            _cv.draw_color(work, color),
            -1 if filled else int(thickness),
            _cv.draw_type(work),
        )
        return _finish(template, work)


class Circle(Operator):
    """在图像上绘制圆形。"""

    spec = OperatorSpec(
        id="draw.circle",
        name="绘制圆形",
        category=("交互处理", "基础图形"),
        description=(
            "在图像上绘制圆形边框或实心圆，用于圈选目标。未指定区域时以图像中心为"
            "圆心、短边的一半为半径。"
        ),
        keywords=("圆", "circle", "圈选", "标注", "draw"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec(
                "roi",
                "圆形区域",
                "roi",
                default=None,
                placeholder="[x, y, w, h]",
                tooltip="按矩形的中心与短边确定圆；缺省使用图像中心区域。",
            ),
            ParamSpec("color", "颜色", "color", default="#00a0ff"),
            ParamSpec(
                "thickness",
                "线宽",
                "int",
                default=2,
                min=1,
                max=51,
                step=1,
                unit="px",
                visible_when="not filled",
                enabled_when="not filled",
            ),
            ParamSpec("filled", "填充", "bool", default=False),
        ),
    )

    def process(self, image=None, roi=None, color="#00a0ff", thickness=2, filled=False, **_: Any):
        template, work = _prepare(image)
        height, width = work.shape[:2]
        x, y, rect_w, rect_h = _cv.resolve_rect(roi, width, height, fraction=0.5)
        center = (x + rect_w // 2, y + rect_h // 2)
        radius = max(1, min(rect_w, rect_h) // 2)
        cv2.circle(
            work,
            center,
            radius,
            _cv.draw_color(work, color),
            -1 if filled else int(thickness),
            _cv.draw_type(work),
        )
        return _finish(template, work)


class Ellipse(Operator):
    """在图像上绘制椭圆。"""

    spec = OperatorSpec(
        id="draw.ellipse",
        name="绘制椭圆",
        category=("交互处理", "基础图形"),
        description=(
            "在图像上绘制椭圆边框或实心椭圆，可设置旋转角度。未指定区域时使用图像"
            "中心 50% 区域作为外接矩形。"
        ),
        keywords=("椭圆", "ellipse", "旋转", "标注", "draw"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec(
                "roi",
                "椭圆区域",
                "roi",
                default=None,
                placeholder="[x, y, w, h]",
                tooltip="矩形的中心作为椭圆中心，宽高的一半作为长短轴。",
            ),
            ParamSpec(
                "angle",
                "旋转角度",
                "int",
                default=0,
                min=0,
                max=360,
                step=1,
                unit="°",
            ),
            ParamSpec("color", "颜色", "color", default="#34c759"),
            ParamSpec(
                "thickness",
                "线宽",
                "int",
                default=2,
                min=1,
                max=51,
                step=1,
                unit="px",
                visible_when="not filled",
                enabled_when="not filled",
            ),
            ParamSpec("filled", "填充", "bool", default=False),
        ),
    )

    def process(
        self,
        image=None,
        roi=None,
        angle=0,
        color="#34c759",
        thickness=2,
        filled=False,
        **_: Any,
    ):
        template, work = _prepare(image)
        height, width = work.shape[:2]
        x, y, rect_w, rect_h = _cv.resolve_rect(roi, width, height, fraction=0.5)
        center = (x + rect_w // 2, y + rect_h // 2)
        axes = (max(1, rect_w // 2), max(1, rect_h // 2))
        cv2.ellipse(
            work,
            center,
            axes,
            float(angle),
            0.0,
            360.0,
            _cv.draw_color(work, color),
            -1 if filled else int(thickness),
            _cv.draw_type(work),
        )
        return _finish(template, work)


class Polygon(Operator):
    """在图像上绘制多边形。"""

    spec = OperatorSpec(
        id="draw.polygon",
        name="绘制多边形",
        category=("交互处理", "基础图形"),
        description=(
            "按点集绘制闭合多边形或实心多边形，可交互拾取任意数量的顶点。未指定点集"
            "时在图像中心生成一个正五边形。"
        ),
        keywords=("多边形", "polygon", "顶点", "标注", "draw"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec(
                "points",
                "顶点",
                "points",
                default=[],
                tooltip="在预览中逐个点击顶点，按 Enter 完成；缺省使用中心五边形。",
            ),
            ParamSpec("color", "颜色", "color", default="#ffcc00"),
            ParamSpec(
                "thickness",
                "线宽",
                "int",
                default=2,
                min=1,
                max=51,
                step=1,
                unit="px",
                visible_when="not filled",
                enabled_when="not filled",
            ),
            ParamSpec("filled", "填充", "bool", default=False),
        ),
    )

    def process(
        self,
        image=None,
        points=None,
        color="#ffcc00",
        thickness=2,
        filled=False,
        **_: Any,
    ):
        template, work = _prepare(image)
        height, width = work.shape[:2]
        resolved = _cv.resolve_points(points, width, height)
        if len(resolved) < 3:
            center_x, center_y = width // 2, height // 2
            radius = max(2, min(width, height) // 3)
            resolved = [
                (
                    int(round(center_x + radius * math.cos(-math.pi / 2 + 2 * math.pi * i / 5))),
                    int(round(center_y + radius * math.sin(-math.pi / 2 + 2 * math.pi * i / 5))),
                )
                for i in range(5)
            ]
        vertices = np.asarray(resolved, dtype=np.int32)
        if filled:
            cv2.fillPoly(work, [vertices], _cv.draw_color(work, color), _cv.draw_type(work))
        else:
            cv2.polylines(
                work,
                [vertices],
                True,
                _cv.draw_color(work, color),
                int(thickness),
                _cv.draw_type(work),
            )
        return _finish(template, work)


# ---------------------------------------------------------------------------
# 文字 / 箭头 / 标记点 / 十字线
# ---------------------------------------------------------------------------
class Text(Operator):
    """在图像上写文字。"""

    spec = OperatorSpec(
        id="draw.text",
        name="绘制文字",
        category=("交互处理", "文字与标记"),
        description=(
            "在指定位置写一段文字，可用于标注名称、参数或结论。未指定位置时文字"
            "居中显示。"
        ),
        keywords=("文字", "text", "标注", "标题", "draw"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec("text", "文字内容", "string", default="MineImage 示例"),
            ParamSpec(
                "points",
                "文字位置",
                "points",
                default=[],
                tooltip="在预览中点击文字中心位置；缺省为图像中心。",
            ),
            ParamSpec(
                "font_scale",
                "字号",
                "float",
                default=0.9,
                min=0.2,
                max=5.0,
                step=0.1,
            ),
            ParamSpec("color", "颜色", "color", default="#ffffff"),
            ParamSpec(
                "thickness",
                "线宽",
                "int",
                default=2,
                min=1,
                max=10,
                step=1,
                unit="px",
            ),
        ),
    )

    def process(
        self,
        image=None,
        text="MineImage 示例",
        points=None,
        font_scale=0.9,
        color="#ffffff",
        thickness=2,
        **_: Any,
    ):
        template, work = _prepare(image)
        height, width = work.shape[:2]
        content = str(text or "").strip()
        if not content:
            return _finish(template, work)
        center = _center_point(points, width, height)
        scale = float(font_scale)
        (text_w, text_h), _baseline = cv2.getTextSize(content, FONT, scale, int(thickness))
        origin = (
            int(center[0] - text_w // 2),
            int(center[1] + text_h // 2),
        )
        cv2.putText(
            work,
            content,
            origin,
            FONT,
            scale,
            _cv.draw_color(work, color),
            int(thickness),
            _cv.draw_type(work),
        )
        return _finish(template, work)


class Arrow(Operator):
    """在图像上绘制带箭头的线段。"""

    spec = OperatorSpec(
        id="draw.arrow",
        name="绘制箭头",
        category=("交互处理", "文字与标记"),
        description=(
            "绘制从起点指向终点的箭头，用于指示方向或标注位移。未指定线段时使用"
            "图像对角线。"
        ),
        keywords=("箭头", "arrow", "方向", "标注", "draw"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec(
                "line",
                "箭头线段",
                "line",
                default=None,
                placeholder="[x1, y1, x2, y2]",
                tooltip="在预览中拖出线段；缺省使用图像对角线。",
            ),
            ParamSpec("color", "颜色", "color", default="#ff9500"),
            ParamSpec(
                "thickness",
                "线宽",
                "int",
                default=2,
                min=1,
                max=31,
                step=1,
                unit="px",
            ),
            ParamSpec(
                "tip_length",
                "箭头长度",
                "float",
                default=0.25,
                min=0.05,
                max=0.6,
                step=0.01,
                tooltip="箭头尖端的相对长度。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        line=None,
        color="#ff9500",
        thickness=2,
        tip_length=0.25,
        **_: Any,
    ):
        template, work = _prepare(image)
        height, width = work.shape[:2]
        x1, y1, x2, y2 = _cv.resolve_line(line, width, height)
        cv2.arrowedLine(
            work,
            (x1, y1),
            (x2, y2),
            _cv.draw_color(work, color),
            int(thickness),
            _cv.draw_type(work),
            tipLength=float(tip_length),
        )
        return _finish(template, work)


class Marker(Operator):
    """在指定点绘制标记符号。"""

    spec = OperatorSpec(
        id="draw.marker",
        name="绘制标记点",
        category=("交互处理", "文字与标记"),
        description=(
            "在点集位置绘制十字、星形、菱形等标记，用于标出特征点或关键位置。"
            "未指定点时标记图像中心。"
        ),
        keywords=("标记", "marker", "特征点", "十字", "draw"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec(
                "points",
                "标记位置",
                "points",
                default=[],
                tooltip="在预览中点击标记位置；缺省为图像中心。",
            ),
            ParamSpec(
                "marker",
                "标记形状",
                "enum",
                choices=(
                    ("cross", "十字"),
                    ("tilted_cross", "斜十字"),
                    ("star", "星形"),
                    ("diamond", "菱形"),
                    ("square", "方形"),
                    ("triangle_up", "上三角"),
                    ("triangle_down", "下三角"),
                ),
                default="cross",
            ),
            ParamSpec(
                "size",
                "标记大小",
                "int",
                default=20,
                min=2,
                max=100,
                step=1,
                unit="px",
            ),
            ParamSpec("color", "颜色", "color", default="#ff2d55"),
            ParamSpec(
                "thickness",
                "线宽",
                "int",
                default=2,
                min=1,
                max=10,
                step=1,
                unit="px",
            ),
        ),
    )

    def process(
        self,
        image=None,
        points=None,
        marker="cross",
        size=20,
        color="#ff2d55",
        thickness=2,
        **_: Any,
    ):
        template, work = _prepare(image)
        height, width = work.shape[:2]
        point = _center_point(points, width, height)
        marker_type = _MARKERS.get(str(marker), cv2.MARKER_CROSS)
        cv2.drawMarker(
            work,
            point,
            _cv.draw_color(work, color),
            marker_type,
            int(size),
            int(thickness),
            _cv.draw_type(work),
        )
        return _finish(template, work)


class Crosshair(Operator):
    """在指定点绘制十字准线。"""

    spec = OperatorSpec(
        id="draw.crosshair",
        name="绘制十字线",
        category=("交互处理", "文字与标记"),
        description=(
            "在指定位置绘制水平 + 垂直的十字准线，用于定位或对齐。长度设置为 0 时"
            "按图像短边自动计算（四分之一），未指定点时位于图像中心。"
        ),
        keywords=("十字线", "crosshair", "准线", "定位", "draw"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec(
                "points",
                "中心位置",
                "points",
                default=[],
                tooltip="缺省为图像中心。",
            ),
            ParamSpec(
                "size",
                "准线长度",
                "int",
                default=0,
                min=0,
                max=500,
                step=2,
                unit="px",
                tooltip="十字线的一半长度；0 表示自动取短边的四分之一。",
            ),
            ParamSpec("color", "颜色", "color", default="#00ffcc"),
            ParamSpec(
                "thickness",
                "线宽",
                "int",
                default=1,
                min=1,
                max=10,
                step=1,
                unit="px",
            ),
        ),
    )

    def process(
        self,
        image=None,
        points=None,
        size=0,
        color="#00ffcc",
        thickness=1,
        **_: Any,
    ):
        template, work = _prepare(image)
        height, width = work.shape[:2]
        center_x, center_y = _center_point(points, width, height)
        half = int(size) // 2 if int(size) > 0 else max(1, min(width, height) // 4)
        draw_color = _cv.draw_color(work, color)
        line_type = _cv.draw_type(work)
        cv2.line(
            work, (center_x - half, center_y), (center_x + half, center_y), draw_color, int(thickness), line_type
        )
        cv2.line(
            work, (center_x, center_y - half), (center_x, center_y + half), draw_color, int(thickness), line_type
        )
        return _finish(template, work)


# ---------------------------------------------------------------------------
# 网格 / 比例尺 / 水印 / ROI 高亮
# ---------------------------------------------------------------------------
class Grid(Operator):
    """在图像上叠加网格线。"""

    spec = OperatorSpec(
        id="draw.grid",
        name="绘制网格",
        category=("交互处理", "辅助元素"),
        description=(
            "按固定间距绘制横竖网格线，用于观察像素位置、比例或构图。网格间距会"
            "自动收敛到适合图像尺寸的大小。"
        ),
        keywords=("网格", "grid", "参考线", "构图", "draw"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec(
                "spacing",
                "网格间距",
                "int",
                default=64,
                min=4,
                max=400,
                step=4,
                unit="px",
            ),
            ParamSpec("color", "颜色", "color", default="#808080"),
            ParamSpec(
                "thickness",
                "线宽",
                "int",
                default=1,
                min=1,
                max=10,
                step=1,
                unit="px",
            ),
        ),
    )

    def process(self, image=None, spacing=64, color="#808080", thickness=1, **_: Any):
        template, work = _prepare(image)
        height, width = work.shape[:2]
        step = max(4, min(int(spacing), max(1, min(width, height) // 2)))
        draw_color = _cv.draw_color(work, color)
        for x in range(step, width, step):
            cv2.line(work, (x, 0), (x, height - 1), draw_color, int(thickness), _cv.draw_type(work))
        for y in range(step, height, step):
            cv2.line(work, (0, y), (width - 1, y), draw_color, int(thickness), _cv.draw_type(work))
        return _finish(template, work)


class ScaleBar(Operator):
    """在图像角落绘制比例尺。"""

    spec = OperatorSpec(
        id="draw.scale_bar",
        name="绘制比例尺",
        category=("交互处理", "辅助元素"),
        description=(
            "在图像角落绘制带端点刻度的比例尺，并显示对应像素长度或自定义文字，"
            "适合报告与测量示意。"
        ),
        keywords=("比例尺", "scale", "bar", "测量", "标注"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec(
                "ratio",
                "长度占比",
                "float",
                default=0.25,
                min=0.05,
                max=0.9,
                step=0.05,
                tooltip="比例尺长度占图像宽度的比例。",
            ),
            ParamSpec(
                "position",
                "放置位置",
                "enum",
                choices=_POSITIONS,
                default="bottom_left",
            ),
            ParamSpec(
                "label",
                "标注文字",
                "string",
                default="",
                placeholder="留空显示像素长度",
            ),
            ParamSpec("color", "颜色", "color", default="#ffffff"),
            ParamSpec(
                "thickness",
                "线宽",
                "int",
                default=3,
                min=1,
                max=20,
                step=1,
                unit="px",
            ),
        ),
    )

    def process(
        self,
        image=None,
        ratio=0.25,
        position="bottom_left",
        label="",
        color="#ffffff",
        thickness=3,
        **_: Any,
    ):
        template, work = _prepare(image)
        height, width = work.shape[:2]
        bar_length = max(4, int(round(width * float(ratio))))
        bar_length = min(bar_length, max(4, width - 2))
        margin = max(4, min(width, height) // 15)
        cap = max(3, int(thickness) * 2)
        draw_color = _cv.draw_color(work, color)
        line_type = _cv.draw_type(work)
        left = position in ("top_left", "bottom_left")
        top = position in ("top_left", "top_right")
        x = margin if left else max(0, width - margin - bar_length)
        y = margin + cap if top else max(cap, height - margin - cap)
        x2 = min(width - 1, x + bar_length)
        y = min(max(y, cap), height - 1 - cap)
        cv2.line(work, (x, y), (x2, y), draw_color, int(thickness), line_type)
        cv2.line(work, (x, y - cap), (x, y + cap), draw_color, int(thickness), line_type)
        cv2.line(work, (x2, y - cap), (x2, y + cap), draw_color, int(thickness), line_type)
        content = str(label or "").strip() or f"{bar_length}px"
        scale = max(0.3, min(1.0, _text_scale(width, height, 1, content)))
        (text_w, text_h), _baseline = cv2.getTextSize(content, FONT, scale, 1)
        text_x = min(max(0, x), max(0, width - text_w - 1))
        text_y = y - cap - 4 if not top else min(height - 1, y + cap + text_h + 4)
        text_y = max(text_h + 1, text_y)
        cv2.putText(work, content, (text_x, text_y), FONT, scale, draw_color, 1, line_type)
        return _finish(template, work)


class Watermark(Operator):
    """在图像上叠加半透明文字水印。"""

    spec = OperatorSpec(
        id="draw.watermark",
        name="添加水印",
        category=("交互处理", "辅助元素"),
        description=(
            "在图像角落叠加半透明文字水印，可按图像尺寸自动缩放字号。适合给结果图"
            "添加来源或版权说明。"
        ),
        keywords=("水印", "watermark", "版权", "文字", "叠加"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec("text", "水印文字", "string", default="MineImage"),
            ParamSpec(
                "position",
                "放置位置",
                "enum",
                choices=_POSITIONS,
                default="bottom_right",
            ),
            ParamSpec(
                "opacity",
                "不透明度",
                "float",
                default=0.35,
                min=0.0,
                max=1.0,
                step=0.05,
            ),
            ParamSpec(
                "scale",
                "字号缩放",
                "float",
                default=1.0,
                min=0.2,
                max=5.0,
                step=0.1,
                unit="×",
            ),
            ParamSpec("color", "颜色", "color", default="#ffffff"),
            ParamSpec(
                "thickness",
                "线宽",
                "int",
                default=2,
                min=1,
                max=10,
                step=1,
                unit="px",
            ),
        ),
    )

    def process(
        self,
        image=None,
        text="MineImage",
        position="bottom_right",
        opacity=0.35,
        scale=1.0,
        color="#ffffff",
        thickness=2,
        **_: Any,
    ):
        template, work = _prepare(image)
        height, width = work.shape[:2]
        content = str(text or "").strip()
        if not content or float(opacity) <= 0.0:
            return _finish(template, work)
        font_scale = max(0.2, min(_text_scale(width, height, int(thickness), content) * float(scale), 5.0))
        (text_w, text_h), _baseline = cv2.getTextSize(content, FONT, font_scale, int(thickness))
        margin = max(4, min(width, height) // 15)
        left = position in ("top_left", "bottom_left")
        top = position in ("top_left", "top_right")
        x = margin if left else max(0, width - text_w - margin)
        y = margin + text_h if top else max(text_h + 1, height - margin)
        overlay = work.copy()
        cv2.putText(
            overlay,
            content,
            (x, y),
            FONT,
            font_scale,
            _cv.draw_color(overlay, color),
            int(thickness),
            _cv.draw_type(overlay),
        )
        result = cv2.addWeighted(overlay, float(opacity), work, 1.0 - float(opacity), 0.0)
        return _finish(template, result)


class RoiHighlight(Operator):
    """压暗 ROI 以外的区域并描边。"""

    spec = OperatorSpec(
        id="draw.roi_highlight",
        name="ROI 高亮",
        category=("交互处理", "辅助元素"),
        description=(
            "压暗矩形区域之外的画面并给区域描边，把注意力集中到目标上。可切换为"
            "只压暗区域内。未指定区域时使用图像中心 50% 区域。"
        ),
        keywords=("roi", "高亮", "聚光", "压暗", "标注"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec(
                "roi",
                "高亮区域",
                "roi",
                default=None,
                placeholder="[x, y, w, h]",
                tooltip="缺省使用图像中心区域。",
            ),
            ParamSpec(
                "dim",
                "压暗程度",
                "float",
                default=0.45,
                min=0.0,
                max=0.9,
                step=0.05,
            ),
            ParamSpec(
                "mode",
                "压暗范围",
                "enum",
                choices=(("outside", "区域外压暗"), ("inside", "区域内压暗")),
                default="outside",
            ),
            ParamSpec("color", "边框颜色", "color", default="#ffffff"),
            ParamSpec(
                "thickness",
                "边框线宽",
                "int",
                default=2,
                min=1,
                max=20,
                step=1,
                unit="px",
            ),
        ),
    )

    def process(
        self,
        image=None,
        roi=None,
        dim=0.45,
        mode="outside",
        color="#ffffff",
        thickness=2,
        **_: Any,
    ):
        template, work = _prepare(image)
        height, width = work.shape[:2]
        x, y, rect_w, rect_h = _cv.resolve_rect(roi, width, height, fraction=0.5)
        factor = 1.0 - float(dim)
        dimmed = np.clip(work.astype(np.float32) * factor, 0.0, 255.0).astype(np.uint8)
        result = dimmed if mode == "outside" else work.copy()
        result[y : y + rect_h, x : x + rect_w] = work[y : y + rect_h, x : x + rect_w]
        if mode == "inside":
            result[y : y + rect_h, x : x + rect_w] = dimmed[y : y + rect_h, x : x + rect_w]
        cv2.rectangle(
            result,
            (x, y),
            (x + rect_w - 1, y + rect_h - 1),
            _cv.draw_color(result, color),
            int(thickness),
            _cv.draw_type(result),
        )
        return _finish(template, result)


#: 模块导出的算子类；loader 支持一个模块注册多个算子。
OPERATORS = (
    Rectangle,
    Circle,
    Ellipse,
    Polygon,
    Text,
    Arrow,
    Marker,
    Crosshair,
    Grid,
    ScaleBar,
    Watermark,
    RoiHighlight,
)
