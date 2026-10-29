"""右侧图像预览控件。

职责
----
* 绘制 :class:`~app.core.types.ImageData`（BGR/BGRA/灰度 → 显示 RGB）；
* 缩放 / 平移：二维模式下 ``Ctrl + 滚轮`` 缩放，左键拖动平移；
* 显示模式：Imshow / Imagesc / Mesh；Mesh 支持拖动旋转、滚轮缩放、双击重置；
* 对比：单图、左右对比、滑动对比（分割线以图像坐标保存，随缩放/平移对齐）；
* 交互拾取：点集、矩形 ROI、直线，屏幕坐标与图像坐标双向映射；
* 像素采样：通过 :func:`app.core.types.sample_pixel` 返回左上原点的行/列与原始值。

本控件只负责显示，不参与任何计算与下游失效逻辑。
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, Signal
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetrics,
    QGuiApplication,
    QImage,
    QKeyEvent,
    QLinearGradient,
    QMouseEvent,
    QOffscreenSurface,
    QOpenGLContext,
    QPainter,
    QPen,
    QPolygonF,
    QSurfaceFormat,
    QWheelEvent,
)
from PySide6.QtOpenGL import (
    QOpenGLBuffer,
    QOpenGLFramebufferObject,
    QOpenGLFramebufferObjectFormat,
    QOpenGLShader,
    QOpenGLShaderProgram,
    QOpenGLVertexArrayObject,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from app.core.constants import DEFAULT_IMAGE_DISPLAY_MODE, IMAGE_DISPLAY_MODES
from app.core.types import (
    ImageData,
    PixelSample,
    display_colormap_rgb,
    intensity_grid,
    sample_pixel,
    to_display_rgb,
    to_scaled_rgb,
)
from app.ui.display_modes import display_mode_title

_MODES = ("single", "side_by_side", "swipe")
_MODE_ALIASES = {
    "side-by-side": "side_by_side",
    "sidebyside": "side_by_side",
    "compare": "side_by_side",
    "左右对比": "side_by_side",
    "single_view": "single",
    "单图": "single",
    "滑动对比": "swipe",
}
_TOOL_ALIASES = {
    "roi": "rect",
    "rectangle": "rect",
    "box": "rect",
    "rect": "rect",
    "point": "points",
    "points": "points",
    "line": "line",
}
_TOOL_HINTS = {
    "points": "左键添加点",
    "rect": "拖动框选矩形 ROI",
    "line": "拖动拾取直线",
}
_PICK_TOOL_NAMES = {"points", "rect", "line"}
_DISPLAY_MODE_ALIASES = {
    "image": "imshow",
    "2d": "imshow",
    "二维": "imshow",
    "colormap": "imagesc",
    "伪彩色": "imagesc",
    "3d": "mesh",
    "网格": "mesh",
}
#: mesh 高度场的最长边采样数；块平均后再做平滑着色，避免出现明显多边形。
_MESH_MAX_SIDE = 320
#: 初始视角与 MATLAB `view(-37.5, 30)` 一致。
_MESH_AZIMUTH = math.radians(-37.5)
_MESH_ELEVATION = math.radians(30.0)
#: 拖动旋转的灵敏度（弧度 / 屏幕像素）。
_MESH_ROTATE_SPEED_X = math.radians(0.55)
_MESH_ROTATE_SPEED_Y = math.radians(0.4)
#: 曲面高度（相对图像宽高比）。
_MESH_HEIGHT = 0.42
#: 投影后内容占虚拟图像框的比例，留一点边距。
_MESH_PADDING = 0.92
#: OpenGL FBO 超采样倍数上限与最大边长。
_MESH_GL_SUPERSAMPLE = 2.0
_MESH_GL_MAX_SIDE = 2400
#: 软件回退时的采样上限（无 GL 环境下控制旋转 / 缩放的重绘耗时）。
_MESH_SOFTWARE_MAX_SIDE = 160

LOG = logging.getLogger(__name__)

_GL_FLOAT = 0x1406
_GL_TRIANGLE_STRIP = 0x0005
_GL_DEPTH_TEST = 0x0B71
_GL_LESS = 0x0201
_GL_COLOR_BUFFER_BIT = 0x4000
_GL_DEPTH_BUFFER_BIT = 0x0100

#: mesh 曲面着色器：顶点直接输出屏幕 NDC + 平滑插值颜色。
_MESH_GL_VERTEX_SHADER = """
#version 330 core
layout(location = 0) in vec3 aPos;
layout(location = 1) in vec3 aColor;
out vec3 vColor;
void main() {
    vColor = aColor;
    gl_Position = vec4(aPos, 1.0);
}
"""

_MESH_GL_FRAGMENT_SHADER = """
#version 330 core
in vec3 vColor;
out vec4 fragColor;
void main() {
    fragColor = vec4(vColor, 1.0);
}
"""


def _normalize_display_mode(mode: str) -> str:
    normalized = str(mode or "").strip().lower()
    normalized = _DISPLAY_MODE_ALIASES.get(normalized, normalized)
    return normalized if normalized in IMAGE_DISPLAY_MODES else DEFAULT_IMAGE_DISPLAY_MODE


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _round(value: float) -> float:
    rounded = round(float(value), 2)
    return 0.0 if rounded == 0 else rounded


@dataclass
class _Pane:
    """一个显示面板：视口矩形 + 图像 + 该图像的屏幕变换。"""

    rect: QRectF
    offset: QPointF
    image: ImageData | None
    qimage: QImage | None
    title: str
    kind: str  # current | original


@dataclass
class _MeshField:
    """与相机无关的 mesh 数据：归一化高度场 + 已做明暗的色带颜色。"""

    grid: np.ndarray  # (rows, cols) float32，0–1
    colors: np.ndarray  # (rows, cols, 3) float32，0–255
    width: int
    height: int


@dataclass
class _MeshGeometry:
    """mesh 模式下按当前视角预计算的投影网格。"""

    points: np.ndarray  # (rows, cols, 2) float32
    colors: np.ndarray  # (rows, cols, 3) float32
    depth: np.ndarray  # (rows, cols) float32，视图方向深度（用于 GL 消隐）


class PreviewWidget(QWidget):
    """图像预览控件（缩放 / 平移 / 对比 / 拾取）。"""

    pixel_info = Signal(object)  # PixelSample | None
    zoom_changed = Signal(float)  # 百分比，例如 100.0
    pick_finished = Signal(str, object)  # tool, points / rect / line 载荷
    pick_cancelled = Signal(str)

    MIN_ZOOM = 0.02
    MAX_ZOOM = 64.0
    MARGIN = 14.0
    MATTE = QColor("#e9eef5")
    PANE_MATTE = QColor("#f2f5fa")
    PANE_BORDER = QColor("#d8dee8")
    IMAGE_BORDER = QColor(15, 23, 42, 52)
    IMAGE_SHADOW = QColor(15, 23, 42, 26)
    ACCENT = QColor("#2563eb")
    LABEL_BG = QColor(255, 255, 255, 232)
    LABEL_BORDER = QColor(203, 213, 225, 220)
    LABEL_TEXT = QColor("#1f2937")
    LABEL_TEXT_MUTED = QColor("#475569")
    EMPTY_TEXT = QColor("#64748b")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(260, 190)
        self.setAttribute(Qt.WidgetAttribute.WA_OpaquePaintEvent, True)

        self._image: ImageData | None = None
        self._original: ImageData | None = None
        self._node_title = ""
        self._info = ""
        self._original_title = "原始图像"
        self._mode = "single"
        self._display_mode = DEFAULT_IMAGE_DISPLAY_MODE
        self._zoom = 1.0
        self._offset = QPointF(0.0, 0.0)
        self._auto_fit = True
        self._fit_on_new_image = True

        self._quality_enabled = False
        self._quality_max_dim = 2048
        self._image_qimage: QImage | None = None
        self._original_qimage: QImage | None = None
        # mesh 相机：方位角 / 仰角，初始与 MATLAB view(-37.5, 30) 一致。
        self._mesh_azimuth = _MESH_AZIMUTH
        self._mesh_elevation = _MESH_ELEVATION
        self._mesh_rotating = False
        self._mesh_rotate_last = QPointF()
        # id(ImageData) -> (ImageData, _MeshField)；保留图像引用，避免 id 复用。
        self._mesh_field_cache: dict[int, tuple[ImageData, _MeshField]] = {}
        # (id(ImageData), 方位角, 仰角) -> (ImageData, _MeshGeometry)
        self._mesh_cache: dict[tuple[int, float, float], tuple[ImageData, _MeshGeometry]] = {}
        # OpenGL 离屏曲面渲染（懒初始化；失败时回退到软件条带渲染）。
        self._gl_ready = False
        self._gl_failed = False
        self._gl_context: QOpenGLContext | None = None
        self._gl_surface: QOffscreenSurface | None = None
        self._gl_program: QOpenGLShaderProgram | None = None
        self._gl_vbo: QOpenGLBuffer | None = None
        self._gl_vao: QOpenGLVertexArrayObject | None = None
        self._gl_fbo: QOpenGLFramebufferObject | None = None
        self._gl_fbo_size = QSize(0, 0)

        self._panning = False
        self._pan_last = QPointF()
        self._dragging_divider = False
        self._swipe_image_x: float | None = None

        self._pick_tool: str | None = None
        self._pick_count = 1
        self._pick_points: list[list[float]] = []
        self._pick_start: tuple[float, float] | None = None
        self._pick_current: tuple[float, float] | None = None
        self._pick_dragging = False

        self._last_view_pos: QPointF | None = None
        self._last_pixel_key: tuple[Any, ...] | None = None

    # ------------------------------------------------------------------ 公共 API
    def set_image(self, image: ImageData | None, node_title: str = "", info: str = "") -> None:
        """设置当前结果图；同尺寸更新会保留缩放/平移，尺寸变化时自动适应。"""
        if self._pick_tool is not None:
            self.cancel_pick()
        previous_size = self.image_size()
        self._image = image
        self._node_title = node_title or ""
        self._info = info or ""
        self._image_qimage = self._build_qimage(image)
        new_size = self.image_size()
        if image is None:
            self._focus_on_available()
        else:
            if self._swipe_image_x is None:
                self._swipe_image_x = image.width * 0.5
            else:
                self._swipe_image_x = _clamp(self._swipe_image_x, 0.0, float(image.width))
            if previous_size is None or new_size != previous_size:
                if self._fit_on_new_image:
                    self.fit_to_window()
                elif previous_size is None:
                    self.reset_zoom()
        if self._last_view_pos is not None:
            self._emit_pixel_info_at(self._last_view_pos)
        self.update()

    def set_auto_fit_on_open(self, enabled: bool) -> None:
        """设置“首次显示/尺寸变化时是否自动适应窗口”。"""
        self._fit_on_new_image = bool(enabled)
        if not enabled:
            self._auto_fit = False

    def set_original_image(self, image: ImageData | None, title: str = "原始图像") -> None:
        """设置对比模式使用的原始图像。"""
        self._original = image
        self._original_title = title or "原始图像"
        self._original_qimage = self._build_qimage(image)
        if image is not None and self._swipe_image_x is None:
            self._swipe_image_x = image.width * 0.5
        if self._auto_fit:
            self.fit_to_window()
        self.update()

    def set_compare_mode(self, mode: str) -> None:
        """切换显示模式：``single`` / ``side_by_side`` / ``swipe``。"""
        normalized = _MODE_ALIASES.get(mode, mode)
        if normalized not in _MODES:
            normalized = "single"
        if normalized == self._mode:
            return
        self._mode = normalized
        if self._mode == "swipe" and self._swipe_image_x is None:
            image = self._image or self._original
            if image is not None:
                self._swipe_image_x = image.width * 0.5
        if self._auto_fit:
            self.fit_to_window()
        self.update()
        if self._last_view_pos is not None:
            self._emit_pixel_info_at(self._last_view_pos)

    def set_display_mode(self, mode: str) -> None:
        """切换图像显示模式：``imshow`` / ``imagesc`` / ``mesh``。

        与 :meth:`set_compare_mode`（单图 / 对比布局）相互独立：显示模式只
        决定每幅图如何渲染，切换节点或对比布局都不会重置它。
        """
        normalized = _normalize_display_mode(mode)
        if normalized == self._display_mode:
            return
        self._display_mode = normalized
        self._image_qimage = self._build_qimage(self._image)
        self._original_qimage = self._build_qimage(self._original)
        self._mesh_rotating = False
        if self._display_mode == "mesh" and self._pick_tool is not None:
            self.cancel_pick()
        self.update()
        if self._last_view_pos is not None:
            self._emit_pixel_info_at(self._last_view_pos)

    def display_mode(self) -> str:
        return self._display_mode

    def mesh_camera(self) -> tuple[float, float]:
        """返回 mesh 当前视角（方位角, 仰角），单位度。"""

        return (math.degrees(self._mesh_azimuth), math.degrees(self._mesh_elevation))

    def reset_mesh_view(self) -> None:
        """恢复 mesh 默认视角并适应窗口。"""

        self._mesh_azimuth = _MESH_AZIMUTH
        self._mesh_elevation = _MESH_ELEVATION
        self._mesh_rotating = False
        self.fit_to_window()
        self.update()

    def set_preview_quality(self, enabled: bool, max_dim: int) -> None:
        """仅影响显示分辨率的降采样设置，不改变引擎缓存。"""
        max_dim = max(32, int(max_dim or 2048))
        if enabled == self._quality_enabled and max_dim == self._quality_max_dim:
            return
        self._quality_enabled = bool(enabled)
        self._quality_max_dim = max_dim
        self._image_qimage = self._build_qimage(self._image)
        self._original_qimage = self._build_qimage(self._original)
        self.update()

    def begin_pick(self, tool: str, count: int = 1, initial: Any = None) -> None:
        """进入交互拾取模式。

        ``tool``：``points`` / ``rect``（也接受 ``roi``）/ ``line``。
        ``count``：点集需要收集的点数，0 表示数量不限。
        ``initial``：点集可用 ``[[x, y], ...]`` 预填已有结果。
        """
        normalized = _TOOL_ALIASES.get(str(tool), str(tool))
        if normalized not in _PICK_TOOL_NAMES:
            raise ValueError(f"不支持的拾取工具：{tool!r}")
        if self._display_mode == "mesh":
            raise ValueError("三维网格（mesh）模式下不支持拾取，请先切换到 Imshow / Imagesc")
        if self._pick_tool is not None:
            self.cancel_pick()
        self._pick_tool = normalized
        self._pick_count = max(0, int(count or 0))
        self._pick_points = []
        self._pick_start = None
        self._pick_current = None
        self._pick_dragging = False
        if normalized == "points" and isinstance(initial, (list, tuple)):
            for point in initial:
                try:
                    if len(point) >= 2:
                        self._pick_points.append([_round(float(point[0])), _round(float(point[1]))])
                except (TypeError, ValueError):
                    continue
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.update()

    def cancel_pick(self) -> None:
        """取消当前拾取并发出 :attr:`pick_cancelled`。"""
        if self._pick_tool is None:
            return
        tool = self._pick_tool
        self._clear_pick_state()
        self.pick_cancelled.emit(tool)
        self.update()

    def is_picking(self) -> bool:
        return self._pick_tool is not None

    def picking_tool(self) -> str | None:
        return self._pick_tool

    def fit_to_window(self) -> None:
        """按当前模式适应窗口并居中。"""
        self._auto_fit = True
        area = self._image_area()
        images = [img for img in (self._original, self._image) if img is not None]
        if not images:
            self._zoom = 1.0
            self._offset = QPointF(0.0, 0.0)
            self.update()
            self.zoom_changed.emit(self.zoom_percent())
            return
        if self._mode == "side_by_side" and len(images) >= 2:
            gap = 10.0
            pane_w = max(10.0, (area.width() - gap) / 2.0)
            pane_h = max(10.0, area.height())
            fit = min(
                min(pane_w / max(1, img.width), pane_h / max(1, img.height)) for img in images
            )
        else:
            image = self._image or self._original
            assert image is not None
            fit = min(
                area.width() / max(1, image.width),
                area.height() / max(1, image.height),
            )
        self._zoom = _clamp(fit, self.MIN_ZOOM, self.MAX_ZOOM)
        self._recenter_offset()
        self.update()
        self.zoom_changed.emit(self.zoom_percent())

    def reset_zoom(self) -> None:
        """恢复 100%（1 图像像素 = 1 屏幕像素）并居中。"""
        self._auto_fit = False
        self._zoom = 1.0
        self._recenter_offset()
        self.update()
        self.zoom_changed.emit(self.zoom_percent())
        if self._last_view_pos is not None:
            self._emit_pixel_info_at(self._last_view_pos)

    def zoom_percent(self) -> float:
        return round(self._zoom * 100.0, 2)

    def image_size(self) -> tuple[int, int] | None:
        image = self._image or self._original
        if image is None:
            return None
        return (image.width, image.height)

    def compare_mode(self) -> str:
        return self._mode

    # ------------------------------------------------------------ 坐标映射辅助
    def image_to_view(self, x: float, y: float) -> QPointF:
        """图像坐标 → 控件坐标（多面板模式下使用“当前结果”面板）。"""
        pane = self._active_pane()
        if pane is None:
            return QPointF(0.0, 0.0)
        return QPointF(pane.offset.x() + float(x) * self._zoom, pane.offset.y() + float(y) * self._zoom)

    def view_to_image(self, point: QPointF) -> tuple[float, float] | None:
        """控件坐标 → 图像坐标（多面板模式下取光标所在面板）。"""
        pane = self._pane_at(point)
        if pane is None or pane.image is None:
            return None
        return self._to_image_point(pane, point)

    def current_image(self) -> ImageData | None:
        return self._image

    # ------------------------------------------------------------------ 绘制
    def paintEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        painter.fillRect(self.rect(), self.MATTE)
        if self._image is None and self._original is None:
            self._paint_empty(painter)
            return
        if self._mode == "side_by_side":
            self._paint_side_by_side(painter)
        elif self._mode == "swipe":
            self._paint_swipe(painter)
        else:
            self._paint_single(painter)
        self._paint_info_caption(painter)
        if self._display_mode == "mesh":
            self._paint_mesh_hint(painter)
        self._paint_pick_overlay(painter)
        painter.end()

    def _paint_empty(self, painter: QPainter) -> None:
        title_font = QFont(self.font())
        title_font.setPointSizeF(13.0)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(self.EMPTY_TEXT)
        rect = QRectF(self.rect()).adjusted(12, 0, -12, -26)
        title = self._node_title or "暂无预览"
        painter.drawText(
            rect,
            int(Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap),
            title,
        )
        body_font = QFont(self.font())
        body_font.setBold(False)
        body_font.setPointSizeF(9.5)
        painter.setFont(body_font)
        painter.setPen(QColor("#7c8797"))
        body_rect = QRectF(self.rect()).adjusted(28, 40, -28, 0)
        body = self._info or "在画布中单击节点查看结果\n双击节点可编辑参数"
        painter.drawText(
            body_rect,
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop | Qt.TextFlag.TextWordWrap),
            body,
        )
        pen = QPen(QColor(15, 23, 42, 44), 1.0, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        frame = QRectF(self.rect()).adjusted(22, 22, -22, -22)
        painter.drawRoundedRect(frame, 10, 10)

    def _paint_single(self, painter: QPainter) -> None:
        panes = self._panes()
        pane = panes[0] if panes else None
        if pane is None:
            return
        self._paint_pane_background(painter, QRectF(self.rect()).adjusted(4, 4, -4, -4))
        if pane.image is not None:
            self._draw_pane_image(painter, pane)

    def _paint_side_by_side(self, painter: QPainter) -> None:
        panes = self._panes()
        for index, pane in enumerate(panes):
            if index:
                painter.setPen(QPen(QColor(15, 23, 42, 34), 1.0))
                seam = pane.rect.left() - 5.0
                painter.drawLine(QPointF(seam, 8.0), QPointF(seam, self.height() - 8.0))
            self._paint_pane_background(painter, pane.rect)
            if pane.image is not None:
                self._draw_pane_image(painter, pane)
            title = pane.title or ("原始图像" if pane.kind == "original" else "当前结果")
            lines = [title]
            if pane.image is not None:
                lines.append(f"{pane.image.width}×{pane.image.height}")
            max_width = max(90.0, pane.rect.width() - 16.0)
            rect = self._chip_rect(painter, lines, max_width=max_width)
            self._paint_chip(
                painter,
                rect,
                lines,
                pane.rect.left() + 8.0,
                pane.rect.top() + 8.0,
            )

    def _paint_swipe(self, painter: QPainter) -> None:
        panes = self._panes()
        if not panes:
            return
        pane = panes[0]
        self._paint_pane_background(painter, QRectF(self.rect()).adjusted(4, 4, -4, -4))
        if pane.image is not None:
            self._draw_pane_image(painter, pane)

        both = self._original is not None and self._image is not None
        if both:
            original = self._original
            assert original is not None
            divider = self._divider_view_x()
            clip_left = QRectF(
                pane.rect.left(),
                pane.rect.top(),
                max(0.0, divider - pane.rect.left()),
                pane.rect.height(),
            )
            painter.save()
            painter.setClipRect(clip_left)
            original_pane = _Pane(
                rect=pane.rect,
                offset=QPointF(pane.offset),
                image=original,
                qimage=self._original_qimage,
                title=self._original_title,
                kind="original",
            )
            self._draw_pane_image(painter, original_pane)
            painter.restore()

            # 分割线 + 手柄
            top = max(pane.offset.y(), pane.rect.top())
            bottom = min(pane.offset.y() + (self._image.height if self._image else 0) * self._zoom, pane.rect.bottom())
            if bottom <= top:
                top, bottom = pane.rect.top(), pane.rect.bottom()
            pen = QPen(self.ACCENT, 2.0)
            painter.setPen(pen)
            painter.drawLine(QPointF(divider, top), QPointF(divider, bottom))
            mid = (top + bottom) / 2.0
            painter.setPen(QPen(QColor("#ffffff"), 1.5))
            painter.setBrush(self.ACCENT)
            painter.drawEllipse(QPointF(divider, mid), 9.0, 9.0)
            painter.drawLine(QPointF(divider - 3.5, mid - 4.0), QPointF(divider - 3.5, mid + 4.0))
            painter.drawLine(QPointF(divider + 3.5, mid - 4.0), QPointF(divider + 3.5, mid + 4.0))

            left_lines = [self._original_title or "原始图像"]
            if self._original is not None:
                left_lines.append(f"{self._original.width}×{self._original.height}")
            max_width = max(90.0, pane.rect.width() * 0.46)
            left_rect = self._chip_rect(painter, left_lines, max_width=max_width)
            self._paint_chip(
                painter,
                left_rect,
                left_lines,
                pane.rect.left() + 8.0,
                pane.rect.top() + 8.0,
            )
            title_lines = [self._node_title or "当前结果"]
            if self._image is not None:
                title_lines.append(f"{self._image.width}×{self._image.height}")
            right_rect = self._chip_rect(painter, title_lines, max_width=max_width)
            self._paint_chip(
                painter,
                right_rect,
                title_lines,
                pane.rect.right() - 8.0,
                pane.rect.top() + 8.0,
                align_right=True,
            )
        else:
            label = (
                self._original_title
                if self._image is None and self._original is not None
                else (self._node_title or "当前结果")
            )
            lines = [label]
            image = self._image or self._original
            if image is not None:
                lines.append(f"{image.width}×{image.height}")
            max_width = max(90.0, pane.rect.width() - 16.0)
            rect = self._chip_rect(painter, lines, max_width=max_width)
            self._paint_chip(
                painter,
                rect,
                lines,
                pane.rect.left() + 8.0,
                pane.rect.top() + 8.0,
            )

    def _paint_pane_background(self, painter: QPainter, rect: QRectF) -> None:
        painter.setPen(QPen(self.PANE_BORDER, 1.0))
        painter.setBrush(self.PANE_MATTE)
        painter.drawRoundedRect(rect, 6.0, 6.0)

    def _draw_pane_image(self, painter: QPainter, pane: _Pane) -> None:
        image = pane.image
        if image is None:
            return
        if self._display_mode == "mesh":
            self._draw_mesh(painter, pane)
            return
        qimage = pane.qimage
        if qimage is None:
            return
        target = QRectF(
            pane.offset.x(),
            pane.offset.y(),
            image.width * self._zoom,
            image.height * self._zoom,
        )
        painter.save()
        painter.setClipRect(pane.rect)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self.IMAGE_SHADOW)
        painter.drawRoundedRect(
            target.translated(0.0, 3.0).adjusted(-1.0, -1.0, 1.0, 1.0), 3.0, 3.0
        )
        painter.drawImage(target, qimage)
        painter.setPen(QPen(self.IMAGE_BORDER, 1.0))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(target)
        painter.restore()

    # ------------------------------------------------------------ 三维网格
    def _mesh_field_for(self, image: ImageData | None) -> _MeshField | None:
        """缓存与相机无关的高度场 / 颜色；旋转时不会重复降采样原图。"""

        if image is None or image.width <= 0 or image.height <= 0:
            return None
        key = id(image)
        cached = self._mesh_field_cache.get(key)
        if cached is not None and cached[0] is image:
            return cached[1]
        field = self._build_mesh_field(image)
        if field is None:
            return None
        self._mesh_field_cache[key] = (image, field)
        while len(self._mesh_field_cache) > 4:
            self._mesh_field_cache.pop(next(iter(self._mesh_field_cache)))
        return field

    def _build_mesh_field(self, image: ImageData) -> _MeshField | None:
        grid = intensity_grid(image, _MESH_MAX_SIDE).astype(np.float32)
        if grid.size == 0:
            return None
        # 单行 / 单列矩阵（例如 CSV 导入的一维序列）没有成对的行或列，
        # np.gradient 与三角化都要求至少 2×2，因此把退化维度复制一份；
        # 缩放仍使用原始图像宽高，普通图像（≥2×2）的网格不受影响。
        rows, cols = grid.shape
        if rows < 2:
            grid = np.repeat(grid, 2, axis=0)
        if cols < 2:
            grid = np.repeat(grid, 2, axis=1)
        low = float(np.min(grid))
        high = float(np.max(grid))
        if high > low:
            normalized = (grid - low) / (high - low)
        else:
            normalized = np.zeros_like(grid)
        colors = self._shade_mesh_colors(normalized)
        return _MeshField(normalized, colors, image.width, image.height)

    @staticmethod
    def _shade_mesh_colors(normalized: np.ndarray) -> np.ndarray:
        """按高度梯度做简单的漫反射着色，让曲面看起来连续而不是平坦多边形。"""

        colors = display_colormap_rgb(normalized).astype(np.float32)
        gradient_y, gradient_x = np.gradient(normalized.astype(np.float32))
        rows, cols = normalized.shape
        slope_x = gradient_x * max(1, cols) * 0.55
        slope_y = gradient_y * max(1, rows) * 0.55
        normal_x = -slope_x
        normal_y = -slope_y
        normal_z = np.ones_like(normalized)
        length = np.sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z)
        light = np.asarray([-0.42, -0.52, 0.74], dtype=np.float32)
        light /= float(np.linalg.norm(light))
        diffuse = (
            normal_x * light[0] + normal_y * light[1] + normal_z * light[2]
        ) / np.maximum(length, 1e-6)
        shade = 0.68 + 0.32 * np.clip(diffuse, 0.0, 1.0)
        return np.clip(colors * shade[:, :, None], 0.0, 255.0)

    def _mesh_geometry_for(self, image: ImageData | None) -> _MeshGeometry | None:
        """按（图像, 方位角, 仰角）缓存投影网格。"""

        field = self._mesh_field_for(image)
        if field is None or image is None:
            return None
        key = (id(image), round(self._mesh_azimuth, 5), round(self._mesh_elevation, 5))
        cached = self._mesh_cache.get(key)
        if cached is not None and cached[0] is image:
            return cached[1]
        geometry = self._project_mesh_field(field)
        if geometry is None:
            return None
        self._mesh_cache[key] = (image, geometry)
        while len(self._mesh_cache) > 8:
            self._mesh_cache.pop(next(iter(self._mesh_cache)))
        return geometry

    def _project_mesh_field(self, field: _MeshField) -> _MeshGeometry | None:
        grid = field.grid
        row_count, col_count = grid.shape
        if row_count <= 0 or col_count <= 0:
            return None
        xs = np.linspace(-0.5, 0.5, col_count, dtype=np.float32)
        ys = np.linspace(-0.5, 0.5, row_count, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xs, ys)
        grid_z = (grid - 0.5) * _MESH_HEIGHT

        cos_az = math.cos(self._mesh_azimuth)
        sin_az = math.sin(self._mesh_azimuth)
        cos_el = math.cos(self._mesh_elevation)
        sin_el = math.sin(self._mesh_elevation)
        projected_x = grid_x * cos_az - grid_y * sin_az
        rotated_y = grid_x * sin_az + grid_y * cos_az
        projected_y = rotated_y * sin_el - grid_z * cos_el
        view_depth = rotated_y * cos_el + grid_z * sin_el

        span_x = max(0.0, float(projected_x.max() - projected_x.min()))
        span_y = max(0.0, float(projected_y.max() - projected_y.min()))
        # 仰角接近 0° 且图像完全平坦时 span_y 可能为 0；不要让退化的
        # 维度参与缩放，否则会把所有点放大到极端坐标。
        limits: list[float] = []
        if span_x > 1e-6:
            limits.append((float(field.width) * _MESH_PADDING) / span_x)
        if span_y > 1e-6:
            limits.append((float(field.height) * _MESH_PADDING) / span_y)
        scale = min(limits) if limits else 1.0
        points = np.empty((row_count, col_count, 2), dtype=np.float32)
        points[:, :, 0] = (projected_x - projected_x.min()) * scale + (
            float(field.width) - span_x * scale
        ) / 2.0
        points[:, :, 1] = (projected_y - projected_y.min()) * scale + (
            float(field.height) - span_y * scale
        ) / 2.0
        return _MeshGeometry(
            points=points,
            colors=field.colors,
            depth=view_depth.astype(np.float32),
        )

    def _draw_mesh(self, painter: QPainter, pane: _Pane) -> None:
        geometry = self._mesh_geometry_for(pane.image)
        if geometry is None:
            return
        if self._render_mesh_gl(painter, pane, geometry):
            return
        self._draw_mesh_software(painter, pane, geometry)

    # -------------------------------------------------------- OpenGL 曲面
    def _ensure_mesh_gl(self) -> bool:
        """懒初始化离屏 OpenGL；不可用时静默回退到软件渲染。"""

        if self._gl_ready:
            return True
        if self._gl_failed:
            return False
        app = QGuiApplication.instance()
        if app is not None and app.platformName().lower() in {"offscreen", "minimal"}:
            # 离屏平台没有 GL 上下文，直接使用软件回退，避免无意义的警告。
            self._gl_failed = True
            return False
        context: QOpenGLContext | None = None
        try:
            fmt = QSurfaceFormat()
            fmt.setVersion(3, 3)
            fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
            fmt.setDepthBufferSize(24)
            fmt.setSamples(0)
            context = QOpenGLContext(self)
            context.setFormat(fmt)
            if not context.create():
                raise RuntimeError("无法创建 OpenGL 3.3 上下文")
            surface = QOffscreenSurface(None, self)
            surface.setFormat(context.format())
            surface.create()
            if not surface.isValid():
                raise RuntimeError("无法创建 OpenGL 离屏表面")
            if not context.makeCurrent(surface):
                raise RuntimeError("无法激活 OpenGL 上下文")

            program = QOpenGLShaderProgram(self)
            if not program.addShaderFromSourceCode(
                QOpenGLShader.ShaderTypeBit.Vertex, _MESH_GL_VERTEX_SHADER
            ):
                raise RuntimeError(program.log())
            if not program.addShaderFromSourceCode(
                QOpenGLShader.ShaderTypeBit.Fragment, _MESH_GL_FRAGMENT_SHADER
            ):
                raise RuntimeError(program.log())
            if not program.link():
                raise RuntimeError(program.log())
            vbo = QOpenGLBuffer(QOpenGLBuffer.Type.VertexBuffer)
            if not vbo.create():
                raise RuntimeError("无法创建顶点缓冲")
            vao = QOpenGLVertexArrayObject(self)
            if not vao.create():
                raise RuntimeError("无法创建顶点数组对象")
            vao.bind()
            vbo.bind()
            program.bind()
            program.enableAttributeArray(0)
            program.setAttributeBuffer(0, _GL_FLOAT, 0, 3, 6 * 4)
            program.enableAttributeArray(1)
            program.setAttributeBuffer(1, _GL_FLOAT, 3 * 4, 3, 6 * 4)
            vao.release()
            program.release()

            self._gl_context = context
            self._gl_surface = surface
            self._gl_program = program
            self._gl_vbo = vbo
            self._gl_vao = vao
            self._gl_ready = True
            self._gl_failed = False
            return True
        except Exception as exc:  # noqa: BLE001 - GL 不可用时必须安静回退
            LOG.info("mesh OpenGL 渲染不可用，改用软件渲染：%s", exc)
            self._gl_failed = True
            self._gl_ready = False
            return False
        finally:
            if context is not None:
                try:
                    context.doneCurrent()
                except Exception:  # noqa: BLE001
                    pass

    def _render_mesh_gl(
        self, painter: QPainter, pane: _Pane, geometry: _MeshGeometry
    ) -> bool:
        """用离屏 GL 渲染曲面，返回是否成功（失败后本控件永久回退软件渲染）。"""

        if not self._ensure_mesh_gl():
            return False
        context = self._gl_context
        surface = self._gl_surface
        program = self._gl_program
        vbo = self._gl_vbo
        vao = self._gl_vao
        if context is None or surface is None or program is None or vbo is None or vao is None:
            return False
        width = float(pane.rect.width())
        height = float(pane.rect.height())
        if width < 2.0 or height < 2.0:
            return False
        try:
            if not context.makeCurrent(surface):
                raise RuntimeError("无法激活 OpenGL 上下文")
            scale = min(
                _MESH_GL_SUPERSAMPLE,
                max(0.5, _MESH_GL_MAX_SIDE / max(width, height)),
            )
            fbo_size = QSize(
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            if self._gl_fbo is None or self._gl_fbo_size != fbo_size:
                fbo_format = QOpenGLFramebufferObjectFormat()
                fbo_format.setAttachment(QOpenGLFramebufferObject.Attachment.Depth)
                fbo_format.setSamples(4)
                framebuffer = QOpenGLFramebufferObject(fbo_size, fbo_format)
                if not framebuffer.isValid():
                    raise RuntimeError("无法创建 OpenGL 帧缓冲")
                self._gl_fbo = framebuffer
                self._gl_fbo_size = fbo_size
            framebuffer = self._gl_fbo
            if framebuffer is None:
                return False
            framebuffer.bind()

            data, row_count, col_count = self._build_mesh_gl_vertices(
                geometry, pane, width, height
            )

            gl = context.functions()
            gl.glViewport(0, 0, fbo_size.width(), fbo_size.height())
            gl.glClearColor(0.0, 0.0, 0.0, 0.0)
            gl.glEnable(_GL_DEPTH_TEST)
            gl.glDepthFunc(_GL_LESS)
            gl.glClear(_GL_COLOR_BUFFER_BIT | _GL_DEPTH_BUFFER_BIT)
            program.bind()
            raw = data.tobytes()
            vbo.bind()
            vbo.allocate(raw, len(raw))
            vao.bind()
            vertices_per_strip = 2 * col_count
            offset = 0
            for _ in range(row_count - 1):
                gl.glDrawArrays(_GL_TRIANGLE_STRIP, offset, vertices_per_strip)
                offset += vertices_per_strip
            vao.release()
            program.release()
            vbo.release()
            image = framebuffer.toImage()
            framebuffer.release()
            context.doneCurrent()
            if image.isNull():
                raise RuntimeError("OpenGL 帧缓冲读回失败")
            painter.save()
            painter.setClipRect(pane.rect)
            painter.drawImage(QRectF(pane.rect), image)
            painter.restore()
            return True
        except Exception as exc:  # noqa: BLE001 - 回退软件渲染
            LOG.info("mesh OpenGL 渲染失败，改用软件渲染：%s", exc)
            self._gl_failed = True
            self._gl_ready = False
            try:
                context.doneCurrent()
            except Exception:  # noqa: BLE001
                pass
            return False

    def _build_mesh_gl_vertices(
        self, geometry: _MeshGeometry, pane: _Pane, width: float, height: float
    ) -> tuple[np.ndarray, int, int]:
        """把投影网格组装成 GL 三角形条带顶点（位置 NDC + 颜色 + 网格坐标）。"""

        points = geometry.points
        colors = geometry.colors
        depth = geometry.depth
        row_count, col_count = points.shape[:2]
        if row_count < 2 or col_count < 2:
            return np.zeros((0, 6), dtype=np.float32), row_count, col_count
        zoom = self._zoom
        local_x = (
            points[:, :, 0] * zoom + pane.offset.x() - float(pane.rect.left())
        ) / max(1.0, float(width))
        local_y = (
            points[:, :, 1] * zoom + pane.offset.y() - float(pane.rect.top())
        ) / max(1.0, float(height))
        low = float(np.min(depth))
        high = float(np.max(depth))
        if high > low:
            # OpenGL 的窗口深度越小越靠近相机；view_depth 越大越靠近相机，
            # 因此这里做一次翻转，保证 GL_LESS 消隐的是背面。
            depth_ndc = 1.0 - (depth - low) / (high - low) * 2.0
        else:
            depth_ndc = np.zeros_like(depth)
        full = np.empty((row_count, col_count, 6), dtype=np.float32)
        full[:, :, 0] = local_x * 2.0 - 1.0
        full[:, :, 1] = 1.0 - local_y * 2.0
        full[:, :, 2] = depth_ndc
        full[:, :, 3:6] = colors * (1.0 / 255.0)
        strips = np.empty((row_count - 1, 2 * col_count, 6), dtype=np.float32)
        strips[:, 0::2, :] = full[:-1]
        strips[:, 1::2, :] = full[1:]
        return np.ascontiguousarray(strips), row_count, col_count

    # ------------------------------------------------------ 软件回退渲染
    def _draw_mesh_software(
        self, painter: QPainter, pane: _Pane, geometry: _MeshGeometry
    ) -> None:
        points = geometry.points
        colors = geometry.colors
        row_count, col_count = points.shape[:2]
        if row_count < 2 or col_count < 2:
            self._draw_mesh_fallback(painter, pane, points, colors)
            return
        stride = max(
            1,
            int(math.ceil(max(row_count, col_count) / float(_MESH_SOFTWARE_MAX_SIDE))),
        )
        if stride > 1:
            points = points[::stride, ::stride]
            colors = colors[::stride, ::stride]
            row_count, col_count = points.shape[:2]
            if row_count < 2 or col_count < 2:
                self._draw_mesh_fallback(painter, pane, points, colors)
                return
        zoom = self._zoom
        screen = np.empty_like(points)
        screen[:, :, 0] = points[:, :, 0] * zoom + pane.offset.x()
        screen[:, :, 1] = points[:, :, 1] * zoom + pane.offset.y()
        cos_az = math.cos(self._mesh_azimuth)
        sin_az = math.sin(self._mesh_azimuth)
        painter.save()
        painter.setClipRect(pane.rect)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if abs(sin_az) <= abs(cos_az):
            order = range(row_count - 1) if cos_az >= 0 else range(row_count - 2, -1, -1)
            for index in order:
                self._draw_mesh_strip(
                    painter,
                    pane,
                    screen[index, :, :],
                    screen[index + 1, :, :],
                    colors[index, :, :],
                    colors[index + 1, :, :],
                )
        else:
            order = range(col_count - 1) if sin_az >= 0 else range(col_count - 2, -1, -1)
            for index in order:
                self._draw_mesh_strip(
                    painter,
                    pane,
                    screen[:, index, :],
                    screen[:, index + 1, :],
                    colors[:, index, :],
                    colors[:, index + 1, :],
                )
        painter.restore()

    def _draw_mesh_strip(
        self,
        painter: QPainter,
        pane: _Pane,
        line_a: np.ndarray,
        line_b: np.ndarray,
        colors_a: np.ndarray,
        colors_b: np.ndarray,
    ) -> None:
        """画一条连续的曲面条带（两排高度点之间），用渐变色平滑过渡。"""

        point_count = len(line_a)
        if point_count < 2:
            return
        xs = np.concatenate((line_a[:, 0], line_b[:, 0]))
        ys = np.concatenate((line_a[:, 1], line_b[:, 1]))
        if (
            float(xs.max()) < pane.rect.left()
            or float(xs.min()) > pane.rect.right()
            or float(ys.max()) < pane.rect.top()
            or float(ys.min()) > pane.rect.bottom()
        ):
            return

        polygon = QPolygonF(
            [QPointF(float(x), float(y)) for x, y in line_a]
            + [QPointF(float(x), float(y)) for x, y in line_b[::-1]]
        )
        first = QPointF(float(line_a[0, 0]), float(line_a[0, 1]))
        last = QPointF(float(line_a[-1, 0]), float(line_a[-1, 1]))
        if first == last:
            mean = (colors_a.mean(axis=0) + colors_b.mean(axis=0)) / 2.0
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(int(mean[0]), int(mean[1]), int(mean[2])))
        else:
            gradient = QLinearGradient(first, last)
            stride = max(1, point_count // 64)
            last_index = max(1, point_count - 1)
            for index in range(0, point_count, stride):
                color = (colors_a[index] + colors_b[index]) * 0.5
                gradient.setColorAt(
                    index / last_index,
                    QColor(int(color[0]), int(color[1]), int(color[2])),
                )
            color = (colors_a[-1] + colors_b[-1]) * 0.5
            gradient.setColorAt(1.0, QColor(int(color[0]), int(color[1]), int(color[2])))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QBrush(gradient))
        painter.drawPolygon(polygon)

    def _draw_mesh_fallback(
        self,
        painter: QPainter,
        pane: _Pane,
        points: np.ndarray,
        colors: np.ndarray,
    ) -> None:
        """退化网格（单行 / 单列）无法构面片时，仍按折线绘制。"""

        painter.save()
        painter.setClipRect(pane.rect)
        line_width = _clamp(self._zoom * 0.9, 0.6, 2.0)
        if points.shape[0] >= 2:
            for col in range(points.shape[1]):
                self._draw_mesh_line(
                    painter, pane, points[:, col, :], colors[:, col, :], line_width
                )
        elif points.shape[1] >= 2:
            for row in range(points.shape[0]):
                self._draw_mesh_line(
                    painter, pane, points[row, :, :], colors[row, :, :], line_width
                )
        else:
            self._draw_mesh_line(painter, pane, points[0, :, :], colors[0, :, :], line_width)
        painter.restore()

    def _draw_mesh_line(
        self,
        painter: QPainter,
        pane: _Pane,
        points: np.ndarray,
        colors: np.ndarray,
        width: float,
    ) -> None:
        if len(points) < 1:
            return
        if len(points) == 1:
            point = points[0]
            color = colors[0]
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(int(color[0]), int(color[1]), int(color[2])))
            radius = max(1.5, width * 1.4)
            painter.drawEllipse(
                QPointF(
                    pane.offset.x() + float(point[0]) * self._zoom,
                    pane.offset.y() + float(point[1]) * self._zoom,
                ),
                radius,
                radius,
            )
            return
        polygon = QPolygonF(
            [
                QPointF(
                    pane.offset.x() + float(point[0]) * self._zoom,
                    pane.offset.y() + float(point[1]) * self._zoom,
                )
                for point in points
            ]
        )
        first = polygon.at(0)
        last = polygon.at(len(polygon) - 1)
        if first == last:
            pen = QPen(QColor(int(colors[0][0]), int(colors[0][1]), int(colors[0][2])), width)
        else:
            gradient = QLinearGradient(first, last)
            last_index = max(1, len(colors) - 1)
            for index, color in enumerate(colors):
                gradient.setColorAt(
                    index / last_index,
                    QColor(int(color[0]), int(color[1]), int(color[2])),
                )
            pen = QPen(QBrush(gradient), width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPolyline(polygon)

    def _paint_mesh_hint(self, painter: QPainter) -> None:
        """mesh 模式左下角的操作提示与当前视角。"""

        if self._image is None and self._original is None:
            return
        azimuth = math.degrees(self._mesh_azimuth) % 360.0
        if azimuth > 180.0:
            azimuth -= 360.0
        elevation = math.degrees(self._mesh_elevation)
        elevation = (elevation + 180.0) % 360.0 - 180.0
        lines = [
            "Mesh · 左键拖动旋转 · 滚轮缩放",
            (
                f"Shift/右键拖动平移 · 双击重置 · "
                f"方位 {azimuth:.0f}° / 仰角 {elevation:.0f}°"
            ),
        ]
        max_width = max(120.0, self.width() * 0.62)
        rect = self._chip_rect(painter, lines, max_width=max_width)
        self._paint_chip(
            painter,
            rect,
            lines,
            12.0,
            self.height() - 12.0,
            align_bottom=True,
        )

    def _paint_info_caption(self, painter: QPainter) -> None:
        lines: list[str] = []
        title = self._node_title or (
            "原始图像" if self._image is None and self._original is not None else ""
        )
        if title:
            lines.append(title)
        if self._info:
            lines.append(self._info)
        image = self._image or self._original
        if image is not None:
            zoom_text = f"{self.zoom_percent():.0f}%"
            mode_text = display_mode_title(self._display_mode)
            suffix = f" · {mode_text}" if self._display_mode != "imshow" else ""
            if len(lines) < 2:
                lines.append(f"{image.width}×{image.height} · {zoom_text}{suffix}")
            else:
                lines.append(f"{zoom_text} · {image.width}×{image.height}{suffix}")
        if not lines:
            return
        max_width = max(120.0, self.width() - 24.0)
        rect = self._chip_rect(painter, lines, max_width=max_width)
        self._paint_chip(
            painter,
            rect,
            lines,
            self.width() - 12.0,
            self.height() - 12.0,
            align_right=True,
            align_bottom=True,
        )

    def _chip_font(self, painter: QPainter, *, bold: bool) -> QFont:
        font = QFont(painter.font())
        font.setPointSizeF(8.8)
        font.setBold(bold)
        return font

    def _chip_rect(
        self, painter: QPainter, lines: list[str], *, max_width: float | None = None
    ) -> QRectF:
        bold_metrics = QFontMetrics(self._chip_font(painter, bold=True))
        normal_metrics = QFontMetrics(self._chip_font(painter, bold=False))
        widths: list[float] = []
        heights: list[float] = []
        for index, line in enumerate(lines):
            metrics = bold_metrics if index == 0 else normal_metrics
            widths.append(float(metrics.horizontalAdvance(line)))
            heights.append(float(metrics.height()))
        width = max(widths or [0.0]) + 16.0
        if max_width is not None:
            width = min(width, max(48.0, float(max_width)))
        height = max(heights or [16.0]) * len(lines) + 8.0
        return QRectF(0.0, 0.0, width, height)

    def _paint_chip(
        self,
        painter: QPainter,
        rect: QRectF,
        lines: list[str],
        x: float,
        y: float,
        *,
        align_right: bool = False,
        align_bottom: bool = False,
    ) -> None:
        left = x - rect.width() if align_right else x
        top = y - rect.height() if align_bottom else y
        rect.moveTo(left, top)
        painter.save()
        painter.setPen(QPen(self.LABEL_BORDER, 1.0))
        painter.setBrush(self.LABEL_BG)
        painter.drawRoundedRect(rect, 5.0, 5.0)
        line_height = (rect.height() - 8.0) / max(1, len(lines))
        available = max(0.0, rect.width() - 16.0)
        for index, text in enumerate(lines):
            font = self._chip_font(painter, bold=index == 0)
            painter.setFont(font)
            metrics = QFontMetrics(font)
            painter.setPen(self.LABEL_TEXT if index == 0 else self.LABEL_TEXT_MUTED)
            text = metrics.elidedText(text, Qt.TextElideMode.ElideRight, int(available))
            text_rect = QRectF(
                rect.left() + 8.0,
                rect.top() + 4.0 + index * line_height,
                available,
                line_height,
            )
            painter.drawText(
                text_rect,
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                text,
            )
        painter.restore()

    def _paint_pick_overlay(self, painter: QPainter) -> None:
        if self._pick_tool is None:
            return
        tool = self._pick_tool
        painter.save()
        if tool == "points":
            for index, point in enumerate(self._pick_points):
                view = self._image_point_to_view(point[0], point[1])
                painter.setPen(QPen(QColor("#ffffff"), 1.6))
                painter.setBrush(self.ACCENT)
                painter.drawEllipse(view, 4.5, 4.5)
                painter.setPen(QColor("#1f2937"))
                font = QFont(self.font())
                font.setPointSizeF(8.0)
                painter.setFont(font)
                painter.drawText(QPointF(view.x() + 7.0, view.y() - 7.0), str(index + 1))
        elif self._pick_start is not None and self._pick_current is not None:
            start = self._image_point_to_view(*self._pick_start)
            current = self._image_point_to_view(*self._pick_current)
            pen = QPen(self.ACCENT, 1.6, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            if tool == "rect":
                rect = QRectF(start, current).normalized()
                painter.setBrush(QColor(59, 130, 246, 48))
                painter.drawRect(rect)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor("#ffffff"), 1.4))
                painter.drawEllipse(start, 3.0, 3.0)
                painter.drawEllipse(current, 3.0, 3.0)
            else:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawLine(start, current)
                painter.setPen(QPen(QColor("#ffffff"), 1.4))
                painter.setBrush(self.ACCENT)
                painter.drawEllipse(start, 3.5, 3.5)
                painter.drawEllipse(current, 3.5, 3.5)
        painter.restore()

        hint = self._pick_hint_text()
        if hint:
            rect = self._chip_rect(
                painter, [hint], max_width=max(120.0, self.width() - 24.0)
            )
            x = (self.width() - rect.width()) / 2.0
            y = self.height() - rect.height() - 14.0
            self._paint_chip(painter, rect, [hint], x, y)

    def _pick_hint_text(self) -> str:
        tool = self._pick_tool
        if tool == "points":
            total = str(self._pick_count) if self._pick_count else "任意"
            return f"左键添加点（{len(self._pick_points)}/{total}） · Enter 或右键完成 · Esc 取消"
        if tool == "rect":
            return "拖动框选矩形 ROI · Esc 取消"
        if tool == "line":
            return "拖动拾取直线 · Esc 取消"
        return ""

    # ------------------------------------------------------------ 布局与变换
    def _image_area(self) -> QRectF:
        rect = QRectF(self.rect()).adjusted(self.MARGIN, self.MARGIN, -self.MARGIN, -self.MARGIN)
        if rect.width() < 10.0:
            rect.setWidth(10.0)
        if rect.height() < 10.0:
            rect.setHeight(10.0)
        return rect

    def _recenter_offset(self) -> None:
        image = self._image or self._original
        if image is None:
            self._offset = QPointF(0.0, 0.0)
            return
        area = self._image_area()
        width = image.width * self._zoom
        height = image.height * self._zoom
        self._offset = QPointF(
            area.left() + (area.width() - width) / 2.0,
            area.top() + (area.height() - height) / 2.0,
        )

    def _focus_on_available(self) -> None:
        if self._original is not None:
            self.fit_to_window()
            return
        self._zoom = 1.0
        self._offset = QPointF(0.0, 0.0)
        self.zoom_changed.emit(self.zoom_percent())

    def _panes(self) -> list[_Pane]:
        if self._mode == "side_by_side":
            return self._side_panes()
        image = self._image or self._original
        if image is None:
            return []
        qimage = self._image_qimage if self._image is not None else self._original_qimage
        kind = "current" if self._image is not None else "original"
        title = self._node_title if kind == "current" else self._original_title
        return [
            _Pane(
                rect=QRectF(self.rect()).adjusted(4, 4, -4, -4),
                offset=QPointF(self._offset),
                image=image,
                qimage=qimage,
                title=title,
                kind=kind,
            )
        ]

    def _side_panes(self) -> list[_Pane]:
        area = self._image_area()
        images: list[tuple[str, ImageData | None, QImage | None, str]] = []
        if self._original is not None:
            images.append(("original", self._original, self._original_qimage, self._original_title))
        if self._image is not None:
            images.append(("current", self._image, self._image_qimage, self._node_title or "当前结果"))
        if not images:
            return []
        if len(images) == 1:
            kind, image, qimage, title = images[0]
            offset = QPointF(
                area.left() + (area.width() - image.width * self._zoom) / 2.0,
                area.top() + (area.height() - image.height * self._zoom) / 2.0,
            )
            return [_Pane(area, offset, image, qimage, title, kind)]
        gap = 10.0
        pane_width = max(10.0, (area.width() - gap) / 2.0)
        rects = [
            QRectF(area.left(), area.top(), pane_width, area.height()),
            QRectF(area.right() - pane_width, area.top(), pane_width, area.height()),
        ]
        panes: list[_Pane] = []
        for index, (kind, image, qimage, title) in enumerate(images):
            rect = rects[index]
            offset = QPointF(
                rect.left() + (rect.width() - image.width * self._zoom) / 2.0,
                rect.top() + (rect.height() - image.height * self._zoom) / 2.0,
            )
            panes.append(_Pane(rect, offset, image, qimage, title, kind))
        return panes

    def _active_pane(self) -> _Pane | None:
        panes = self._panes()
        if not panes:
            return None
        for pane in panes:
            if pane.kind == "current":
                return pane
        return panes[0]

    def _pane_at(self, point: QPointF) -> _Pane | None:
        panes = self._panes()
        if not panes:
            return None
        if len(panes) == 1:
            return panes[0]
        if point.x() <= panes[0].rect.right():
            return panes[0]
        return panes[1]

    def _to_image_point(self, pane: _Pane, point: QPointF) -> tuple[float, float]:
        zoom = self._zoom if self._zoom > 1e-9 else 1.0
        return (
            (point.x() - pane.offset.x()) / zoom,
            (point.y() - pane.offset.y()) / zoom,
        )

    def _image_point_to_view(self, x: float, y: float) -> QPointF:
        pane = self._active_pane()
        if pane is None:
            return QPointF(0.0, 0.0)
        return QPointF(pane.offset.x() + x * self._zoom, pane.offset.y() + y * self._zoom)

    def _divider_view_x(self) -> float:
        pane = self._panes()[0] if self._panes() else None
        image = self._image or self._original
        if pane is None or image is None:
            return self.width() / 2.0
        value = self._swipe_image_x if self._swipe_image_x is not None else image.width * 0.5
        return pane.offset.x() + value * self._zoom

    def _build_qimage(self, image: ImageData | None) -> QImage | None:
        if image is None or self._display_mode == "mesh":
            return None
        max_dim = self._quality_max_dim if self._quality_enabled else None
        if self._display_mode == "imagesc":
            rgb = to_scaled_rgb(image, max_dim=max_dim)
        else:
            rgb = to_display_rgb(image, max_dim=max_dim)
        height, width = rgb.shape[:2]
        if height <= 0 or width <= 0:
            return None
        rgb = np.ascontiguousarray(rgb)
        qimage = QImage(
            rgb.data,
            width,
            height,
            int(rgb.strides[0]),
            QImage.Format.Format_RGB888,
        )
        return qimage.copy()

    # ------------------------------------------------------------------ 事件
    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        super().resizeEvent(event)
        if self._auto_fit and (self._image is not None or self._original is not None):
            self.fit_to_window()

    def wheelEvent(self, event: QWheelEvent) -> None:  # noqa: N802 - Qt 命名
        if self._image is None and self._original is None:
            event.ignore()
            return
        # mesh 是三维视图：滚轮直接缩放；二维模式仍保持 Ctrl+滚轮，避免误触。
        mesh_mode = self._display_mode == "mesh"
        if not mesh_mode and not (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
        ):
            event.ignore()
            return
        steps = event.angleDelta().y() / 120.0
        if steps == 0:
            event.accept()
            return
        old_zoom = self._zoom
        new_zoom = _clamp(old_zoom * (1.25**steps), self.MIN_ZOOM, self.MAX_ZOOM)
        if abs(new_zoom - old_zoom) < 1e-9:
            event.accept()
            return
        position = event.position()
        if self._mode != "side_by_side":
            pane = self._active_pane()
            if pane is not None:
                image_point = self._to_image_point(pane, position)
                self._offset = QPointF(
                    position.x() - image_point[0] * new_zoom,
                    position.y() - image_point[1] * new_zoom,
                )
        self._auto_fit = False
        self._zoom = new_zoom
        self.update()
        self.zoom_changed.emit(self.zoom_percent())
        if self._last_view_pos is not None:
            self._emit_pixel_info_at(self._last_view_pos)
        event.accept()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt 命名
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        position = event.position()
        if self._display_mode == "mesh" and (
            self._image is not None or self._original is not None
        ):
            # mesh：左键拖动旋转，Shift+左键或右键拖动平移。
            if event.button() == Qt.MouseButton.LeftButton:
                if event.modifiers() & Qt.KeyboardModifier.ShiftModifier:
                    self._panning = True
                    self._pan_last = position
                    self._auto_fit = False
                else:
                    self._mesh_rotating = True
                    self._mesh_rotate_last = position
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
            if event.button() in (Qt.MouseButton.RightButton, Qt.MouseButton.MiddleButton):
                self._panning = True
                self._pan_last = position
                self._auto_fit = False
                self.setCursor(Qt.CursorShape.ClosedHandCursor)
                event.accept()
                return
        if event.button() == Qt.MouseButton.RightButton:
            if self._pick_tool is not None:
                if self._pick_tool == "points":
                    self._try_finish_points()
                else:
                    self.cancel_pick()
                event.accept()
                return
            super().mousePressEvent(event)
            return
        if event.button() != Qt.MouseButton.LeftButton:
            super().mousePressEvent(event)
            return

        if self._pick_tool is not None:
            point = self._clamped_image_point(position, edge=(self._pick_tool != "points"))
            if point is None:
                event.accept()
                return
            if self._pick_tool == "points":
                if self._pick_count and len(self._pick_points) >= self._pick_count:
                    event.accept()
                    return
                self._pick_points.append([point[0], point[1]])
            else:
                self._pick_start = point
                self._pick_current = point
                self._pick_dragging = True
            self.update()
            event.accept()
            return

        if (
            self._mode == "swipe"
            and self._original is not None
            and self._image is not None
            and abs(position.x() - self._divider_view_x()) <= 9.0
        ):
            self._dragging_divider = True
            self.setCursor(Qt.CursorShape.SizeHorCursor)
            event.accept()
            return

        if self._mode != "side_by_side":
            self._panning = True
            self._pan_last = position
            self._auto_fit = False
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt 命名
        position = event.position()
        self._last_view_pos = QPointF(position)
        if self._mesh_rotating:
            delta = position - self._mesh_rotate_last
            self._mesh_rotate_last = position
            self._mesh_azimuth -= delta.x() * _MESH_ROTATE_SPEED_X
            # 仰角不设限制：可连续拖过顶部 / 底部，便于从任意方向观察曲面。
            self._mesh_elevation += delta.y() * _MESH_ROTATE_SPEED_Y
            self.update()
            return
        if self._dragging_divider:
            pane = self._panes()[0] if self._panes() else None
            image = self._image or self._original
            if pane is not None and image is not None and self._zoom > 1e-9:
                value = (position.x() - pane.offset.x()) / self._zoom
                self._swipe_image_x = _clamp(value, 0.0, float(image.width))
            self.update()
        elif self._panning:
            delta = position - self._pan_last
            self._pan_last = position
            self._offset += delta
            self.update()
        elif self._pick_tool in ("rect", "line") and self._pick_start is not None:
            point = self._clamped_image_point(position, edge=True)
            if point is not None:
                self._pick_current = point
                self.update()
        self._emit_pixel_info_at(position)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt 命名
        position = event.position()
        if self._mesh_rotating and event.button() == Qt.MouseButton.LeftButton:
            self._mesh_rotating = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            if self._dragging_divider:
                self._dragging_divider = False
                self.setCursor(Qt.CursorShape.ArrowCursor)
                event.accept()
                return
            if self._pick_tool in ("rect", "line") and self._pick_start is not None:
                point = self._clamped_image_point(position, edge=True)
                if point is not None:
                    self._pick_current = point
                self._finish_drag_pick()
                event.accept()
                return
            if self._panning:
                self._panning = False
                self.setCursor(
                    Qt.CursorShape.CrossCursor if self._pick_tool else Qt.CursorShape.ArrowCursor
                )
                event.accept()
                return
        elif event.button() in (
            Qt.MouseButton.RightButton,
            Qt.MouseButton.MiddleButton,
        ) and self._panning:
            self._panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt 命名
        # mesh：双击恢复默认视角与适应窗口，方便旋转/缩放后快速回到初始状态。
        if (
            self._display_mode == "mesh"
            and event.button() == Qt.MouseButton.LeftButton
            and (self._image is not None or self._original is not None)
        ):
            self.reset_mesh_view()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        self._last_view_pos = None
        if self._last_pixel_key is not None:
            self._last_pixel_key = None
            self.pixel_info.emit(None)
        super().leaveEvent(event)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt 命名
        if event.key() == Qt.Key.Key_Escape:
            if self._pick_tool is not None:
                self.cancel_pick()
                event.accept()
                return
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            if self._pick_tool == "points":
                self._try_finish_points()
                event.accept()
                return
        super().keyPressEvent(event)

    # ------------------------------------------------------------ 拾取内部实现
    def _clamped_image_point(
        self, position: QPointF, *, edge: bool = False
    ) -> tuple[float, float] | None:
        pane = self._active_pane()
        if pane is None or pane.image is None:
            return None
        x, y = self._to_image_point(pane, position)
        width = float(pane.image.width)
        height = float(pane.image.height)
        high_x = width if edge else max(0.0, width - 1.0)
        high_y = height if edge else max(0.0, height - 1.0)
        x = _clamp(x, 0.0, high_x)
        y = _clamp(y, 0.0, high_y)
        return (_round(x), _round(y))

    def _finish_drag_pick(self) -> None:
        tool = self._pick_tool
        if tool not in ("rect", "line") or self._pick_start is None:
            self._pick_start = None
            self._pick_current = None
            self._pick_dragging = False
            return
        start = self._pick_start
        current = self._pick_current or start
        if tool == "rect":
            x = min(start[0], current[0])
            y = min(start[1], current[1])
            width = abs(current[0] - start[0])
            height = abs(current[1] - start[1])
            if width < 0.5 or height < 0.5:
                self._pick_start = None
                self._pick_current = None
                self._pick_dragging = False
                self.update()
                return
            payload: Any = [_round(x), _round(y), _round(width), _round(height)]
        else:
            if math.hypot(current[0] - start[0], current[1] - start[1]) < 0.5:
                self._pick_start = None
                self._pick_current = None
                self._pick_dragging = False
                self.update()
                return
            payload = [
                _round(start[0]),
                _round(start[1]),
                _round(current[0]),
                _round(current[1]),
            ]
        self._pick_start = None
        self._pick_current = None
        self._pick_dragging = False
        self._finish_pick(payload)

    def _try_finish_points(self) -> None:
        if self._pick_tool != "points":
            return
        count = self._pick_count
        if not self._pick_points:
            self.cancel_pick()
            return
        if count and len(self._pick_points) < count:
            return
        payload = [list(point) for point in self._pick_points]
        self._pick_points = []
        self._finish_pick(payload)

    def _finish_pick(self, payload: Any) -> None:
        tool = self._pick_tool
        if tool is None:
            return
        self._clear_pick_state()
        self.update()
        self.pick_finished.emit(tool, payload)

    def _clear_pick_state(self) -> None:
        self._pick_tool = None
        self._pick_count = 1
        self._pick_points = []
        self._pick_start = None
        self._pick_current = None
        self._pick_dragging = False
        self._dragging_divider = False
        self.setCursor(Qt.CursorShape.ArrowCursor)

    # ------------------------------------------------------------ 像素采样
    def _emit_pixel_info_at(self, position: QPointF) -> None:
        sample = self._sample_at(position)
        key: tuple[Any, ...] | None
        if sample is None:
            key = None
        else:
            key = (
                id(self._image),
                id(self._original),
                sample.row,
                sample.col,
                sample.in_bounds,
                sample.dtype_name,
            )
        if key == self._last_pixel_key:
            return
        self._last_pixel_key = key
        self.pixel_info.emit(sample)

    def _sample_at(self, position: QPointF) -> PixelSample | None:
        if self._image is None and self._original is None:
            return None
        # 三维网格是投影视图，屏幕坐标不再对应图像行列，不做像素采样。
        if self._display_mode == "mesh":
            return None
        if (
            self._mode == "swipe"
            and self._image is not None
            and self._original is not None
            and position.x() < self._divider_view_x()
        ):
            pane = self._panes()[0]
            image = self._original
            assert image is not None
            x, y = self._to_image_point(pane, position)
        else:
            pane = self._pane_at(position)
            if pane is None or pane.image is None:
                return None
            image = pane.image
            x, y = self._to_image_point(pane, position)
        column = int(math.floor(x))
        row = int(math.floor(y))
        if column < 0 or row < 0 or column >= image.width or row >= image.height:
            return PixelSample(row, column, (), (), False, image.dtype_name)
        return sample_pixel(image, row, column)
