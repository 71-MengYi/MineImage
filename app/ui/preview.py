"""右侧图像预览控件。

职责
----
* 绘制 :class:`~app.core.types.ImageData`（BGR/BGRA/灰度 → 显示 RGB）；
* 缩放 / 平移：``Ctrl + 滚轮`` 以光标为中心缩放，左键拖动平移；
* 对比：单图、左右对比、滑动对比（分割线以图像坐标保存，随缩放/平移对齐）；
* 交互拾取：点集、矩形 ROI、直线，屏幕坐标与图像坐标双向映射；
* 像素采样：通过 :func:`app.core.types.sample_pixel` 返回左上原点的行/列与原始值。

本控件只负责显示，不参与任何计算与下游失效逻辑。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QWheelEvent,
)
from PySide6.QtWidgets import QSizePolicy, QWidget

from app.core.types import ImageData, PixelSample, sample_pixel, to_display_rgb

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
        self._zoom = 1.0
        self._offset = QPointF(0.0, 0.0)
        self._auto_fit = True
        self._fit_on_new_image = True

        self._quality_enabled = False
        self._quality_max_dim = 2048
        self._image_qimage: QImage | None = None
        self._original_qimage: QImage | None = None

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
        if pane.image is not None and pane.qimage is not None:
            self._draw_pane_image(painter, pane)

    def _paint_side_by_side(self, painter: QPainter) -> None:
        panes = self._panes()
        for index, pane in enumerate(panes):
            if index:
                painter.setPen(QPen(QColor(15, 23, 42, 34), 1.0))
                seam = pane.rect.left() - 5.0
                painter.drawLine(QPointF(seam, 8.0), QPointF(seam, self.height() - 8.0))
            self._paint_pane_background(painter, pane.rect)
            if pane.image is not None and pane.qimage is not None:
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
        if pane.image is not None and pane.qimage is not None:
            self._draw_pane_image(painter, pane)

        both = self._original is not None and self._image is not None
        if both and self._original_qimage is not None:
            divider = self._divider_view_x()
            clip_left = QRectF(pane.rect.left(), pane.rect.top(), max(0.0, divider - pane.rect.left()), pane.rect.height())
            painter.save()
            painter.setClipRect(clip_left)
            original = self._original
            assert original is not None
            target = QRectF(
                pane.offset.x(),
                pane.offset.y(),
                original.width * self._zoom,
                original.height * self._zoom,
            )
            painter.drawImage(target, self._original_qimage)
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
        qimage = pane.qimage
        if image is None or qimage is None:
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
            if len(lines) < 2:
                lines.append(f"{image.width}×{image.height} · {zoom_text}")
            else:
                lines.append(f"{zoom_text} · {image.width}×{image.height}")
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
        if image is None:
            return None
        max_dim = self._quality_max_dim if self._quality_enabled else None
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
        if not (event.modifiers() & Qt.KeyboardModifier.ControlModifier):
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
        super().mouseReleaseEvent(event)

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
