"""预览显示模式（Imshow / Imagesc / Mesh）的界面元数据与矢量图标。

图标全部用 :class:`QPainter` 绘制，不依赖外部资源文件，因此在打包版里也能
随系统 DPI 清晰缩放：

* **Imshow**：带太阳与山峦的相框，代表按原始颜色 / 灰度显示；
* **Imagesc**：色带填充的网格，代表把数值范围映射到伪彩色；
* **Mesh**：透视网格曲面，代表以像素强度为高度的三维网格。
"""

from __future__ import annotations

import math

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import (
    QColor,
    QIcon,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QPolygonF,
)

from app.core.constants import IMAGE_DISPLAY_MODES, DEFAULT_IMAGE_DISPLAY_MODE

#: 工具栏 / 菜单里使用的模式元数据：mode, 标题, 悬浮说明。
DISPLAY_MODE_SPECS: tuple[tuple[str, str, str], ...] = (
    (
        "imshow",
        "Imshow",
        "Imshow：按原始颜色 / 灰度显示当前节点结果（MATLAB imshow）",
    ),
    (
        "imagesc",
        "Imagesc",
        "Imagesc：把数值范围映射到色带，突出弱对比结构（MATLAB imagesc）",
    ),
    (
        "mesh",
        "Mesh",
        "Mesh：以像素强度为高度绘制三维曲面网格；拖动旋转、滚轮缩放、双击重置（MATLAB mesh）",
    ),
)

_MODE_TITLES = {mode: title for mode, title, _tip in DISPLAY_MODE_SPECS}
_ICON_SIZES = (16, 18, 20, 24, 32, 40, 48, 64, 96)
_ICON_CACHE: dict[str, QIcon] = {}

_ACCENT = QColor("#2563eb")
_ACCENT_DARK = QColor("#1d4ed8")
_ACCENT_LIGHT = QColor("#93c5fd")
_FRAME = QColor("#334155")
_PAPER = QColor("#ffffff")

#: parula / jet 混合的色带控制点，用于 imagesc 图标。
_COLORMAP_STOPS = (
    (0.00, QColor("#352a87")),
    (0.18, QColor("#0f5cdd")),
    (0.36, QColor("#00b5a3")),
    (0.54, QColor("#f2c14e")),
    (0.72, QColor("#f98c40")),
    (0.88, QColor("#e2462f")),
    (1.00, QColor("#a3123a")),
)


def display_mode_title(mode: str) -> str:
    """返回显示模式的展示名称（``Imshow`` / ``Imagesc`` / ``Mesh``）。"""

    fallback = _MODE_TITLES[DEFAULT_IMAGE_DISPLAY_MODE]
    return _MODE_TITLES.get(str(mode or "").strip().lower(), fallback)


def display_mode_icon(mode: str) -> QIcon:
    """返回显示模式图标（带多档尺寸，按需生成并缓存）。"""

    normalized = str(mode or "").strip().lower()
    if normalized not in IMAGE_DISPLAY_MODES:
        normalized = DEFAULT_IMAGE_DISPLAY_MODE
    cached = _ICON_CACHE.get(normalized)
    if cached is not None:
        return cached
    icon = QIcon()
    for size in _ICON_SIZES:
        icon.addPixmap(_paint_mode_pixmap(normalized, size))
    _ICON_CACHE[normalized] = icon
    return icon


def clear_icon_cache() -> None:
    """清空图标缓存（仅测试或主题切换时使用）。"""

    _ICON_CACHE.clear()


# ---------------------------------------------------------------------------
# 绘制实现
# ---------------------------------------------------------------------------
def _paint_mode_pixmap(mode: str, size: int) -> QPixmap:
    pixmap = QPixmap(QSize(size, size))
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
    try:
        unit = size / 64.0
        if mode == "imshow":
            _paint_imshow(painter, unit)
        elif mode == "imagesc":
            _paint_imagesc(painter, unit)
        else:
            _paint_mesh(painter, unit)
    finally:
        painter.end()
    return pixmap


def _frame_rect(unit: float) -> QRectF:
    return QRectF(7.0 * unit, 13.0 * unit, 50.0 * unit, 38.0 * unit)


def _stroke(unit: float, scale: float = 2.4) -> float:
    return max(1.0, scale * unit)


def _paint_imshow(painter: QPainter, unit: float) -> None:
    frame = _frame_rect(unit)
    radius = 5.0 * unit
    # 相框底色 + 天空
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(_PAPER)
    painter.drawRoundedRect(frame, radius, radius)

    clip = QPainterPath()
    clip.addRoundedRect(frame, radius, radius)
    painter.save()
    painter.setClipPath(clip)
    sky = QLinearGradient(frame.topLeft(), frame.bottomLeft())
    sky.setColorAt(0.0, QColor("#dbeafe"))
    sky.setColorAt(1.0, QColor("#f8fbff"))
    painter.setBrush(sky)
    painter.drawRect(frame)

    # 太阳
    painter.setBrush(QColor("#fbbf24"))
    painter.setPen(Qt.PenStyle.NoPen)
    painter.drawEllipse(
        QPointF(frame.left() + 36.0 * unit, frame.top() + 12.0 * unit),
        5.0 * unit,
        5.0 * unit,
    )

    # 两层山峦
    painter.setBrush(QColor("#60a5fa"))
    painter.drawPolygon(
        QPolygonF(
            [
                QPointF(frame.left() + 2.0 * unit, frame.bottom()),
                QPointF(frame.left() + 19.0 * unit, frame.top() + 17.0 * unit),
                QPointF(frame.left() + 32.0 * unit, frame.bottom()),
            ]
        )
    )
    painter.setBrush(_ACCENT)
    painter.drawPolygon(
        QPolygonF(
            [
                QPointF(frame.left() + 20.0 * unit, frame.bottom()),
                QPointF(frame.left() + 35.0 * unit, frame.top() + 20.0 * unit),
                QPointF(frame.left() + 50.0 * unit, frame.bottom()),
            ]
        )
    )
    painter.restore()

    # 外框
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(_FRAME, _stroke(unit, 2.2)))
    painter.drawRoundedRect(frame, radius, radius)

    # 底部相框底座，强调“图像窗口”
    painter.setPen(QPen(_FRAME, _stroke(unit, 2.0)))
    painter.drawLine(
        QPointF(frame.left() + 14.0 * unit, frame.bottom() + 6.0 * unit),
        QPointF(frame.right() - 14.0 * unit, frame.bottom() + 6.0 * unit),
    )


def _paint_imagesc(painter: QPainter, unit: float) -> None:
    frame = _frame_rect(unit)
    radius = 5.0 * unit
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(_PAPER)
    painter.drawRoundedRect(frame, radius, radius)

    clip = QPainterPath()
    clip.addRoundedRect(frame, radius, radius)
    painter.save()
    painter.setClipPath(clip)
    gradient = QLinearGradient(frame.topLeft(), frame.bottomRight())
    for position, color in _COLORMAP_STOPS:
        gradient.setColorAt(position, color)
    painter.setBrush(gradient)
    painter.drawRect(frame)

    # 网格线：隐式矩阵被映射到色带
    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(QColor(255, 255, 255, 150), max(0.8, 1.6 * unit)))
    columns = 6
    rows = 4
    for index in range(1, columns):
        x = frame.left() + frame.width() * index / columns
        painter.drawLine(QPointF(x, frame.top()), QPointF(x, frame.bottom()))
    for index in range(1, rows):
        y = frame.top() + frame.height() * index / rows
        painter.drawLine(QPointF(frame.left(), y), QPointF(frame.right(), y))
    painter.restore()

    painter.setBrush(Qt.BrushStyle.NoBrush)
    painter.setPen(QPen(_FRAME, _stroke(unit, 2.2)))
    painter.drawRoundedRect(frame, radius, radius)

    # 右下角色带端点的“标尺”小方块
    dot = 3.2 * unit
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor("#352a87"))
    painter.drawRect(QRectF(frame.left() + 4.0 * unit, frame.bottom() + 3.0 * unit, dot * 2.4, dot))
    painter.setBrush(QColor("#a3123a"))
    painter.drawRect(
        QRectF(frame.right() - 4.0 * unit - dot * 2.4, frame.bottom() + 3.0 * unit, dot * 2.4, dot)
    )


def _paint_mesh(painter: QPainter, unit: float) -> None:
    frame = _frame_rect(unit)
    center_x = frame.center().x()
    base_y = frame.top() + 18.0 * unit
    width = frame.width() * 0.78
    depth = 9.0 * unit
    grid = 6
    heights: list[list[float]] = []
    for row in range(grid):
        values: list[float] = []
        for col in range(grid):
            u = col / (grid - 1)
            v = row / (grid - 1)
            # 中央隆起的平滑曲面，让图标一眼看出是“高度场”。
            height = math.exp(-((u - 0.5) ** 2 + (v - 0.5) ** 2) / 0.55)
            values.append(height)
        heights.append(values)
    max_height = max(max(row) for row in heights) or 1.0

    def project(col: int, row: int) -> QPointF:
        u = col / (grid - 1) - 0.5
        v = row / (grid - 1) - 0.5
        x = center_x + u * width + v * depth * 0.65
        y = base_y + v * depth * 0.9 - (heights[row][col] / max_height) * 7.0 * unit
        return QPointF(x, y)

    # 行线：从后往前由浅到深，制造纵深感。
    for row in range(grid):
        points = [project(col, row) for col in range(grid)]
        shade = row / (grid - 1)
        color = QColor(_ACCENT_LIGHT).lighter(int(108 - 8 * shade))
        color.setAlpha(130 + int(100 * shade))
        painter.setPen(QPen(color, _stroke(unit, 1.6)))
        painter.drawPolyline(QPolygonF(points))
    # 列线：用主色强调高度变化。
    for col in range(grid):
        points = [project(col, row) for row in range(grid)]
        color = QColor(_ACCENT if col % 2 else _ACCENT_DARK)
        color.setAlpha(215)
        painter.setPen(QPen(color, _stroke(unit, 1.8)))
        painter.drawPolyline(QPolygonF(points))