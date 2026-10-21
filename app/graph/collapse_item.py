"""分支节点的折叠 / 展开按钮。

多分支连接点（有至少两个下游子节点的节点）下方显示一个圆形按钮：

* 展开状态：圆形框内的减号，点击折叠全部下游节点；
* 折叠状态：圆形框内的数字表示被隐藏的下游节点数量，点击展开。

按钮不是图模型的一部分，由 :class:`~app.graph.view.GraphView` 根据图结构
动态创建、定位和显示 / 隐藏。
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from app.graph.view import GraphView


class CollapseBadgeItem(QGraphicsItem):
    """节点下方圆形折叠按钮（减号 / 数字）。"""

    #: 圆形按钮半径。
    RADIUS = 13.0
    #: 按钮圆心相对节点下边缘的垂直偏移（留出与连线分叉的间距）。
    BOTTOM_OFFSET = 38.0

    def __init__(self, view: "GraphView", node_id: str) -> None:
        super().__init__()
        self._view = view
        self.node_id = node_id
        self._collapsed = False
        self._count = 0
        self._hover = False
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setZValue(3.0)
        self._refresh_tooltip()

    # ------------------------------------------------------------------ 状态
    @property
    def collapsed(self) -> bool:
        return self._collapsed

    @property
    def count(self) -> int:
        return self._count

    def set_state(self, collapsed: bool, count: int) -> None:
        """更新按钮状态；只在状态变化时重绘。"""

        collapsed = bool(collapsed)
        count = max(0, int(count))
        if (collapsed, count) == (self._collapsed, self._count):
            return
        self._collapsed = collapsed
        self._count = count
        self._refresh_tooltip()
        self.update()

    def _refresh_tooltip(self) -> None:
        action = "展开" if self._collapsed else "折叠"
        self.setToolTip(f"{action}子节点（{self._count} 个下游节点）")

    # ------------------------------------------------------------------ 几何
    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt 接口
        radius = self.RADIUS + 2.0
        return QRectF(-radius, -radius, radius * 2.0, radius * 2.0)

    def shape(self) -> QPainterPath:  # noqa: N802 - Qt 接口
        radius = self.RADIUS
        path = QPainterPath()
        path.addEllipse(QRectF(-radius, -radius, radius * 2.0, radius * 2.0))
        return path

    def hit_test(self, scene_pos: QPointF) -> bool:
        return self.shape().contains(self.mapFromScene(scene_pos))

    # ------------------------------------------------------------------ 绘制
    def paint(  # noqa: N802 - Qt 接口
        self,
        painter: QPainter,
        option: QStyleOptionGraphicsItem,
        widget: QWidget | None = None,
    ) -> None:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        radius = self.RADIUS
        circle = QRectF(-radius, -radius, radius * 2.0, radius * 2.0)
        if self._collapsed:
            fill = QColor("#2563eb")
            border = QColor("#1d4ed8") if self._hover else QColor("#2563eb")
            foreground = QColor("#ffffff")
        else:
            fill = QColor("#eff6ff") if self._hover else QColor("#ffffff")
            border = QColor("#2563eb") if self._hover else QColor("#94a3b8")
            foreground = QColor("#2563eb") if self._hover else QColor("#475569")
        painter.setPen(QPen(border, 1.4))
        painter.setBrush(fill)
        painter.drawEllipse(circle)

        if self._collapsed:
            font = QFont("Microsoft YaHei UI")
            font.setPixelSize(10 if self._count < 100 else 9)
            font.setBold(True)
            painter.setFont(font)
            painter.setPen(foreground)
            label = str(self._count) if self._count < 100 else "99+"
            painter.drawText(circle, int(Qt.AlignmentFlag.AlignCenter), label)
        else:
            pen = QPen(foreground, 2.0)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawLine(QPointF(-5.0, 0.0), QPointF(5.0, 0.0))

    # ------------------------------------------------------------------ 交互
    def hoverEnterEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        self._hover = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        self._hover = False
        self.update()
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        if event.button() == Qt.MouseButton.LeftButton:
            self._view.toggle_node_collapsed(self.node_id)
            event.accept()
            return
        super().mousePressEvent(event)


__all__ = ["CollapseBadgeItem"]
