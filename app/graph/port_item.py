"""节点端口条目（输入 / 输出）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from app.graph.node_item import NodeItem


class PortItem(QGraphicsItem):
    """节点边缘的圆形端口：输入在上边缘、输出在下边缘，输出端口可拖拽出连线。"""

    RADIUS = 5.5
    HIT_RADIUS = 9.0

    def __init__(
        self,
        node_item: "NodeItem",
        key: str,
        label: str,
        is_input: bool,
        index: int,
    ) -> None:
        super().__init__(node_item)
        self.node_item = node_item
        self.key = key
        self.label = label
        self.is_input = is_input
        self.index = index
        self._connected = False
        self._hover = False
        self._highlight = False
        self.setZValue(3.0)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        role = "输入" if is_input else "输出"
        accepts = ""
        if is_input and node_item.spec is not None:
            try:
                accepts = node_item.spec.input(key).accept_text()
            except KeyError:
                accepts = ""
        tip = f"{role}端口：{label}"
        if accepts:
            tip += f"\n接受图像：{accepts}"
        self.setToolTip(tip)

    # ------------------------------------------------------------------ 状态
    def set_connected(self, connected: bool) -> None:
        connected = bool(connected)
        if connected != self._connected:
            self._connected = connected
            self.update()

    @property
    def connected(self) -> bool:
        return self._connected

    def set_highlight(self, highlight: bool) -> None:
        highlight = bool(highlight)
        if highlight != self._highlight:
            self._highlight = highlight
            self.update()

    def center_scene_pos(self) -> QPointF:
        return self.scenePos()

    # --------------------------------------------------------------- 几何绘制
    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt 接口
        radius = self.HIT_RADIUS + 3.0
        return QRectF(-radius, -radius, radius * 2.0, radius * 2.0)

    def shape(self) -> QPainterPath:  # noqa: N802 - Qt 接口
        path = QPainterPath()
        path.addEllipse(QPointF(0.0, 0.0), self.HIT_RADIUS, self.HIT_RADIUS)
        return path

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:  # noqa: N802
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self._highlight:
            pen = QPen(QColor("#2563eb"), 1.6, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawEllipse(QPointF(0.0, 0.0), self.HIT_RADIUS, self.HIT_RADIUS)

        if self._connected:
            brush = QColor("#3b82f6")
            pen_color = QColor("#1d4ed8")
        else:
            brush = QColor("#ffffff")
            pen_color = QColor("#64748b")
        if self._hover:
            pen_color = QColor("#2563eb")
        painter.setBrush(brush)
        painter.setPen(QPen(pen_color, 1.6))
        painter.drawEllipse(QPointF(0.0, 0.0), self.RADIUS, self.RADIUS)

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
        # 端口上的点击不应改变节点选中状态，避免误触发预览。
        if self.is_input:
            event.accept()
            return
        scene = self.scene()
        begin = getattr(scene, "begin_connection_drag", None)
        if callable(begin):
            begin(self)
            event.accept()
            return
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        event.accept()


__all__ = ["PortItem"]
