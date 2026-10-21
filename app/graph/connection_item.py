"""节点之间的连线条目。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPainterPathStroker, QPen, QPolygonF
from PySide6.QtWidgets import QGraphicsItem, QGraphicsPathItem, QStyleOptionGraphicsItem, QWidget

from app.core.model import ConnectionModel
from app.graph.curve import connection_path

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from app.graph.view import GraphView


class ConnectionItem(QGraphicsPathItem):
    """三次贝塞尔连线，支持悬停 / 选中与命中测试。"""

    HIT_WIDTH = 12.0

    def __init__(self, view: "GraphView", connection: ConnectionModel) -> None:
        super().__init__()
        self._view = view
        self.connection_id = connection.id
        self.src_node = connection.src_node
        self.src_port = connection.src_port
        self.dst_node = connection.dst_node
        self.dst_port = connection.dst_port
        self._hover = False
        self._highlight = False
        self.setZValue(-2.0)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True)
        self.setAcceptHoverEvents(True)
        self.setToolTip("连线")
        self.update_path()

    # ------------------------------------------------------------------ 路径
    def update_path(self) -> None:
        src_port = self._view.port_item(self.src_node, self.src_port, False)
        dst_port = self._view.port_item(self.dst_node, self.dst_port, True)
        if src_port is None or dst_port is None:
            self.setPath(QPainterPath())
            return
        start = src_port.center_scene_pos()
        end = dst_port.center_scene_pos()
        # 输出端口在节点下边缘、输入端口在上边缘，统一使用纵向 S 形曲线。
        self.setPath(connection_path(start, end))

    def set_highlight(self, highlight: bool) -> None:
        """自动连线悬停到候选输入端口时的高亮（交互层调用）。"""

        highlight = bool(highlight)
        if highlight != self._highlight:
            self._highlight = highlight
            self.update()

    @property
    def highlighted(self) -> bool:
        return self._highlight

    def hit_test(self, scene_pos: QPointF) -> bool:
        if self.path().isEmpty():
            return False
        return self._stroke().contains(scene_pos)

    def _stroke(self) -> QPainterPath:
        stroker = QPainterPathStroker()
        stroker.setWidth(self.HIT_WIDTH)
        return stroker.createStroke(self.path())

    # --------------------------------------------------------------- 几何绘制
    def shape(self) -> QPainterPath:  # noqa: N802 - Qt 接口
        return self._stroke()

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt 接口
        return super().boundingRect().adjusted(-6.0, -6.0, 6.0, 6.0)

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:  # noqa: N802
        path = self.path()
        if path.isEmpty():
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        if self.isSelected():
            # 选中优先于自动连线高亮。
            color = QColor("#2563eb")
            width = 2.6
        elif self._highlight:
            color = QColor("#2563eb")
            width = 3.0
        elif self._hover:
            color = QColor("#3b82f6")
            width = 2.4
        else:
            color = QColor("#8a97a6")
            width = 2.0
        pen = QPen(color, width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.setBrush(QColor(0, 0, 0, 0))
        painter.drawPath(path)

        # 终点箭头
        end = path.pointAtPercent(1.0)
        before = path.pointAtPercent(0.965)
        dx = end.x() - before.x()
        dy = end.y() - before.y()
        if abs(dx) > 1e-9 or abs(dy) > 1e-9:
            angle = math.degrees(math.atan2(dy, dx))
            painter.save()
            painter.translate(end)
            painter.rotate(angle)
            arrow = QPolygonF(
                [
                    QPointF(0.0, 0.0),
                    QPointF(-8.0, 4.0),
                    QPointF(-8.0, -4.0),
                ]
            )
            painter.setPen(QPen(color, 1.0))
            painter.setBrush(color)
            painter.drawPolygon(arrow)
            painter.restore()

    # ------------------------------------------------------------------ 交互
    def hoverEnterEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        self._hover = True
        self.update()
        super().hoverEnterEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        self._hover = False
        self.update()
        super().hoverLeaveEvent(event)


__all__ = ["ConnectionItem"]
