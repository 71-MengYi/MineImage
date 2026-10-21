"""节点画布场景：网格背景、连线拖拽与视图交互转发。"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import QGraphicsPathItem, QGraphicsScene, QGraphicsSceneMouseEvent

from app.graph.curve import connection_path
from app.graph.port_item import PortItem

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from app.graph.view import GraphView


class GraphScene(QGraphicsScene):
    """画布场景。把端口拖拽等交互集中在场景内处理。"""

    GRID = 28.0

    def __init__(self, view: "GraphView") -> None:
        super().__init__(view)
        self._view = view
        self._temp_connection: QGraphicsPathItem | None = None
        self._drag_source: PortItem | None = None
        self._highlighted: PortItem | None = None

    # --------------------------------------------------------------- 背景绘制
    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802 - Qt 接口
        painter.fillRect(rect, QColor("#f6f7f9"))
        grid = self.GRID
        left = math.floor(rect.left() / grid) * grid
        top = math.floor(rect.top() / grid) * grid
        columns = int((rect.right() - left) / grid) + 1
        rows = int((rect.bottom() - top) / grid) + 1
        if columns <= 0 or rows <= 0 or columns * rows > 20000:
            return
        pen = QPen(QColor("#e6eaef"))
        pen.setWidth(0)
        painter.setPen(pen)
        first_x = left
        while first_x <= rect.right():
            painter.drawLine(QLineF(first_x, rect.top(), first_x, rect.bottom()))
            first_x += grid
        first_y = top
        while first_y <= rect.bottom():
            painter.drawLine(QLineF(rect.left(), first_y, rect.right(), first_y))
            first_y += grid

    # ------------------------------------------------------------------ 连线
    def begin_connection_drag(self, port: PortItem) -> None:
        if not port.is_input:
            self.cancel_connection_drag()
            self._drag_source = port
            self._temp_connection = QGraphicsPathItem()
            pen = QPen(QColor("#3b82f6"), 2.0, Qt.PenStyle.DashLine)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            self._temp_connection.setPen(pen)
            self._temp_connection.setZValue(20.0)
            self.addItem(self._temp_connection)
            self._update_temp(port.center_scene_pos())

    def cancel_connection_drag(self) -> None:
        if self._temp_connection is not None:
            if self._temp_connection.scene() is self:
                self.removeItem(self._temp_connection)
            self._temp_connection = None
        if self._highlighted is not None:
            self._highlighted.set_highlight(False)
            self._highlighted = None
        self._drag_source = None

    @property
    def dragging_connection(self) -> bool:
        return self._drag_source is not None

    def _update_temp(self, scene_pos: QPointF) -> None:
        if self._temp_connection is None or self._drag_source is None:
            return
        # 与正式连线共用 curve.connection_path，保证拖拽中的纵向曲线与落定后一致。
        start = self._drag_source.center_scene_pos()
        self._temp_connection.setPath(connection_path(start, scene_pos))

    def _update_highlight(self, scene_pos: QPointF) -> None:
        target = self._target_input_at(scene_pos)
        if target is self._highlighted:
            return
        if self._highlighted is not None:
            self._highlighted.set_highlight(False)
        self._highlighted = target
        if target is not None:
            target.set_highlight(True)

    def _target_input_at(self, scene_pos: QPointF) -> PortItem | None:
        source = self._drag_source
        if source is None:
            return None
        best: PortItem | None = None
        best_distance = 16.0
        for port in self._view.port_items(is_input=True):
            if port.node_item.node_id == source.node_item.node_id:
                continue
            if not self._view.ports_compatible(
                source.node_item.node_id,
                source.key,
                port.node_item.node_id,
                port.key,
            ):
                continue
            distance = QLineF(port.center_scene_pos(), scene_pos).length()
            if distance <= best_distance:
                best = port
                best_distance = distance
        return best

    def _port_at(self, scene_pos: QPointF, is_input: bool) -> PortItem | None:
        best: PortItem | None = None
        best_distance = 12.0
        for port in self._view.port_items(is_input=is_input):
            distance = QLineF(port.center_scene_pos(), scene_pos).length()
            if distance <= best_distance:
                best = port
                best_distance = distance
        return best

    # ------------------------------------------------------------------ 事件
    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802 - Qt 接口
        if event.button() == Qt.MouseButton.LeftButton:
            source = self._port_at(event.scenePos(), is_input=False)
            if source is not None:
                self.begin_connection_drag(source)
                event.accept()
                return
        super().mousePressEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._view.begin_move_tracking()
            top_item = self._view.item_at_scene(event.scenePos())
            # 点击节点时启动延迟选中信号；点击端口 / 空白处不触发。
            if top_item is not None and hasattr(top_item, "node_id") and not isinstance(top_item, PortItem):
                self._view.schedule_selection_signal()

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802 - Qt 接口
        if self._drag_source is not None:
            self._update_temp(event.scenePos())
            self._update_highlight(event.scenePos())
            event.accept()
            return
        if event.buttons() & Qt.MouseButton.LeftButton:
            grabber = self.mouseGrabberItem()
            if grabber is not None and hasattr(grabber, "node_id"):
                # 拖动节点时不应在 220ms 后误触发预览。
                self._view.cancel_selection_timer()
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802 - Qt 接口
        if self._drag_source is not None and event.button() == Qt.MouseButton.LeftButton:
            source = self._drag_source
            target = self._target_input_at(event.scenePos())
            self.cancel_connection_drag()
            if target is not None and source is not None:
                self._view.connect_ports(
                    source.node_item.node_id,
                    source.key,
                    target.node_item.node_id,
                    target.key,
                )
            elif source is not None:
                # 拖到空白处：自动创建一个输出节点并接上，省去手动找输出节点。
                self._view.drop_connection_on_blank(source, event.scenePos())
            event.accept()
            return
        super().mouseReleaseEvent(event)
        if event.button() == Qt.MouseButton.LeftButton:
            self._view.finish_move_tracking()

    def mouseDoubleClickEvent(self, event: QGraphicsSceneMouseEvent) -> None:  # noqa: N802 - Qt 接口
        super().mouseDoubleClickEvent(event)


__all__ = ["GraphScene"]
