"""节点端口条目（输入 / 输出）。"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from app.core.data_types import DataType
from app.operators.spec import PortSpec

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from app.graph.node_item import NodeItem


class PortItem(QGraphicsItem):
    """节点边缘的圆形端口：输入在上边缘、输出在下边缘，输出端口可拖拽出连线。

    端口颜色由 :class:`~app.core.data_types.DataType` 决定，同类型的输出 / 输入
    才能连接；悬停提示会显示数据类型、可选状态与端口说明。
    """

    RADIUS = 5.5
    HIT_RADIUS = 9.0

    def __init__(
        self,
        node_item: "NodeItem",
        key: str,
        label: str,
        is_input: bool,
        index: int,
        port_spec: PortSpec | None = None,
        pending: bool = False,
    ) -> None:
        super().__init__(node_item)
        self.node_item = node_item
        self.key = key
        self.label = label
        self.is_input = is_input
        self.index = index
        self.port_spec = port_spec
        self.data_type = port_spec.data_type if port_spec is not None else DataType.IMAGE
        self.any_type = bool(port_spec.any_type) if port_spec is not None else False
        #: 流程框的待定端口：类型未知，使用灰色虚线绘制。
        self.pending = bool(pending)
        self._connected = False
        self._hover = False
        self._highlight = False
        self.setZValue(3.0)
        self.setAcceptHoverEvents(True)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)
        self.setToolTip(self._tooltip_text())

    def _tooltip_text(self) -> str:
        role = "输入" if self.is_input else "输出"
        if self.pending:
            type_text = "待定（连接后自动确定类型）"
        elif self.any_type:
            type_text = "任意数据（图像 / 数值 / 边缘特征 / 轮廓特征 / 连通域）"
        else:
            type_text = self.data_type.label
        lines = [f"{role}端口：{self.label}", f"数据类型：{type_text}"]
        spec = self.port_spec
        if spec is not None:
            if self.data_type is DataType.IMAGE and not self.any_type:
                prefix = "接受图像" if self.is_input else "图像颜色空间"
                lines.append(f"{prefix}：{spec.accept_text()}")
            if spec.optional:
                lines.append("可选：未连接时算子收到 None")
            if spec.description:
                lines.append(spec.description)
        if not self.is_input:
            lines.append("拖动此端口可创建连线")
        return "\n".join(lines)

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

    def type_color(self) -> QColor:
        if self.any_type:
            # 万能端口使用中性色，避免被误认为固定为图像类型。
            return QColor("#64748b")
        return QColor(self.data_type.color)

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

        color = self.type_color()
        if self._connected:
            brush = QColor(color)
            pen_color = color.darker(125)
        else:
            brush = QColor("#ffffff")
            pen_color = color
        if self._hover:
            if not self._connected:
                brush = QColor(color)
                brush.setAlpha(52)
            pen_color = color.darker(115)
        painter.setBrush(brush)
        port_pen = QPen(pen_color, 1.7)
        if self.pending:
            # 待定端口：灰色虚线圆，提示类型尚未确定。
            port_pen.setStyle(Qt.PenStyle.DashLine)
            port_pen.setDashPattern([2.0, 2.0])
        painter.setPen(port_pen)
        painter.drawEllipse(QPointF(0.0, 0.0), self.RADIUS, self.RADIUS)
        if self._connected:
            # 已连接端口画一个小白点，和未连接端口区分更明显。
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor("#ffffff"))
            painter.drawEllipse(QPointF(0.0, 0.0), 1.8, 1.8)

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
