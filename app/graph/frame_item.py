"""流程框节点条目：可嵌套的容器节点。

流程框在画布上表现为一个圆角容器：标题栏显示名称与状态（待定 / 失效），
输入端口在外框顶边、输出端口在外框底边；每个外侧端口在框内约 14px 处有一个
同款圆形的「内侧端口」，分别承担输出 / 输入的角色，供框内节点连接。

框的尺寸与位置由 GraphView 根据直接子节点的包围盒反推后调用
set_frame_geometry 同步，本类只负责绘制与端口摆放。
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from app.graph.node_item import NodeItem, _elide
from app.graph.port_item import PortItem

#: 外侧端口与内侧端口之间的偏移（像素）。
INNER_OFFSET = 14.0
#: 无子节点时的默认尺寸。
DEFAULT_FRAME_WIDTH = 420.0
DEFAULT_FRAME_HEIGHT = 220.0
#: 子节点包围盒之外的最小留白；顶部额外留给标题栏。
FRAME_PADDING_X = 26.0
FRAME_PADDING_TOP = 52.0
FRAME_PADDING_BOTTOM = 52.0
#: 标题栏高度。
TITLE_HEIGHT = 24.0


class FrameItem(NodeItem):
    """流程框容器条目（输入在顶边、输出在底边，含内外两侧端口）。"""

    CORNER = 12.0
    PORT_MARGIN = 28.0
    PORT_SPACING = 64.0

    def __init__(
        self,
        view,
        graph,
        node,
        spec,
        thumbnails_enabled: bool = True,
        thumbnail_box: int = 72,
        parent: QGraphicsItem | None = None,
    ) -> None:
        self.frame = node.frame
        self._requested_width = DEFAULT_FRAME_WIDTH
        self._requested_height = DEFAULT_FRAME_HEIGHT
        self._outer_inputs: dict[str, PortItem] = {}
        self._inner_outputs: dict[str, PortItem] = {}
        self._inner_inputs: dict[str, PortItem] = {}
        self._outer_outputs: dict[str, PortItem] = {}
        self._input_label_rects = []
        self._output_label_rects = []
        self._stale = False
        super().__init__(
            view,
            graph,
            node,
            spec,
            thumbnails_enabled,
            thumbnail_box,
            parent,
        )
        # 容器背景必须位于连线（z=-2）之下，否则框内连线会被填充色盖住；
        # 端口作为子条目随后单独抬高，保证仍可拖拽 / 命中。
        self.setZValue(-3.0)
        self._refresh_stale()
        self._refresh_tooltip()

    # ------------------------------------------------------------------ 端口
    @property
    def outer_input_ports(self) -> dict[str, PortItem]:
        return self._outer_inputs

    @property
    def outer_output_ports(self) -> dict[str, PortItem]:
        return self._outer_outputs

    @property
    def inner_input_ports(self) -> dict[str, PortItem]:
        return self._inner_inputs

    @property
    def inner_output_ports(self) -> dict[str, PortItem]:
        return self._inner_outputs

    def _frame_inputs(self):
        return list(self.frame.inputs) if self.frame is not None else []

    def _frame_outputs(self):
        return list(self.frame.outputs) if self.frame is not None else []

    def _build_ports(self) -> None:
        """按 FrameModel 重建外侧 / 内侧端口（同一连接点图形）。"""

        self._outer_inputs = {}
        self._inner_outputs = {}
        self._inner_inputs = {}
        self._outer_outputs = {}
        input_ports: dict[str, PortItem] = {}
        output_ports: dict[str, PortItem] = {}
        self._clear_child_ports()

        for index, port in enumerate(self._frame_inputs()):
            spec = port.port_spec()
            outer = PortItem(
                self, port.key, port.label, True, index, port_spec=spec,
                pending=bool(port.pending),
            )
            inner = PortItem(
                self, port.key, port.label, False, index, port_spec=spec,
                pending=bool(port.pending),
            )
            outer.setToolTip(
                f"流程框输入端口：{port.label}\n数据类型：{port.type_text}\n"
                "从框内节点连接到这里时，数据来自框外上游"
            )
            inner.setToolTip(
                f"流程框输入（内侧）：{port.label}\n数据类型：{port.type_text}\n"
                "拖动此端口可连接到框内节点"
            )
            outer.setZValue(4.0)
            inner.setZValue(4.0)
            self._outer_inputs[port.key] = outer
            self._inner_outputs[port.key] = inner
            input_ports[port.key] = outer
            output_ports[port.key] = inner

        for index, port in enumerate(self._frame_outputs()):
            spec = port.port_spec()
            inner = PortItem(
                self, port.key, port.label, True, index, port_spec=spec,
                pending=bool(port.pending),
            )
            outer = PortItem(
                self, port.key, port.label, False, index, port_spec=spec,
                pending=bool(port.pending),
            )
            inner.setToolTip(
                f"流程框输出（内侧）：{port.label}\n数据类型：{port.type_text}\n"
                "框内节点连接到这里时，数据输出到框外"
            )
            outer.setToolTip(
                f"流程框输出端口：{port.label}\n数据类型：{port.type_text}\n"
                "拖动此端口可连接到框外节点"
            )
            inner.setZValue(4.0)
            outer.setZValue(4.0)
            self._inner_inputs[port.key] = inner
            self._outer_outputs[port.key] = outer
            input_ports[port.key] = inner
            output_ports[port.key] = outer

        self._input_ports = input_ports
        self._output_ports = output_ports

    def _clear_child_ports(self) -> None:
        scene = self.scene()
        for port in list(self._input_ports.values()) + list(self._output_ports.values()):
            port.setParentItem(None)
            if scene is not None and port.scene() is scene:
                scene.removeItem(port)
        self._input_ports = {}
        self._output_ports = {}

    def rebuild_ports(self) -> None:
        """外框端口数量 / 类型变化后重建端口，并刷新几何与相连连线。"""

        self._build_ports()
        self._refresh_stale()
        self._rebuild_geometry()
        self.update_port_states()
        self._refresh_connections()
        self._refresh_tooltip()
        self.update()

    # ---------------------------------------------------------------- 几何布局
    def frame_input_ports(self) -> list[PortItem]:
        return list(self._outer_inputs.values())

    def frame_output_ports(self) -> list[PortItem]:
        return list(self._outer_outputs.values())

    def set_frame_geometry(self, width: float, height: float) -> None:
        """由视图根据子节点包围盒设置框尺寸。"""

        width = max(DEFAULT_FRAME_WIDTH if not self._has_children() else 320.0, float(width))
        height = max(DEFAULT_FRAME_HEIGHT if not self._has_children() else 180.0, float(height))
        if (
            abs(self._requested_width - width) < 0.01
            and abs(self._requested_height - height) < 0.01
        ):
            return
        self._requested_width = width
        self._requested_height = height
        self._rebuild_geometry()
        self._refresh_connections()
        self.update()

    def _has_children(self) -> bool:
        try:
            return bool(self._graph.direct_children(self.node.id))
        except Exception:  # noqa: BLE001 - 模型接口异常时按无子节点处理
            return False

    def _rebuild_geometry(self) -> None:
        self.prepareGeometryChange()
        input_count = max(len(self._outer_inputs), 1)
        output_count = max(len(self._outer_outputs), 1)
        needed = 2.0 * self.PORT_MARGIN + self.PORT_SPACING * (max(input_count, output_count) - 1)
        self._width = max(float(self._requested_width), needed)
        self._height = max(float(self._requested_height), 180.0, TITLE_HEIGHT + 120.0)

        for index, port in enumerate(self._outer_inputs.values()):
            x = self.port_x(index, len(self._outer_inputs), self._width)
            port.setPos(QPointF(x, 0.0))
            inner = self._inner_outputs.get(port.key)
            if inner is not None:
                inner.setPos(QPointF(x, INNER_OFFSET))
        for index, port in enumerate(self._outer_outputs.values()):
            x = self.port_x(index, len(self._outer_outputs), self._width)
            port.setPos(QPointF(x, self._height))
            inner = self._inner_inputs.get(port.key)
            if inner is not None:
                inner.setPos(QPointF(x, self._height - INNER_OFFSET))

        self._input_label_rects = self._label_rects(
            len(self._outer_inputs), INNER_OFFSET + 9.0
        )
        self._output_label_rects = self._label_rects(
            len(self._outer_outputs), self._height - INNER_OFFSET - 9.0 - 16.0
        )
        self.update()

    def _label_rects(self, count: int, top: float) -> list[QRectF]:
        if count <= 0:
            return []
        xs = [self.port_x(index, count, self._width) for index in range(count)]
        rects: list[QRectF] = []
        for index, x in enumerate(xs):
            left = 8.0 if index == 0 else (xs[index - 1] + x) / 2.0
            right = self._width - 8.0 if index == count - 1 else (x + xs[index + 1]) / 2.0
            rects.append(QRectF(left + 2.0, top, max(12.0, right - left - 4.0), 16.0))
        return rects

    # ---------------------------------------------------------------- 状态更新
    def update_port_states(self) -> None:
        for key, port in self._outer_inputs.items():
            port.set_connected(self._graph.input_connection(self.node.id, key) is not None)
        for key, port in self._inner_inputs.items():
            port.set_connected(self._graph.input_connection(self.node.id, key) is not None)
        outgoing = self._graph.outgoing(self.node.id)
        for key, port in self._inner_outputs.items():
            port.set_connected(any(connection.src_port == key for connection in outgoing))
        for key, port in self._outer_outputs.items():
            port.set_connected(any(connection.src_port == key for connection in outgoing))

    def _refresh_stale(self) -> None:
        operator_id = getattr(self.frame, "operator_id", None) if self.frame is not None else None
        stale = False
        checker = getattr(self._view, "is_operator_registered", None)
        if operator_id:
            registered = bool(checker(operator_id)) if callable(checker) else True
            stale = not registered
        elif not self.node.is_frame and self.frame is not None:
            registered = bool(checker(self.node.op_id)) if callable(checker) else True
            stale = not registered
        if stale != self._stale:
            self._stale = stale
            self.update()

    @property
    def stale(self) -> bool:
        return self._stale

    def _refresh_connections(self) -> None:
        refresh = getattr(self._view, "on_frame_item_geometry_changed", None)
        if callable(refresh):
            refresh(self)
            return
        super()._refresh_connections()

    def refresh_from_model(self) -> None:
        # 撤销 / 重做时模型可能整体替换 FrameModel 实例，这里重新同步引用。
        self.frame = self.node.frame
        ports_changed = self._ports_signature() != self._current_ports_signature()
        if ports_changed:
            self.rebuild_ports()
        self._refresh_stale()
        self._refresh_tooltip()
        self.update()

    def _ports_signature(self) -> tuple:
        data = []
        for port in self._frame_inputs():
            data.append(("i", port.key, port.data_type.value if port.data_type else None))
        for port in self._frame_outputs():
            data.append(("o", port.key, port.data_type.value if port.data_type else None))
        return tuple(data)

    def _current_ports_signature(self) -> tuple:
        data = []
        for key, port in self._outer_inputs.items():
            dtype = port.data_type.value if not port.any_type else None
            data.append(("i", key, dtype))
        for key, port in self._outer_outputs.items():
            dtype = port.data_type.value if not port.any_type else None
            data.append(("o", key, dtype))
        return tuple(data)

    def _refresh_tooltip(self) -> None:
        lines: list[str] = [self.node.display_title(), "流程框"]
        if self._stale:
            lines.append("依赖的自定义算子不存在，框内节点可能失效；可右键“替换算子…”")
        if self.frame is not None:
            for port in self.frame.inputs:
                lines.append(f"输入 {port.label}：{port.type_text}")
            for port in self.frame.outputs:
                lines.append(f"输出 {port.label}：{port.type_text}")
        lines.append("拖动框体可整体移动框内节点")
        self.setToolTip("\n".join(lines))

    def _header_chip(self) -> tuple[str, QColor] | None:
        if self._stale:
            operator_id = getattr(self.frame, "operator_id", None) if self.frame is not None else None
            return ("依赖缺失" if operator_id else "失效"), QColor("#b45309")
        if self.frame is not None and (
            self.frame.has_pending(True) or self.frame.has_pending(False)
        ):
            return "待定", QColor("#64748b")
        return None

    # -------------------------------------------------------------------- 绘制
    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt 接口
        margin = PortItem.HIT_RADIUS + 4.0
        return QRectF(
            -margin,
            -margin,
            self._width + 2.0 * margin,
            self._height + 2.0 * margin,
        )

    def shape(self) -> QPainterPath:  # noqa: N802
        path = QPainterPath()
        path.addRoundedRect(QRectF(0.0, 0.0, self._width, self._height), self.CORNER, self.CORNER)
        return path

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:  # noqa: N802
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(0.0, 0.0, self._width, self._height)
        body = QPainterPath()
        body.addRoundedRect(rect, self.CORNER, self.CORNER)

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(15, 23, 42, 16))
        shadow = QPainterPath()
        shadow.addRoundedRect(rect.translated(0.0, 3.0), self.CORNER, self.CORNER)
        painter.drawPath(shadow)

        painter.setBrush(QColor(248, 250, 252, 235))
        painter.drawPath(body)

        painter.save()
        painter.setClipPath(body)
        title_rect = QRectF(0.0, 0.0, self._width, TITLE_HEIGHT + 8.0)
        painter.fillRect(title_rect, QColor(241, 245, 249, 240))
        painter.restore()

        if self._stale:
            border_color = QColor("#b45309")
            border_width = 2.0
        elif self.isSelected():
            border_color = QColor("#2563eb")
            border_width = 2.0
        else:
            border_color = QColor("#94a3b8")
            border_width = 1.4
        pen = QPen(border_color, border_width)
        if self._stale:
            pen.setStyle(Qt.PenStyle.DashLine)
            pen.setDashPattern([5.0, 3.0])
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(pen)
        painter.drawRoundedRect(rect.adjusted(0.7, 0.7, -0.7, -0.7), self.CORNER, self.CORNER)

        painter.setPen(QPen(QColor("#dbe2ea"), 1.0))
        painter.drawLine(QPointF(1.0, TITLE_HEIGHT), QPointF(self._width - 1.0, TITLE_HEIGHT))

        title_font = QFont("Microsoft YaHei UI")
        title_font.setPointSizeF(9.5)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor("#1f2937"))
        chip = self._header_chip()
        chip_width = 52.0 if chip is not None else 44.0
        painter.drawText(
            QRectF(12.0, 2.0, self._width - 24.0 - chip_width - 8.0, TITLE_HEIGHT),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            _elide(title_font, self.node.display_title(), self._width - 24.0 - chip_width - 8.0),
        )

        chip_font = QFont("Microsoft YaHei UI")
        chip_font.setPointSizeF(7.0)
        painter.setFont(chip_font)
        x_cursor = self._width - 10.0
        if chip is not None:
            text_width = QFontMetrics(chip_font).horizontalAdvance(chip[0]) + 14.0
            chip_rect = QRectF(x_cursor - text_width, 3.0, text_width, TITLE_HEIGHT - 6.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(chip[1])
            painter.drawRoundedRect(chip_rect, 7.0, 7.0)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(chip_rect, int(Qt.AlignmentFlag.AlignCenter), chip[0])
            x_cursor = chip_rect.left() - 6.0
        frame_rect = QRectF(x_cursor - 42.0, 3.0, 42.0, TITLE_HEIGHT - 6.0)
        painter.setPen(QColor("#64748b"))
        painter.setBrush(QColor(226, 232, 240, 180))
        painter.drawRoundedRect(frame_rect, 7.0, 7.0)
        painter.setPen(QColor("#475569"))
        painter.drawText(frame_rect, int(Qt.AlignmentFlag.AlignCenter), "流程框")
        if self.frame is not None and getattr(self.frame, "operator_id", None):
            sub_font = QFont("Microsoft YaHei UI")
            sub_font.setPointSizeF(7.0)
            painter.setFont(sub_font)
            painter.setPen(QColor("#64748b"))
            subtitle = f"引用：{self.frame.operator_id}"
            painter.drawText(
                QRectF(12.0, TITLE_HEIGHT + 5.0, self._width - 24.0, 14.0),
                int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                _elide(sub_font, subtitle, self._width - 24.0),
            )

        label_font = QFont("Microsoft YaHei UI")
        label_font.setPointSizeF(7.5)
        painter.setFont(label_font)
        for port, label_rect in zip(self._outer_inputs.values(), self._input_label_rects):
            pending = port.any_type
            painter.setPen(QColor("#9aa4b2") if pending else QColor("#4b5563"))
            text = f"{port.label}（待定）" if pending else port.label
            painter.drawText(
                label_rect,
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
                _elide(label_font, text, label_rect.width()),
            )
        for port, label_rect in zip(self._outer_outputs.values(), self._output_label_rects):
            pending = port.any_type
            painter.setPen(QColor("#9aa4b2") if pending else QColor("#4b5563"))
            text = f"{port.label}（待定）" if pending else port.label
            painter.drawText(
                label_rect,
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
                _elide(label_font, text, label_rect.width()),
            )

        if not self._has_children():
            hint_font = QFont("Microsoft YaHei UI")
            hint_font.setPointSizeF(9.0)
            painter.setFont(hint_font)
            painter.setPen(QColor("#94a3b8"))
            painter.drawText(
                QRectF(16.0, self._height / 2.0 - 12.0, self._width - 32.0, 24.0),
                int(Qt.AlignmentFlag.AlignCenter),
                "将算子拖入流程框，组合为可复用的整体",
            )

    # ------------------------------------------------------------------ 交互
    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: Any) -> Any:  # noqa: N802
        if change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            # 流程框始终位于连线（z=-2）与框内节点下方，选中时也不能盖住它们。
            self.setZValue(-2.8 if bool(value) else -3.0)
            return QGraphicsItem.itemChange(self, change, value)
        return super().itemChange(change, value)


__all__ = [
    "DEFAULT_FRAME_HEIGHT",
    "DEFAULT_FRAME_WIDTH",
    "FRAME_PADDING_BOTTOM",
    "FRAME_PADDING_TOP",
    "FRAME_PADDING_X",
    "FrameItem",
    "INNER_OFFSET",
]
