"""画布上的节点卡片。

节点外观为浅色圆角卡片：标题、分类 / 算子 id、上边缘的输入端口、下边缘的输出端口、
右下角耗时、可选的结果缩略图以及出错时的红色边框与 tooltip。
节点条目的位置始终与 :class:`NodeModel.pos`（左上角场景坐标）保持一致。

端口改为上下方向后，输入端口沿节点上边缘水平分布，输出端口沿下边缘水平分布；
端口标签分别位于节点内部顶部一行（圆点下方）与底部一行（圆点上方），保证标签、
标题、缩略图与底部信息互不重叠。
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSizeF, Qt
from PySide6.QtGui import (
    QColor,
    QFont,
    QFontMetrics,
    QImage,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from app.core.model import NodeModel
from app.core.types import make_thumbnail
from app.graph.port_item import PortItem
from app.operators.spec import OperatorSpec

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from app.graph.view import GraphView


def _elide(font: QFont, text: str, width: float) -> str:
    metrics = QFontMetrics(font)
    return metrics.elidedText(text, Qt.TextElideMode.ElideRight, max(8, int(width)))


def _rgb_to_pixmap(rgb: np.ndarray) -> QPixmap:
    array = np.ascontiguousarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"缩略图必须是 RGB 三通道，收到 {array.shape}")
    height, width = array.shape[:2]
    image = QImage(
        array.data,
        int(width),
        int(height),
        int(array.strides[0]),
        QImage.Format.Format_RGB888,
    ).copy()
    return QPixmap.fromImage(image)


class NodeItem(QGraphicsItem):
    """浅色节点卡片，输入端口在顶边、输出端口在底边。"""

    WIDTH = 222.0
    CORNER = 10.0

    # 端口沿边缘分布时距左右边缘的最小留白。
    PORT_MARGIN = 24.0
    # 顶部：节点上边缘 -> 输入端口圆点 -> 输入标签 -> 标题区。
    TOP_LABEL_GAP = 8.0
    LABEL_H = 16.0
    LABEL_GAP = 2.0
    HEADER_H = 44.0
    # 中部内容（缩略图 / 空白）与底部区域。
    CONTENT_GAP = 6.0
    MIN_CONTENT_H = 8.0
    FOOTER_H = 18.0
    FOOTER_LABEL_GAP = 3.0
    BOTTOM_LABEL_GAP = 8.0
    THUMB_MARGIN = 8.0

    def __init__(
        self,
        view: "GraphView",
        graph,
        node: NodeModel,
        spec: OperatorSpec | None,
        thumbnails_enabled: bool = True,
        thumbnail_box: int = 72,
        parent: QGraphicsItem | None = None,
    ) -> None:
        super().__init__(parent)
        self._view = view
        self._graph = graph
        self.node = node
        self.spec = spec
        self._width = self.WIDTH
        self._height = 120.0
        self._thumbnails_enabled = bool(thumbnails_enabled)
        self._thumbnail_box = max(24, int(thumbnail_box))
        self._result: Any = None
        self._thumb_pixmap: QPixmap | None = None
        self._elapsed: float | None = None
        self._error: str | None = None
        self._thumb_rect = QRectF()
        self._header_rect = QRectF()
        self._footer_rect = QRectF()
        self._content_top = 0.0
        self._input_label_rects: list[QRectF] = []
        self._output_label_rects: list[QRectF] = []
        self._syncing = False
        self._input_ports: dict[str, PortItem] = {}
        self._output_ports: dict[str, PortItem] = {}

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(0.0)
        self.setPos(QPointF(float(node.pos[0]), float(node.pos[1])))
        self._build_ports()
        self._rebuild_geometry()
        self._refresh_tooltip()

    # ------------------------------------------------------------------ 属性
    @property
    def node_id(self) -> str:
        return self.node.id

    @property
    def elapsed_ms(self) -> float | None:
        return self._elapsed

    @property
    def error_message(self) -> str | None:
        return self._error

    @property
    def thumbnail_pixmap(self) -> QPixmap | None:
        return self._thumb_pixmap

    @property
    def input_ports(self) -> dict[str, PortItem]:
        return self._input_ports

    @property
    def output_ports(self) -> dict[str, PortItem]:
        return self._output_ports

    def node_size(self) -> QSizeF:
        return QSizeF(self._width, self._height)

    def size(self) -> QSizeF:
        return self.node_size()

    def _refresh_connections(self) -> None:
        """节点高度变化会移动上下边缘端口，通知视图刷新相连连线。"""

        refresh = getattr(self._view, "on_node_item_moved", None)
        if callable(refresh):
            refresh(self)

    # ------------------------------------------------------------------ 端口
    def _port_specs(self) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
        if self.node.is_source:
            return [], [("image", "图像")]
        if self.spec is None:
            # 未注册的算子仍然允许连线，便于打开旧工程或临时删除算子文件。
            return [("image", "图像")], [("image", "图像")]
        inputs = [(port.key, port.label) for port in self.spec.inputs]
        outputs = [(port.key, port.label) for port in self.spec.outputs]
        return inputs, outputs

    def _build_ports(self) -> None:
        inputs, outputs = self._port_specs()
        for index, (key, label) in enumerate(inputs):
            self._input_ports[key] = PortItem(self, key, label, True, index)
        for index, (key, label) in enumerate(outputs):
            self._output_ports[key] = PortItem(self, key, label, False, index)

    def port_item(self, key: str, is_input: bool) -> PortItem | None:
        return (self._input_ports if is_input else self._output_ports).get(key)

    @staticmethod
    def port_x(index: int, count: int, width: float | None = None) -> float:
        """返回第 ``index`` 个端口相对节点左边缘的水平偏移。

        单端口居中；多端口在左右各留出 :attr:`PORT_MARGIN`（端口很多时自动收窄）
        后水平均分。交互层用它估算自动连线的落点，无需访问条目实例。
        """

        count = max(1, int(count))
        index = max(0, min(int(index), count - 1))
        width = float(NodeItem.WIDTH if width is None else width)
        if count == 1:
            return width / 2.0
        margin = min(NodeItem.PORT_MARGIN, width / (2.0 * count))
        usable = max(0.0, width - 2.0 * margin)
        return margin + usable * (index + 0.5) / count

    def update_port_states(self) -> None:
        for key, port in self._input_ports.items():
            port.set_connected(self._graph.input_connection(self.node.id, key) is not None)
        outgoing = self._graph.outgoing(self.node.id)
        for key, port in self._output_ports.items():
            port.set_connected(any(connection.src_port == key for connection in outgoing))

    # ---------------------------------------------------------------- 几何布局
    def _label_rects(self, count: int, top: float) -> list[QRectF]:
        """按端口位置把标签行切分成互不重叠的槽位。"""

        if count <= 0:
            return []
        xs = [self.port_x(index, count, self._width) for index in range(count)]
        rects: list[QRectF] = []
        for index, x in enumerate(xs):
            left = 8.0 if index == 0 else (xs[index - 1] + x) / 2.0
            right = self._width - 8.0 if index == count - 1 else (x + xs[index + 1]) / 2.0
            rects.append(
                QRectF(left + 2.0, top, max(12.0, right - left - 4.0), self.LABEL_H)
            )
        return rects

    def _rebuild_geometry(self) -> None:
        self.prepareGeometryChange()
        input_count = len(self._input_ports)
        output_count = len(self._output_ports)

        header_top = self.TOP_LABEL_GAP
        if input_count:
            header_top += self.LABEL_H + self.LABEL_GAP
        self._header_rect = QRectF(0.0, header_top, self._width, self.HEADER_H)
        self._content_top = header_top + self.HEADER_H + self.CONTENT_GAP

        thumb_area = 0.0
        self._thumb_rect = QRectF()
        if self._thumb_pixmap is not None and not self._thumb_pixmap.isNull():
            thumb_area = max(
                40.0,
                min(self._thumbnail_box + 8.0, float(self._thumb_pixmap.height()) + 8.0),
            )
            self._thumb_rect = QRectF(
                self.THUMB_MARGIN,
                self._content_top,
                self._width - 2.0 * self.THUMB_MARGIN,
                thumb_area,
            )
        content_height = thumb_area if thumb_area > 0.0 else self.MIN_CONTENT_H

        bottom_stack = self.BOTTOM_LABEL_GAP + self.FOOTER_H + self.CONTENT_GAP
        if output_count:
            bottom_stack += self.LABEL_H + self.FOOTER_LABEL_GAP
        self._height = self._content_top + content_height + bottom_stack

        # 输入端口贴上边缘，输出端口贴下边缘，多端口水平均分。
        for index, port in enumerate(self._input_ports.values()):
            x = self.port_x(index, input_count, self._width)
            port.setPos(QPointF(x, 0.0))
        for index, port in enumerate(self._output_ports.values()):
            x = self.port_x(index, output_count, self._width)
            port.setPos(QPointF(x, self._height))

        # 标签槽位：输入在圆点下方一行，输出在圆点上方一行。
        self._input_label_rects = self._label_rects(input_count, self.TOP_LABEL_GAP)
        self._output_label_rects = self._label_rects(
            output_count,
            self._height - self.BOTTOM_LABEL_GAP - self.LABEL_H,
        )

        footer_top = self._height - self.BOTTOM_LABEL_GAP - self.FOOTER_H
        if output_count:
            footer_top -= self.LABEL_H + self.FOOTER_LABEL_GAP
        self._footer_rect = QRectF(0.0, footer_top, self._width, self.FOOTER_H)
        self.update()

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt 接口
        # 端口圆点位于上下边缘之外，需要纳入包围盒（半径约 9 + 3）。
        margin = PortItem.HIT_RADIUS + 3.0
        return QRectF(
            -margin,
            -margin,
            self._width + 2.0 * margin,
            self._height + 2.0 * margin,
        )

    def shape(self) -> QPainterPath:  # noqa: N802 - Qt 接口
        path = QPainterPath()
        path.addRoundedRect(QRectF(0.0, 0.0, self._width, self._height), self.CORNER, self.CORNER)
        return path

    # ---------------------------------------------------------------- 状态更新
    def sync_position(self, pos: tuple[float, float]) -> None:
        target = QPointF(float(pos[0]), float(pos[1]))
        if self.pos() == target:
            return
        self._syncing = True
        try:
            self.setPos(target)
        finally:
            self._syncing = False

    def set_result(self, result: Any, thumbnails_enabled: bool | None = None) -> None:
        if thumbnails_enabled is not None:
            self._thumbnails_enabled = bool(thumbnails_enabled)
        self._result = result
        self._elapsed = float(getattr(result, "elapsed_ms", 0.0) or 0.0)
        self._error = None
        self._update_thumbnail()
        self._rebuild_geometry()
        self._refresh_connections()
        self._refresh_tooltip()
        self.update()

    def set_thumbnails_enabled(self, enabled: bool, thumbnail_box: int | None = None) -> None:
        enabled = bool(enabled)
        changed = enabled != self._thumbnails_enabled
        self._thumbnails_enabled = enabled
        if thumbnail_box is not None:
            box = max(24, int(thumbnail_box))
            if box != self._thumbnail_box:
                self._thumbnail_box = box
                changed = True
        if changed:
            self._update_thumbnail()
            self._rebuild_geometry()
            self._refresh_connections()
            self.update()

    def set_error(self, message: str | None) -> None:
        message = message or None
        if self._error == message:
            return
        self._error = message
        self._refresh_tooltip()
        self.update()

    def refresh_from_model(self) -> None:
        self._refresh_tooltip()
        self.update()

    def _update_thumbnail(self) -> None:
        self._thumb_pixmap = None
        if not self._thumbnails_enabled or self._result is None:
            return
        primary = getattr(self._result, "primary", None)
        if primary is None:
            return
        try:
            rgb = make_thumbnail(primary, self._thumbnail_box)
            self._thumb_pixmap = _rgb_to_pixmap(rgb)
        except Exception:  # noqa: BLE001 - 缩略图失败不能影响画布
            self._thumb_pixmap = None

    def _refresh_tooltip(self) -> None:
        lines: list[str] = [self.node.display_title()]
        if self.node.is_source:
            lines.append("原始图像节点")
        elif self.spec is not None:
            lines.append(f"{self.spec.name} · {self.spec.id}")
            if self.spec.description:
                lines.append(self.spec.description.strip().splitlines()[0])
        else:
            lines.append(f"未注册的算子：{self.node.op_id}")
        if self._error:
            lines.append(f"错误：{self._error}")
        elif self._elapsed is not None:
            lines.append(f"耗时：{self._elapsed:.1f} ms")
        self.setToolTip("\n".join(lines))

    # -------------------------------------------------------------------- 绘制
    def _header_chip(self) -> tuple[str, QColor] | None:
        if self._error:
            return "错误", QColor("#dc2626")
        if self.node.is_source:
            return "原始图像", QColor("#0f766e")
        return None

    def paint(self, painter: QPainter, option: QStyleOptionGraphicsItem, widget: QWidget | None = None) -> None:  # noqa: N802
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        rect = QRectF(0.0, 0.0, self._width, self._height)
        body = QPainterPath()
        body.addRoundedRect(rect, self.CORNER, self.CORNER)

        # 轻微投影，增加卡片层次感。
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(15, 23, 42, 22))
        shadow = QPainterPath()
        shadow.addRoundedRect(rect.translated(0.0, 2.0), self.CORNER, self.CORNER)
        painter.drawPath(shadow)

        painter.fillPath(body, QColor("#ffffff"))

        painter.save()
        painter.setClipPath(body)
        header_color = QColor("#f6f8fb")
        if self.node.is_source:
            header_color = QColor("#eef7f5")
        painter.fillRect(self._header_rect, header_color)
        if self.node.is_source:
            painter.fillRect(QRectF(0.0, 0.0, 3.0, self._height), QColor("#0f766e"))
        painter.restore()

        error = self._error is not None
        if error:
            border_color = QColor("#dc2626")
            border_width = 2.0
        elif self.isSelected():
            border_color = QColor("#2563eb")
            border_width = 2.0
        elif self.node.is_source:
            border_color = QColor("#7fbfb5")
            border_width = 1.2
        else:
            border_color = QColor("#cfd6de")
            border_width = 1.2
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(border_color, border_width))
        painter.drawRoundedRect(rect.adjusted(0.6, 0.6, -0.6, -0.6), self.CORNER, self.CORNER)

        header_top = self._header_rect.top()

        # 标题与副标题
        chip = self._header_chip()
        chip_width = 0.0
        if chip is not None:
            chip_width = 58.0 if chip[0] != "错误" else 40.0
        title_font = QFont("Microsoft YaHei UI")
        title_font.setPointSizeF(9.5)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QColor("#1f2937"))
        title_text = _elide(title_font, self.node.display_title(), self._width - 24.0 - chip_width)
        painter.drawText(
            QRectF(12.0, header_top + 5.0, self._width - 24.0 - chip_width, 20.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            title_text,
        )

        subtitle_font = QFont("Microsoft YaHei UI")
        subtitle_font.setPointSizeF(7.5)
        painter.setFont(subtitle_font)
        painter.setPen(QColor("#6b7280"))
        if self.node.is_source:
            subtitle = "原始图像 · 源节点"
        elif self.spec is not None:
            subtitle = f"{self.spec.tab} · {self.spec.id}"
        else:
            subtitle = self.node.op_id
        subtitle_text = _elide(subtitle_font, subtitle, self._width - 24.0 - chip_width)
        painter.drawText(
            QRectF(12.0, header_top + 25.0, self._width - 24.0, 16.0),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            subtitle_text,
        )

        if chip is not None:
            chip_font = QFont("Microsoft YaHei UI")
            chip_font.setPointSizeF(7.0)
            painter.setFont(chip_font)
            chip_rect = QRectF(self._width - chip_width - 10.0, header_top + 13.0, chip_width, 18.0)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(chip[1])
            painter.drawRoundedRect(chip_rect, 9.0, 9.0)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(chip_rect, int(Qt.AlignmentFlag.AlignCenter), chip[0])

        # 端口标签：输入在节点内部顶部一行（圆点下方），输出在底部一行（圆点上方）。
        label_font = QFont("Microsoft YaHei UI")
        label_font.setPointSizeF(7.5)
        painter.setFont(label_font)
        painter.setPen(QColor("#4b5563"))
        for port, label_rect in zip(self._input_ports.values(), self._input_label_rects):
            text = _elide(label_font, port.label, label_rect.width())
            painter.drawText(
                label_rect,
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
                text,
            )
        for port, label_rect in zip(self._output_ports.values(), self._output_label_rects):
            text = _elide(label_font, port.label, label_rect.width())
            painter.drawText(
                label_rect,
                int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter),
                text,
            )

        # 缩略图
        if self._thumb_pixmap is not None and self._thumb_rect.height() > 0.0:
            painter.setPen(QPen(QColor("#e2e7ec"), 1.0))
            painter.setBrush(QColor("#f6f8fa"))
            painter.drawRoundedRect(self._thumb_rect, 6.0, 6.0)
            pixmap = self._thumb_pixmap
            x = self._thumb_rect.x() + (self._thumb_rect.width() - pixmap.width()) / 2.0
            y = self._thumb_rect.y() + (self._thumb_rect.height() - pixmap.height()) / 2.0
            painter.drawPixmap(QPointF(round(x), round(y)), pixmap)

        # 底部信息（位于输出标签上方，避免与标签重叠）
        footer_font = QFont("Microsoft YaHei UI")
        footer_font.setPointSizeF(7.0)
        painter.setFont(footer_font)
        painter.setPen(QColor("#94a3b8"))
        footer_y = self._footer_rect.top()
        output_count = len(self._output_ports)
        painter.drawText(
            QRectF(12.0, footer_y, self._width / 2.0, self.FOOTER_H),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            f"输出 {output_count}",
        )
        if error:
            footer_text = "错误"
            painter.setPen(QColor("#dc2626"))
        elif self._elapsed is not None:
            footer_text = f"{self._elapsed:.1f} ms"
            painter.setPen(QColor("#64748b"))
        else:
            footer_text = "未运行"
        painter.drawText(
            QRectF(self._width - 92.0, footer_y, 80.0, self.FOOTER_H),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            footer_text,
        )

    # ------------------------------------------------------------------ 交互
    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: Any) -> Any:  # noqa: N802 - Qt 接口
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and not self._syncing:
            if self._view is not None:
                self._view.on_node_item_moved(self)
        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.setZValue(1.0 if bool(value) else 0.0)
        return super().itemChange(change, value)


__all__ = ["NodeItem"]
