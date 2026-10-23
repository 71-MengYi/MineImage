"""画布上的节点卡片。

节点外观为浅色圆角卡片：标题、分类 / 算子 id、上边缘的输入端口、下边缘的输出端口、
右下角耗时、结果区（图像缩略图 / 单个数值 / 数组提示 / 便签文字）以及出错时的红色
边框与 tooltip。算子通过 ``OperatorSpec.canvas_text_param`` 声明便签文字参数后，
卡片改为暖色便签样式，只显示该参数的文字，不显示任何结果数据。
节点条目的位置始终与 :class:`NodeModel.pos`（左上角场景坐标）保持一致。

端口改为上下方向后，输入端口沿节点上边缘水平分布，输出端口沿下边缘水平分布；
端口标签分别位于节点内部顶部一行（圆点下方）与底部一行（圆点上方），保证标签、
标题、缩略图与底部信息互不重叠。
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

import math

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
    QTextLayout,
    QTextOption,
)
from PySide6.QtWidgets import QApplication, QGraphicsItem, QStyleOptionGraphicsItem, QWidget

from app.core.constants import MATRIX_SUFFIXES
from app.core.data_types import DataType, NumberValue, data_type_of, format_number
from app.core.model import NodeModel
from app.core.types import ImageData, image_values, make_thumbnail
from app.graph.port_item import PortItem
from app.operators.spec import OperatorSpec, PortSpec

if TYPE_CHECKING:  # pragma: no cover - 仅类型检查
    from app.graph.view import GraphView


#: 节点卡片右下角耗时文本的常规颜色。
ELAPSED_TEXT_COLOR = "#64748b"
#: 计算完成时耗时文本的起始绿色；动画从该颜色渐暗回 ELAPSED_TEXT_COLOR。
ELAPSED_FLASH_COLOR = "#22c55e"


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
    # 端口数量较多时节点自动加宽，避免圆点 / 标签挤在一起。
    PORT_SPACING = 62.0
    MAX_WIDTH = 460.0
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
    # 便签卡片（OperatorSpec.canvas_text_param）的文字内边距、最小 / 最大文字高度。
    NOTE_WIDTH = 240.0
    NOTE_PADDING = 10.0
    NOTE_MIN_TEXT_H = 34.0
    NOTE_MAX_TEXT_H = 150.0
    # 图像矩阵网格：方形单元格、单元格间距与序号徽标尺寸。
    GRID_SPACING = 6.0
    GRID_BADGE_W = 15.0
    GRID_BADGE_H = 13.0

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
        #: 图像矩阵网格（仅 _content_mode == "grid" 时非空）。
        self._grid_images: tuple[ImageData, ...] = ()
        #: 单图节点当前显示的图像；使单图缩略图也能拖出副本。
        self._single_image: ImageData | None = None
        self._grid_pixmaps: list[QPixmap | None] = []
        self._grid_rect = QRectF()
        self._image_tile_rects: list[QRectF] = []
        #: 结果中图像矩阵的总张数（缩略图关闭时仍保留，供文字提示 / 恢复选择）。
        self._matrix_image_count = 0
        self._selected_image_index = 0
        self._grid_press_active = False
        #: 缩略图上按下 / 拖动的状态：超过系统拖拽阈值后启动 QDrag。
        self._grid_press_index: int | None = None
        self._grid_press_pos = QPointF()
        self._grid_drag_started = False
        #: 单图缩略图按下 / 拖动状态；点击仍选中节点，拖动超过阈值则拖出副本。
        self._thumb_press_active = False
        self._thumb_press_pos = QPointF()
        self._thumb_drag_started = False
        #: 图像拖拽 / 源节点合并时的高亮状态（虚线蓝框 + 浅色遮罩）。
        self._image_drop_target = False
        self._content_mode = (
            "empty"  # empty | thumb | grid | matrix | number | numbers | array | note
        )
        self._content_label = ""
        self._content_value = ""
        self._content_hint = ""
        self._footer_number = ""
        # 由 OperatorSpec.canvas_text_param 声明的卡片文字（便签 / 注释类节点）。
        self._text_param_key = (
            str(getattr(spec, "canvas_text_param", "") or "") if spec is not None else ""
        )
        self._canvas_text = ""
        self._canvas_lines: list[str] = []
        self._canvas_line_spacing = 14.0
        self._elapsed: float | None = None
        #: 正在运行时的 monotonic 起始时间；None 表示当前没有在执行。
        self._running_since: float | None = None
        #: 计算完成后的绿色渐暗动画进度（0→1）；None 表示未播放。
        self._flash_progress: float | None = None
        self._error: str | None = None
        self._thumb_rect = QRectF()
        self._content_rect = QRectF()
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
        self._update_content()
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
    def running(self) -> bool:
        """节点是否正在后台执行（右下角显示实时耗时）。"""

        return self._running_since is not None

    @property
    def elapsed_flash_progress(self) -> float | None:
        """完成后的绿色渐暗动画进度；None 表示动画未播放。"""

        return self._flash_progress

    @property
    def elapsed_display_text(self) -> str | None:
        """右下角耗时文本：运行中为实时耗时，完成后为最终耗时。"""

        return self._elapsed_footer_text()

    @property
    def error_message(self) -> str | None:
        return self._error

    @property
    def thumbnail_pixmap(self) -> QPixmap | None:
        return self._thumb_pixmap

    @property
    def canvas_text(self) -> str:
        """便签卡片显示的文字（非便签卡片为空字符串）。"""

        return self._canvas_text

    @property
    def image_count(self) -> int:
        """当前网格中的缩略图数量；非网格模式（含缩略图关闭）为 0。"""

        return len(self._grid_images)

    @property
    def matrix_image_count(self) -> int:
        """结果中第一组图像输出的总张数（单图为 1，无图像为 0）。"""

        return self._matrix_image_count

    @property
    def has_image_grid(self) -> bool:
        return self._content_mode == "grid" and bool(self._grid_images)

    @property
    def selected_image_index(self) -> int:
        return self._selected_image_index

    @property
    def image_drop_target(self) -> bool:
        """当前节点是否被标记为拖入 / 合并目标。"""

        return self._image_drop_target

    def set_image_drop_target(self, active: bool) -> None:
        """设置合并目标高亮：虚线蓝框 + 浅色遮罩。"""

        active = bool(active)
        if active == self._image_drop_target:
            return
        self._image_drop_target = active
        self.update()

    @property
    def is_stale(self) -> bool:
        """节点引用的算子未注册 / 已失效。"""

        return (
            self.spec is None
            and not self.node.is_source
            and getattr(self.node, "group", None) is None
        )

    @property
    def is_note_card(self) -> bool:
        """卡片是否显示 ``canvas_text_param`` 指定的文字（便签 / 注释样式）。"""

        return bool(self._text_param_key)

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

    def width(self) -> float:
        """节点卡片宽度（局部坐标，不含端口圆点的包围盒）。"""

        return self._width

    def height(self) -> float:
        """节点卡片高度（局部坐标，不含端口圆点的包围盒）。"""

        return self._height

    # ------------------------------------------------------------ 图像矩阵 API
    @staticmethod
    def grid_columns(count: int) -> int:
        """图像矩阵的列数：1 张 1 列、2-4 张 2 列、5 张及以上 3 列。"""

        count = max(1, int(count))
        if count <= 1:
            return 1
        if count <= 4:
            return 2
        return 3

    def image_at(self, index: int) -> ImageData | None:
        try:
            index = int(index)
        except (TypeError, ValueError):
            return None
        if 0 <= index < len(self._grid_images):
            return self._grid_images[index]
        if index == 0 and self._single_image is not None:
            return self._single_image
        return None

    def image_pixmap(self, index: int) -> QPixmap | None:
        """返回第 index 张图的缩略图，供 QDrag 图标使用；越界返回 None。"""

        try:
            index = int(index)
        except (TypeError, ValueError):
            return None
        if index < 0:
            return None
        if index < len(self._grid_pixmaps):
            pixmap = self._grid_pixmaps[index]
            if pixmap is not None and not pixmap.isNull():
                return pixmap
        if index == 0 and self._thumb_pixmap is not None and not self._thumb_pixmap.isNull():
            # 单图节点虽然不走网格，也可以把当前缩略图作为拖动图标。
            return self._thumb_pixmap
        image = self.image_at(index)
        if image is None:
            image = self._source_image_at(index)
        if image is None:
            return None
        return self._make_grid_pixmap(image)

    def _source_image_at(self, index: int) -> ImageData | None:
        """源节点即使还没有引擎结果，也能按 source_ids 顺序取到图像。"""

        if not self.node.is_source:
            return None
        workspace = getattr(self._view, "workspace", None)
        getter = getattr(workspace, "images_for_node", None)
        if not callable(getter):
            return None
        try:
            index = int(index)
            if index < 0:
                return None
            images = getter(self.node)
            return images[index]
        except Exception:  # noqa: BLE001 - 取不到按无图处理
            return None

    def image_tile_rects(self) -> list[QRectF]:
        """返回每个缩略图单元格的局部矩形（按行优先顺序）。"""

        return [QRectF(rect) for rect in self._image_tile_rects]

    def image_index_at(self, pos: QPointF) -> int | None:
        """命中测试：返回局部坐标下命中的缩略图序号，未命中返回 None。"""

        if not self.has_image_grid:
            return None
        try:
            point = QPointF(pos)
        except (TypeError, ValueError):
            return None
        for index, rect in enumerate(self._image_tile_rects):
            if rect.contains(point):
                return index
        return None

    def set_selected_image_index(self, index: int) -> None:
        """选中矩阵中的第 index 张图；越界时钳制到有效范围。"""

        count = len(self._grid_images) or self._matrix_image_count
        if count <= 0:
            index = 0
        else:
            try:
                index = int(index)
            except (TypeError, ValueError):
                index = 0
            index = max(0, min(index, count - 1))
        if index != self._selected_image_index:
            self._selected_image_index = index
            self.update()

    def _refresh_connections(self) -> None:
        """节点高度变化会移动上下边缘端口，通知视图刷新相连连线。"""

        refresh = getattr(self._view, "on_node_item_moved", None)
        if callable(refresh):
            refresh(self)

    # ------------------------------------------------------------------ 端口
    def _port_specs(self) -> tuple[list[PortSpec], list[PortSpec]]:
        if self.node.is_source:
            return [], [PortSpec("image", "图像", description="原始图像输出")]
        if self.spec is None:
            # 未注册的算子仍然允许连线，便于打开旧工程或临时删除算子文件。
            return [PortSpec("image", "图像")], [PortSpec("image", "图像")]
        return list(self.spec.inputs), list(self.spec.outputs)

    def _build_ports(self) -> None:
        inputs, outputs = self._port_specs()
        for index, port_spec in enumerate(inputs):
            self._input_ports[port_spec.key] = PortItem(
                self, port_spec.key, port_spec.label, True, index, port_spec=port_spec
            )
        for index, port_spec in enumerate(outputs):
            self._output_ports[port_spec.key] = PortItem(
                self, port_spec.key, port_spec.label, False, index, port_spec=port_spec
            )

    def port_item(self, key: str, is_input: bool) -> PortItem | None:
        return (self._input_ports if is_input else self._output_ports).get(key)

    def port_spec(self, key: str, is_input: bool) -> PortSpec | None:
        port = self.port_item(key, is_input)
        return port.port_spec if port is not None else None

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
        port_count = max(input_count, output_count, 1)
        needed = 2.0 * self.PORT_MARGIN + self.PORT_SPACING * (port_count - 1)
        base_width = self.NOTE_WIDTH if self.is_note_card else self.WIDTH
        self._width = min(self.MAX_WIDTH, max(base_width, needed))

        # 便签文字按最终卡片宽度排版，下面用它计算文字区高度。
        if self.is_note_card:
            self._layout_canvas_text()

        header_top = self.TOP_LABEL_GAP
        if input_count:
            header_top += self.LABEL_H + self.LABEL_GAP
        self._header_rect = QRectF(0.0, header_top, self._width, self.HEADER_H)
        self._content_top = header_top + self.HEADER_H + self.CONTENT_GAP

        thumb_area = 0.0
        self._thumb_rect = QRectF()
        self._grid_rect = QRectF()
        self._image_tile_rects = []
        self._content_rect = QRectF()
        if self._content_mode == "grid" and self._grid_images:
            # 图像矩阵：中部 content 区域按方形网格排布，四周留 THUMB_MARGIN。
            self._layout_image_grid()
            content_height = self._grid_rect.height() + 2.0 * self.THUMB_MARGIN
        else:
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
            text_area = 0.0
            if not thumb_area:
                if self.is_note_card:
                    text_area = self._canvas_text_height()
                elif self._content_mode == "number":
                    text_area = 48.0
                elif self._content_mode in ("array", "numbers", "matrix"):
                    text_area = 40.0
                if text_area:
                    self._content_rect = QRectF(
                        self.THUMB_MARGIN,
                        self._content_top,
                        self._width - 2.0 * self.THUMB_MARGIN,
                        text_area,
                    )
            content_height = thumb_area or text_area or self.MIN_CONTENT_H

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

    def _layout_image_grid(self) -> None:
        """计算方形单元格网格；节点宽度不足时按可用宽度自动收缩。"""

        count = len(self._grid_images)
        columns = self.grid_columns(count)
        rows = (count + columns - 1) // columns
        spacing = self.GRID_SPACING
        available = max(1.0, self._width - 2.0 * self.THUMB_MARGIN)
        cell = min(
            float(self._thumbnail_box),
            (available - spacing * (columns - 1)) / columns,
        )
        cell = max(12.0, cell)
        row_width = columns * cell + spacing * (columns - 1)
        grid_height = rows * cell + spacing * (rows - 1)
        left = (self._width - row_width) / 2.0
        top = self._content_top + self.THUMB_MARGIN
        rects: list[QRectF] = []
        for index in range(count):
            row, column = divmod(index, columns)
            rects.append(
                QRectF(
                    left + column * (cell + spacing),
                    top + row * (cell + spacing),
                    cell,
                    cell,
                )
            )
        self._image_tile_rects = rects
        self._grid_rect = QRectF(left, top, row_width, grid_height)

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
        self._running_since = None
        self._elapsed = float(getattr(result, "elapsed_ms", 0.0) or 0.0)
        self._error = None
        self._update_content()
        self._rebuild_geometry()
        self._refresh_connections()
        self._refresh_tooltip()
        self.update()

    def set_running(self) -> None:
        """开始运行：右下角从“未运行”切换为实时增加的耗时。"""

        self._running_since = time.monotonic()
        self._flash_progress = None
        if self._error is not None:
            # 重跑时旧错误不再成立，先恢复普通卡片样式。
            self._error = None
            self._refresh_tooltip()
        self.update(self._footer_rect)

    def clear_running(self) -> None:
        """结束运行计时（最终耗时由后续 ``set_result`` 写入）。"""

        if self._running_since is None:
            return
        self._running_since = None
        self.update(self._footer_rect)

    def set_elapsed_flash(self, progress: float | None) -> None:
        """设置完成动画进度；None 表示恢复常规耗时样式。"""

        if progress is None:
            self._flash_progress = None
        else:
            self._flash_progress = max(0.0, min(1.0, float(progress)))
        self.update(self._footer_rect)

    def refresh_running_elapsed(self) -> None:
        """运行中由画布定时器调用：只重绘底部耗时区域。"""

        if self._running_since is not None:
            self.update(self._footer_rect)

    def _elapsed_footer_text(self) -> str | None:
        if self._running_since is not None:
            seconds = max(0.0, time.monotonic() - self._running_since)
            if seconds < 1.0:
                return f"{seconds * 1000.0:.0f} ms"
            return f"{seconds:.1f} s"
        if self._elapsed is not None:
            return f"{self._elapsed:.1f} ms"
        return None

    def _elapsed_footer_color(self) -> QColor:
        base = QColor(ELAPSED_TEXT_COLOR)
        if self._flash_progress is None:
            return base
        start = QColor(ELAPSED_FLASH_COLOR)
        t = self._flash_progress
        return QColor(
            round(start.red() + (base.red() - start.red()) * t),
            round(start.green() + (base.green() - start.green()) * t),
            round(start.blue() + (base.blue() - start.blue()) * t),
        )

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
            self._update_content()
            self._rebuild_geometry()
            self._refresh_connections()
            self.update()

    def set_error(self, message: str | None) -> None:
        message = message or None
        was_running = self._running_since is not None
        self._running_since = None
        if self._error == message and not was_running:
            return
        self._error = message
        self._refresh_tooltip()
        self.update()

    def refresh_from_model(self) -> None:
        if self.is_note_card:
            # 便签文字来自参数：文字变化会改变卡片高度，需要重新排版。
            self._update_content()
            self._rebuild_geometry()
            self._refresh_connections()
        self._refresh_tooltip()
        self.update()

    def _update_content(self) -> None:
        """按结果类型决定节点中部显示缩略图、图像矩阵、数值、数组提示或便签文字。"""

        self._thumb_pixmap = None
        self._single_image = None
        self._grid_images = ()
        self._grid_pixmaps = []
        self._matrix_image_count = 0
        self._content_mode = "empty"
        self._content_label = ""
        self._content_value = ""
        self._content_hint = ""
        self._footer_number = ""
        if self.is_note_card:
            self._update_canvas_text()
            return
        if self._result is None:
            self._selected_image_index = 0
            return

        outputs = getattr(self._result, "outputs", {}) or {}
        numbers = [value for value in outputs.values() if isinstance(value, NumberValue)]
        # 找到第一个含图像的输出（单张 ImageData 或 ImageList），多图时展开成矩阵。
        images: tuple[ImageData, ...] = ()
        for value in outputs.values():
            images = image_values(value)
            if images:
                break
        if not images:
            preview = getattr(self._result, "preview_image", None)
            if isinstance(preview, ImageData):
                images = (preview,)
        self._matrix_image_count = len(images)

        if len(images) > 1:
            # 缩略图开启时在节点中部渲染网格；关闭时退回文字提示。
            self._selected_image_index = max(
                0, min(self._selected_image_index, len(images) - 1)
            )
            if self._thumbnails_enabled:
                self._grid_images = images
                self._grid_pixmaps = [self._make_grid_pixmap(image) for image in images]
                self._content_mode = "grid"
            else:
                self._content_mode = "matrix"
                self._content_label = f"图像矩阵 ×{len(images)}"
                self._content_hint = "开启节点缩略图查看图像矩阵"
            if len(numbers) == 1:
                self._footer_number = self._footer_number_text(numbers[0])
            return

        self._selected_image_index = 0
        if len(images) == 1:
            self._single_image = images[0]
        if len(images) == 1 and self._thumbnails_enabled:
            try:
                rgb = make_thumbnail(images[0], self._thumbnail_box)
                self._thumb_pixmap = _rgb_to_pixmap(rgb)
                self._content_mode = "thumb"
                if len(numbers) == 1:
                    self._footer_number = self._footer_number_text(numbers[0])
                return
            except Exception:  # noqa: BLE001 - 缩略图失败不能影响画布
                self._thumb_pixmap = None
        if len(numbers) == 1:
            value = numbers[0]
            self._content_mode = "number"
            self._content_label = str(value.label or "数值").strip()
            unit = str(value.unit or "").strip()
            self._content_value = format_number(value.value) + (f" {unit}" if unit else "")
            return

        array_types = (DataType.EDGES, DataType.CONTOURS, DataType.COMPONENTS)
        array_values = [
            value for value in outputs.values() if data_type_of(value) in array_types
        ]
        if array_values:
            self._content_mode = "array"
            self._content_label = "数组数据"
            self._content_hint = "双击节点查看详情"
            return

        if numbers:
            self._content_mode = "numbers"
            self._content_label = f"数值 ×{len(numbers)}"
            self._content_hint = "双击节点查看详情"

    # ------------------------------------------------------------- 便签文字
    @staticmethod
    def _canvas_text_font() -> QFont:
        font = QFont("Microsoft YaHei UI")
        font.setPointSizeF(8.6)
        return font

    @staticmethod
    def _canvas_text_placeholder() -> str:
        return "双击节点输入便签文字"

    def _update_canvas_text(self) -> None:
        params = getattr(self.node, "params", None) or {}
        value = params.get(self._text_param_key, "")
        self._canvas_text = "" if value is None else str(value)
        self._content_mode = "note"

    @staticmethod
    def _wrap_canvas_text(text: str, width: float, font: QFont) -> list[str]:
        """按给定宽度换行；保留显式换行产生的空行。"""

        option = QTextOption()
        option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        wrapped: list[str] = []
        for paragraph in text.splitlines() or [""]:
            if not paragraph:
                wrapped.append("")
                continue
            layout = QTextLayout(paragraph, font)
            layout.setTextOption(option)
            layout.beginLayout()
            while True:
                line = layout.createLine()
                if not line.isValid():
                    break
                line.setLineWidth(width)
                start = line.textStart()
                wrapped.append(paragraph[start : start + line.textLength()])
            layout.endLayout()
        return wrapped or [""]

    def _layout_canvas_text(self) -> None:
        """按当前卡片宽度排版便签文字；超出上限时在最后一行加省略号。"""

        text = self._canvas_text.strip() or self._canvas_text_placeholder()
        width = max(
            40.0,
            self._width - 2.0 * self.THUMB_MARGIN - 2.0 * self.NOTE_PADDING,
        )
        font = self._canvas_text_font()
        metrics = QFontMetrics(font)
        self._canvas_line_spacing = max(12.0, float(metrics.lineSpacing()))
        lines = self._wrap_canvas_text(text, width, font)
        max_lines = max(1, int(self.NOTE_MAX_TEXT_H // self._canvas_line_spacing))
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = _elide(font, lines[-1].rstrip() + "…", width)
        self._canvas_lines = lines

    def _canvas_text_height(self) -> float:
        lines = max(1, len(self._canvas_lines))
        text_height = min(self.NOTE_MAX_TEXT_H, lines * self._canvas_line_spacing)
        return max(self.NOTE_MIN_TEXT_H, text_height) + 2.0 * self.NOTE_PADDING

    def _is_matrix_source(self) -> bool:
        """源节点是否由 CSV / 文本矩阵导入（画布用「矩阵」区分）。"""

        if not self.node.is_source or not self.node.source_path:
            return False
        return Path(self.node.source_path).suffix.lower() in MATRIX_SUFFIXES

    @staticmethod
    def _footer_number_text(value: NumberValue) -> str:
        unit = str(value.unit or "").strip()
        label = str(value.label or "数值").strip()
        return f"{label} {format_number(value.value)}" + (f" {unit}" if unit else "")

    def _make_grid_pixmap(self, image: ImageData) -> QPixmap | None:
        try:
            return _rgb_to_pixmap(make_thumbnail(image, self._thumbnail_box))
        except Exception:  # noqa: BLE001 - 单张缩略图失败时保留空占位格
            return None

    def _refresh_tooltip(self) -> None:
        lines: list[str] = [self.node.display_title()]
        if self.node.is_source:
            if self._is_matrix_source():
                lines.append("矩阵数据节点（由 CSV / 文本矩阵导入）")
            else:
                lines.append("原始图像节点")
        elif self.spec is not None:
            lines.append(f"{self.spec.name} · {self.spec.id}")
            if self.spec.description:
                lines.append(self.spec.description.strip().splitlines()[0])
            if self.is_note_card:
                lines.append("便签文字（双击节点编辑）：")
                lines.append(self._canvas_text.strip() or "（空便签）")
            else:
                for port in self.spec.outputs:
                    lines.append(f"输出 {port.label}：{port.type_text()}")
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
            if self._is_matrix_source():
                return "矩阵", QColor("#0f766e")
            return "原始图像", QColor("#0f766e")
        if self.is_stale:
            return "失效", QColor("#b45309")
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

        note_card = self.is_note_card
        painter.fillPath(body, QColor("#fffdf2") if note_card else QColor("#ffffff"))

        painter.save()
        painter.setClipPath(body)
        header_color = QColor("#f6f8fb")
        if self.node.is_source:
            header_color = QColor("#eef7f5")
        elif note_card:
            header_color = QColor("#fbf3d6")
        painter.fillRect(self._header_rect, header_color)
        if self.node.is_source or note_card:
            accent = "#0f766e" if self.node.is_source else "#d9b84a"
            painter.fillRect(QRectF(0.0, 0.0, 3.0, self._height), QColor(accent))
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
        elif note_card:
            border_color = QColor("#e6d7a4")
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
            if self._is_matrix_source():
                subtitle = "矩阵数据 · 源节点"
            else:
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

        # 便签文字：直接绘制在卡片中部，不画结果数据框。
        if self._content_mode == "note" and self._content_rect.height() > 0.0:
            self._paint_canvas_text(painter)
        # 图像矩阵网格：每格一张缩略图，选中格加蓝色描边。
        elif self.has_image_grid:
            self._paint_image_grid(painter)
        # 数值 / 数组提示：没有缩略图时占用节点中部
        elif self._content_rect.height() > 0.0:
            painter.setPen(QPen(QColor("#e2e7ec"), 1.0))
            painter.setBrush(QColor("#f8fafc"))
            painter.drawRoundedRect(self._content_rect, 6.0, 6.0)
            content_rect = self._content_rect.adjusted(10.0, 4.0, -10.0, -4.0)
            if self._content_mode == "number":
                label_font = QFont("Microsoft YaHei UI")
                label_font.setPointSizeF(7.0)
                painter.setFont(label_font)
                painter.setPen(QColor("#64748b"))
                painter.drawText(
                    QRectF(content_rect.x(), content_rect.y(), content_rect.width(), 14.0),
                    int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                    _elide(label_font, self._content_label, content_rect.width()),
                )
                value_font = QFont("Microsoft YaHei UI")
                value_font.setPointSizeF(12.5)
                value_font.setBold(True)
                painter.setFont(value_font)
                painter.setPen(QColor("#1f2937"))
                painter.drawText(
                    QRectF(content_rect.x(), content_rect.y() + 14.0, content_rect.width(), 26.0),
                    int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
                    _elide(value_font, self._content_value, content_rect.width()),
                )
            else:
                label_font = QFont("Microsoft YaHei UI")
                label_font.setPointSizeF(8.5)
                label_font.setBold(True)
                painter.setFont(label_font)
                painter.setPen(QColor("#475569"))
                painter.drawText(
                    QRectF(content_rect.x(), content_rect.y(), content_rect.width(), 17.0),
                    int(Qt.AlignmentFlag.AlignCenter),
                    _elide(label_font, self._content_label, content_rect.width()),
                )
                hint_font = QFont("Microsoft YaHei UI")
                hint_font.setPointSizeF(7.0)
                painter.setFont(hint_font)
                painter.setPen(QColor("#94a3b8"))
                painter.drawText(
                    QRectF(content_rect.x(), content_rect.y() + 17.0, content_rect.width(), 15.0),
                    int(Qt.AlignmentFlag.AlignCenter),
                    _elide(hint_font, self._content_hint, content_rect.width()),
                )

        # 底部信息（位于输出标签上方，避免与标签重叠）
        footer_font = QFont("Microsoft YaHei UI")
        footer_font.setPointSizeF(7.0)
        painter.setFont(footer_font)
        painter.setPen(QColor("#a29464") if note_card else QColor("#94a3b8"))
        footer_y = self._footer_rect.top()
        output_count = len(self._output_ports)
        if note_card:
            footer_left = "便签"
        else:
            footer_left = self._footer_number or f"输出 {output_count}"
        painter.drawText(
            QRectF(12.0, footer_y, self._width / 2.0, self.FOOTER_H),
            int(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter),
            _elide(footer_font, footer_left, self._width / 2.0 - 20.0),
        )
        if error:
            footer_text = "错误"
            painter.setPen(QColor("#dc2626"))
        elif note_card:
            count = len(self._canvas_text.strip())
            footer_text = f"{count} 字" if count else "空便签"
            painter.setPen(QColor("#a29464"))
        else:
            elapsed_text = self._elapsed_footer_text()
            if elapsed_text is None:
                footer_text = "未运行"
            else:
                footer_text = elapsed_text
                if self._running_since is None:
                    # 完成动画：亮绿渐暗回常规耗时颜色。
                    painter.setPen(self._elapsed_footer_color())
                else:
                    painter.setPen(QColor(ELAPSED_TEXT_COLOR))
        painter.drawText(
            QRectF(self._width - 92.0, footer_y, 80.0, self.FOOTER_H),
            int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter),
            footer_text,
        )

        if self._image_drop_target:
            self._paint_image_drop_target(painter, body)

    def _paint_canvas_text(self, painter: QPainter) -> None:
        """绘制便签文字；文字为空时显示「双击节点输入便签文字」占位。"""

        text_rect = self._content_rect.adjusted(
            self.NOTE_PADDING,
            self.NOTE_PADDING,
            -self.NOTE_PADDING,
            -self.NOTE_PADDING,
        )
        if text_rect.width() <= 4.0 or text_rect.height() <= 4.0:
            return
        font = self._canvas_text_font()
        metrics = QFontMetrics(font)
        painter.setFont(font)
        placeholder = not self._canvas_text.strip()
        painter.setPen(QColor("#b5a86f") if placeholder else QColor("#564a20"))
        painter.save()
        painter.setClipRect(text_rect)
        y = text_rect.top() + float(metrics.ascent())
        for line in self._canvas_lines:
            if y - metrics.ascent() > text_rect.bottom():
                break
            painter.drawText(QPointF(text_rect.left(), y), line)
            y += self._canvas_line_spacing
        painter.restore()

    def _paint_image_drop_target(self, painter: QPainter, body: QPainterPath) -> None:
        """合并目标高亮：浅色遮罩 + 虚线蓝框，提示松开即可合并。"""

        rect = QRectF(0.0, 0.0, self._width, self._height)
        painter.save()
        try:
            painter.setClipPath(body)
            painter.fillRect(rect, QColor(37, 99, 235, 30))
        finally:
            painter.restore()
        pen = QPen(QColor("#2563eb"), 2.0, Qt.PenStyle.DashLine)
        pen.setWidthF(2.0)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(pen)
        painter.drawRoundedRect(
            rect.adjusted(2.0, 2.0, -2.0, -2.0), self.CORNER, self.CORNER
        )

    def _paint_image_grid(self, painter: QPainter) -> None:
        """绘制图像矩阵：方形单元格 + 居中缩略图 + 左上角序号徽标。"""

        badge_font = QFont("Microsoft YaHei UI")
        badge_font.setPointSizeF(6.5)
        badge_font.setBold(True)
        painter.save()
        try:
            for index, rect in enumerate(self._image_tile_rects):
                selected = index == self._selected_image_index
                border_color = QColor("#2563eb") if selected else QColor("#e2e7ec")
                painter.setPen(QPen(border_color, 2.0 if selected else 1.0))
                painter.setBrush(QColor("#f6f8fa"))
                painter.drawRoundedRect(rect, 5.0, 5.0)
                pixmap = (
                    self._grid_pixmaps[index]
                    if index < len(self._grid_pixmaps)
                    else None
                )
                if pixmap is not None and not pixmap.isNull():
                    scaled = pixmap.scaled(
                        max(1, int(rect.width() - 4.0)),
                        max(1, int(rect.height() - 4.0)),
                        Qt.AspectRatioMode.KeepAspectRatio,
                        Qt.TransformationMode.SmoothTransformation,
                    )
                    x = rect.center().x() - scaled.width() / 2.0
                    y = rect.center().y() - scaled.height() / 2.0
                    painter.drawPixmap(QPointF(round(x), round(y)), scaled)
                # 左上角序号徽标，便于与工作区预览的“第 n/N 张”对应。
                badge_rect = QRectF(
                    rect.x() + 4.0,
                    rect.y() + 4.0,
                    self.GRID_BADGE_W,
                    self.GRID_BADGE_H,
                )
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(
                    QColor(37, 99, 235, 235) if selected else QColor(15, 23, 42, 160)
                )
                painter.drawRoundedRect(badge_rect, 4.0, 4.0)
                painter.setFont(badge_font)
                painter.setPen(QColor("#ffffff"))
                painter.drawText(
                    badge_rect,
                    int(Qt.AlignmentFlag.AlignCenter),
                    str(index + 1),
                )
        finally:
            painter.restore()

    # ------------------------------------------------------------------ 交互
    def _select_grid_image(self, index: int) -> None:
        """选中网格中的一张图，并通知视图切换工作区预览（不改节点选中状态）。"""

        self.set_selected_image_index(index)
        callback = getattr(self._view, "on_node_image_selected", None)
        if callable(callback):
            try:
                callback(self, self._selected_image_index)
            except Exception:  # noqa: BLE001 - 预览切换失败不能影响画布交互
                pass

    def _reset_grid_press(self) -> None:
        self._grid_press_active = False
        self._grid_press_index = None
        self._grid_drag_started = False

    def _reset_thumb_press(self) -> None:
        self._thumb_press_active = False
        self._thumb_drag_started = False

    def _start_thumb_image_drag(self) -> None:
        """单图缩略图拖动超过阈值后把图像副本交给视图启动 QDrag。"""

        self._thumb_drag_started = True
        starter = getattr(self._view, "start_image_drag", None)
        if not callable(starter):
            return
        try:
            starter(self, 0)
        except Exception:  # noqa: BLE001 - 拖拽启动失败不能影响画布交互
            return

    def _start_grid_image_drag(self) -> None:
        """超过系统拖拽阈值后把拖动交给视图启动 QDrag。"""

        self._grid_drag_started = True
        starter = getattr(self._view, "start_image_drag", None)
        if not callable(starter):
            return
        index = self._grid_press_index
        try:
            starter(self, 0 if index is None else int(index))
        except Exception:  # noqa: BLE001 - 拖拽启动失败不能影响画布交互
            return

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        if event.button() == Qt.MouseButton.LeftButton:
            index = self.image_index_at(event.pos())
            if index is not None:
                # 缩略图点击只切换选中的图：不选中节点、不移动节点，也不清空其它选择。
                self._reset_thumb_press()
                self._grid_press_active = True
                self._grid_press_index = index
                self._grid_press_pos = QPointF(event.pos())
                self._grid_drag_started = False
                self._select_grid_image(index)
                event.accept()
                return
            if self._content_mode == "thumb" and self._thumb_rect.contains(event.pos()):
                # 单图缩略图：先走默认按下逻辑（保持“点击选中节点”），拖动超过阈值时
                # 由 mouseMoveEvent 改为拖出图像副本。
                self._reset_grid_press()
                self._thumb_press_active = True
                self._thumb_press_pos = QPointF(event.pos())
                self._thumb_drag_started = False
                super().mousePressEvent(event)
                return
        self._reset_grid_press()
        self._reset_thumb_press()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        left_pressed = bool(event.buttons() & Qt.MouseButton.LeftButton)
        if self._thumb_press_active:
            if not left_pressed:
                self._reset_thumb_press()
                super().mouseMoveEvent(event)
                return
            if not self._thumb_drag_started:
                delta = event.pos() - self._thumb_press_pos
                if math.hypot(delta.x(), delta.y()) >= QApplication.startDragDistance():
                    self._start_thumb_image_drag()
            event.accept()
            return
        if self._grid_press_active:
            if not left_pressed:
                self._reset_grid_press()
                super().mouseMoveEvent(event)
                return
            # 从缩略图上拖动不应带动节点；超过阈值后改为拖出图像副本。
            if not self._grid_drag_started:
                delta = event.pos() - self._grid_press_pos
                if math.hypot(delta.x(), delta.y()) >= QApplication.startDragDistance():
                    self._start_grid_image_drag()
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        if self._thumb_press_active:
            self._reset_thumb_press()
            event.accept()
            return
        if self._grid_press_active:
            self._reset_grid_press()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt 接口
        if event.button() == Qt.MouseButton.LeftButton:
            index = self.image_index_at(event.pos())
            if index is not None:
                # 双击缩略图也只切换选中图，不触发参数面板。
                self._reset_grid_press()
                self._reset_thumb_press()
                self._select_grid_image(index)
                event.accept()
                return
        self._reset_thumb_press()
        super().mouseDoubleClickEvent(event)

    def itemChange(self, change: QGraphicsItem.GraphicsItemChange, value: Any) -> Any:  # noqa: N802 - Qt 接口
        if change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged and not self._syncing:
            if self._view is not None:
                self._view.on_node_item_moved(self)
        elif change == QGraphicsItem.GraphicsItemChange.ItemSelectedHasChanged:
            self.setZValue(1.0 if bool(value) else 0.0)
        return super().itemChange(change, value)


__all__ = ["NodeItem"]
