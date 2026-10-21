"""节点参数面板：按 :class:`OperatorSpec` 自动生成编辑器。

设计要点
--------
* 按 ``ParamSpec.group`` 分组，``visible_when`` / ``enabled_when`` 实时生效；
* 任意编辑都会走 120ms 防抖后发出 ``preview_requested``；
* 使用 :func:`app.operators.spec.normalize_params` 做声明式校验，非法时显示
  行内错误并禁用「保存」；
* 交互参数（点集 / 矩形 / 直线）只发出 ``pick_requested``，真正的拾取由预览
  控件完成；不参与下游重算；
* 只要当前结果包含数值 / 结构化数组输出，参数区下方就显示只读的「输出数据」；
  数组以文本详情输出，不再渲染成预览图，也不进入画布缩略图与主预览。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from PySide6.QtCore import QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QPainter, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core.constants import IMAGE_FILTER, PARAM_DEBOUNCE_MS
from app.core.data_types import (
    ComponentsData,
    ContoursData,
    DataType,
    EdgesData,
    NumberValue,
    data_type_of,
    format_number,
    summarize_value,
)
from app.core.model import NodeModel
from app.operators.spec import (
    Operator,
    ParamSpec,
    ParamValidationError,
    coerce_param_value,
    is_param_enabled,
    is_param_visible,
    normalize_params,
    param_defaults,
)

_INTERACTIVE_TYPES = ("points", "roi", "line")
_PICK_TOOLS = {"points", "rect", "line"}


@dataclass
class _ParamRow:
    spec: ParamSpec
    row: QWidget
    label: QLabel
    field: QWidget
    getter: Callable[[], Any]
    setter: Callable[[Any], None]


def _fmt_number(value: float) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return f"{number:g}"


def _points_text(points: Any) -> str:
    if not isinstance(points, (list, tuple)) or not points:
        return "未设置（0 个点）"
    pairs = []
    for point in points[:3]:
        try:
            pairs.append(f"({_fmt_number(point[0])}, {_fmt_number(point[1])})")
        except (TypeError, IndexError, ValueError):
            continue
    text = f"{len(points)} 个点 · " + " ".join(pairs)
    if len(points) > 3:
        text += " …"
    return text


def _points_full_text(points: Any) -> str:
    if not isinstance(points, (list, tuple)) or not points:
        return "未设置"
    pairs = []
    for point in points:
        try:
            pairs.append(f"({_fmt_number(point[0])}, {_fmt_number(point[1])})")
        except (TypeError, IndexError, ValueError):
            continue
    return "  ".join(pairs)


def _rect_text(value: Any) -> str:
    if value in (None, [], ()):
        return "未设置"
    try:
        x, y, width, height = (float(item) for item in value)
    except (TypeError, ValueError):
        return "未设置"
    return f"x={_fmt_number(x)}  y={_fmt_number(y)}  w={_fmt_number(width)}  h={_fmt_number(height)}"


def _line_text(value: Any) -> str:
    if value in (None, [], ()):
        return "未设置"
    try:
        x1, y1, x2, y2 = (float(item) for item in value)
    except (TypeError, ValueError):
        return "未设置"
    return f"({_fmt_number(x1)}, {_fmt_number(y1)}) → ({_fmt_number(x2)}, {_fmt_number(y2)})"


def _clear_layout(layout: QVBoxLayout) -> None:
    while layout.count():
        item = layout.takeAt(0)
        widget = item.widget()
        if widget is not None:
            widget.setParent(None)
            widget.deleteLater()


def _color_icon(color: str) -> QIcon:
    pixmap = QPixmap(20, 20)
    pixmap.fill(Qt.GlobalColor.transparent)
    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.setPen(QColor("#94a3b8"))
    painter.setBrush(QColor(color))
    painter.drawRoundedRect(QRectF(1.5, 1.5, 17.0, 17.0), 3.0, 3.0)
    painter.end()
    return QIcon(pixmap)


class ParamPanel(QWidget):
    """节点参数编辑面板。"""

    preview_requested = Signal(str, dict)
    save_requested = Signal(str, dict)
    back_requested = Signal()
    pick_requested = Signal(str, str, object)
    status_message = Signal(str)  # 提示 / 错误信息，供工作区状态栏展示

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # 参数面板与画布共用中间一栏；默认三栏比例 1:2:4 时约 420px 宽，
        # 面板内部有滚动区，用户把分隔条拖窄到 220px 时仍可正常使用。
        self.setMinimumWidth(220)
        self._node: NodeModel | None = None
        self._operator: Operator | None = None
        self._spec = None
        self._node_id = ""
        self._rows: dict[str, _ParamRow] = {}
        self._groups: list[tuple[QGroupBox, list[str]]] = []
        self._pick_values: dict[str, Any] = {}
        self._baseline: dict[str, Any] = {}
        self._normalized: dict[str, Any] = {}
        self._valid = False
        self._loading = False
        self._last_status_message = ""
        self._last_result: Any = None
        self._output_group: QGroupBox | None = None
        self._output_layout: QVBoxLayout | None = None
        self._empty_params_label: QLabel | None = None

        self._preview_timer = QTimer(self)
        self._preview_timer.setSingleShot(True)
        self._preview_timer.setInterval(PARAM_DEBOUNCE_MS)
        self._preview_timer.timeout.connect(self._emit_preview)

        self._build_ui()
        self._show_placeholder()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        self.header = QFrame(self)
        self.header.setObjectName("panelHeader")
        header_layout = QVBoxLayout(self.header)
        header_layout.setContentsMargins(10, 8, 10, 8)
        header_layout.setSpacing(3)
        top_row = QHBoxLayout()
        top_row.setSpacing(8)
        self.back_button = QPushButton("← 返回流程", self.header)
        self.back_button.setObjectName("linkButton")
        self.back_button.setToolTip("返回画布（Esc）")
        self.back_button.clicked.connect(self.back_requested.emit)
        self.live_badge = QLabel("● 实时预览", self.header)
        self.live_badge.setObjectName("liveBadge")
        self.live_badge.setProperty("state", "ok")
        top_row.addWidget(self.back_button, 0)
        top_row.addStretch(1)
        top_row.addWidget(self.live_badge, 0)
        header_layout.addLayout(top_row)
        self.title_label = QLabel("未选择节点", self.header)
        self.title_label.setObjectName("sectionTitle")
        self.title_label.setWordWrap(False)
        self.meta_label = QLabel("双击画布中的节点可编辑参数", self.header)
        self.meta_label.setObjectName("hintLabel")
        self.meta_label.setWordWrap(False)
        header_layout.addWidget(self.title_label)
        header_layout.addWidget(self.meta_label)
        layout.addWidget(self.header)

        self._stack = QStackedWidget(self)
        self.placeholder = QLabel(
            "选择一个节点以编辑参数\n单击节点可预览，双击进入参数面板",
            self,
        )
        self.placeholder.setObjectName("hintLabel")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)

        self.editor_page = QWidget(self)
        editor_layout = QVBoxLayout(self.editor_page)
        editor_layout.setContentsMargins(0, 0, 0, 0)
        editor_layout.setSpacing(0)
        self.scroll = QScrollArea(self.editor_page)
        self.scroll.setWidgetResizable(True)
        self.scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.content = QWidget(self.scroll)
        self.content_layout = QVBoxLayout(self.content)
        self.content_layout.setContentsMargins(2, 2, 2, 2)
        self.content_layout.setSpacing(8)
        self.content_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        self.scroll.setWidget(self.content)
        editor_layout.addWidget(self.scroll)

        self._stack.addWidget(self.placeholder)
        self._stack.addWidget(self.editor_page)
        layout.addWidget(self._stack, 1)

        self.footer = QFrame(self)
        self.footer.setObjectName("panelFooter")
        footer_layout = QVBoxLayout(self.footer)
        footer_layout.setContentsMargins(10, 8, 10, 8)
        footer_layout.setSpacing(6)
        self.error_label = QLabel("", self.footer)
        self.error_label.setObjectName("errorLabel")
        self.error_label.setWordWrap(True)
        self.error_label.setVisible(False)
        footer_layout.addWidget(self.error_label)
        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        # 中间栏在 1:2:4 下约 420px；按钮文案保持简短，完整说明放在 tooltip 中。
        self.reset_button = QPushButton("重置", self.footer)
        self.reset_button.setObjectName("subtleButton")
        self.reset_button.setToolTip("把所有参数恢复为算子声明的默认值")
        self.reset_button.clicked.connect(self._on_reset_clicked)
        self.undo_button = QPushButton("撤销修改", self.footer)
        self.undo_button.setObjectName("subtleButton")
        self.undo_button.setToolTip("恢复到打开面板时的参数")
        self.undo_button.clicked.connect(self._on_undo_clicked)
        self.save_button = QPushButton("保存", self.footer)
        self.save_button.setObjectName("primaryButton")
        self.save_button.setToolTip("保存并触发下游节点重新计算")
        self.save_button.clicked.connect(self._on_save_clicked)
        buttons.addWidget(self.reset_button, 0)
        buttons.addWidget(self.undo_button, 0)
        buttons.addStretch(1)
        buttons.addWidget(self.save_button, 0)
        footer_layout.addLayout(buttons)
        layout.addWidget(self.footer)

    # ------------------------------------------------------------------ 数据加载
    def load_node(self, node: NodeModel | None, operator: Operator | None) -> None:
        """载入节点与算子，重建编辑器。"""
        self._preview_timer.stop()
        if node is None or operator is None:
            self._node = None
            self._operator = None
            self._spec = None
            self._node_id = ""
            self._rows.clear()
            self._groups.clear()
            self._pick_values.clear()
            self._baseline = {}
            self._normalized = {}
            self._valid = False
            self._last_result = None
            self._output_group = None
            self._output_layout = None
            self._empty_params_label = None
            self._show_placeholder()
            return

        self._node = node
        self._operator = operator
        self._spec = operator.spec
        self._node_id = node.id
        self._last_result = None
        self._output_group = None
        self._output_layout = None
        self._empty_params_label = None
        defaults = param_defaults(self._spec)
        merged = dict(defaults)
        merged.update(node.params or {})
        try:
            baseline = normalize_params(self._spec, merged)
        except ParamValidationError:
            baseline = merged
        self._baseline = dict(baseline)

        self.title_label.setText(node.display_title())
        category = " / ".join(self._spec.category) or "未分类"
        self.meta_label.setText(f"{self._spec.name} · {category}")
        self.meta_label.setToolTip(f"算子 id：{self._spec.id}")
        self.live_badge.setVisible(True)

        self._build_editors()
        self._apply_values(self._baseline)
        self._show_editor()
        self._validate()
        self._preview_timer.stop()

    def set_result(self, result: Any | None) -> None:
        """更新参数面板中的「输出数据」区；不影响参数控件。"""

        self._last_result = result
        self._refresh_output_section()

    def current_params(self) -> dict[str, Any]:
        """返回当前参数（合法时为规范化结果，非法时为尽力收集的原始值）。"""
        if self._spec is None:
            return {}
        raw = self._collect()
        try:
            return normalize_params(self._spec, raw)
        except ParamValidationError:
            return raw

    def set_pick_result(self, key: str, value: Any) -> None:
        """预览拾取完成后回填参数并触发防抖预览。"""
        row = self._rows.get(key)
        if row is None:
            return
        row.setter(value)
        self._on_editor_changed()

    def is_valid(self) -> bool:
        return self._valid

    def field_widget(self, key: str) -> QWidget | None:
        row = self._rows.get(key)
        return row.field if row else None

    def set_param_value(self, key: str, value: Any) -> None:
        row = self._rows.get(key)
        if row is None:
            return
        row.setter(value)
        self._on_editor_changed()

    def param_row_visible(self, key: str) -> bool:
        row = self._rows.get(key)
        return bool(row and not row.row.isHidden())

    def baseline_params(self) -> dict[str, Any]:
        return dict(self._baseline)

    def node_id(self) -> str:
        return self._node_id

    # ------------------------------------------------------------------ 编辑器构建
    def _clear_editors(self) -> None:
        self._output_group = None
        self._output_layout = None
        self._empty_params_label = None
        while self.content_layout.count():
            item = self.content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.setParent(None)
                widget.deleteLater()
        self._rows.clear()
        self._groups.clear()
        self._pick_values.clear()

    def _build_editors(self) -> None:
        assert self._spec is not None
        self._clear_editors()
        ordered_groups: dict[str, list[ParamSpec]] = {}
        for param in self._spec.params:
            ordered_groups.setdefault(param.group_name, []).append(param)
        if not ordered_groups:
            empty = QLabel("该算子没有可配置参数。", self.content)
            empty.setObjectName("hintLabel")
            self._empty_params_label = empty
            self.content_layout.addWidget(empty)
            self._build_output_section()
            self.content_layout.addStretch(1)
            return
        for group_name, params in ordered_groups.items():
            group = QGroupBox(group_name, self.content)
            group_layout = QVBoxLayout(group)
            group_layout.setContentsMargins(10, 14, 10, 8)
            group_layout.setSpacing(4)
            keys: list[str] = []
            for param in params:
                row = self._build_row(param, group)
                group_layout.addWidget(row.row)
                self._rows[param.key] = row
                keys.append(param.key)
            self.content_layout.addWidget(group)
            self._groups.append((group, keys))
        self._build_output_section()
        self.content_layout.addStretch(1)

    # ------------------------------------------------------------------ 输出数据
    def _build_output_section(self) -> None:
        group = QGroupBox("输出数据", self.content)
        group.setVisible(False)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(10, 14, 10, 8)
        layout.setSpacing(6)
        self._output_group = group
        self._output_layout = layout
        self.content_layout.addWidget(group)
        self._refresh_output_section()

    def _refresh_output_section(self) -> None:
        layout = self._output_layout
        group = self._output_group
        if layout is None or group is None:
            return
        _clear_layout(layout)
        entries: list[tuple[str, Any]] = []
        if self._last_result is not None:
            outputs = getattr(self._last_result, "outputs", {}) or {}
            labels = getattr(self._last_result, "port_labels", {}) or {}
            entries = [
                (labels.get(key, key), value)
                for key, value in outputs.items()
                if data_type_of(value) is not DataType.IMAGE
            ]
        if not entries:
            group.setVisible(False)
            if self._empty_params_label is not None:
                self._empty_params_label.setVisible(True)
            return
        group.setVisible(True)
        if self._empty_params_label is not None:
            self._empty_params_label.setVisible(False)
        for label, value in entries:
            self._build_output_value(layout, label, value)

    def _build_output_value(self, layout: QVBoxLayout, label: str, value: Any) -> None:
        group = self._output_group
        if group is None:
            return
        data_type = data_type_of(value)
        type_label = data_type.label if data_type is not None else "未知数据"
        title = QLabel(f"{label} · {type_label}", group)
        title_font = QFont(title.font())
        title_font.setBold(True)
        title.setFont(title_font)
        title.setWordWrap(True)
        layout.addWidget(title)

        if isinstance(value, NumberValue):
            number_text = format_number(value.value)
            if value.unit:
                number_text += f" {value.unit}"
            number = QLabel(number_text, group)
            number_font = QFont(number.font())
            number_font.setPointSizeF(13.0)
            number_font.setBold(True)
            number.setFont(number_font)
            number.setWordWrap(True)
            layout.addWidget(number)

        for line in self._output_detail_lines(value):
            detail = QLabel(line, group)
            detail.setObjectName("hintLabel")
            detail.setWordWrap(True)
            layout.addWidget(detail)
        layout.addSpacing(6)

    @staticmethod
    def _output_detail_lines(value: Any) -> list[str]:
        lines = [summarize_value(value)]
        image_size = getattr(value, "image_size", None)
        if image_size:
            lines.append(f"画布：{image_size[0]} × {image_size[1]}")
        if isinstance(value, EdgesData):
            segments = value.segments
            lines.append(f"折线：{len(segments)} 条")
            if segments:
                lines.append(f"首条形状：{segments[0].shape}")
                lines.append(f"前几个点：{segments[0].reshape(-1, 2)[:4].tolist()}")
        elif isinstance(value, ContoursData):
            lines.append(f"轮廓：{len(value.contours)} 个")
            if value.contours:
                lines.append(f"首个形状：{value.contours[0].shape}")
                lines.append(f"前几个点：{value.contours[0].reshape(-1, 2)[:4].tolist()}")
            if value.hierarchy is not None:
                lines.append(f"层级数组：{value.hierarchy.shape}")
        elif isinstance(value, ComponentsData):
            lines.append(f"标签图：{value.labels.shape} · {value.labels.dtype}")
            lines.append(f"连通域：{value.count} 个 · 总面积 {value.total_area} 像素")
            if value.areas:
                lines.append(f"面积前 5：{list(value.areas[:5])}")
            if value.centroids:
                lines.append(f"质心前 3：{list(value.centroids[:3])}")
        return lines

    def _build_row(self, param: ParamSpec, parent: QWidget) -> _ParamRow:
        row_widget = QWidget(parent)
        # 中间一栏在默认 1:2:4 布局下约 420px，但用户可能把分隔条拖窄；横向
        # “标签 + 控件”会挤压数字滑杆 / 拾取按钮，改为标签在上、控件占满整行。
        layout = QVBoxLayout(row_widget)
        layout.setContentsMargins(0, 2, 0, 2)
        layout.setSpacing(3)
        label = QLabel(param.label, row_widget)
        label.setWordWrap(True)
        label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        if param.tooltip:
            label.setToolTip(param.tooltip)
        field, getter, setter = self._create_field(param, row_widget)
        field.setToolTip(param.tooltip or param.label)
        layout.addWidget(label, 0)
        layout.addWidget(field, 1)
        return _ParamRow(param, row_widget, label, field, getter, setter)

    def _create_field(
        self, param: ParamSpec, parent: QWidget
    ) -> tuple[QWidget, Callable[[], Any], Callable[[Any], None]]:
        if param.type == "int":
            return self._create_int(param, parent)
        if param.type == "float":
            return self._create_float(param, parent)
        if param.type == "bool":
            return self._create_bool(param, parent)
        if param.type == "enum":
            return self._create_enum(param, parent)
        if param.type == "color":
            return self._create_color(param, parent)
        if param.type == "path":
            return self._create_path(param, parent)
        if param.type == "string":
            return self._create_string(param, parent)
        if param.type in _INTERACTIVE_TYPES:
            return self._create_interactive(param, parent)
        line = QLineEdit(parent)
        line.setReadOnly(True)
        line.setText(str(param.default_value()))
        return line, line.text, line.setText

    # ------------------------------------------------------------------ int
    def _create_int(
        self, param: ParamSpec, parent: QWidget
    ) -> tuple[QWidget, Callable[[], Any], Callable[[Any], None]]:
        minimum = int(param.min) if param.min is not None else -999999
        maximum = int(param.max) if param.max is not None else 999999
        if maximum < minimum:
            maximum = minimum
        step = max(1, int(round(param.ui_step())))
        if param.odd and step % 2 == 0:
            step = 2
        if param.odd and step == 1:
            step = 2

        def normalize_odd(value: int) -> int:
            if not param.odd or value % 2 == 1:
                return value
            candidates = [value + 1, value - 1, value + 3, value - 3]
            for candidate in candidates:
                if minimum <= candidate <= maximum and candidate % 2 == 1:
                    return candidate
            return value

        spin = QSpinBox(parent)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        if param.unit:
            spin.setSuffix(f" {param.unit}")
        spin.setValue(normalize_odd(int(param.default_value())))

        wide = (
            param.min is not None
            and param.max is not None
            and (int(param.max) - int(param.min)) >= 20
        )
        container = QWidget(parent)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        if wide:
            slider = QSlider(Qt.Orientation.Horizontal, container)
            slider.setRange(minimum, maximum)
            slider.setSingleStep(step)
            slider.setValue(spin.value())
            slider.setMinimumWidth(80)
            layout.addWidget(slider, 1)
        else:
            slider = None
        layout.addWidget(spin, 1 if wide else 1)

        def apply_normalized(value: int) -> None:
            target = normalize_odd(int(value))
            if target != spin.value():
                spin.blockSignals(True)
                spin.setValue(target)
                spin.blockSignals(False)
            if slider is not None and slider.value() != target:
                slider.blockSignals(True)
                slider.setValue(target)
                slider.blockSignals(False)

        def on_spin_changed(value: int) -> None:
            if param.odd and value % 2 == 0:
                apply_normalized(value)
                value = spin.value()
            if slider is not None and slider.value() != value:
                slider.blockSignals(True)
                slider.setValue(value)
                slider.blockSignals(False)
            self._on_editor_changed()

        spin.valueChanged.connect(on_spin_changed)
        if slider is not None:
            slider.valueChanged.connect(lambda value: spin.setValue(value))

        def setter(value: Any) -> None:
            try:
                number = int(value)
            except (TypeError, ValueError):
                number = int(param.default_value())
            number = max(minimum, min(maximum, number))
            spin.blockSignals(True)
            spin.setValue(normalize_odd(number))
            spin.blockSignals(False)
            if slider is not None:
                slider.blockSignals(True)
                slider.setValue(spin.value())
                slider.blockSignals(False)

        return container, spin.value, setter

    # ------------------------------------------------------------------ float
    def _create_float(
        self, param: ParamSpec, parent: QWidget
    ) -> tuple[QWidget, Callable[[], Any], Callable[[Any], None]]:
        minimum = float(param.min) if param.min is not None else -1e6
        maximum = float(param.max) if param.max is not None else 1e6
        if maximum < minimum:
            maximum = minimum
        step = float(param.ui_step())
        if step <= 0:
            step = 0.1
        if step >= 1:
            decimals = 2
        elif step >= 0.1:
            decimals = 2
        elif step >= 0.01:
            decimals = 3
        else:
            decimals = 4
        spin = QDoubleSpinBox(parent)
        spin.setDecimals(decimals)
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        if param.unit:
            spin.setSuffix(f" {param.unit}")
        try:
            initial = float(param.default_value())
        except (TypeError, ValueError):
            initial = 0.0
        spin.setValue(max(minimum, min(maximum, initial)))
        spin.valueChanged.connect(lambda _value: self._on_editor_changed())

        def setter(value: Any) -> None:
            try:
                number = float(value)
            except (TypeError, ValueError):
                number = float(param.default_value())
            spin.blockSignals(True)
            spin.setValue(max(minimum, min(maximum, number)))
            spin.blockSignals(False)

        return spin, spin.value, setter

    # ------------------------------------------------------------------ bool
    def _create_bool(
        self, param: ParamSpec, parent: QWidget
    ) -> tuple[QWidget, Callable[[], Any], Callable[[Any], None]]:
        check = QCheckBox("", parent)
        check.setChecked(bool(param.default_value()))
        check.toggled.connect(lambda _checked: self._on_editor_changed())

        def setter(value: Any) -> None:
            try:
                checked = bool(coerce_param_value(param, value))
            except ParamValidationError:
                checked = bool(param.default_value())
            check.blockSignals(True)
            check.setChecked(checked)
            check.blockSignals(False)

        return check, check.isChecked, setter

    # ------------------------------------------------------------------ enum
    def _create_enum(
        self, param: ParamSpec, parent: QWidget
    ) -> tuple[QWidget, Callable[[], Any], Callable[[Any], None]]:
        combo = QComboBox(parent)
        for value, label in param.choices:
            combo.addItem(label, value)
        initial = param.default_value()
        index = combo.findData(initial)
        if index >= 0:
            combo.setCurrentIndex(index)
        combo.currentIndexChanged.connect(lambda _index: self._on_editor_changed())

        def setter(value: Any) -> None:
            index = combo.findData(value)
            if index < 0:
                for item_index in range(combo.count()):
                    if combo.itemData(item_index) == value:
                        index = item_index
                        break
            combo.blockSignals(True)
            combo.setCurrentIndex(index if index >= 0 else 0)
            combo.blockSignals(False)

        return combo, combo.currentData, setter

    # ------------------------------------------------------------------ color
    def _create_color(
        self, param: ParamSpec, parent: QWidget
    ) -> tuple[QWidget, Callable[[], Any], Callable[[Any], None]]:
        button = QPushButton(parent)
        button.setObjectName("colorSwatch")
        button.setIconSize(QSize(18, 18))
        button.setMinimumWidth(110)
        value = str(param.default_value() or "#ffffff").lower()
        button.setProperty("colorValue", value)
        button.setIcon(_color_icon(value))
        button.setText(value.upper())

        def apply(value_text: str) -> None:
            button.setProperty("colorValue", value_text.lower())
            button.setIcon(_color_icon(value_text))
            button.setText(value_text.upper())
            button.setToolTip(f"{param.label}：{value_text.upper()}")

        apply(value)

        def choose() -> None:
            current = QColor(str(button.property("colorValue") or "#ffffff"))
            color = QColorDialog.getColor(current, self, f"选择{param.label}")
            if color.isValid():
                apply(color.name().lower())
                self._on_editor_changed()

        button.clicked.connect(choose)

        def setter(new_value: Any) -> None:
            text = str(new_value or "#ffffff").lower()
            if not text.startswith("#"):
                text = "#" + text
            apply(text)

        return button, lambda: str(button.property("colorValue") or "#ffffff"), setter

    # ------------------------------------------------------------------ path
    def _create_path(
        self, param: ParamSpec, parent: QWidget
    ) -> tuple[QWidget, Callable[[], Any], Callable[[Any], None]]:
        container = QWidget(parent)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        line = QLineEdit(container)
        line.setPlaceholderText(param.placeholder or "请选择文件路径")
        button = QToolButton(container)
        button.setText("浏览…")
        button.setToolTip("浏览文件")
        layout.addWidget(line, 1)
        layout.addWidget(button, 0)
        line.textChanged.connect(lambda _text: self._on_editor_changed())

        def browse() -> None:
            start = line.text().strip()
            path, _selected = QFileDialog.getOpenFileName(self, "选择文件", start, IMAGE_FILTER)
            if path:
                line.setText(path)
                line.setCursorPosition(0)

        button.clicked.connect(browse)

        def setter(value: Any) -> None:
            line.blockSignals(True)
            line.setText("" if value is None else str(value))
            line.blockSignals(False)

        return container, line.text, setter

    # ------------------------------------------------------------------ string
    def _create_string(
        self, param: ParamSpec, parent: QWidget
    ) -> tuple[QWidget, Callable[[], Any], Callable[[Any], None]]:
        if param.multiline:
            edit = QPlainTextEdit(parent)
            edit.setPlaceholderText(param.placeholder or "")
            edit.setFixedHeight(76)
            edit.textChanged.connect(self._on_editor_changed)

            def getter() -> str:
                return edit.toPlainText()

            def setter(value: Any) -> None:
                edit.blockSignals(True)
                edit.setPlainText("" if value is None else str(value))
                edit.blockSignals(False)

            return edit, getter, setter
        line = QLineEdit(parent)
        line.setPlaceholderText(param.placeholder or "")
        line.textChanged.connect(lambda _text: self._on_editor_changed())
        return (
            line,
            line.text,
            lambda value: line.setText("" if value is None else str(value)),
        )

    # ------------------------------------------------------------------ 交互参数
    def _create_interactive(
        self, param: ParamSpec, parent: QWidget
    ) -> tuple[QWidget, Callable[[], Any], Callable[[Any], None]]:
        container = QWidget(parent)
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        display = QLineEdit(container)
        display.setReadOnly(True)
        display.setObjectName("valueLabel")
        display.setMinimumWidth(80)
        tool = param.tool_name()
        pick_button = QPushButton("拾取", container)
        pick_button.setToolTip("在右侧预览中拾取")
        clear_button = QPushButton("清除", container)
        clear_button.setObjectName("subtleButton")
        layout.addWidget(display, 1)
        if tool in _PICK_TOOLS:
            layout.addWidget(pick_button, 0)
        layout.addWidget(clear_button, 0)

        default = param.default_value()
        if param.type == "points":
            value: Any = list(default) if isinstance(default, (list, tuple)) else []
        else:
            value = default
        self._pick_values[param.key] = value

        def render(new_value: Any) -> None:
            self._pick_values[param.key] = new_value
            if param.type == "points":
                text = _points_text(new_value)
                display.setToolTip(_points_full_text(new_value))
            elif param.type == "roi":
                text = _rect_text(new_value)
                display.setToolTip(text)
            else:
                text = _line_text(new_value)
                display.setToolTip(text)
            display.blockSignals(True)
            display.setText(text)
            display.setCursorPosition(0)
            display.blockSignals(False)

        render(value)

        def setter(new_value: Any) -> None:
            render(new_value)

        def request_pick() -> None:
            if tool in _PICK_TOOLS and self._node_id:
                self.status_message.emit("请在右侧预览中完成拾取，按 Esc 取消。")
                self.pick_requested.emit(self._node_id, param.key, tool)

        def clear() -> None:
            empty: Any = [] if param.type == "points" else None
            render(empty)
            self._on_editor_changed()

        pick_button.clicked.connect(request_pick)
        clear_button.clicked.connect(clear)
        return container, lambda: self._pick_values.get(param.key), setter

    # ------------------------------------------------------------------ 状态
    def _show_placeholder(self) -> None:
        self._stack.setCurrentWidget(self.placeholder)
        self.footer.setVisible(False)
        self.title_label.setText("未选择节点")
        self.meta_label.setText("双击画布中的节点可编辑参数")
        self.live_badge.setVisible(False)
        self.error_label.setVisible(False)

    def _show_editor(self) -> None:
        self._stack.setCurrentWidget(self.editor_page)
        self.footer.setVisible(True)
        self.scroll.verticalScrollBar().setValue(0)

    def _apply_values(self, values: dict[str, Any]) -> None:
        self._loading = True
        try:
            for key, row in self._rows.items():
                value = values.get(key, row.spec.default_value())
                try:
                    coerced = coerce_param_value(row.spec, value)
                except Exception:  # noqa: BLE001 - 非法旧值回退默认值
                    coerced = row.spec.default_value()
                row.setter(coerced)
        finally:
            self._loading = False
        self._refresh_conditions()

    def _collect(self) -> dict[str, Any]:
        values: dict[str, Any] = {}
        for key, row in self._rows.items():
            try:
                values[key] = row.getter()
            except Exception:  # noqa: BLE001 - 单个控件异常不阻塞其它参数
                values[key] = row.spec.default_value()
        return values

    def _refresh_conditions(self) -> None:
        if self._spec is None:
            return
        values = self._collect()
        for group, keys in self._groups:
            any_visible = False
            for key in keys:
                row = self._rows.get(key)
                if row is None:
                    continue
                visible = is_param_visible(row.spec, values)
                row.row.setVisible(visible)
                if visible:
                    any_visible = True
                    enabled = is_param_enabled(row.spec, values)
                    row.field.setEnabled(enabled)
                    row.label.setEnabled(enabled)
            group.setVisible(any_visible)

    def _validate(self) -> bool:
        if self._spec is None:
            self._valid = False
            self.save_button.setEnabled(False)
            return False
        raw = self._collect()
        try:
            normalized = normalize_params(self._spec, raw)
        except ParamValidationError as exc:
            self._valid = False
            self._normalized = raw
            self._set_error(f"⚠ {exc.message}", exc.key)
        except Exception as exc:  # noqa: BLE001 - 兜底，避免界面崩溃
            self._valid = False
            self._normalized = raw
            self._set_error(f"⚠ 参数校验失败：{exc}", None)
        else:
            self._valid = True
            self._normalized = normalized
            self._set_error("", None)
        self.save_button.setEnabled(self._valid)
        self._update_badge()
        return self._valid

    def _set_error(self, message: str, key: str | None) -> None:
        self.error_label.setText(message)
        self.error_label.setVisible(bool(message))
        if message:
            if message != self._last_status_message:
                self._last_status_message = message
                self.status_message.emit(message)
        elif self._last_status_message:
            self._last_status_message = ""
        for row_key, row in self._rows.items():
            has_error = bool(key and row_key == key)
            row.label.setProperty("hasError", has_error)
            row.label.style().unpolish(row.label)
            row.label.style().polish(row.label)
            if has_error and message:
                row.row.setToolTip(message)
            elif row.row.toolTip() == message:
                row.row.setToolTip("")

    def _update_badge(self) -> None:
        if self._spec is None:
            self.live_badge.setVisible(False)
            return
        self.live_badge.setVisible(True)
        if self._valid:
            self.live_badge.setText("● 实时预览")
            self.live_badge.setProperty("state", "ok")
        else:
            self.live_badge.setText("● 参数有误")
            self.live_badge.setProperty("state", "error")
        self.live_badge.style().unpolish(self.live_badge)
        self.live_badge.style().polish(self.live_badge)

    # ------------------------------------------------------------------ 事件
    def _on_editor_changed(self) -> None:
        if self._loading or self._spec is None:
            return
        self._refresh_conditions()
        if self._validate():
            self._preview_timer.start()
        else:
            self._preview_timer.stop()

    def _emit_preview(self) -> None:
        if self._spec is None or not self._node_id:
            return
        if not self._validate():
            return
        self.preview_requested.emit(self._node_id, dict(self._normalized))

    def _on_reset_clicked(self) -> None:
        if self._spec is None:
            return
        self._apply_values(param_defaults(self._spec))
        self._validate()
        if self._valid:
            self._preview_timer.start()

    def _on_undo_clicked(self) -> None:
        if self._spec is None:
            return
        self._apply_values(self._baseline)
        self._validate()
        if self._valid:
            self._preview_timer.start()

    def _on_save_clicked(self) -> None:
        if self._spec is None or not self._node_id:
            return
        if not self._validate():
            return
        self._preview_timer.stop()
        self._baseline = dict(self._normalized)
        self.save_requested.emit(self._node_id, dict(self._normalized))

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        if event.key() == Qt.Key.Key_Escape:
            self.back_requested.emit()
            event.accept()
            return
        super().keyPressEvent(event)
