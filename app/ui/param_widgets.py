"""参数面板的自定义控件 / 行布局注册表。

算子作者可以按需替换某个参数在参数面板中的编辑器（``ParamSpec.widget``）或
参数行的排布方式（``ParamSpec.layout``），无需修改 :mod:`app.ui.params`。
本模块刻意不导入 :mod:`app.ui.params`，避免二者循环导入；注册表在界面创建
参数行时再被查询。

界面扩展只改变“怎么编辑”，不改变参数值与校验：值仍然按 ``ParamSpec.type``
由 :func:`app.operators.spec.coerce_param_value` 规范化。

最小示例::

    from PySide6.QtWidgets import QHBoxLayout, QLineEdit, QWidget

    from app.ui.param_widgets import (
        ParamField,
        ParamLayoutContext,
        ParamWidgetContext,
        register_param_layout,
        register_param_widget,
    )

    def line_widget(context: ParamWidgetContext) -> ParamField:
        line = QLineEdit(context.parent)
        line.setPlaceholderText(str(context.options.get("placeholder", "")))
        line.textChanged.connect(lambda _text: context.panel.notify_edited())
        return ParamField(
            widget=line,
            getter=line.text,
            setter=lambda value: line.setText("" if value is None else str(value)),
        )

    def inline_row(context: ParamLayoutContext) -> QWidget:
        row = QWidget(context.parent)
        row.setObjectName("myInlineRow")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        layout.addWidget(context.label, 0)
        layout.addWidget(context.field, 1)
        return row

    register_param_widget("my.line", line_widget)
    register_param_layout("my.inline", inline_row)

然后在算子声明中引用::

    ParamSpec(
        "endpoint",
        "服务地址",
        "string",
        default="",
        widget="my.line",
        layout="my.inline",
        widget_options={"placeholder": "http://127.0.0.1:8000"},
    )

约定：

* 控件工厂接收 :class:`ParamWidgetContext`，返回 :class:`ParamField`；
  ``getter()`` 返回当前值，``setter(value)`` 接收新值（默认值回填、重置、
  撤销、交互拾取回填都会调用它，setter 内应屏蔽控件自身信号，避免重复预览）。
* 布局工厂接收 :class:`ParamLayoutContext`，返回一个行容器 ``QWidget``；它应把
  ``context.label`` 与 ``context.field`` 放进自己的布局（父子关系会自动转移）。
* 内置控件 ``model_file`` 用于选择 / 下载 OpenCV ONNX 模型，配置项为
  ``{"model": "<MODELS 中的 key>"}``；未知 key 时只禁用下载，不抛异常。
* 内置布局 ``stacked``（标签在上，默认）与 ``inline``（标签在左，固定约 120px）。
* 注册名只允许字母、数字、下划线、点、连字符，且以字母 / 下划线开头；重复注册
  默认报错，可用 ``override=True`` 覆盖（测试或替换内置实现时有用）。
* 下载通过 :func:`app.ui.model_download.start_model_download` 在后台线程执行，
  进度 / 结果以 Qt 信号回到界面线程，不阻塞参数面板。
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from PySide6.QtCore import Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app.core import models
from app.operators.spec import PARAM_UI_NAME_PATTERN, ParamSpec
from app.ui.model_download import DownloadSignals, start_model_download

#: 模型文件选择框的过滤器（OpenCV DNN 常用模型格式，ONNX 优先）。
MODEL_FILE_FILTER = (
    "ONNX 模型 (*.onnx);;"
    "所有模型文件 (*.onnx *.caffemodel *.pb *.tflite *.bin);;"
    "所有文件 (*)"
)

#: inline 布局中标签的固定宽度（像素）。
INLINE_LABEL_WIDTH = 120


@dataclass
class ParamWidgetContext:
    """创建自定义参数控件时传给工厂的上下文。

    * ``param``：对应的 :class:`ParamSpec`；
    * ``panel``：参数面板（:class:`app.ui.params.ParamPanel`），可用
      ``panel.notify_edited()`` 触发联动与防抖预览；
    * ``parent``：建议的父控件；
    * ``options``：``ParamSpec.widget_options`` 的字典副本。
    """

    param: ParamSpec
    panel: Any
    parent: QWidget
    options: Mapping[str, Any]


@dataclass
class ParamField:
    """自定义控件工厂的返回值：控件 + 读写当前值的两个回调。"""

    widget: QWidget
    getter: Callable[[], Any]
    setter: Callable[[Any], None]


@dataclass
class ParamLayoutContext:
    """创建自定义参数行布局时传给工厂的上下文。

    工厂应把 ``label`` 与 ``field`` 放进返回的行容器中；``options`` 是
    ``ParamSpec.widget_options`` 的字典副本。
    """

    param: ParamSpec
    panel: Any
    parent: QWidget
    label: QWidget
    field: QWidget
    options: Mapping[str, Any]


ParamWidgetFactory = Callable[[ParamWidgetContext], ParamField]
ParamLayoutFactory = Callable[[ParamLayoutContext], QWidget]

_PARAM_WIDGETS: dict[str, ParamWidgetFactory] = {}
_PARAM_LAYOUTS: dict[str, ParamLayoutFactory] = {}


# ---------------------------------------------------------------------------
# 注册表
# ---------------------------------------------------------------------------
def _check_registration(
    name: str,
    factory: Any,
    registry: Mapping[str, Any],
    override: bool,
    kind: str,
) -> None:
    if not isinstance(name, str) or not re.match(PARAM_UI_NAME_PATTERN, name):
        raise ValueError(
            f"非法{kind}名：{name!r}（需匹配 {PARAM_UI_NAME_PATTERN}）"
        )
    if not callable(factory):
        raise TypeError(f"{kind} {name!r} 的工厂必须可调用")
    if name in registry and not override:
        raise ValueError(
            f"{kind} {name!r} 已注册；如需替换请传 override=True"
        )


def register_param_widget(
    name: str,
    factory: ParamWidgetFactory,
    *,
    override: bool = False,
) -> ParamWidgetFactory:
    """注册一个自定义参数控件工厂（``ParamSpec.widget``）。

    返回 ``factory``，便于直接引用。同名重复注册默认抛 :class:`ValueError`。
    """

    _check_registration(name, factory, _PARAM_WIDGETS, override, "参数控件")
    _PARAM_WIDGETS[name] = factory
    return factory


def register_param_layout(
    name: str,
    factory: ParamLayoutFactory,
    *,
    override: bool = False,
) -> ParamLayoutFactory:
    """注册一个自定义参数行布局工厂（``ParamSpec.layout``）。"""

    _check_registration(name, factory, _PARAM_LAYOUTS, override, "参数布局")
    _PARAM_LAYOUTS[name] = factory
    return factory


def get_param_widget(name: str) -> ParamWidgetFactory | None:
    """按注册名取控件工厂；未注册返回 ``None``。"""

    if not isinstance(name, str):
        return None
    return _PARAM_WIDGETS.get(name)


def get_param_layout(name: str) -> ParamLayoutFactory | None:
    """按注册名取布局工厂；未注册返回 ``None``。"""

    if not isinstance(name, str):
        return None
    return _PARAM_LAYOUTS.get(name)


def registered_param_widgets() -> tuple[str, ...]:
    """已注册的控件名（排序后的元组，便于显示与测试）。"""

    return tuple(sorted(_PARAM_WIDGETS))


def registered_param_layouts() -> tuple[str, ...]:
    """已注册的布局名（排序后的元组）。"""

    return tuple(sorted(_PARAM_LAYOUTS))


# ---------------------------------------------------------------------------
# 内置布局
# ---------------------------------------------------------------------------
def _stacked_layout(context: ParamLayoutContext) -> QWidget:
    """默认布局：标签在上、控件在下，占满整行。"""

    row = QWidget(context.parent)
    row.setObjectName("paramRow")
    row.setProperty("paramLayout", "stacked")
    layout = QVBoxLayout(row)
    layout.setContentsMargins(0, 2, 0, 2)
    layout.setSpacing(3)
    context.label.setWordWrap(True)
    context.label.setAlignment(
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    )
    layout.addWidget(context.label, 0)
    layout.addWidget(context.field, 1)
    return row


def _inline_layout(context: ParamLayoutContext) -> QWidget:
    """行内布局：标签在左（固定宽度），控件在右。"""

    row = QWidget(context.parent)
    row.setObjectName("paramRowInline")
    row.setProperty("paramLayout", "inline")
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 2, 0, 2)
    layout.setSpacing(8)
    context.label.setWordWrap(True)
    context.label.setFixedWidth(INLINE_LABEL_WIDTH)
    context.label.setAlignment(
        Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter
    )
    layout.addWidget(context.label, 0)
    layout.addWidget(context.field, 1)
    return row


# ---------------------------------------------------------------------------
# 内置控件：model_file
# ---------------------------------------------------------------------------
def _resolve_model(key: str) -> models.ModelSpec | None:
    """按 key 取模型定义；空 / 未知 key 返回 ``None``（不抛异常）。"""

    if not key:
        return None
    try:
        return models.resolve_model(key)
    except (models.ModelError, KeyError):
        return None


def _model_download_url(spec: models.ModelSpec) -> str:
    """模型官方下载地址：优先使用 models 模块的辅助函数，兼容 spec.url。"""

    resolver = getattr(models, "model_url", None)
    if callable(resolver):
        try:
            value = resolver(spec.key)
        except Exception:  # noqa: BLE001 - 辅助函数未来可能改签名
            value = ""
        if value:
            return str(value)
    return str(spec.url or spec.homepage or "")


def _emit_panel_status(panel: Any, message: str) -> None:
    """把提示信息发给参数面板状态栏；控件已销毁时静默忽略。"""

    emit = getattr(getattr(panel, "status_message", None), "emit", None)
    if callable(emit):
        try:
            emit(message)
        except RuntimeError:
            pass


def _notify_panel_edited(panel: Any) -> None:
    notify = getattr(panel, "notify_edited", None)
    if callable(notify):
        try:
            notify()
        except RuntimeError:
            pass


class _ModelFileField(QWidget):
    """模型文件编辑器：路径框 + 浏览 + 后台下载 + 状态 / 官方地址。

    设计要点：

    * 下载信号对象以本控件为 parent，任务引用保存在 ``_download_task``，
      避免线程池任务被回收；
    * 所有槽都是本控件的绑定方法，控件销毁后 Qt 自动断开，迟到信号由
      :class:`app.ui.model_download.ModelDownloadTask` 的安全 emit 兜底；
    * 未知模型 key 只显示「未指定模型」并禁用下载，绝不抛异常。
    """

    def __init__(self, context: ParamWidgetContext) -> None:
        super().__init__(context.parent)
        self.setObjectName("modelFileField")
        self._param = context.param
        self._panel = context.panel
        self._options = dict(context.options)
        self._model_key = str(self._options.get("model") or "")
        self._model_spec = _resolve_model(self._model_key)
        self._download_task: Any = None
        self._download_signals: DownloadSignals | None = None
        self._downloading = False

        self._build_ui()
        self.set_value(self._param.default_value())

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(4)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(6)
        self._line = QLineEdit(self)
        self._line.setObjectName("modelFilePath")
        placeholder = self._param.placeholder or "请选择模型文件（.onnx 等）"
        self._line.setPlaceholderText(placeholder)
        self._line.setClearButtonEnabled(True)
        self._browse_button = QToolButton(self)
        self._browse_button.setObjectName("modelBrowseButton")
        self._browse_button.setText("浏览…")
        self._browse_button.setToolTip("选择本机已下载的模型文件")
        self._browse_button.clicked.connect(self._on_browse_clicked)
        top.addWidget(self._line, 1)
        top.addWidget(self._browse_button, 0)
        root.addLayout(top)

        actions = QHBoxLayout()
        actions.setContentsMargins(0, 0, 0, 0)
        actions.setSpacing(6)
        self._status_label = QLabel(self)
        self._status_label.setObjectName("modelStatusLabel")
        self._status_label.setWordWrap(True)
        self._download_button = QPushButton("下载模型", self)
        self._download_button.setObjectName("modelDownloadButton")
        self._download_button.clicked.connect(self._on_download_clicked)
        self._open_dir_button = QPushButton("打开目录", self)
        self._open_dir_button.setObjectName("subtleButton")
        self._open_dir_button.setToolTip("打开本地模型缓存目录")
        self._open_dir_button.clicked.connect(self._on_open_dir_clicked)
        actions.addWidget(self._status_label, 1)
        actions.addWidget(self._download_button, 0)
        actions.addWidget(self._open_dir_button, 0)
        root.addLayout(actions)

        self._url_label = QLabel(self)
        self._url_label.setObjectName("modelUrlLabel")
        self._url_label.setWordWrap(True)
        self._url_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        self._url_label.setOpenExternalLinks(True)
        root.addWidget(self._url_label)

        self._line.textChanged.connect(self._on_text_changed)
        self._apply_model_meta()

    def _apply_model_meta(self) -> None:
        spec = self._model_spec
        if spec is None:
            self._status_label.setText("未指定模型")
            self._status_label.setToolTip(
                "该参数未关联内置模型；可手动浏览选择模型文件。"
            )
            self._url_label.clear()
            self._url_label.setVisible(False)
            self._download_button.setEnabled(False)
            self._download_button.setToolTip("未指定模型，无法自动下载")
            return
        tooltip = spec.description or spec.name
        self._status_label.setToolTip(tooltip)
        self._download_button.setEnabled(True)
        self._download_button.setToolTip(
            f"从官方地址下载{spec.name}（约 {models.format_size(spec.size)}）"
        )
        url = _model_download_url(spec)
        if url:
            safe_url = html.escape(url, quote=True)
            self._url_label.setText(
                f'官方下载地址：<a href="{safe_url}">{html.escape(url)}</a>'
            )
            self._url_label.setToolTip("点击或选中地址，可手动下载模型文件")
            self._url_label.setVisible(True)
        else:
            self._url_label.clear()
            self._url_label.setVisible(False)
        self._refresh_status()

    def _refresh_status(self) -> None:
        if self._model_spec is None:
            self._status_label.setText("未指定模型")
            return
        try:
            status = models.model_status_text(self._model_key)
        except Exception:  # noqa: BLE001 - 状态文本失败不应影响界面
            status = "状态未知"
        self._status_label.setText(status)
        self._status_label.setToolTip(
            f"{self._model_spec.name} · {status}\n"
            f"缓存路径：{models.model_path(self._model_key)}"
        )

    # ------------------------------------------------------------------ 值
    def value(self) -> str:
        return self._line.text()

    def set_value(self, value: Any) -> None:
        text = "" if value is None else str(value)
        self._line.blockSignals(True)
        self._line.setText(text)
        self._line.blockSignals(False)

    def _on_text_changed(self, _text: str) -> None:
        _notify_panel_edited(self._panel)

    # ------------------------------------------------------------------ 按钮
    def _on_browse_clicked(self) -> None:
        start = self._line.text().strip()
        if not start:
            start = str(models.model_dir())
        path, _selected = QFileDialog.getOpenFileName(
            self, "选择模型文件", start, MODEL_FILE_FILTER
        )
        if not path:
            return
        self.set_value(path)
        _notify_panel_edited(self._panel)

    def _on_open_dir_clicked(self) -> None:
        directory = models.model_dir()
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            _emit_panel_status(self._panel, f"无法创建模型目录：{exc}")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory))):
            _emit_panel_status(self._panel, f"无法打开模型目录：{directory}")

    def _on_download_clicked(self) -> None:
        if self._model_spec is None or self._downloading:
            return
        self._downloading = True
        self._download_button.setEnabled(False)
        self._download_button.setText("下载中…")
        self._status_label.setText("正在下载模型…")
        signals = DownloadSignals(self)
        signals.progress.connect(self._on_download_progress)
        signals.succeeded.connect(self._on_download_succeeded)
        signals.failed.connect(self._on_download_failed)
        signals.cancelled.connect(self._on_download_cancelled)
        signals.finished.connect(self._on_download_finished)
        self._download_signals = signals
        try:
            self._download_task = start_model_download(self._model_key, signals)
        except Exception as exc:  # noqa: BLE001 - 启动失败也不能让面板崩溃
            self._download_task = None
            self._downloading = False
            self._download_signals = None
            self._status_label.setText(f"无法启动下载：{exc}")
            _emit_panel_status(self._panel, f"无法启动模型下载：{exc}")
            self._reset_download_button()

    # ------------------------------------------------------------------ 下载回调
    def _on_download_progress(self, downloaded: int, total: int) -> None:
        if total > 0:
            percent = max(0, min(100, int(downloaded * 100 / total)))
            text = (
                f"正在下载… {percent}%"
                f"（{models.format_size(downloaded)} / {models.format_size(total)}）"
            )
        else:
            text = f"正在下载… 已下载 {models.format_size(downloaded)}"
        self._status_label.setText(text)

    def _on_download_succeeded(self, path: str) -> None:
        self._downloading = False
        current = self._line.text().strip()
        if not current:
            self.set_value(path)
            current = path
        self._refresh_status()
        self._reset_download_button()
        # 把下载路径写回参数并触发预览；行内值可能原本就指向别的文件，保持不动。
        setter = getattr(self._panel, "set_param_value", None)
        if callable(setter):
            try:
                setter(self._param.key, current)
            except Exception:  # noqa: BLE001 - 面板已销毁时忽略
                _notify_panel_edited(self._panel)
        else:
            _notify_panel_edited(self._panel)
        _emit_panel_status(self._panel, f"模型已下载：{path}")

    def _on_download_failed(self, message: str) -> None:
        self._downloading = False
        text = " ".join(str(message).splitlines()).strip() or "下载失败。"
        self._status_label.setText(text)
        self._reset_download_button()
        _emit_panel_status(self._panel, text)

    def _on_download_cancelled(self) -> None:
        self._downloading = False
        self._status_label.setText("已取消下载。")
        self._reset_download_button()
        _emit_panel_status(self._panel, "已取消模型下载。")

    def _on_download_finished(self) -> None:
        self._downloading = False
        self._download_task = None
        self._download_signals = None
        self._reset_download_button()
        if self._status_label.text().startswith("正在下载"):
            self._refresh_status()

    def _reset_download_button(self) -> None:
        self._download_button.setText("下载模型")
        self._download_button.setEnabled(self._model_spec is not None)


def _model_file_widget(context: ParamWidgetContext) -> ParamField:
    """内置 ``model_file`` 控件：模型路径 + 浏览 + 后台下载 + 状态。"""

    field = _ModelFileField(context)
    return ParamField(widget=field, getter=field.value, setter=field.set_value)


# ---------------------------------------------------------------------------
# 内置注册
# ---------------------------------------------------------------------------
# 使用 override=True：模块被重新加载（测试 / 插件热更新）时不会因重复注册失败。
register_param_widget("model_file", _model_file_widget, override=True)
register_param_layout("stacked", _stacked_layout, override=True)
register_param_layout("inline", _inline_layout, override=True)
