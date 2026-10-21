"""参数面板自定义控件 / 布局注册表与内置模型下载控件测试。

全部在 offscreen 平台运行：

* 不访问外网：模型下载按钮只做界面断言，信号回调用模拟的
  :func:`app.ui.param_widgets.start_model_download` 触发；
* 临时目录放在工作区 `tests/.tmp/` 下（DSH 沙箱可能禁止写系统临时目录）。
"""

from __future__ import annotations

import os
import shutil
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from app.core import models
from app.core.model import NodeModel
from app.operators.spec import Operator, OperatorSpec, ParamSpec, coerce_param_value
from app.ui.param_widgets import (
    INLINE_LABEL_WIDTH,
    ParamField,
    ParamLayoutContext,
    ParamWidgetContext,
    get_param_layout,
    get_param_widget,
    register_param_layout,
    register_param_widget,
    registered_param_layouts,
    registered_param_widgets,
)
from app.ui.params import ParamPanel
from app.ui.theme import apply_theme

_APP: QApplication | None = None
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _workspace_tempdir() -> Path:
    """在工作区内创建临时目录（不使用 tempfile.mkdtemp，见 tests/test_widgets.py）。"""

    base = _REPO_ROOT / "tests" / ".tmp"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"param-widgets-{uuid.uuid4().hex[:10]}"
    path.mkdir()
    return path


def _cleanup_tempdir(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def setUpModule() -> None:
    global _APP
    _APP = QApplication.instance() or QApplication([])
    apply_theme(_APP)


def tearDownModule() -> None:
    global _APP
    _APP = None


# ---------------------------------------------------------------------------
# 测试用自定义控件 / 布局
# ---------------------------------------------------------------------------
_WIDGET_NAME = "test.custom_line"
_LAYOUT_NAME = "test.inline_layout"


def _custom_line_widget(context: ParamWidgetContext) -> ParamField:
    """一个最小的自定义控件：QLineEdit + 通知面板刷新。"""

    line = QLineEdit(context.parent)
    line.setObjectName("testCustomLine")
    line.setPlaceholderText(str(context.options.get("placeholder", "")))
    line.textChanged.connect(lambda _text: context.panel.notify_edited())

    def set_value(value: object) -> None:
        line.blockSignals(True)
        line.setText("" if value is None else str(value))
        line.blockSignals(False)

    return ParamField(widget=line, getter=line.text, setter=set_value)


def _custom_inline_layout(context: ParamLayoutContext) -> QWidget:
    """测试用行布局：标签在左、控件在右，记录 options 中的标签宽度。"""

    row = QWidget(context.parent)
    row.setObjectName("testCustomLayoutRow")
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    width = int(context.options.get("label_width", 90))
    context.label.setFixedWidth(width)
    layout.addWidget(context.label, 0)
    layout.addWidget(context.field, 1)
    return row


register_param_widget(_WIDGET_NAME, _custom_line_widget, override=True)
register_param_layout(_LAYOUT_NAME, _custom_inline_layout, override=True)


# ---------------------------------------------------------------------------
# 测试用算子
# ---------------------------------------------------------------------------
_CUSTOM_SPEC = OperatorSpec(
    id="test.param_widgets.custom",
    name="自定义控件测试算子",
    category=("测试", "参数面板"),
    description="用于测试自定义控件与布局。",
    params=(
        ParamSpec(
            "model_path",
            "模型路径",
            "string",
            default="",
            widget=_WIDGET_NAME,
            widget_options={"placeholder": "请选择模型文件"},
            group="自定义控件",
        ),
        ParamSpec(
            "threshold",
            "阈值",
            "int",
            default=7,
            min=0,
            max=20,
            layout=_LAYOUT_NAME,
            widget_options={"label_width": 76},
            group="自定义布局",
        ),
        ParamSpec("inline_value", "行内数值", "int", default=3, layout="inline", group="内置布局"),
    ),
)


class _CustomOperator(Operator):
    spec = _CUSTOM_SPEC

    def process(self, **kwargs):  # pragma: no cover - 不会被调用
        return None


_CUSTOM_OPERATOR = _CustomOperator()


_FALLBACK_SPEC = OperatorSpec(
    id="test.param_widgets.fallback",
    name="未注册控件回退测试算子",
    category=("测试", "参数面板"),
    params=(
        ParamSpec("weird", "未知控件", "int", default=5, widget="test.missing_widget"),
        ParamSpec("weird_layout", "未知布局", "int", default=6, layout="test.missing_layout"),
    ),
)


class _FallbackOperator(Operator):
    spec = _FALLBACK_SPEC

    def process(self, **kwargs):  # pragma: no cover - 不会被调用
        return None


_FALLBACK_OPERATOR = _FallbackOperator()


_MODEL_SPEC = OperatorSpec(
    id="test.param_widgets.model_file",
    name="模型下载控件测试算子",
    category=("测试", "参数面板"),
    params=(
        ParamSpec(
            "model",
            "模型文件",
            "path",
            default="",
            widget="model_file",
            widget_options={"model": "face_detection_yunet"},
        ),
    ),
)


class _ModelOperator(Operator):
    spec = _MODEL_SPEC

    def process(self, **kwargs):  # pragma: no cover - 不会被调用
        return None


_MODEL_OPERATOR = _ModelOperator()


_UNKNOWN_MODEL_SPEC = OperatorSpec(
    id="test.param_widgets.unknown_model",
    name="未知模型测试算子",
    category=("测试", "参数面板"),
    params=(
        ParamSpec(
            "model",
            "模型文件",
            "path",
            default="",
            widget="model_file",
            widget_options={"model": "not_a_real_model"},
        ),
    ),
)


class _UnknownModelOperator(Operator):
    spec = _UNKNOWN_MODEL_SPEC

    def process(self, **kwargs):  # pragma: no cover - 不会被调用
        return None


_UNKNOWN_MODEL_OPERATOR = _UnknownModelOperator()


def _make_node(spec: OperatorSpec, params: dict | None = None) -> NodeModel:
    return NodeModel(
        id="node-param-widgets",
        op_id=spec.id,
        title="参数节点",
        params=params or {},
    )


# ---------------------------------------------------------------------------
# ParamSpec 校验
# ---------------------------------------------------------------------------
class ParamSpecWidgetFieldTests(unittest.TestCase):
    def test_accepts_valid_widget_and_layout_names(self) -> None:
        param = ParamSpec(
            "model",
            "模型",
            "path",
            widget="model_file",
            layout="inline",
            widget_options={"model": "face_detection_yunet"},
        )
        self.assertEqual(param.widget, "model_file")
        self.assertEqual(param.layout, "inline")
        self.assertEqual(param.widget_options, {"model": "face_detection_yunet"})

        named = ParamSpec("value", "数值", "int", widget="a.B-c_1", layout="a.b-c_1")
        self.assertEqual(named.widget, "a.B-c_1")

    def test_rejects_invalid_widget_name(self) -> None:
        for name in ("1bad", "bad name", "bad/name", None, 123):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    ParamSpec("value", "数值", "int", widget=name)
        # 空字符串表示不启用自定义控件。
        self.assertEqual(ParamSpec("value", "数值", "int", widget="").widget, "")

    def test_rejects_invalid_layout_name(self) -> None:
        for name in ("1bad", "bad name", "bad/name", None, 123):
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    ParamSpec("value", "数值", "int", layout=name)
        self.assertEqual(ParamSpec("value", "数值", "int", layout="").layout, "")

    def test_interactive_params_cannot_use_custom_widget(self) -> None:
        for type_ in ("points", "roi", "line"):
            with self.subTest(type=type_):
                with self.assertRaises(ValueError):
                    ParamSpec("value", "交互参数", type_, default=None, widget="model_file")

    def test_widget_options_must_be_mapping(self) -> None:
        for options in ([("model", "x")], "model", 3, None):
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    ParamSpec("value", "数值", "int", widget_options=options)

    def test_widget_options_are_copied(self) -> None:
        source = {"model": "face_detection_yunet"}
        param = ParamSpec("model", "模型", "path", widget="model_file", widget_options=source)
        source["model"] = "changed"
        self.assertEqual(param.widget_options, {"model": "face_detection_yunet"})
        self.assertIsInstance(param.widget_options, dict)
        hash(param)  # widget_options 不参与 hash / 比较

    def test_runtime_value_semantics_unchanged(self) -> None:
        param = ParamSpec("value", "数值", "int", default=5, min=1, max=10, widget="model_file")
        self.assertEqual(coerce_param_value(param, "7"), 7)
        with self.assertRaises(Exception):
            coerce_param_value(param, "99")


# ---------------------------------------------------------------------------
# 参数面板：自定义控件 / 布局
# ---------------------------------------------------------------------------
class ParamPanelCustomWidgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.panel = ParamPanel()
        self.panel.resize(380, 640)
        self.panel.show()
        self.node = _make_node(_CUSTOM_SPEC)

    def tearDown(self) -> None:
        self.panel.deleteLater()

    def test_custom_widget_is_used_and_roundtrips(self) -> None:
        self.panel.load_node(self.node, _CUSTOM_OPERATOR)
        field = self.panel.field_widget("model_path")
        self.assertIsInstance(field, QLineEdit)
        assert isinstance(field, QLineEdit)
        self.assertEqual(field.objectName(), "testCustomLine")
        self.assertEqual(field.placeholderText(), "请选择模型文件")

        path = str(_REPO_ROOT / "models" / "face_detection_yunet_2023mar.onnx")
        self.panel.set_param_value("model_path", path)
        self.assertEqual(field.text(), path)
        self.assertEqual(self.panel.current_params()["model_path"], path)

        # 自定义控件内部编辑时通过 notify_edited() 触发防抖预览。
        spy = QSignalSpy(self.panel.preview_requested)
        field.setText("typed.onnx")
        self.assertEqual(spy.count(), 0)
        QTest.qWait(200)
        self.assertEqual(spy.count(), 1)
        node_id, params = spy.at(0)
        self.assertEqual(node_id, self.node.id)
        self.assertEqual(params["model_path"], "typed.onnx")

    def test_custom_layout_is_used(self) -> None:
        self.panel.load_node(self.node, _CUSTOM_OPERATOR)
        field = self.panel.field_widget("threshold")
        self.assertIsNotNone(field)
        assert field is not None
        row = field.parentWidget()
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.objectName(), "testCustomLayoutRow")
        labels = [label for label in row.findChildren(QLabel) if label.text() == "阈值"]
        self.assertTrue(labels)
        self.assertEqual(labels[0].minimumWidth(), 76)
        self.assertEqual(self.panel.current_params()["threshold"], 7)

    def test_builtin_inline_layout(self) -> None:
        self.panel.load_node(self.node, _CUSTOM_OPERATOR)
        field = self.panel.field_widget("inline_value")
        self.assertIsNotNone(field)
        assert field is not None
        row = field.parentWidget()
        self.assertIsNotNone(row)
        assert row is not None
        self.assertEqual(row.objectName(), "paramRowInline")
        labels = [label for label in row.findChildren(QLabel) if label.text() == "行内数值"]
        self.assertTrue(labels)
        self.assertEqual(labels[0].minimumWidth(), INLINE_LABEL_WIDTH)

    def test_unregistered_widget_and_layout_fall_back(self) -> None:
        panel = ParamPanel()
        panel.resize(380, 640)
        panel.show()
        self.addCleanup(panel.deleteLater)
        status_spy = QSignalSpy(panel.status_message)
        panel.load_node(_make_node(_FALLBACK_SPEC), _FALLBACK_OPERATOR)

        self.assertGreaterEqual(status_spy.count(), 2)
        messages = " ".join(str(status_spy.at(index)[0]) for index in range(status_spy.count()))
        self.assertIn("未注册的参数控件", messages)
        self.assertIn("未注册的参数布局", messages)
        self.assertTrue(panel.is_valid())
        self.assertEqual(panel.current_params()["weird"], 5)
        self.assertEqual(panel.current_params()["weird_layout"], 6)

        field = panel.field_widget("weird")
        self.assertIsNotNone(field)
        assert field is not None
        self.assertIsNotNone(field.findChild(QSpinBox))
        layout_field = panel.field_widget("weird_layout")
        self.assertIsNotNone(layout_field)
        assert layout_field is not None
        self.assertEqual(layout_field.parentWidget().objectName(), "paramRow")

    def test_builtin_registries(self) -> None:
        self.assertIsNotNone(get_param_widget("model_file"))
        self.assertIsNotNone(get_param_layout("stacked"))
        self.assertIsNotNone(get_param_layout("inline"))
        self.assertIn("model_file", registered_param_widgets())
        self.assertIn("stacked", registered_param_layouts())
        self.assertIn("inline", registered_param_layouts())


# ---------------------------------------------------------------------------
# 内置 model_file 控件
# ---------------------------------------------------------------------------
class ModelFileWidgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = _workspace_tempdir()
        self.addCleanup(_cleanup_tempdir, self.tempdir)
        self._env = patch.dict(os.environ, {"MINEIMAGE_MODEL_DIR": str(self.tempdir)})
        self._env.start()
        self.addCleanup(self._env.stop)
        self.panel = ParamPanel()
        self.panel.resize(380, 640)
        self.panel.show()
        self.addCleanup(self.panel.deleteLater)

    @staticmethod
    def _find_label(widget: QWidget, object_name: str) -> QLabel:
        labels = widget.findChildren(QLabel, object_name)
        if not labels:
            raise AssertionError(f"未找到 QLabel#{object_name}")
        return labels[0]

    def test_shows_official_url_status_and_download_button(self) -> None:
        spec = models.MODELS["face_detection_yunet"]
        self.panel.load_node(_make_node(_MODEL_SPEC), _MODEL_OPERATOR)
        field = self.panel.field_widget("model")
        self.assertIsNotNone(field)
        assert field is not None

        self.assertEqual(models.model_dir(), self.tempdir)
        status = self._find_label(field, "modelStatusLabel")
        self.assertEqual(status.text(), models.model_status_text("face_detection_yunet"))
        self.assertEqual(status.text(), "未下载")

        url_label = self._find_label(field, "modelUrlLabel")
        self.assertIn(spec.url, url_label.text())
        self.assertFalse(url_label.isHidden())

        buttons = [button for button in field.findChildren(QPushButton) if button.text() == "下载模型"]
        self.assertTrue(buttons)
        self.assertTrue(buttons[0].isEnabled())
        open_buttons = [button for button in field.findChildren(QPushButton) if button.text() == "打开目录"]
        self.assertTrue(open_buttons)
        self.assertIsInstance(field.findChild(QLineEdit), QLineEdit)

    def test_ready_status_when_model_file_exists(self) -> None:
        target = models.model_path("face_detection_yunet")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"0" * 200_000)
        self.panel.load_node(_make_node(_MODEL_SPEC), _MODEL_OPERATOR)
        field = self.panel.field_widget("model")
        assert field is not None
        status = self._find_label(field, "modelStatusLabel")
        self.assertTrue(status.text().startswith("已就绪"))

    def test_unknown_model_disables_download(self) -> None:
        self.panel.load_node(_make_node(_UNKNOWN_MODEL_SPEC), _UNKNOWN_MODEL_OPERATOR)
        field = self.panel.field_widget("model")
        self.assertIsNotNone(field)
        assert field is not None
        status = self._find_label(field, "modelStatusLabel")
        self.assertEqual(status.text(), "未指定模型")
        self.assertTrue(self._find_label(field, "modelUrlLabel").isHidden())

        download = [button for button in field.findChildren(QPushButton) if button.text() == "下载模型"]
        self.assertTrue(download)
        self.assertFalse(download[0].isEnabled())
        self.assertTrue(self.panel.is_valid())

    def test_download_progress_and_success_write_back(self) -> None:
        """用模拟的下载任务触发信号，验证进度文本与成功后回填（不访问网络）。"""

        captured: dict[str, object] = {}

        def fake_start(key, signals, **_kwargs):  # noqa: ANN001 - 与 start_model_download 对齐
            captured["key"] = key
            captured["signals"] = signals
            return object()

        self.panel.load_node(_make_node(_MODEL_SPEC), _MODEL_OPERATOR)
        field = self.panel.field_widget("model")
        assert field is not None
        line = field.findChild(QLineEdit)
        assert line is not None
        status = self._find_label(field, "modelStatusLabel")
        download = next(
            button for button in field.findChildren(QPushButton) if button.text() == "下载模型"
        )
        status_spy = QSignalSpy(self.panel.status_message)

        with patch("app.ui.param_widgets.start_model_download", side_effect=fake_start):
            download.click()
            self.assertEqual(captured["key"], "face_detection_yunet")
            signals = captured["signals"]
            signals.progress.emit(512, 1024)
            self.assertIn("50%", status.text())
            self.assertIn("1.0 KB", status.text())

            target = self.tempdir / "face_detection_yunet_2023mar.onnx"
            target.write_bytes(b"0" * 200_000)
            signals.succeeded.emit(str(target))
            signals.finished.emit()

        self.assertEqual(line.text(), str(target))
        self.assertEqual(self.panel.current_params()["model"], str(target))
        self.assertGreaterEqual(status_spy.count(), 1)

    def test_download_failure_shows_chinese_message(self) -> None:
        captured: dict[str, object] = {}

        def fake_start(_key, signals, **_kwargs):  # noqa: ANN001
            captured["signals"] = signals
            return object()

        self.panel.load_node(_make_node(_MODEL_SPEC), _MODEL_OPERATOR)
        field = self.panel.field_widget("model")
        assert field is not None
        status = self._find_label(field, "modelStatusLabel")
        download = next(
            button for button in field.findChildren(QPushButton) if button.text() == "下载模型"
        )
        status_spy = QSignalSpy(self.panel.status_message)
        with patch("app.ui.param_widgets.start_model_download", side_effect=fake_start):
            download.click()
            signals = captured["signals"]
            signals.failed.emit("模型下载失败：网络不可用")
            signals.finished.emit()

        self.assertIn("模型下载失败", status.text())
        self.assertIn("网络不可用", status.text())
        self.assertGreaterEqual(status_spy.count(), 1)

    def test_download_cancelled_shows_chinese_message(self) -> None:
        captured: dict[str, object] = {}

        def fake_start(_key, signals, **_kwargs):  # noqa: ANN001
            captured["signals"] = signals
            return object()

        self.panel.load_node(_make_node(_MODEL_SPEC), _MODEL_OPERATOR)
        field = self.panel.field_widget("model")
        assert field is not None
        status = self._find_label(field, "modelStatusLabel")
        download = next(
            button for button in field.findChildren(QPushButton) if button.text() == "下载模型"
        )
        with patch("app.ui.param_widgets.start_model_download", side_effect=fake_start):
            download.click()
            signals = captured["signals"]
            signals.cancelled.emit()
            signals.finished.emit()

        self.assertIn("已取消", status.text())
        self.assertTrue(download.isEnabled())


if __name__ == "__main__":
    unittest.main()
