"""界面控件单元测试：预览 / 算子库 / 参数面板 / 设置 / 主题。

全部使用 offscreen 平台与内存生成的小图，不弹出任何模态对话框。
"""

from __future__ import annotations

import json
import os
import shutil
import unittest
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QEvent, QPoint, QPointF, Qt, QTimer
from PySide6.QtGui import QKeyEvent, QMouseEvent, QPalette, QWheelEvent
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QDialog,
    QLineEdit,
    QListWidgetItem,
    QMessageBox,
    QWidget,
)

from app.config import AppSettings
from app.core.constants import (
    CUSTOM_OPERATOR_CATEGORY,
    MIME_LIBRARY_MOVE,
    MIME_OPERATOR,
)
from app.core.editor import EditorInfo
from app.core.model import NodeModel
from app.core.types import ImageData
from app.operators.loader import OperatorRegistry
from app.operators.spec import Operator, OperatorSpec, ParamSpec
from app.ui.display_modes import (
    DISPLAY_MODE_SPECS,
    display_mode_icon,
    display_mode_title,
)
from app.ui.library import OperatorLibrary
from app.ui.params import ParamPanel
from app.ui.preview import PreviewWidget
from app.ui.replace_operator import ReplaceOperatorDialog
from app.ui.settings_dialog import SettingsDialog
from app.ui.theme import apply_theme

_APP: QApplication | None = None
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _workspace_tempdir() -> Path:
    """在工作区内创建临时目录（DSH 沙箱可能禁止写系统临时目录）。

    注意：不使用 ``tempfile.mkdtemp``，它在 Windows 上会把目录权限改为 0700，
    当前沙箱下会导致目录不可写；这里用普通 ``mkdir``。
    """
    base = _REPO_ROOT / ".test_tmp"
    base.mkdir(parents=True, exist_ok=True)
    path = base / f"widgets-{uuid.uuid4().hex[:10]}"
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


def _app() -> QApplication:
    assert _APP is not None
    return _APP


def _mouse_event(
    widget: QWidget,
    event_type: QEvent.Type,
    position: QPointF,
    *,
    button: Qt.MouseButton = Qt.MouseButton.LeftButton,
    buttons: Qt.MouseButton | None = None,
) -> QMouseEvent:
    if buttons is None:
        buttons = (
            Qt.MouseButton.NoButton
            if event_type == QEvent.Type.MouseButtonRelease
            else button
        )
    global_position = widget.mapToGlobal(position)
    event = QMouseEvent(
        event_type,
        position,
        global_position,
        button,
        buttons,
        Qt.KeyboardModifier.NoModifier,
    )
    QApplication.sendEvent(widget, event)
    return event


def _click_image(widget: PreviewWidget, x: float, y: float) -> None:
    view = widget.image_to_view(x, y)
    _mouse_event(widget, QEvent.Type.MouseButtonPress, view)
    _mouse_event(widget, QEvent.Type.MouseButtonRelease, view)


def _send_key(widget: QWidget, key: Qt.Key) -> None:
    QApplication.sendEvent(
        widget,
        QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.NoModifier),
    )


def _send_wheel(
    widget: QWidget,
    position: QPointF,
    *,
    delta_y: int = 120,
    modifiers: Qt.KeyboardModifier = Qt.KeyboardModifier.NoModifier,
) -> None:
    event = QWheelEvent(
        position,
        widget.mapToGlobal(position),
        QPoint(0, 0),
        QPoint(0, delta_y),
        Qt.MouseButton.NoButton,
        modifiers,
        Qt.ScrollPhase.NoScrollPhase,
        False,
    )
    QApplication.sendEvent(widget, event)


def _make_color_image(width: int = 200, height: int = 100, b: int = 10) -> ImageData:
    array = np.zeros((height, width, 3), np.uint8)
    array[:, :, 0] = b
    return ImageData(array, "test")


# ---------------------------------------------------------------------------
# 测试用算子
# ---------------------------------------------------------------------------
_PARAM_SPEC = OperatorSpec(
    id="test.params",
    name="参数测试算子",
    category=("交互处理", "测试"),
    description="用于测试参数面板的算子。",
    params=(
        ParamSpec("value", "数值", "int", default=5, min=1, max=100, group="基本"),
        ParamSpec("gain", "增益", "float", default=1.0, min=0.0, max=10.0, step=0.1, group="基本"),
        ParamSpec("enabled", "启用", "bool", default=True, group="高级"),
        ParamSpec(
            "mode",
            "模式",
            "enum",
            default="a",
            choices=("a", "b"),
            group="高级",
        ),
        ParamSpec("name", "名称", "string", default="示例", group="高级"),
        ParamSpec("extra", "附加", "int", default=3, visible_when="enabled", enabled_when="enabled", group="高级"),
        ParamSpec("color", "颜色", "color", default="#ff0000", group="外观"),
        ParamSpec("path", "路径", "path", default="", group="外观"),
        ParamSpec(
            "points",
            "点集",
            "points",
            default=[[10.0, 10.0], [20.0, 20.0]],
            count=2,
            group="交互",
        ),
        ParamSpec("roi", "ROI", "roi", default=None, group="交互"),
        ParamSpec("line", "直线", "line", default=None, interactive="line", group="交互"),
    ),
)


class _ParamTestOperator(Operator):
    spec = _PARAM_SPEC

    def process(self, **kwargs):  # pragma: no cover - 不会被调用
        return None


_PARAM_OPERATOR = _ParamTestOperator()


# ---------------------------------------------------------------------------
# PreviewWidget
# ---------------------------------------------------------------------------
class PreviewWidgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.widget = PreviewWidget()
        self.widget.resize(400, 300)
        self.image = _make_color_image()
        self.widget.set_image(self.image)

    def tearDown(self) -> None:
        self.widget.deleteLater()

    def test_fit_and_image_size(self) -> None:
        self.assertEqual(self.widget.image_size(), (200, 100))
        self.widget.fit_to_window()
        zoom = self.widget.zoom_percent()
        self.assertGreater(zoom, 100.0)
        top_left = self.widget.image_to_view(0, 0)
        bottom_right = self.widget.image_to_view(200, 100)
        self.assertGreaterEqual(top_left.x(), 0.0)
        self.assertGreaterEqual(top_left.y(), 0.0)
        self.assertLessEqual(bottom_right.x(), self.widget.width())
        self.assertLessEqual(bottom_right.y(), self.widget.height())

    def test_info_text_for_status_bar(self) -> None:
        """预览右下角信息标签的详情现在由状态栏「节点」字段展示。"""
        self.widget.set_image(self.image, "高斯模糊", "BGR · uint8")
        self.assertEqual(self.widget.info_text(), "BGR · uint8")
        self.widget.set_image(None, "高斯模糊", "正在计算…")
        self.assertEqual(self.widget.info_text(), "正在计算…")

    def test_view_to_image_roundtrip(self) -> None:
        self.widget.fit_to_window()
        view = self.widget.image_to_view(100.5, 50.5)
        mapped = self.widget.view_to_image(view)
        self.assertIsNotNone(mapped)
        assert mapped is not None
        self.assertAlmostEqual(mapped[0], 100.5, places=3)
        self.assertAlmostEqual(mapped[1], 50.5, places=3)

    def test_pixel_info_and_outside(self) -> None:
        spy = QSignalSpy(self.widget.pixel_info)
        self.widget.fit_to_window()
        view = self.widget.image_to_view(100.5, 50.5)
        _mouse_event(self.widget, QEvent.Type.MouseMove, view)
        self.assertGreaterEqual(spy.count(), 1)
        sample = spy.at(spy.count() - 1)[0]
        self.assertIsNotNone(sample)
        self.assertTrue(sample.in_bounds)
        self.assertEqual((sample.row, sample.col), (50, 100))
        self.assertEqual(sample.channels, ("R", "G", "B"))
        self.assertEqual(sample.values[2], 10.0)
        self.assertEqual(sample.value_text(), "R=0  G=0  B=10")

        _mouse_event(self.widget, QEvent.Type.MouseMove, QPointF(2.0, 2.0))
        last = spy.at(spy.count() - 1)[0]
        self.assertIsNotNone(last)
        self.assertFalse(last.in_bounds)
        self.assertEqual(last.value_text(), "--")

    def test_ctrl_wheel_zoom_about_cursor(self) -> None:
        self.widget.fit_to_window()
        before = self.widget.zoom_percent()
        anchor = self.widget.image_to_view(100.5, 50.5)
        wheel = QWheelEvent(
            anchor,
            self.widget.mapToGlobal(anchor),
            QPoint(0, 0),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.ControlModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        QApplication.sendEvent(self.widget, wheel)
        self.assertGreater(self.widget.zoom_percent(), before)
        mapped = self.widget.view_to_image(anchor)
        assert mapped is not None
        self.assertAlmostEqual(mapped[0], 100.5, delta=0.8)
        self.assertAlmostEqual(mapped[1], 50.5, delta=0.8)

        self.widget.reset_zoom()
        self.assertAlmostEqual(self.widget.zoom_percent(), 100.0, places=2)

    def test_pick_rect_payload(self) -> None:
        spy = QSignalSpy(self.widget.pick_finished)
        self.widget.fit_to_window()
        self.widget.begin_pick("rect")
        self.assertTrue(self.widget.is_picking())
        start = self.widget.image_to_view(10.0, 10.0)
        end = self.widget.image_to_view(60.0, 40.0)
        _mouse_event(self.widget, QEvent.Type.MouseButtonPress, start)
        _mouse_event(self.widget, QEvent.Type.MouseMove, end)
        _mouse_event(self.widget, QEvent.Type.MouseButtonRelease, end)
        self.assertEqual(spy.count(), 1)
        tool, payload = spy.at(0)
        self.assertEqual(tool, "rect")
        self.assertEqual(len(payload), 4)
        expected = [10.0, 10.0, 50.0, 30.0]
        for value, target in zip(payload, expected):
            self.assertAlmostEqual(value, target, delta=0.15)
        self.assertFalse(self.widget.is_picking())

    def test_pick_points_and_enter(self) -> None:
        spy = QSignalSpy(self.widget.pick_finished)
        self.widget.fit_to_window()
        self.widget.begin_pick("points", count=2)
        _click_image(self.widget, 5.0, 6.0)
        _click_image(self.widget, 30.0, 40.0)
        _send_key(self.widget, Qt.Key.Key_Return)
        self.assertEqual(spy.count(), 1)
        tool, payload = spy.at(0)
        self.assertEqual(tool, "points")
        self.assertEqual(len(payload), 2)
        self.assertAlmostEqual(payload[0][0], 5.0, delta=0.15)
        self.assertAlmostEqual(payload[0][1], 6.0, delta=0.15)
        self.assertAlmostEqual(payload[1][0], 30.0, delta=0.15)
        self.assertAlmostEqual(payload[1][1], 40.0, delta=0.15)

    def test_pick_cancelled_by_escape_and_set_image(self) -> None:
        cancelled = QSignalSpy(self.widget.pick_cancelled)
        self.widget.begin_pick("line")
        _send_key(self.widget, Qt.Key.Key_Escape)
        self.assertEqual(cancelled.count(), 1)
        self.assertEqual(cancelled.at(0)[0], "line")
        self.assertFalse(self.widget.is_picking())

        self.widget.begin_pick("points")
        self.widget.set_image(_make_color_image())
        self.assertEqual(cancelled.count(), 2)
        self.assertEqual(cancelled.at(1)[0], "points")
        self.assertFalse(self.widget.is_picking())

    def test_compare_modes_and_swipe_split(self) -> None:
        original = _make_color_image(b=0)
        current_bright = ImageData(
            np.full((100, 200, 3), 255, np.uint8), "current"
        )
        self.widget.set_original_image(original, "原始图像")
        self.widget.set_image(current_bright, "当前结果")
        self.widget.set_compare_mode("side_by_side")
        self.assertEqual(self.widget.compare_mode(), "side_by_side")
        self.widget.grab()

        self.widget.set_compare_mode("swipe")
        self.widget.fit_to_window()
        spy = QSignalSpy(self.widget.pixel_info)
        left = self.widget.image_to_view(40.5, 50.5)
        _mouse_event(self.widget, QEvent.Type.MouseMove, left)
        left_sample = spy.at(spy.count() - 1)[0]
        self.assertEqual(left_sample.values[2], 0.0)
        right = self.widget.image_to_view(160.5, 50.5)
        _mouse_event(self.widget, QEvent.Type.MouseMove, right)
        right_sample = spy.at(spy.count() - 1)[0]
        self.assertEqual(right_sample.values[2], 255.0)
        self.widget.grab()

    def test_display_modes_switch_and_render(self) -> None:
        self.widget.set_original_image(_make_color_image(b=0), "原始图像")
        for mode in ("imagesc", "mesh"):
            self.widget.set_display_mode(mode)
            self.assertEqual(self.widget.display_mode(), mode)
            self.widget.fit_to_window()
            self.assertFalse(self.widget.grab().isNull())
            for compare in ("side_by_side", "swipe"):
                self.widget.set_compare_mode(compare)
                self.assertFalse(self.widget.grab().isNull())
            self.widget.set_compare_mode("single")
        # 切换节点（重新 set_image）不重置显示模式
        self.widget.set_display_mode("imagesc")
        self.widget.set_image(_make_color_image(b=90))
        self.assertEqual(self.widget.display_mode(), "imagesc")
        # 非法值回退到 imshow
        self.widget.set_display_mode("未知模式")
        self.assertEqual(self.widget.display_mode(), "imshow")

    def test_mesh_mode_disables_pixel_sampling_and_picking(self) -> None:
        self.widget.fit_to_window()
        spy = QSignalSpy(self.widget.pixel_info)
        center = QPointF(self.widget.width() / 2.0, self.widget.height() / 2.0)
        _mouse_event(self.widget, QEvent.Type.MouseMove, center)
        self.assertIsNotNone(spy.at(spy.count() - 1)[0])
        self.widget.set_display_mode("mesh")
        self.assertIsNone(spy.at(spy.count() - 1)[0])
        with self.assertRaises(ValueError):
            self.widget.begin_pick("rect")
        self.assertFalse(self.widget.is_picking())

    def test_mesh_rotation_zoom_and_double_click_reset(self) -> None:
        self.widget.set_display_mode("mesh")
        self.widget.fit_to_window()
        azimuth, elevation = self.widget.mesh_camera()
        self.assertAlmostEqual(azimuth, -37.5, places=3)
        self.assertAlmostEqual(elevation, 30.0, places=3)

        start = QPointF(self.widget.width() / 2.0, self.widget.height() / 2.0)
        _mouse_event(self.widget, QEvent.Type.MouseButtonPress, start)
        _mouse_event(
            self.widget,
            QEvent.Type.MouseMove,
            start + QPointF(80.0, 40.0),
        )
        _mouse_event(
            self.widget,
            QEvent.Type.MouseButtonRelease,
            start + QPointF(80.0, 40.0),
            buttons=Qt.MouseButton.NoButton,
        )
        rotated = self.widget.mesh_camera()
        self.assertNotAlmostEqual(rotated[0], azimuth, places=2)
        self.assertNotAlmostEqual(rotated[1], elevation, places=2)

        # 仰角不限制：继续向下拖可以越过 90°（旧实现会卡在 82°）。
        _mouse_event(self.widget, QEvent.Type.MouseButtonPress, start)
        _mouse_event(
            self.widget,
            QEvent.Type.MouseMove,
            start + QPointF(0.0, 140.0),
        )
        _mouse_event(
            self.widget,
            QEvent.Type.MouseButtonRelease,
            start + QPointF(0.0, 140.0),
            buttons=Qt.MouseButton.NoButton,
        )
        self.assertGreater(self.widget.mesh_camera()[1], 90.0)

        zoom_before = self.widget.zoom_percent()
        _send_wheel(self.widget, start)
        self.assertGreater(self.widget.zoom_percent(), zoom_before)

        double_click = QMouseEvent(
            QEvent.Type.MouseButtonDblClick,
            start,
            self.widget.mapToGlobal(start),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(self.widget, double_click)
        self.assertAlmostEqual(self.widget.mesh_camera()[0], -37.5, places=3)
        self.assertAlmostEqual(self.widget.mesh_camera()[1], 30.0, places=3)

    def test_mesh_flat_image_at_horizon_stays_finite(self) -> None:
        """仰角可到 0°：平坦图像在该角度下不应产生无穷大的缩放坐标。"""

        self.widget.set_image(ImageData(np.full((32, 48), 7, np.uint8), "flat"))
        self.widget.set_display_mode("mesh")
        self.widget._mesh_elevation = 0.0
        self.widget.fit_to_window()
        self.assertFalse(self.widget.grab().isNull())
        geometry = self.widget._mesh_geometry_for(self.widget.current_image())
        self.assertIsNotNone(geometry)
        assert geometry is not None
        self.assertTrue(np.isfinite(geometry.points).all())
        self.assertLessEqual(float(np.abs(geometry.points).max()), 5000.0)

    def test_mouse_wheel_without_ctrl_only_zooms_mesh(self) -> None:
        center = QPointF(self.widget.width() / 2.0, self.widget.height() / 2.0)
        zoom_before = self.widget.zoom_percent()
        _send_wheel(self.widget, center)
        self.assertAlmostEqual(self.widget.zoom_percent(), zoom_before, places=3)

        self.widget.set_display_mode("mesh")
        _send_wheel(self.widget, center)
        self.assertGreater(self.widget.zoom_percent(), zoom_before)

    def test_mesh_degenerate_matrices_render(self) -> None:
        """单行 / 单列 / 单像素矩阵在 mesh 模式下补齐为 2×2，不再因 np.gradient 崩溃。"""

        for shape in ((3, 1), (1, 3), (1, 1)):
            with self.subTest(shape=shape):
                matrix = ImageData(
                    np.arange(int(np.prod(shape)), dtype=np.uint8).reshape(shape), "矩阵"
                )
                self.widget.set_display_mode("mesh")
                self.widget.set_image(matrix)
                self.widget.fit_to_window()
                self.assertEqual(self.widget.image_size(), (shape[1], shape[0]))
                field = self.widget._mesh_field_for(matrix)
                self.assertIsNotNone(field)
                # 退化维度被复制到至少 2，网格可以计算梯度并三角化。
                self.assertGreaterEqual(field.grid.shape[0], 2)
                self.assertGreaterEqual(field.grid.shape[1], 2)
                self.assertEqual(field.colors.shape, field.grid.shape + (3,))
                # 布局宽高仍来自原始图像，缩放不受补齐影响。
                self.assertEqual((field.width, field.height), (shape[1], shape[0]))
                self.assertFalse(self.widget.grab().isNull())

    def test_mesh_regular_matrix_keeps_grid_shape(self) -> None:
        """普通 2×2 矩阵的网格保持原尺寸，不被退化补齐逻辑改动。"""

        matrix = ImageData(np.arange(4, dtype=np.uint8).reshape(2, 2), "矩阵")
        self.widget.set_display_mode("mesh")
        self.widget.set_image(matrix)
        field = self.widget._mesh_field_for(matrix)
        self.assertIsNotNone(field)
        self.assertEqual(field.grid.shape, (2, 2))
        self.assertFalse(self.widget.grab().isNull())

    def test_empty_state_paints(self) -> None:
        self.widget.set_image(None)
        self.assertIsNone(self.widget.image_size())
        pixmap = self.widget.grab()
        self.assertFalse(pixmap.isNull())


# ---------------------------------------------------------------------------
# 显示模式图标
# ---------------------------------------------------------------------------
class DisplayModeIconTests(unittest.TestCase):
    def test_three_modes_have_icons_and_titles(self) -> None:
        modes = [mode for mode, _title, _tip in DISPLAY_MODE_SPECS]
        self.assertEqual(modes, ["imshow", "imagesc", "mesh"])
        for mode, title, tip in DISPLAY_MODE_SPECS:
            icon = display_mode_icon(mode)
            self.assertFalse(icon.isNull())
            self.assertFalse(icon.pixmap(18, 18).isNull())
            self.assertEqual(display_mode_title(mode), title)
            self.assertTrue(tip)

    def test_unknown_mode_falls_back_to_imshow(self) -> None:
        self.assertEqual(display_mode_title("not-a-mode"), "Imshow")
        self.assertFalse(display_mode_icon("not-a-mode").isNull())


# ---------------------------------------------------------------------------
# ParamPanel
# ---------------------------------------------------------------------------
class ParamPanelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.panel = ParamPanel()
        self.panel.resize(380, 640)
        self.panel.show()
        self.node = NodeModel(
            id="node-1",
            op_id=_PARAM_SPEC.id,
            title="参数节点",
            params={},
        )

    def tearDown(self) -> None:
        self.panel.deleteLater()

    def test_load_node_has_no_preview_and_shows_defaults(self) -> None:
        spy = QSignalSpy(self.panel.preview_requested)
        self.panel.load_node(self.node, _PARAM_OPERATOR)
        QTest.qWait(180)
        self.assertEqual(spy.count(), 0)
        params = self.panel.current_params()
        self.assertEqual(params["value"], 5)
        self.assertAlmostEqual(params["gain"], 1.0)
        self.assertTrue(params["enabled"])
        self.assertEqual(params["mode"], "a")
        self.assertEqual(len(params["points"]), 2)
        self.assertTrue(self.panel.is_valid())
        self.assertTrue(self.panel.save_button.isEnabled())

    def test_debounced_preview(self) -> None:
        self.panel.load_node(self.node, _PARAM_OPERATOR)
        spy = QSignalSpy(self.panel.preview_requested)
        self.panel.set_param_value("value", 42)
        self.assertEqual(spy.count(), 0)
        QTest.qWait(200)
        self.assertEqual(spy.count(), 1)
        node_id, params = spy.at(0)
        self.assertEqual(node_id, "node-1")
        self.assertEqual(params["value"], 42)

        self.panel.set_param_value("value", 43)
        self.panel.set_param_value("value", 44)
        self.assertEqual(spy.count(), 1)
        QTest.qWait(200)
        self.assertEqual(spy.count(), 2)
        self.assertEqual(spy.at(1)[1]["value"], 44)

    def test_visible_when_and_enabled_when(self) -> None:
        self.panel.load_node(self.node, _PARAM_OPERATOR)
        self.assertTrue(self.panel.param_row_visible("extra"))
        self.panel.set_param_value("enabled", False)
        self.assertFalse(self.panel.param_row_visible("extra"))
        field = self.panel.field_widget("extra")
        self.assertIsNotNone(field)
        self.panel.set_param_value("enabled", True)
        self.assertTrue(self.panel.param_row_visible("extra"))
        self.assertTrue(field.isEnabled())

    def test_invalid_roi_disables_save_and_shows_error(self) -> None:
        self.panel.load_node(self.node, _PARAM_OPERATOR)
        status_spy = QSignalSpy(self.panel.status_message)
        preview_spy = QSignalSpy(self.panel.preview_requested)
        self.panel.set_pick_result("roi", [1.0, 1.0, 0.0, 5.0])
        self.assertFalse(self.panel.is_valid())
        self.assertFalse(self.panel.save_button.isEnabled())
        self.assertTrue(self.panel.error_label.isVisible())
        self.assertIn("宽高", self.panel.error_label.text())
        self.assertGreaterEqual(status_spy.count(), 1)
        QTest.qWait(200)
        self.assertEqual(preview_spy.count(), 0)

        self.panel.set_pick_result("roi", [1.0, 1.0, 10.0, 5.0])
        self.assertTrue(self.panel.is_valid())
        self.assertTrue(self.panel.save_button.isEnabled())
        QTest.qWait(200)
        self.assertEqual(preview_spy.count(), 1)
        self.assertEqual(preview_spy.at(0)[1]["roi"], [1.0, 1.0, 10.0, 5.0])

    def test_set_pick_result_points_and_save(self) -> None:
        self.panel.load_node(self.node, _PARAM_OPERATOR)
        self.panel.set_pick_result("points", [[1.0, 2.0], [3.0, 4.0]])
        self.assertEqual(self.panel.current_params()["points"], [[1.0, 2.0], [3.0, 4.0]])
        save_spy = QSignalSpy(self.panel.save_requested)
        self.panel.save_button.click()
        self.assertEqual(save_spy.count(), 1)
        self.assertEqual(save_spy.at(0)[0], "node-1")
        self.assertEqual(save_spy.at(0)[1]["points"], [[1.0, 2.0], [3.0, 4.0]])

    def test_reset_and_undo(self) -> None:
        self.node.params = {"value": 7, "gain": 2.0}
        self.panel.load_node(self.node, _PARAM_OPERATOR)
        self.assertEqual(self.panel.current_params()["value"], 7)
        self.panel.set_param_value("value", 33)
        self.panel.undo_button.click()
        self.assertEqual(self.panel.current_params()["value"], 7)
        self.panel.reset_button.click()
        self.assertEqual(self.panel.current_params()["value"], 5)
        self.assertEqual(self.panel.baseline_params()["value"], 7)

    def test_pick_request_signal(self) -> None:
        self.panel.load_node(self.node, _PARAM_OPERATOR)
        spy = QSignalSpy(self.panel.pick_requested)
        field = self.panel.field_widget("roi")
        self.assertIsNotNone(field)
        buttons = field.findChildren(type(self.panel.reset_button))
        pick_buttons = [button for button in buttons if button.text() == "拾取"]
        self.assertTrue(pick_buttons)
        pick_buttons[0].click()
        self.assertEqual(spy.count(), 1)
        self.assertEqual(spy.at(0), ["node-1", "roi", "rect"])

    def test_status_message_signal_exists(self) -> None:
        self.assertTrue(hasattr(ParamPanel, "status_message"))


# ---------------------------------------------------------------------------
# OperatorLibrary
# ---------------------------------------------------------------------------
_OPERATOR_TEMPLATE = '''\
from app.operators.spec import Operator, OperatorSpec, ParamSpec


class _TestOperator(Operator):
    spec = OperatorSpec(
        id="{op_id}",
        name="{name}",
        category={category!r},
        description="{description}",
        keywords=({keyword!r},),
        params=(
            ParamSpec("strength", "强度", "int", default=3, min=1, max=9),
        ),
    )

    def process(self, image, strength=3, **kwargs):
        return image


OPERATOR = _TestOperator
'''


class OperatorLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = _workspace_tempdir()
        cls.addClassCleanup(_cleanup_tempdir, cls.root)
        root = cls.root
        files = {
            "brightness.py": ("test.brightness", "亮度调整", ("基础调整", "颜色", "亮度"), "调整图像亮度", "亮度"),
            "contrast.py": ("test.contrast", "对比度调整", ("基础调整", "颜色"), "调整图像对比度", "对比度"),
            "blur.py": ("test.blur", "高斯模糊", ("滤波与平滑", "模糊"), "对图像进行高斯模糊", "模糊"),
        }
        for filename, (op_id, name, category, description, keyword) in files.items():
            (root / filename).write_text(
                _OPERATOR_TEMPLATE.format(
                    op_id=op_id,
                    name=name,
                    category=category,
                    description=description,
                    keyword=keyword,
                ),
                encoding="utf-8",
            )
        cls.root = root
        cls.registry = OperatorRegistry.build([root], include_builtin=False)

    def setUp(self) -> None:
        self.library = OperatorLibrary()
        self.library.resize(300, 600)
        self.library.show()
        self.library.set_registry(self.registry)
        QApplication.processEvents()

    def tearDown(self) -> None:
        self.library.deleteLater()

    def _find_leaf(self, tree, op_id: str):
        stack = [tree.topLevelItem(index) for index in range(tree.topLevelItemCount())]
        while stack:
            item = stack.pop()
            if item.data(0, Qt.ItemDataRole.UserRole) == op_id:
                return item
            stack.extend(item.child(index) for index in range(item.childCount()))
        return None

    @staticmethod
    def _item_reachable(item) -> bool:
        """按 Qt 的展开语义判断条目是否真实可见（父级折叠时不可见）。"""
        current = item
        while current is not None:
            if current.isHidden():
                return False
            parent = current.parent()
            if parent is not None and not parent.isExpanded():
                return False
            current = parent
        return True

    def _visible_leaf_ids(self, tree) -> set[str]:
        result: set[str] = set()

        def walk(item) -> None:
            op_id = item.data(0, Qt.ItemDataRole.UserRole)
            if op_id:
                if self._item_reachable(item):
                    result.add(str(op_id))
                return
            if item.isHidden():
                return
            for index in range(item.childCount()):
                walk(item.child(index))

        for index in range(tree.topLevelItemCount()):
            walk(tree.topLevelItem(index))
        return result

    def _all_visible_ids(self) -> set[str]:
        tree = self.library.current_tree()
        return self._visible_leaf_ids(tree) if tree is not None else set()

    def test_accordion_categories_default_collapsed(self) -> None:
        self.assertEqual(self.library.category_names(), ["基础调整", "滤波与平滑"])
        self.assertIsNone(self.library.current_category())
        self.assertEqual(self.library.operator_count(), 3)
        self.assertEqual(self.library.load_error_count(), 0)
        self.assertIn("3", self.library.count_label.text())
        self.assertIn("全部折叠", self.library.category_label.text())

        tree = self.library.current_tree()
        self.assertIsNotNone(tree)
        assert tree is not None
        self.assertEqual(tree.topLevelItemCount(), 2)
        items = self.library.category_items()
        self.assertEqual([item.text(0) for item in items], ["基础调整", "滤波与平滑"])
        self.assertEqual([item.text(1) for item in items], ["2", "1"])
        for item in items:
            self.assertFalse(item.isExpanded())
            self.assertTrue(item.font(0).bold())
        self.assertEqual(self._visible_leaf_ids(tree), set())

    def test_click_category_expands_in_place_and_multi_open(self) -> None:
        tree = self.library.current_tree()
        self.assertIsNotNone(tree)
        assert tree is not None
        base, filter_category = self.library.category_items()

        tree.itemClicked.emit(base, 0)
        QApplication.processEvents()
        self.assertTrue(base.isExpanded())
        self.assertFalse(filter_category.isExpanded())
        self.assertEqual(self.library.current_category(), "基础调整")
        # 点击一级分类会直接展开到算子叶子，而不只是显示分组标题。
        self.assertEqual(
            self._visible_leaf_ids(tree), {"test.brightness", "test.contrast"}
        )
        color_group = base.child(0)
        self.assertEqual(color_group.text(0), "颜色")
        self.assertTrue(color_group.isExpanded())

        tree.itemClicked.emit(filter_category, 0)
        QApplication.processEvents()
        self.assertTrue(base.isExpanded())
        self.assertTrue(filter_category.isExpanded())
        self.assertEqual(self.library.current_category(), "滤波与平滑")
        # 多开：两个一级分类可以同时展开。
        self.assertEqual(
            self._visible_leaf_ids(tree),
            {"test.brightness", "test.contrast", "test.blur"},
        )

        # 收起其中一个，不影响另一个。
        tree.itemClicked.emit(base, 0)
        QApplication.processEvents()
        self.assertFalse(base.isExpanded())
        self.assertTrue(filter_category.isExpanded())
        self.assertEqual(self._visible_leaf_ids(tree), {"test.blur"})

    def test_real_mouse_click_category_row_expands(self) -> None:
        """真实 QTest 点击分类行，覆盖 itemClicked → 原地展开事件链。"""

        tree = self.library.current_tree()
        self.assertIsNotNone(tree)
        assert tree is not None
        item = self.library.category_items()[0]
        tree.scrollToItem(item)
        QApplication.processEvents()
        rect = tree.visualItemRect(item)
        QTest.mouseClick(
            tree.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            rect.center(),
        )
        QApplication.processEvents()
        self.assertTrue(item.isExpanded())
        self.assertEqual(
            self._visible_leaf_ids(tree), {"test.brightness", "test.contrast"}
        )

        # 与第一次点击隔开，避免被 Qt 识别成双击。
        QTest.qWait(300)
        QTest.mouseClick(
            tree.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            tree.visualItemRect(item).center(),
        )
        QApplication.processEvents()
        self.assertFalse(item.isExpanded())
        self.assertEqual(self._visible_leaf_ids(tree), set())

    def test_nested_groups_can_collapse_independently(self) -> None:
        tree = self.library.current_tree()
        self.assertIsNotNone(tree)
        assert tree is not None
        self.assertTrue(self.library.select_category("基础调整"))
        base = self.library.category_items()[0]
        color_group = base.child(0)
        self.assertTrue(color_group.isExpanded())

        brightness = self._find_leaf(tree, "test.brightness")
        contrast = self._find_leaf(tree, "test.contrast")
        self.assertIsNotNone(brightness)
        self.assertIsNotNone(contrast)
        assert brightness is not None and contrast is not None
        # 三级分类：叶子 -> 亮度(二级) -> 颜色(一级)
        self.assertEqual(brightness.parent().text(0), "亮度")
        self.assertEqual(brightness.parent().parent().text(0), "颜色")
        self.assertTrue(brightness.parent().isExpanded())
        self.assertEqual(contrast.parent().text(0), "颜色")
        self.assertIn("调整图像亮度", brightness.toolTip(0))
        self.assertTrue(brightness.flags() & Qt.ItemFlag.ItemIsDragEnabled)

        # 单独折叠二级分组：只影响该分组，不影响一级分类。
        brightness.parent().setExpanded(False)
        QApplication.processEvents()
        self.assertEqual(self._visible_leaf_ids(tree), {"test.contrast"})
        self.assertTrue(base.isExpanded())
        # 单独折叠一级分组。
        color_group.setExpanded(False)
        QApplication.processEvents()
        self.assertEqual(self._visible_leaf_ids(tree), set())
        # 点击分组行也可以原地展开。
        tree.itemClicked.emit(color_group, 0)
        QApplication.processEvents()
        self.assertTrue(color_group.isExpanded())
        self.assertFalse(brightness.parent().isExpanded())
        self.assertEqual(self._visible_leaf_ids(tree), {"test.contrast"})

    def test_expand_all_and_collapse_all(self) -> None:
        tree = self.library.current_tree()
        self.assertIsNotNone(tree)
        assert tree is not None
        self.assertEqual(self._visible_leaf_ids(tree), set())

        self.library.expand_all_button.click()
        self.assertEqual(
            self._visible_leaf_ids(tree),
            {"test.brightness", "test.contrast", "test.blur"},
        )
        self.assertIn("全部展开", self.library.category_label.text())

        self.library.collapse_all_button.click()
        self.assertEqual(self._visible_leaf_ids(tree), set())
        self.assertIn("全部折叠", self.library.category_label.text())

        self.library.expand_all_button.click()
        self.assertIn("test.brightness", self._visible_leaf_ids(tree))

    def test_search_across_categories_and_auto_expand(self) -> None:
        self.library.search_edit.setText("高斯模糊")
        QApplication.processEvents()
        self.assertEqual(self.library.current_category(), "滤波与平滑")
        self.assertIn("匹配", self.library.count_label.text())
        tree = self.library.current_tree()
        self.assertIsNotNone(tree)
        assert tree is not None
        self.assertEqual(self._visible_leaf_ids(tree), {"test.blur"})
        self.assertEqual(self._all_visible_ids(), {"test.blur"})

        # 深层命中项在搜索时自动展开可见。
        self.library.search_edit.setText("调整图像亮度")
        QApplication.processEvents()
        self.assertEqual(self.library.current_category(), "基础调整")
        tree = self.library.current_tree()
        self.assertIsNotNone(tree)
        assert tree is not None
        self.assertEqual(self._visible_leaf_ids(tree), {"test.brightness"})

        self.library.search_edit.setText("不存在的算子")
        QApplication.processEvents()
        self.assertIn("匹配 0 个", self.library.count_label.text())
        self.assertEqual(self._all_visible_ids(), set())
        self.assertEqual(
            self.library._content_stack.currentWidget(),
            self.library.filter_empty_page,
        )

        self.library.search_edit.clear()
        QApplication.processEvents()
        self.assertNotIn("匹配", self.library.count_label.text())
        self.assertIn("全部折叠", self.library.category_label.text())
        self.assertEqual(self._all_visible_ids(), set())
        # 清空搜索后再次点击分类可直接展开算子。
        tree.itemClicked.emit(self.library.category_items()[0], 0)
        QApplication.processEvents()
        self.assertEqual(
            self._all_visible_ids(), {"test.brightness", "test.contrast"}
        )

    def test_refresh_button_emits(self) -> None:
        # 刷新按钮只显示图标（与工具栏一致），不显示文字。
        self.assertEqual(self.library.refresh_button.text(), "")
        self.assertFalse(self.library.refresh_button.icon().isNull())
        self.assertEqual(self.library.refresh_button.accessibleName(), "刷新算子库")
        spy = QSignalSpy(self.library.refresh_requested)
        self.library.refresh_button.click()
        self.assertEqual(spy.count(), 1)

    def test_info_panel_and_escape(self) -> None:
        self.assertTrue(self.library.show_operator_info("test.brightness"))
        self.assertTrue(self.library.info_panel_visible())
        self.assertEqual(self.library.info_name.text(), "亮度调整")
        self.assertIn("调整图像亮度", self.library.info_desc.text())
        self.assertEqual(self.library.info_table.rowCount(), 1)
        self.assertIn("输入端口", self.library.info_ports.text())

        _send_key(self.library, Qt.Key.Key_Escape)
        QApplication.processEvents()
        self.assertFalse(self.library.info_panel_visible())

    def test_info_panel_closes_on_outside_click(self) -> None:
        self.assertTrue(self.library.show_operator_info("test.contrast"))
        self.assertTrue(self.library.info_panel_visible())
        global_position = self.library.search_edit.mapToGlobal(QPoint(4, 4))
        event = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(4, 4),
            QPointF(global_position),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(self.library.search_edit, event)
        QApplication.processEvents()
        self.assertFalse(self.library.info_panel_visible())

    def test_drag_mime_data(self) -> None:
        mime = self.library.mime_data_for("test.blur")
        payload = json.loads(bytes(mime.data(MIME_OPERATOR)).decode("utf-8"))
        self.assertEqual(payload, {"id": "test.blur"})
        self.assertTrue(mime.hasFormat(MIME_OPERATOR))

        captured = {}

        class _FakeDrag:
            def __init__(self, source):
                captured["source"] = source

            def setMimeData(self, mime_data):
                captured["mime"] = mime_data

            def setPixmap(self, pixmap):
                captured["pixmap"] = pixmap

            def setHotSpot(self, point):
                captured["hotspot"] = point

            def exec(self, action):
                captured["action"] = action
                return Qt.DropAction.CopyAction

        self.assertTrue(self.library.select_category("滤波与平滑"))
        tree = self.library.current_tree()
        self.assertIsNotNone(tree)
        assert tree is not None
        leaf = self._find_leaf(tree, "test.blur")
        self.assertIsNotNone(leaf)
        assert leaf is not None
        tree.setCurrentItem(leaf)
        with patch("app.ui.library.QDrag", _FakeDrag):
            tree.startDrag(Qt.DropAction.CopyAction)
        self.assertIn("mime", captured)
        payload = json.loads(bytes(captured["mime"].data(MIME_OPERATOR)).decode("utf-8"))
        self.assertEqual(payload["id"], "test.blur")

        # 非叶子节点不允许拖拽。
        self.library.collapse_all_button.click()
        group = tree.topLevelItem(0)
        tree.setCurrentItem(group)
        captured.clear()
        with patch("app.ui.library.QDrag", _FakeDrag):
            tree.startDrag(Qt.DropAction.CopyAction)
        self.assertNotIn("mime", captured)

    def test_empty_registry(self) -> None:
        empty = OperatorRegistry.build([], include_builtin=False)
        library = OperatorLibrary()
        library.show()
        library.set_registry(empty)
        try:
            self.assertEqual(library.category_names(), [])
            self.assertEqual(library.category_buttons(), [])
            self.assertIsNone(library.current_category())
            self.assertIsNone(library.current_tree())
            self.assertEqual(library.operator_count(), 0)
            self.assertIn("共 0 个算子", library.count_label.text())
            self.assertFalse(library.tree_bar.isVisible())
        finally:
            library.deleteLater()


# ---------------------------------------------------------------------------
# OperatorLibrary：用户自定义算子置顶 / 算子组算子标识
# ---------------------------------------------------------------------------
_GROUP_OPERATOR_TEMPLATE = '''import json

from app.operators.group import GroupOperator
from app.operators.spec import OperatorSpec, PortSpec

GROUP_DEFINITION = json.loads({definition!r})


class _TestGroupOperator(GroupOperator):
    spec = OperatorSpec(
        id="{op_id}",
        name="{name}",
        category=("{category}",),
        description="{description}",
        keywords=("算子组", "自定义算子"),
        inputs=(
            PortSpec("input_1", "输入 1", any_type=True),
        ),
        outputs=(
            PortSpec("output_1", "输出 1", any_type=True),
        ),
    )
    group_definition = GROUP_DEFINITION


OPERATOR = _TestGroupOperator
'''

_GROUP_DEFINITION = {
    "format": "mineimage-group-definition",
    "version": 1,
    "name": "演示算子组",
    "inputs": [{"key": "input_1", "label": "输入 1", "data_type": None}],
    "outputs": [{"key": "output_1", "label": "输出 1", "data_type": "image"}],
    "nodes": [],
    "connections": [],
}


def _build_group_fixture(root: Path) -> None:
    """写入普通算子与算子组算子，构造含「用户自定义算子」分类的临时注册表。"""

    files = {
        "brightness.py": (
            "test.brightness",
            "亮度调整",
            ("基础调整", "颜色"),
            "调整图像亮度",
            "亮度",
        ),
        "blur.py": (
            "test.blur",
            "高斯模糊",
            ("滤波与平滑", "模糊"),
            "对图像进行高斯模糊",
            "模糊",
        ),
        "user_op.py": (
            "test.user",
            "我的亮度",
            (CUSTOM_OPERATOR_CATEGORY, "我的算子"),
            "用户保存的普通算子",
            "自定义",
        ),
    }
    for filename, (op_id, name, category, description, keyword) in files.items():
        (root / filename).write_text(
            _OPERATOR_TEMPLATE.format(
                op_id=op_id,
                name=name,
                category=category,
                description=description,
                keyword=keyword,
            ),
            encoding="utf-8",
        )
    (root / "group_op.py").write_text(
        _GROUP_OPERATOR_TEMPLATE.format(
            definition=json.dumps(_GROUP_DEFINITION, ensure_ascii=False),
            op_id="group.demo",
            name="演示算子组",
            category=CUSTOM_OPERATOR_CATEGORY,
            description="由算子组保存生成的演示算子。",
        ),
        encoding="utf-8",
    )


class GroupOperatorLibraryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = _workspace_tempdir()
        cls.addClassCleanup(_cleanup_tempdir, cls.root)
        _build_group_fixture(cls.root)
        cls.registry = OperatorRegistry.build([cls.root], include_builtin=False)

    def setUp(self) -> None:
        self.library = OperatorLibrary()
        self.library.resize(300, 600)
        self.library.show()
        self.library.set_registry(self.registry)
        QApplication.processEvents()

    def tearDown(self) -> None:
        self.library.deleteLater()

    def test_custom_category_is_first(self) -> None:
        names = self.library.category_names()
        self.assertEqual(names, [CUSTOM_OPERATOR_CATEGORY, "基础调整", "滤波与平滑"])
        first = self.library.category_items()[0]
        self.assertEqual(first.text(0), CUSTOM_OPERATOR_CATEGORY)
        self.assertEqual(first.text(1), "2")
        self.assertEqual(self.library.operator_count(), 4)

    def test_group_operator_info_panel_badge_ports_and_no_param_table(self) -> None:
        self.assertTrue(self.library.show_operator_info("group.demo"))
        self.assertIn("· 算子组", self.library.info_meta.text())
        self.assertIn("演示算子组", self.library.info_name.text())
        self.assertIn("由算子组保存生成", self.library.info_desc.text())
        self.assertIn("输入端口", self.library.info_ports.text())
        self.assertIn("输出端口", self.library.info_ports.text())
        self.assertFalse(self.library.info_table.isVisible())
        self.assertTrue(self.library.info_no_params.isVisible())
        self.assertIn("没有可配置参数", self.library.info_no_params.text())

    def test_tree_tooltip_marks_group_operator(self) -> None:
        self.assertTrue(self.library.select_category(CUSTOM_OPERATOR_CATEGORY))
        tree = self.library.current_tree()
        self.assertIsNotNone(tree)
        assert tree is not None
        item = self.library._entry_items["group.demo"]  # noqa: SLF001 - 测试内部定位
        self.assertIn("类型：算子组", item.toolTip(0))

    def test_regular_operator_has_no_group_badge(self) -> None:
        self.assertTrue(self.library.show_operator_info("test.brightness"))
        self.assertNotIn("· 算子组", self.library.info_meta.text())
        self.assertIn("参数", self.library.info_params_title.text())


# ---------------------------------------------------------------------------
# OperatorLibrary：右键新建 / 原地改名 / 用户算子菜单 / 空分类
# ---------------------------------------------------------------------------
class OperatorLibraryCreateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.root = _workspace_tempdir()
        self.addCleanup(_cleanup_tempdir, self.root)
        self.settings = AppSettings(self.root / "settings.ini")
        self.registry = OperatorRegistry.build([self.root], include_builtin=False)
        self.library = OperatorLibrary(
            settings=self.settings, user_operator_dir=self.root
        )
        self.library.resize(320, 640)
        self.library.show()
        self.library.set_registry(self.registry)
        self.messages: list[str] = []
        self.library.status_message.connect(self.messages.append)
        self.library.refresh_requested.connect(self._reload)
        QApplication.processEvents()

    def tearDown(self) -> None:
        self.library.deleteLater()

    def _reload(self) -> None:
        self.registry = OperatorRegistry.build([self.root], include_builtin=False)
        self.library.set_registry(self.registry)
        QApplication.processEvents()

    def _cancel_inline_editor(self) -> None:
        editor = self.library.tree.findChild(QLineEdit, "operatorNameEditor")
        if editor is not None:
            QTest.keyClick(editor, Qt.Key.Key_Escape)
        QApplication.processEvents()

    def _seed_operator(
        self,
        filename: str,
        *,
        op_id: str,
        name: str,
        category,
    ) -> None:
        (self.root / filename).write_text(
            _OPERATOR_TEMPLATE.format(
                op_id=op_id,
                name=name,
                category=category,
                description="测试算子",
                keyword="测试",
            ),
            encoding="utf-8",
        )
        self._reload()

    @staticmethod
    def _actions(menu) -> list[str]:
        return [
            str(action.data()["action"])
            for action in menu.actions()
            if action.data() and action.data().get("action")
        ]

    def test_create_operator_creates_file_and_starts_inline_rename(self) -> None:
        op_id = self.library.create_operator()
        self.assertIsNotNone(op_id)
        assert op_id is not None
        category_dir = self.root / CUSTOM_OPERATOR_CATEGORY
        self.assertTrue(category_dir.is_dir())
        files = sorted(category_dir.glob("*.py"))
        self.assertEqual(len(files), 1)
        self.assertEqual(files[0].stem, "新建算子")

        entry = self.registry.entry(op_id)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.display_name, "新建算子")
        self.assertFalse(entry.builtin)
        self.assertEqual(self.library.category_names(), [CUSTOM_OPERATOR_CATEGORY])
        item = self.library._entry_items[op_id]  # noqa: SLF001 - 测试内部定位
        self.assertEqual(item.text(0), "新建算子")
        self.assertTrue(item.flags() & Qt.ItemFlag.ItemIsEditable)
        self.assertTrue(item.isSelected())
        self.assertEqual(self.library.current_category(), CUSTOM_OPERATOR_CATEGORY)

        editor = self.library.tree.findChild(QLineEdit, "operatorNameEditor")
        self.assertIsNotNone(editor)
        assert editor is not None
        self.assertEqual(editor.text(), "新建算子")
        self.assertNotIn(".py", editor.text())
        self._cancel_inline_editor()

    def test_create_operator_clears_active_search_filter(self) -> None:
        self._seed_operator(
            "seed.py",
            op_id="test.seed",
            name="种子算子",
            category=("基础调整",),
        )
        self.library.search_edit.setText("不存在的关键字")
        self.assertIs(
            self.library._content_stack.currentWidget(),  # noqa: SLF001
            self.library.filter_empty_page,
        )
        op_id = self.library.create_operator()
        self.assertIsNotNone(op_id)
        assert op_id is not None
        self.assertEqual(self.library.search_edit.text(), "")
        self.assertIs(
            self.library._content_stack.currentWidget(),  # noqa: SLF001
            self.library.tree,
        )
        item = self.library._entry_items[op_id]  # noqa: SLF001
        self.assertFalse(item.isHidden())
        self._cancel_inline_editor()

    def test_rename_trims_py_extension_and_info_uses_display_name(self) -> None:
        op_id = self.library.create_operator()
        self.assertIsNotNone(op_id)
        assert op_id is not None
        self._cancel_inline_editor()
        self.assertTrue(self.library.rename_operator_to(op_id, "我的算子.py"))
        self._cancel_inline_editor()

        category_dir = self.root / CUSTOM_OPERATOR_CATEGORY
        self.assertTrue((category_dir / "我的算子.py").is_file())
        self.assertFalse((category_dir / "我的算子.py.py").exists())
        entry = self.registry.entry(op_id)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.path.stem, "我的算子")
        self.assertTrue(self.library.show_operator_info(op_id))
        self.assertEqual(self.library.info_name.text(), "我的算子")

    def test_blank_context_menu_creates_in_default_category(self) -> None:
        # 空列表页面本身启用自定义右键菜单。
        self.assertIs(
            self.library._content_stack.currentWidget(), self.library.empty_page
        )
        self.assertEqual(
            self.library.empty_page.contextMenuPolicy(),
            Qt.ContextMenuPolicy.CustomContextMenu,
        )
        blank_menu = self.library.build_context_menu(None)
        self.assertEqual(
            self._actions(blank_menu), ["create_operator", "create_category"]
        )

        # 先选中一个非默认分类，空白处新建仍应固定落到「用户自定义算子」。
        self._seed_operator(
            "brightness.py",
            op_id="test.brightness",
            name="亮度调整",
            category=("基础调整", "颜色"),
        )
        self.assertTrue(self.library.select_category("基础调整"))
        create_action = next(
            action
            for action in blank_menu.actions()
            if action.data() and action.data()["action"] == "create_operator"
        )
        create_action.trigger()
        QApplication.processEvents()
        self._cancel_inline_editor()
        self.assertTrue((self.root / CUSTOM_OPERATOR_CATEGORY).is_dir())
        self.assertFalse((self.root / "基础调整").exists())

        # 分类行右键：动作携带该分类完整路径。
        category_item = self.library._category_items[CUSTOM_OPERATOR_CATEGORY]
        menu = self.library.build_context_menu(category_item)
        self.assertEqual(
            self._actions(menu), ["create_operator", "create_category"]
        )
        payload = next(
            action.data()["payload"]
            for action in menu.actions()
            if action.data() and action.data()["action"] == "create_operator"
        )
        self.assertEqual(tuple(payload), (CUSTOM_OPERATOR_CATEGORY,))

    def test_empty_category_context_menu_creates_operator(self) -> None:
        path = self.library.create_category_named(None, "空分类")
        self.assertIsNotNone(path)
        empty_item = self.library._category_items["空分类"]
        self.assertEqual(empty_item.text(1), "0")

        menu = self.library.build_context_menu(empty_item)
        self.assertEqual(
            self._actions(menu), ["create_operator", "create_category"]
        )
        create_action = next(
            action
            for action in menu.actions()
            if action.data() and action.data()["action"] == "create_operator"
        )
        self.assertEqual(tuple(create_action.data()["payload"]), ("空分类",))
        create_action.trigger()
        QApplication.processEvents()
        self._cancel_inline_editor()
        self.assertEqual(len(list((self.root / "空分类").glob("*.py"))), 1)
        self.assertEqual(self.library._category_items["空分类"].text(1), "1")

    def test_group_row_context_menu_uses_full_category_path(self) -> None:
        self.library.create_category_named(None, "父分类")
        self.library.create_operator(("父分类", "子分组"))
        self._cancel_inline_editor()
        parent_item = self.library._category_items["父分类"]
        group_item = next(
            parent_item.child(index)
            for index in range(parent_item.childCount())
            if parent_item.child(index).text(0) == "子分组"
        )
        menu = self.library.build_context_menu(group_item)
        payloads = [
            tuple(action.data()["payload"])
            for action in menu.actions()
            if action.data()
        ]
        self.assertEqual(
            payloads, [("父分类", "子分组"), ("父分类", "子分组")]
        )

    def test_user_and_builtin_leaf_context_menu_actions(self) -> None:
        op_id = self.library.create_operator(("基础调整", "颜色"))
        self.assertIsNotNone(op_id)
        self._cancel_inline_editor()
        leaf = self.library._entry_items[op_id]  # noqa: SLF001 - 测试内部定位
        menu = self.library.build_context_menu(leaf)
        self.assertEqual(
            self._actions(menu),
            ["rename_operator", "open_editor", "delete_operator"],
        )
        self.assertEqual(
            [action.text() for action in menu.actions() if action.text()],
            ["重命名", "打开编辑器", "删除算子…"],
        )

        fake_builtin = SimpleNamespace(id="builtin.demo", builtin=True)
        with patch.object(self.library, "_entry", return_value=fake_builtin):
            builtin_menu = self.library.build_context_menu(leaf)
        self.assertEqual(self._actions(builtin_menu), ["show_info"])
        self.assertEqual(builtin_menu.actions()[0].text(), "查看说明")

    def test_inline_editor_commit_renames_file(self) -> None:
        op_id = self.library.create_operator()
        self.assertIsNotNone(op_id)
        assert op_id is not None
        editor = self.library.tree.findChild(QLineEdit, "operatorNameEditor")
        self.assertIsNotNone(editor)
        assert editor is not None
        editor.setText("输入框改名")
        QTest.keyClick(editor, Qt.Key.Key_Return)
        QApplication.processEvents()
        # 提交由 QTimer.singleShot(0) 延后执行，再跑一轮事件循环。
        QApplication.processEvents()
        category_dir = self.root / CUSTOM_OPERATOR_CATEGORY
        self.assertTrue((category_dir / "输入框改名.py").exists())
        self.assertFalse((category_dir / "新建算子.py").exists())
        entry = self.registry.entry(op_id)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.display_name, "输入框改名")
        self.assertIn(op_id, self.library._entry_items)  # noqa: SLF001

    def test_rename_operator_file_updates_display_and_keeps_selection(self) -> None:
        op_id = self.library.create_operator()
        self.assertIsNotNone(op_id)
        assert op_id is not None
        self._cancel_inline_editor()
        self.assertTrue(self.library.rename_operator_to(op_id, "我的算子"))
        old_path = self.root / CUSTOM_OPERATOR_CATEGORY / "新建算子.py"
        new_path = self.root / CUSTOM_OPERATOR_CATEGORY / "我的算子.py"
        self.assertFalse(old_path.exists())
        self.assertTrue(new_path.exists())
        entry = self.registry.entry(op_id)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.display_name, "我的算子")
        self.assertEqual(entry.spec.name, "我的算子")  # 生成模板的 name 行已同步
        item = self.library._entry_items[op_id]  # noqa: SLF001 - 测试内部定位
        self.assertEqual(item.text(0), "我的算子")
        self.assertTrue(item.isSelected())

        # 重名 / 空名只走 status_message，不改文件。
        duplicate = self.root / CUSTOM_OPERATOR_CATEGORY / "占位.py"
        duplicate.write_text(
            _OPERATOR_TEMPLATE.format(
                op_id="test.placeholder",
                name="占位算子",
                category=(CUSTOM_OPERATOR_CATEGORY,),
                description="占位算子",
                keyword="占位",
            ),
            encoding="utf-8",
        )
        self._reload()
        self.assertFalse(self.library.rename_operator_to(op_id, "占位"))
        self.assertTrue(new_path.exists())
        self.assertTrue(any("同名文件已存在" in message for message in self.messages))
        self.assertFalse(self.library.rename_operator_to(op_id, "   "))
        self.assertTrue(any("文件名不能为空" in message for message in self.messages))
        self.assertTrue(new_path.exists())

    def test_rename_legacy_user_operator_keeps_spec_name(self) -> None:
        # 旧的手写用户算子没有 UI_CREATED 标记：改名后显示名仍来自 spec.name。
        self._seed_operator(
            "legacy.py",
            op_id="test.legacy",
            name="旧算子",
            category=(CUSTOM_OPERATOR_CATEGORY, "旧分组"),
        )
        leaf = self.library._entry_items["test.legacy"]  # noqa: SLF001 - 测试内部定位
        self.assertEqual(
            self._actions(self.library.build_context_menu(leaf))[0],
            "rename_operator",
        )
        self.assertTrue(
            self.library.rename_operator_to("test.legacy", "legacy_renamed")
        )
        self.assertTrue((self.root / "legacy_renamed.py").exists())
        entry = self.registry.entry("test.legacy")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.path.stem, "legacy_renamed")
        self.assertEqual(entry.display_name, "旧算子")
        self.assertEqual(
            self.library._entry_items["test.legacy"].text(0),  # noqa: SLF001
            "旧算子",
        )

    def test_delete_operator_requires_confirmation(self) -> None:
        op_id = self.library.create_operator()
        self.assertIsNotNone(op_id)
        assert op_id is not None
        self._cancel_inline_editor()
        path = self.root / CUSTOM_OPERATOR_CATEGORY / "新建算子.py"
        with patch(
            "app.ui.library.QMessageBox.question",
            return_value=QMessageBox.StandardButton.No,
        ):
            self.assertFalse(self.library.delete_operator(op_id))
        self.assertTrue(path.exists())
        with patch(
            "app.ui.library.QMessageBox.question",
            return_value=QMessageBox.StandardButton.Yes,
        ):
            self.assertTrue(self.library.delete_operator(op_id))
        self.assertFalse(path.exists())
        self.assertIsNone(self.registry.entry(op_id))
        self.assertNotIn(op_id, self.library._entry_items)  # noqa: SLF001
        # 分类目录仍然存在（供右键继续新建），空分类继续显示、计数为 0。
        self.assertIn(CUSTOM_OPERATOR_CATEGORY, self.library.category_names())
        self.assertEqual(
            self.library._category_items[CUSTOM_OPERATOR_CATEGORY].text(1), "0"
        )
        self.assertIs(self.library._content_stack.currentWidget(), self.library.tree)

    def test_open_editor_uses_settings_default_editor(self) -> None:
        self.settings.default_editor = "vscode-test"
        op_id = self.library.create_operator()
        self.assertIsNotNone(op_id)
        assert op_id is not None
        self._cancel_inline_editor()
        entry = self.registry.entry(op_id)
        self.assertIsNotNone(entry)
        assert entry is not None
        with patch(
            "app.core.editor.launch_editor", return_value=(True, "")
        ) as launch:
            self.assertTrue(self.library.open_editor(op_id))
        launch.assert_called_once_with(str(entry.path), "vscode-test")
        with patch(
            "app.core.editor.launch_editor",
            return_value=(False, "未找到可用编辑器"),
        ):
            self.assertFalse(self.library.open_editor(op_id))
        self.assertTrue(any("未找到可用编辑器" in item for item in self.messages))
        self.assertFalse(self.library.open_editor("missing.operator"))

    def test_empty_categories_shown_and_user_category_first(self) -> None:
        (self.root / "自定义空分类").mkdir()
        (self.root / CUSTOM_OPERATOR_CATEGORY).mkdir()
        self._reload()
        self.assertEqual(
            self.library.category_names(),
            [CUSTOM_OPERATOR_CATEGORY, "自定义空分类"],
        )
        self.assertEqual(
            [item.text(1) for item in self.library.category_items()],
            ["0", "0"],
        )
        # 没有算子但已知空分类时显示树，而不是「尚未加载算子」空页面。
        self.assertIs(self.library._content_stack.currentWidget(), self.library.tree)

    def test_create_category_named_creates_and_selects_empty_category(self) -> None:
        path = self.library.create_category_named(None, "我的分类")
        self.assertIsNotNone(path)
        assert path is not None
        self.assertTrue(path.is_dir())
        self.assertIn("我的分类", self.library.category_names())
        item = self.library._category_items["我的分类"]
        self.assertEqual(item.text(1), "0")
        self.assertEqual(self.library.current_category(), "我的分类")
        self.assertIs(self.library._content_stack.currentWidget(), self.library.tree)

        child = self.library.create_category_named(("我的分类",), "子分类")
        self.assertIsNotNone(child)
        assert child is not None
        self.assertTrue(child.is_dir())
        # 第二次创建会重建整棵树，旧的 QTreeWidgetItem 已失效，需要重新获取。
        item = self.library._category_items["我的分类"]
        self.library.select_category("我的分类")
        group = item.child(0)
        self.assertEqual(group.text(0), "子分类")
        self.assertEqual(group.text(1), "0")

        self.assertIsNone(self.library.create_category_named(None, "a/b"))
        self.assertTrue(any("新建分类失败" in message for message in self.messages))
        self.assertIsNone(self.library.create_category_named(None, "我的分类"))
        self.assertTrue(any("已存在" in message for message in self.messages))


# ---------------------------------------------------------------------------
# OperatorLibrary：库内拖动整理用户算子 / 用户分类
# ---------------------------------------------------------------------------
_DYNAMIC_CATEGORY_TEMPLATE = '''\
from app.operators.spec import Operator, OperatorSpec

CATEGORY = {category!r}


class _TestOperator(Operator):
    spec = OperatorSpec(
        id="{op_id}",
        name="{name}",
        category=CATEGORY,
        description="动态分类测试算子",
    )

    def process(self, image, **kwargs):
        return image


OPERATOR = _TestOperator
'''

_NO_CATEGORY_TEMPLATE = '''\
from app.operators.spec import Operator, OperatorSpec


class _TestOperator(Operator):
    spec = OperatorSpec(
        id="{op_id}",
        name="{name}",
        description="无分类声明的旧版测试算子",
    )

    def process(self, image, **kwargs):
        return image


OPERATOR = _TestOperator
'''


class OperatorLibraryMoveTests(unittest.TestCase):
    """覆盖算子库内拖动整理的 payload / 移动 / no-op / 失败分支。"""

    def setUp(self) -> None:
        self.root = _workspace_tempdir()
        self.addCleanup(_cleanup_tempdir, self.root)
        self.scan_roots = [self.root]
        self.settings = AppSettings(self.root / "settings.ini")
        self.registry = OperatorRegistry.build(self.scan_roots, include_builtin=False)
        self.library = OperatorLibrary(
            settings=self.settings, user_operator_dir=self.root
        )
        self.library.resize(360, 720)
        self.library.show()
        self.library.set_registry(self.registry)
        self.messages: list[str] = []
        self.library.status_message.connect(self.messages.append)
        self.refreshes = QSignalSpy(self.library.refresh_requested)
        self.library.refresh_requested.connect(self._reload)
        QApplication.processEvents()

    def tearDown(self) -> None:
        self.library.deleteLater()

    def _reload(self) -> None:
        self.registry = OperatorRegistry.build(
            self.scan_roots, include_builtin=False
        )
        self.library.set_registry(self.registry)
        QApplication.processEvents()

    def _seed(
        self,
        relative: str,
        *,
        op_id: str,
        name: str | None = None,
        category=("基础调整",),
        dynamic: bool = False,
    ) -> Path:
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        if dynamic:
            content = _DYNAMIC_CATEGORY_TEMPLATE.format(
                op_id=op_id, name=name or op_id, category=category
            )
        else:
            content = _OPERATOR_TEMPLATE.format(
                op_id=op_id,
                name=name or op_id,
                category=category,
                description="移动整理测试算子",
                keyword="移动",
            )
        path.write_text(content, encoding="utf-8")
        self._reload()
        return path

    def _category_item(self, path):
        item = self.library._category_items[path[0]]  # noqa: SLF001 - 测试内部定位
        for part in path[1:]:
            found = None
            for index in range(item.childCount()):
                child = item.child(index)
                if child.text(0) == part:
                    found = child
                    break
            self.assertIsNotNone(found, f"分类 {path} 缺少子项 {part}")
            item = found
        return item

    def _last_message(self) -> str:
        return self.messages[-1] if self.messages else ""

    # ------------------------------------------------------------ payload / MIME
    def test_user_operator_mime_has_both_formats(self) -> None:
        self._seed("A/op.py", op_id="test.move", name="移动算子", category=("A",))
        item = self.library._entry_items["test.move"]  # noqa: SLF001
        self.assertEqual(
            self.library.library_drag_payload(item),
            {"kind": "operator", "id": "test.move"},
        )
        mime = self.library.mime_data_for_item(item)
        self.assertTrue(mime.hasFormat(MIME_OPERATOR))
        self.assertTrue(mime.hasFormat(MIME_LIBRARY_MOVE))
        self.assertEqual(
            json.loads(bytes(mime.data(MIME_OPERATOR)).decode("utf-8")),
            {"id": "test.move"},
        )
        self.assertEqual(
            json.loads(bytes(mime.data(MIME_LIBRARY_MOVE)).decode("utf-8")),
            {"kind": "operator", "id": "test.move"},
        )
        # 兼容旧接口：mime_data_for 仍只返回 MIME_OPERATOR。
        legacy = self.library.mime_data_for("test.move")
        self.assertTrue(legacy.hasFormat(MIME_OPERATOR))
        self.assertFalse(legacy.hasFormat(MIME_LIBRARY_MOVE))

    def test_category_payload_and_virtual_group_not_draggable(self) -> None:
        self._seed("A/B/op.py", op_id="test.cat", category=("A", "B"))
        self._seed("C/other.py", op_id="test.c", category=("C",))
        self._seed("virtual.py", op_id="test.virtual", category=("虚拟", "分组"))

        group = self._category_item(("A", "B"))
        self.assertEqual(
            self.library.library_drag_payload(group),
            {"kind": "category", "path": ["A", "B"]},
        )
        self.assertTrue(group.flags() & Qt.ItemFlag.ItemIsDragEnabled)
        mime = self.library.mime_data_for_item(group)
        self.assertTrue(mime.hasFormat(MIME_LIBRARY_MOVE))
        self.assertFalse(mime.hasFormat(MIME_OPERATOR))
        self.assertEqual(
            json.loads(bytes(mime.data(MIME_LIBRARY_MOVE)).decode("utf-8")),
            {"kind": "category", "path": ["A", "B"]},
        )
        self.assertEqual(
            self.library.library_drag_payload(self._category_item(("A",))),
            {"kind": "category", "path": ["A"]},
        )

        # 仅由 spec.category 形成的虚拟分组没有对应目录，不可拖动。
        virtual_top = self._category_item(("虚拟",))
        virtual_group = self._category_item(("虚拟", "分组"))
        self.assertIsNone(self.library.library_drag_payload(virtual_top))
        self.assertIsNone(self.library.library_drag_payload(virtual_group))
        self.assertFalse(virtual_top.flags() & Qt.ItemFlag.ItemIsDragEnabled)
        self.assertFalse(virtual_group.flags() & Qt.ItemFlag.ItemIsDragEnabled)

        tree = self.library.current_tree()
        self.assertIsNotNone(tree)
        assert tree is not None
        self.assertEqual(
            tree.dragDropMode(), QAbstractItemView.DragDropMode.DragDrop
        )
        self.assertTrue(tree.acceptDrops())

    def test_start_drag_uses_library_payloads(self) -> None:
        self._seed("A/B/op.py", op_id="test.cat", category=("A", "B"))
        self._seed("virtual.py", op_id="test.virtual", category=("虚拟", "分组"))
        tree = self.library.current_tree()
        self.assertIsNotNone(tree)
        assert tree is not None
        captured: dict[str, object] = {}

        class _FakeDrag:
            def __init__(self, source):
                captured["source"] = source

            def setMimeData(self, mime_data):
                captured["mime"] = mime_data

            def setPixmap(self, pixmap):
                captured["pixmap"] = pixmap

            def setHotSpot(self, point):
                captured["hotspot"] = point

            def exec(self, action):
                captured["action"] = action
                return Qt.DropAction.CopyAction

        leaf = self.library._entry_items["test.cat"]  # noqa: SLF001
        tree.setCurrentItem(leaf)
        with patch("app.ui.library.QDrag", _FakeDrag):
            tree.startDrag(Qt.DropAction.CopyAction)
        mime = captured["mime"]
        self.assertTrue(mime.hasFormat(MIME_OPERATOR))
        self.assertTrue(mime.hasFormat(MIME_LIBRARY_MOVE))

        captured.clear()
        category = self._category_item(("A", "B"))
        tree.setCurrentItem(category)
        with patch("app.ui.library.QDrag", _FakeDrag):
            tree.startDrag(Qt.DropAction.CopyAction)
        mime = captured["mime"]
        self.assertTrue(mime.hasFormat(MIME_LIBRARY_MOVE))
        self.assertFalse(mime.hasFormat(MIME_OPERATOR))

        # 虚拟分组没有库内 payload，也不会发起拖动。
        captured.clear()
        tree.setCurrentItem(self._category_item(("虚拟",)))
        with patch("app.ui.library.QDrag", _FakeDrag):
            tree.startDrag(Qt.DropAction.CopyAction)
        self.assertNotIn("mime", captured)

    def test_can_accept_library_drop(self) -> None:
        self._seed("A/B/op.py", op_id="test.ab", category=("A", "B"))
        self._seed("C/op.py", op_id="test.c", category=("C",))
        op_payload = self.library.library_drag_payload(
            self.library._entry_items["test.ab"]  # noqa: SLF001
        )
        self.assertIsNotNone(op_payload)
        self.assertTrue(
            self.library.can_accept_library_drop(
                op_payload, self._category_item(("C",))
            )
        )
        self.assertTrue(self.library.can_accept_library_drop(op_payload, None))
        self.assertFalse(self.library.can_accept_library_drop(None, None))

        payload = {"kind": "category", "path": ["A", "B"]}
        self.assertTrue(
            self.library.can_accept_library_drop(payload, self._category_item(("C",)))
        )
        self.assertTrue(self.library.can_accept_library_drop(payload, None))
        # 分类不能拖进自身或后代。
        self.assertFalse(
            self.library.can_accept_library_drop(
                payload, self._category_item(("A", "B"))
            )
        )
        self.assertFalse(
            self.library.can_accept_library_drop(
                payload, self.library._entry_items["test.ab"]  # noqa: SLF001
            )
        )
        # 拖到算子行时目标分类 = 该算子所在分类。
        self.assertEqual(
            self.library.drop_target_category(
                self.library._entry_items["test.ab"]  # noqa: SLF001
            ),
            ("A", "B"),
        )
        self.assertFalse(
            self.library.can_accept_library_drop(
                {"kind": "category", "path": ["虚拟"]}, None
            )
        )

    # ------------------------------------------------------------ 移动成功
    def test_move_operator_moves_file_and_selects_new_category(self) -> None:
        old = self._seed(
            "旧分类/op.py",
            op_id="test.move",
            name="移动算子",
            category=("旧分类",),
        )
        before = self.refreshes.count()
        self.assertTrue(
            self.library.move_operator_to_category(
                "test.move", ("基础调整", "颜色")
            )
        )
        self.assertEqual(self.refreshes.count(), before + 1)
        self.assertFalse(old.exists())
        self.assertTrue((self.root / "基础调整" / "颜色" / "op.py").is_file())

        entry = self.registry.entry("test.move")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.spec.category, ("基础调整", "颜色"))
        item = self.library._entry_items["test.move"]  # noqa: SLF001
        self.assertEqual(item.parent().text(0), "颜色")
        self.assertEqual(item.parent().parent().text(0), "基础调整")
        self.assertTrue(item.isSelected())
        self.assertEqual(self.library.current_category(), "基础调整")
        self.assertIn("已移动用户算子", self._last_message())
        self.assertIn("基础调整 / 颜色", self._last_message())

    def test_move_operator_to_blank_moves_to_root(self) -> None:
        old = self._seed("旧分类/op.py", op_id="test.root", category=("旧分类",))
        before = self.refreshes.count()
        self.assertTrue(self.library.move_operator_to_category("test.root", ()))
        self.assertEqual(self.refreshes.count(), before + 1)
        self.assertFalse(old.exists())
        self.assertTrue((self.root / "op.py").is_file())
        self.assertIn("根目录", self._last_message())
        # 根目录下的算子由 loader 归入「其他」。
        self.assertIn("其他", self.library.category_names())

    def test_virtual_category_operator_moves_into_directory(self) -> None:
        # 文件在根目录、但 spec.category 已声明分类：拖到同分类会真实移动文件。
        self._seed("root_level.py", op_id="test.virtual", category=("基础调整",))
        before = self.refreshes.count()
        self.assertTrue(
            self.library.move_operator_to_category("test.virtual", ("基础调整",))
        )
        self.assertEqual(self.refreshes.count(), before + 1)
        self.assertTrue(
            (self.root / "基础调整" / "root_level.py").is_file()
        )

    def test_legacy_operator_without_category_can_move(self) -> None:
        # 旧算子 / 手写算子没有 category 声明：移动到分类后由 loader 按新目录推导。
        legacy = self.root / "legacy.py"
        legacy.write_text(
            _NO_CATEGORY_TEMPLATE.format(op_id="legacy.demo", name="旧版算子"),
            encoding="utf-8",
        )
        self._reload()
        entry = self.registry.entry("legacy.demo")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.spec.category, ("其他",))

        before = self.refreshes.count()
        self.assertTrue(
            self.library.move_operator_to_category(
                "legacy.demo", ("基础调整", "颜色")
            )
        )
        self.assertEqual(self.refreshes.count(), before + 1)
        self.assertFalse(legacy.exists())
        self.assertTrue(
            (self.root / "基础调整" / "颜色" / "legacy.py").is_file()
        )
        entry = self.registry.entry("legacy.demo")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.spec.category, ("基础调整", "颜色"))
        item = self.library._entry_items["legacy.demo"]  # noqa: SLF001
        self.assertTrue(item.isSelected())
        self.assertEqual(item.parent().text(0), "颜色")
        self.assertIn("已移动用户算子", self._last_message())

    def test_operator_can_move_out_of_root_after_blank_drop(self) -> None:
        # 先拖到空白处（根目录，归入「其他」），再拖到别的分类。
        self._seed("A/op.py", op_id="test.roundtrip", category=("A",))
        before = self.refreshes.count()
        self.assertTrue(
            self.library.move_operator_to_category("test.roundtrip", ())
        )
        self.assertEqual(self.refreshes.count(), before + 1)
        self.assertTrue((self.root / "op.py").is_file())
        self.assertIn("其他", self.library.category_names())
        entry = self.registry.entry("test.roundtrip")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.spec.category, ("其他",))
        self.assertEqual(self.library.current_category(), "其他")

        before = self.refreshes.count()
        self.assertTrue(
            self.library.move_operator_to_category("test.roundtrip", ("B",))
        )
        self.assertEqual(self.refreshes.count(), before + 1)
        self.assertFalse((self.root / "op.py").exists())
        self.assertTrue((self.root / "B" / "op.py").is_file())
        entry = self.registry.entry("test.roundtrip")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.spec.category, ("B",))
        item = self.library._entry_items["test.roundtrip"]  # noqa: SLF001
        self.assertTrue(item.isSelected())
        self.assertEqual(item.parent().text(0), "B")
        self.assertEqual(self.library.current_category(), "B")

    def test_move_category_moves_directory_and_children(self) -> None:
        self._seed("A/B/op.py", op_id="test.child", category=("A", "B"))
        self._seed("C/other.py", op_id="test.other", category=("C",))
        before = self.refreshes.count()
        self.assertTrue(self.library.move_category_to_category(("A", "B"), ("C",)))
        self.assertEqual(self.refreshes.count(), before + 1)
        self.assertFalse((self.root / "A" / "B").exists())
        self.assertTrue((self.root / "C" / "B" / "op.py").is_file())
        entry = self.registry.entry("test.child")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.spec.category, ("C", "B"))

        group = self._category_item(("C", "B"))
        self.assertEqual(group.text(1), "1")
        self.assertEqual(self.library.tree.currentItem(), group)
        self.assertEqual(self.library.current_category(), "C")
        message = self._last_message()
        self.assertIn("已移动分类", message)
        self.assertIn("A / B", message)
        self.assertIn("C", message)

        # 分类拖到空白处 = 移到根目录。
        before = self.refreshes.count()
        self.assertTrue(self.library.move_category_to_category(("C", "B"), ()))
        self.assertEqual(self.refreshes.count(), before + 1)
        self.assertTrue((self.root / "B" / "op.py").is_file())
        entry = self.registry.entry("test.child")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.spec.category, ("B",))
        self.assertEqual(
            self.library.tree.currentItem(), self.library._category_items["B"]  # noqa: SLF001
        )
        self.assertIn("根目录", self._last_message())

    def test_move_category_uses_owning_user_root(self) -> None:
        root_a = self.root / "root_a"
        root_b = self.root / "root_b"
        root_a.mkdir()
        (root_b / "A").mkdir(parents=True)
        (root_b / "A" / "op.py").write_text(
            _OPERATOR_TEMPLATE.format(
                op_id="test.multi",
                name="多根算子",
                category=("A",),
                description="多扫描根测试",
                keyword="多根",
            ),
            encoding="utf-8",
        )
        self.scan_roots = [root_a, root_b]
        self._reload()
        before = self.refreshes.count()
        self.assertTrue(
            self.library.move_operator_to_category("test.multi", ("基础调整",))
        )
        self.assertEqual(self.refreshes.count(), before + 1)
        self.assertTrue((root_b / "基础调整" / "op.py").is_file())
        # 不能把算子搬到默认目录 root_a。
        self.assertFalse((root_a / "基础调整").exists())

        before = self.refreshes.count()
        self.assertTrue(self.library.move_category_to_category(("A",), ("基础调整",)))
        self.assertEqual(self.refreshes.count(), before + 1)
        self.assertTrue((root_b / "基础调整" / "A").is_dir())
        self.assertFalse((root_a / "基础调整" / "A").exists())

    def test_move_category_reports_dynamic_category_warnings(self) -> None:
        self._seed("A/static.py", op_id="test.static", category=("A",))
        self._seed(
            "A/dynamic.py",
            op_id="test.dynamic",
            category=("A",),
            dynamic=True,
        )
        before = self.refreshes.count()
        self.assertTrue(self.library.move_category_to_category(("A",), ("B",)))
        self.assertEqual(self.refreshes.count(), before + 1)
        self.assertTrue((self.root / "B" / "A" / "static.py").is_file())
        self.assertTrue((self.root / "B" / "A" / "dynamic.py").is_file())
        message = self._last_message()
        self.assertIn("已移动分类", message)
        self.assertIn("提示：", message)
        self.assertIn("未自动同步", message)

    # ------------------------------------------------------------ no-op
    def test_move_operator_noop_does_not_refresh(self) -> None:
        path = self._seed("A/op.py", op_id="test.same", category=("A",))
        before = self.refreshes.count()
        self.assertTrue(self.library.move_operator_to_category("test.same", ("A",)))
        self.assertEqual(self.refreshes.count(), before)
        self.assertTrue(path.is_file())
        self.assertIn("已在目标分类中", self._last_message())

    def test_move_category_noop_does_not_refresh(self) -> None:
        self._seed("A/B/op.py", op_id="test.same", category=("A", "B"))
        before = self.refreshes.count()
        self.assertTrue(
            self.library.move_category_to_category(("A", "B"), ("A",))
        )
        self.assertEqual(self.refreshes.count(), before)
        self.assertTrue((self.root / "A" / "B" / "op.py").is_file())
        self.assertIn("已在目标分类中", self._last_message())

    # ------------------------------------------------------------ 失败分支
    def test_move_operator_failures_leave_files_untouched(self) -> None:
        self.assertFalse(
            self.library.move_operator_to_category("missing", ("A",))
        )
        self.assertIn("找不到该算子", self._last_message())

        dynamic = self._seed(
            "动态/op.py",
            op_id="test.dynamic",
            category=("动态",),
            dynamic=True,
        )
        before = self.refreshes.count()
        self.assertFalse(
            self.library.move_operator_to_category("test.dynamic", ("目标",))
        )
        self.assertEqual(self.refreshes.count(), before)
        self.assertTrue(dynamic.is_file())
        self.assertFalse((self.root / "目标").exists())
        self.assertIn("移动失败", self._last_message())
        self.assertIn("category 不是字面量", self._last_message())

        source = self._seed("A/dup.py", op_id="test.dup.a", category=("A",))
        target = self._seed("B/dup.py", op_id="test.dup.b", category=("B",))
        target_text = target.read_text(encoding="utf-8")
        before = self.refreshes.count()
        self.assertFalse(
            self.library.move_operator_to_category("test.dup.a", ("B",))
        )
        self.assertEqual(self.refreshes.count(), before)
        self.assertTrue(source.is_file())
        self.assertEqual(target.read_text(encoding="utf-8"), target_text)
        self.assertIn("移动失败", self._last_message())
        self.assertIn("同名", self._last_message())

    def test_move_category_failures_leave_directory_untouched(self) -> None:
        self._seed("A/B/op.py", op_id="test.child", category=("A", "B"))
        self._seed("virtual.py", op_id="test.virtual", category=("虚拟", "分组"))

        before = self.refreshes.count()
        self.assertFalse(
            self.library.move_category_to_category(("虚拟", "分组"), ("A",))
        )
        self.assertEqual(self.refreshes.count(), before)
        self.assertIn("没有对应的用户目录", self._last_message())

        self.assertFalse(
            self.library.move_category_to_category(("A", "B"), ("A", "B"))
        )
        self.assertIn("不能把分类移动到自身或子分类中", self._last_message())
        self.assertFalse(
            self.library.move_category_to_category(("A", "B"), ("A", "B", "C"))
        )
        self.assertIn("不能把分类移动到自身或子分类中", self._last_message())
        self.assertTrue((self.root / "A" / "B" / "op.py").is_file())

        # 目标父级下已有同名目录 → 失败且源目录不动。
        (self.root / "C" / "B").mkdir(parents=True)
        before = self.refreshes.count()
        self.assertFalse(self.library.move_category_to_category(("A", "B"), ("C",)))
        self.assertEqual(self.refreshes.count(), before)
        self.assertTrue((self.root / "A" / "B" / "op.py").is_file())
        self.assertIn("移动失败", self._last_message())
        self.assertIn("同名", self._last_message())

    def test_builtin_operator_and_category_not_movable(self) -> None:
        builtin_dir = _workspace_tempdir()
        self.addCleanup(_cleanup_tempdir, builtin_dir)
        # 内置分类目录：分类行来自磁盘目录，而不是仅由 spec.category 形成。
        (builtin_dir / "内置分类").mkdir()
        (builtin_dir / "内置分类" / "builtin_op.py").write_text(
            _OPERATOR_TEMPLATE.format(
                op_id="builtin.demo",
                name="内置示例",
                category=("内置分类",),
                description="内置测试算子",
                keyword="内置",
            ),
            encoding="utf-8",
        )
        with patch("app.operators.loader.BUILTIN_DIR", builtin_dir):
            self.registry = OperatorRegistry.build(
                [self.root], include_builtin=True
            )
            self.library.set_registry(self.registry)
            QApplication.processEvents()

            entry = self.registry.entry("builtin.demo")
            self.assertIsNotNone(entry)
            assert entry is not None
            self.assertTrue(entry.builtin)
            leaf = self.library._entry_items["builtin.demo"]  # noqa: SLF001
            self.assertIsNone(self.library.library_drag_payload(leaf))
            mime = self.library.mime_data_for_item(leaf)
            self.assertTrue(mime.hasFormat(MIME_OPERATOR))
            self.assertFalse(mime.hasFormat(MIME_LIBRARY_MOVE))

            category_item = self._category_item(("内置分类",))
            self.assertIsNone(self.library.library_drag_payload(category_item))
            self.assertFalse(category_item.flags() & Qt.ItemFlag.ItemIsDragEnabled)

            before = self.refreshes.count()
            self.assertFalse(
                self.library.move_operator_to_category("builtin.demo", ("用户分类",))
            )
            self.assertFalse(
                self.library.move_category_to_category(("内置分类",), ("目标",))
            )
            self.assertEqual(self.refreshes.count(), before)
            self.assertTrue(
                (builtin_dir / "内置分类" / "builtin_op.py").is_file()
            )
            self.assertFalse((self.root / "用户分类").exists())
            self.assertFalse((self.root / "目标").exists())
            self.assertTrue(
                any("内置算子不能移动" in message for message in self.messages)
            )
            self.assertTrue(
                any("内置分类不能移动" in message for message in self.messages)
            )


# ---------------------------------------------------------------------------
# ReplaceOperatorDialog
# ---------------------------------------------------------------------------
class ReplaceOperatorDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.root = _workspace_tempdir()
        cls.addClassCleanup(_cleanup_tempdir, cls.root)
        _build_group_fixture(cls.root)
        cls.registry = OperatorRegistry.build([cls.root], include_builtin=False)

    def setUp(self) -> None:
        self.dialog = ReplaceOperatorDialog(self.registry)

    def tearDown(self) -> None:
        self.dialog.deleteLater()

    def test_lists_non_group_operators_by_category(self) -> None:
        self.assertEqual(
            self.dialog.category_names(),
            [CUSTOM_OPERATOR_CATEGORY, "基础调整", "滤波与平滑"],
        )
        self.assertEqual(
            self.dialog.operator_ids(),
            ["test.user", "test.brightness", "test.blur"],
        )
        self.assertNotIn("group.demo", self.dialog.operator_ids())
        self.assertIsNone(self.dialog.item_for("group.demo"))
        self.assertEqual(self.dialog.category_names()[0], CUSTOM_OPERATOR_CATEGORY)

    def test_search_filter_and_refresh(self) -> None:
        self.dialog.show()
        QApplication.processEvents()
        self.assertEqual(self.dialog.filter("高斯"), 1)
        self.assertEqual(self.dialog.matched_operator_ids(), ["test.blur"])
        self.assertFalse(self.dialog.empty_label.isVisible())

        self.assertEqual(self.dialog.filter("不存在的关键字"), 0)
        self.assertEqual(self.dialog.matched_operator_ids(), [])
        self.assertTrue(self.dialog.empty_label.isVisible())

        self.assertEqual(self.dialog.filter(""), 3)
        self.assertEqual(
            set(self.dialog.matched_operator_ids()),
            {"test.user", "test.brightness", "test.blur"},
        )

        # refresh 后重新应用当前搜索关键字。
        self.dialog.filter("模糊")
        self.dialog.refresh()
        self.assertEqual(self.dialog.matched_operator_ids(), ["test.blur"])
        self.dialog.filter("")
        self.assertEqual(len(self.dialog.matched_operator_ids()), 3)

    def test_selected_operator_id_on_accept_and_reject(self) -> None:
        self.assertIsNone(self.dialog.selected_op_id())
        self.assertFalse(self.dialog.ok_button.isEnabled())
        self.assertFalse(self.dialog.select_operator("group.demo"))

        self.assertTrue(self.dialog.select_operator("test.blur"))
        self.assertEqual(self.dialog.selected_op_id(), "test.blur")
        self.assertTrue(self.dialog.ok_button.isEnabled())
        self.dialog.accept()
        self.assertEqual(self.dialog.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(self.dialog.selected_op_id(), "test.blur")

        other = ReplaceOperatorDialog(self.registry)
        self.addCleanup(other.deleteLater)
        self.assertTrue(other.select_operator("test.user"))
        other.reject()
        self.assertEqual(other.result(), QDialog.DialogCode.Rejected)
        self.assertIsNone(other.selected_op_id())

    def test_double_click_accepts_selection(self) -> None:
        item = self.dialog.item_for("test.blur")
        self.assertIsNotNone(item)
        assert item is not None
        self.dialog.tree.itemDoubleClicked.emit(item, 0)
        self.assertEqual(self.dialog.result(), QDialog.DialogCode.Accepted)
        self.assertEqual(self.dialog.selected_op_id(), "test.blur")

    def test_current_group_operator_allowed_in_list(self) -> None:
        dialog = ReplaceOperatorDialog(self.registry, current_op_id="group.demo")
        self.addCleanup(dialog.deleteLater)
        self.assertIn("group.demo", dialog.operator_ids())
        self.assertIn("当前算子", dialog.current_label.text())
        current_item = dialog.item_for("group.demo")
        self.assertIsNotNone(current_item)
        assert current_item is not None
        self.assertEqual(current_item.text(1), "当前")

    def test_exec_returns_selected_or_none(self) -> None:
        self.assertTrue(self.dialog.select_operator("test.blur"))
        QTimer.singleShot(0, self.dialog.accept)
        result = self.dialog.exec()
        self.assertEqual(result, QDialog.DialogCode.Accepted)
        self.assertEqual(self.dialog.selected_op_id(), "test.blur")

        rejected = ReplaceOperatorDialog(self.registry)
        self.addCleanup(rejected.deleteLater)
        QTimer.singleShot(0, rejected.reject)
        result = rejected.exec()
        self.assertEqual(result, QDialog.DialogCode.Rejected)
        self.assertIsNone(rejected.selected_op_id())


# ---------------------------------------------------------------------------
# SettingsDialog
# ---------------------------------------------------------------------------
class SettingsDialogTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpdir = _workspace_tempdir()
        self.addCleanup(_cleanup_tempdir, self.tmpdir)
        self.ini_path = self.tmpdir / "settings.ini"

    def test_accept_persists_all_values(self) -> None:
        settings = AppSettings(self.ini_path)
        dialog = SettingsDialog(settings)
        try:
            spy = QSignalSpy(dialog.applied)
            dialog.undo_spin.setValue(7)
            dialog.workers_spin.setValue(3)
            dialog.thumbnails_check.setChecked(False)
            dialog.auto_fit_check.setChecked(False)
            dialog.downsample_check.setChecked(True)
            dialog.max_dim_spin.setValue(1024)
            dialog.cursor_info_check.setChecked(False)
            dialog.builtin_check.setChecked(False)
            dialog.watch_check.setChecked(True)
            directory = str(self.tmpdir / "operators")
            item = QListWidgetItem(directory)
            item.setData(Qt.ItemDataRole.UserRole, directory)
            dialog.dirs_list.addItem(item)
            dialog.accept()
            self.assertEqual(spy.count(), 1)

            reloaded = AppSettings(self.ini_path)
            self.assertEqual(reloaded.undo_limit, 7)
            self.assertEqual(reloaded.execution_workers, 3)
            self.assertFalse(reloaded.show_node_thumbnails)
            self.assertFalse(reloaded.auto_fit_on_open)
            self.assertTrue(reloaded.downsample_preview)
            self.assertEqual(reloaded.downsample_max_dim, 1024)
            self.assertFalse(reloaded.cursor_pixel_info)
            self.assertFalse(reloaded.builtin_operators_enabled)
            self.assertTrue(reloaded.watch_operator_dirs)
            self.assertEqual(reloaded.extra_operator_dirs, [directory])
        finally:
            dialog.deleteLater()

    def test_cancel_does_not_persist(self) -> None:
        settings = AppSettings(self.ini_path)
        default_workers = settings.execution_workers
        dialog = SettingsDialog(settings)
        try:
            dialog.undo_spin.setValue(9)
            dialog.workers_spin.setValue(5)
            dialog.downsample_check.setChecked(True)
            dialog.reject()
            reloaded = AppSettings(self.ini_path)
            self.assertEqual(reloaded.undo_limit, 100)
            self.assertEqual(reloaded.execution_workers, default_workers)
            self.assertFalse(reloaded.downsample_preview)
        finally:
            dialog.deleteLater()

    def test_validation_clamps_values(self) -> None:
        settings = AppSettings(self.ini_path)
        dialog = SettingsDialog(settings)
        try:
            dialog.undo_spin.setValue(0)
            dialog.workers_spin.setValue(0)
            dialog.max_dim_spin.setValue(10)
            self.assertGreaterEqual(dialog.undo_spin.value(), 1)
            self.assertGreaterEqual(dialog.workers_spin.value(), 1)
            self.assertGreaterEqual(dialog.max_dim_spin.value(), 256)
            dialog.accept()
            reloaded = AppSettings(self.ini_path)
            self.assertGreaterEqual(reloaded.undo_limit, 1)
            self.assertGreaterEqual(reloaded.downsample_max_dim, 256)
        finally:
            dialog.deleteLater()

    def test_default_editor_options_and_persist(self) -> None:
        settings = AppSettings(self.ini_path)
        detected = [EditorInfo("vscode", "VS Code", r"C:\tools\code.cmd")]
        with patch("app.core.editor.detect_editors", return_value=detected):
            dialog = SettingsDialog(settings)
        try:
            combo = dialog.editor_combo
            data = [combo.itemData(index) for index in range(combo.count())]
            self.assertEqual(data[:2], ["auto", "system"])
            self.assertIn("vscode", data)
            text = combo.itemText(data.index("vscode"))
            self.assertIn("VS Code", text)
            self.assertIn("code.cmd", text)
            combo.setCurrentIndex(combo.findData("vscode"))
            dialog.accept()
            reloaded = AppSettings(self.ini_path)
            self.assertEqual(reloaded.default_editor, "vscode")
        finally:
            dialog.deleteLater()

    def test_default_editor_keeps_undetected_value_and_redetects(self) -> None:
        settings = AppSettings(self.ini_path)
        settings.default_editor = r"path:C:\tools\notepad.exe"
        with patch("app.core.editor.detect_editors", return_value=[]):
            dialog = SettingsDialog(settings)
        try:
            combo = dialog.editor_combo
            index = combo.findData(r"path:C:\tools\notepad.exe")
            self.assertGreaterEqual(index, 0)
            self.assertIn("自定义", combo.itemText(index))

            detected = [
                EditorInfo(
                    "pycharm", "PyCharm", r"C:\pycharm\bin\pycharm64.exe"
                )
            ]
            with patch("app.core.editor.detect_editors", return_value=detected):
                dialog.redetect_button.click()
            data = [combo.itemData(index) for index in range(combo.count())]
            self.assertIn("pycharm", data)
            # 重新检测不改变当前选择。
            self.assertEqual(combo.currentData(), r"path:C:\tools\notepad.exe")
            dialog.accept()
            self.assertEqual(
                AppSettings(self.ini_path).default_editor,
                r"path:C:\tools\notepad.exe",
            )
        finally:
            dialog.deleteLater()


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
class ThemeTests(unittest.TestCase):
    def test_apply_theme_sets_light_palette_and_qss(self) -> None:
        app = _app()
        apply_theme(app)
        stylesheet = app.styleSheet()
        self.assertIn("QSplitter::handle", stylesheet)
        self.assertIn("QTabBar::tab:west", stylesheet)
        self.assertIn("QTreeWidget", stylesheet)
        self.assertIn("QToolTip", stylesheet)
        self.assertGreater(len(stylesheet), 1000)
        window_color = app.palette().color(QPalette.ColorRole.Window)
        self.assertGreater(window_color.lightness(), 200)
        self.assertGreater(app.font().pointSizeF(), 0.0)
        self.assertEqual(app.property("mineimageTheme"), "light")


if __name__ == "__main__":
    unittest.main()
