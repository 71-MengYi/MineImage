"""主窗口 / 工作区的端到端离屏测试（不弹对话框）。"""

from __future__ import annotations

import os
import time
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.config import AppSettings
from app.core.imageio import save_image
from app.core.types import ImageData
from app.ui.main_window import MainWindow
from app.ui.theme import apply_theme

_TMP = Path(__file__).resolve().parent.parent / ".test_tmp"
_TMP.mkdir(exist_ok=True)
_APP = QApplication.instance() or QApplication([])
apply_theme(_APP)


def _sample_image(path: Path) -> Path:
    height, width = 180, 240
    yy, xx = np.mgrid[0:height, 0:width]
    bgr = np.dstack(
        [
            (xx * 255 // max(1, width - 1)).astype(np.uint8),
            (yy * 255 // max(1, height - 1)).astype(np.uint8),
            np.full_like(xx, 90, dtype=np.uint8),
        ]
    )
    save_image(path, ImageData(bgr, "input"))
    return path


class MainWindowIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.image_path = _sample_image(_TMP / "integration_input.png")

    def setUp(self) -> None:  # noqa: N802
        self.win = MainWindow(AppSettings(_TMP / f"integration_{uuid.uuid4().hex[:8]}.ini"))
        self.win.resize(1480, 900)
        self.win.show()
        _APP.processEvents()

    def tearDown(self) -> None:  # noqa: N802
        for widget in list(self.win._workspaces):
            try:
                widget.mark_clean()
            except Exception:  # noqa: BLE001
                pass
        self.win.close()
        self.win.deleteLater()
        _APP.processEvents()

    def _open(self) -> object:
        self.win.open_image_path(str(self.image_path))
        widget = self.win._active_workspace()
        self.assertIsNotNone(widget)
        self._wait_idle(widget)
        return widget

    def _wait_idle(self, widget, timeout: float = 12.0) -> None:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            _APP.processEvents()
            if widget.engine.busy:
                QTest.qWait(15)
                continue
            _APP.processEvents()
            QTest.qWait(60)
            if not widget.engine.busy:
                return
        self.fail("后台计算超时")

    def test_open_image_creates_workspace_and_preview(self) -> None:
        widget = self._open()
        self.assertEqual(self.win.tabs.count(), 1)
        self.assertEqual(len(widget.workspace.graph.source_nodes()), 1)
        self.assertIsNotNone(widget.engine.result(widget._current_node_id))
        self.assertIsNotNone(widget.preview.image_size())
        self.assertIn("节点:", self.win._status_labels["node"].text())
        self.assertNotEqual(self.win._status_labels["size"].text(), "图像: --")
        self.assertIsNotNone(widget.graph_view.items() if hasattr(widget.graph_view, "items") else True)
        # 初始三栏比例为 1:1:4（算子库 : 画布 : 预览）
        sizes = widget.splitter.sizes()
        self.assertAlmostEqual(sizes[0], sizes[1], delta=max(8, sizes[0] * 0.1))
        self.assertAlmostEqual(sizes[2], sizes[0] * 4, delta=max(24, sizes[0] * 0.4))
        self.assertGreater(sizes[2], sizes[0] * 3)
        self.assertGreater(sizes[2], sizes[1] * 3)

    def test_copy_paste_nodes_in_workspace(self) -> None:
        widget = self._open()
        graph = widget.workspace.graph
        source = graph.source_nodes()[0]
        node_id = widget.graph_view.add_operator_at("basic.grayscale")
        self.assertIsNotNone(node_id)
        graph.add_connection(source.id, "image", node_id, "image")
        widget.graph_view.set_selected_node(node_id)

        self.assertTrue(widget.graph_view.copy_selected())
        new_ids = widget.graph_view.paste_nodes()
        self.assertEqual(len(new_ids), 1)
        self.assertIn(new_ids[0], graph.nodes)
        self.assertNotEqual(new_ids[0], node_id)
        self.assertTrue(widget.dirty)
        self._wait_idle(widget)

        widget.undo_stack.undo()
        self.assertNotIn(new_ids[0], graph.nodes)
        self.assertIn(node_id, graph.nodes)

    def test_moving_node_does_not_recompute_cached_result(self) -> None:
        widget = self._open()
        graph = widget.workspace.graph
        source = graph.source_nodes()[0]
        node_id = widget.graph_view.add_operator_at("basic.grayscale")
        self.assertIsNotNone(node_id)
        graph.add_connection(source.id, "image", node_id, "image")
        widget._on_node_selected(node_id)
        self._wait_idle(widget)
        before = widget.engine.result(node_id)
        self.assertIsNotNone(before)

        item = widget.graph_view._nodes[node_id]
        start = item.pos()
        item.setSelected(True)
        widget.graph_view.begin_move_tracking()
        item.setPos(start.x() + 120.0, start.y() + 40.0)
        widget.graph_view.finish_move_tracking()
        QTest.qWait(100)
        self._wait_idle(widget)

        # 纯位置移动不应清空节点快照，也不应触发重算。
        self.assertIs(widget.engine.result(node_id), before)
        self.assertFalse(widget.engine.busy)

        # 撤销移动同样只改变位置。
        widget.undo_stack.undo()
        self._wait_idle(widget)
        self.assertIs(widget.engine.result(node_id), before)

    def test_add_operator_connect_params_undo_and_project(self) -> None:
        widget = self._open()
        graph = widget.workspace.graph
        source = graph.source_nodes()[0]
        op_id = widget.graph_view.add_operator_at("basic.grayscale")
        self.assertIsNotNone(op_id)
        node = graph.nodes[op_id]
        self.assertEqual(node.params.get("mode"), "weighted")
        graph.add_connection(source.id, "image", node.id, "image")
        widget._on_node_selected(op_id)
        self._wait_idle(widget)
        self.assertIsNotNone(widget.engine.result(op_id))
        self.assertEqual(widget.center_stack.currentWidget(), widget.graph_view)

        # 双击进入参数面板
        widget.open_parameters(op_id)
        self.assertEqual(widget.center_stack.currentWidget(), widget.param_panel)
        combo = widget.param_panel.field_widget("mode")
        self.assertIsNotNone(combo)
        combo.setCurrentIndex(1)  # weighted -> average
        QTest.qWait(300)
        self._wait_idle(widget)
        self.assertEqual(node.params["mode"], "average")
        self.assertTrue(widget.undo_stack.canUndo())

        # 保存 -> 下游计算（此处只有当前节点）
        widget.param_panel.save_button.click()
        QTest.qWait(200)
        self._wait_idle(widget)
        self.assertEqual(widget.center_stack.currentWidget(), widget.param_panel)
        widget.back_to_graph()
        self.assertEqual(widget.center_stack.currentWidget(), widget.graph_view)

        # 撤销参数会话
        widget.undo_stack.undo()
        self._wait_idle(widget)
        self.assertEqual(graph.nodes[op_id].params.get("mode"), "weighted")

        # 保存工程并重新打开
        project = _TMP / f"integration_{uuid.uuid4().hex[:8]}.mi"
        self.addCleanup(lambda: project.unlink(missing_ok=True))
        widget.save_to(project)
        self.assertTrue(project.exists())
        self.assertFalse(widget.dirty)
        self.win.open_project_path(project)
        loaded = self.win._active_workspace()
        self.assertEqual(self.win.tabs.count(), 2)
        self.assertEqual(loaded.workspace.name, widget.workspace.name)
        self.assertEqual(len(loaded.workspace.graph.nodes), 2)
        self.assertEqual(len(loaded.workspace.images), 1)
        self.assertFalse(loaded.dirty)
        self._wait_idle(loaded)
        self.assertIsNotNone(loaded._current_node_id)
        self.assertIsNotNone(loaded.engine.result(loaded._current_node_id))
        loaded.mark_clean()

    def test_interactive_roi_pick_updates_params_and_preview(self) -> None:
        from PySide6.QtCore import QPoint, Qt
        from PySide6.QtWidgets import QPushButton

        widget = self._open()
        graph = widget.workspace.graph
        source = graph.source_nodes()[0]
        op_id = widget.graph_view.add_operator_at("interactive.roi_crop")
        node = graph.nodes[op_id]
        graph.add_connection(source.id, "image", node.id, "image")
        widget._on_node_selected(op_id)
        self._wait_idle(widget)
        # 默认 ROI 为空，算子会报错；预览应能回退到上游图供拾取
        self.assertIsNotNone(widget.engine.error(op_id))
        widget.open_parameters(op_id)
        pick = next(
            (button for button in widget.param_panel.findChildren(QPushButton) if button.text() == "拾取"),
            None,
        )
        self.assertIsNotNone(pick)
        pick.click()
        _APP.processEvents()
        self.assertIsNotNone(widget._active_pick)
        preview = widget.preview
        start = QPoint(int(preview.width() * 0.40), int(preview.height() * 0.40))
        end = QPoint(int(preview.width() * 0.68), int(preview.height() * 0.68))
        QTest.mousePress(preview, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, start)
        QTest.mouseMove(preview, end, 30)
        QTest.mouseRelease(preview, Qt.MouseButton.LeftButton, Qt.KeyboardModifier.NoModifier, end)
        QTest.qWait(350)
        self._wait_idle(widget)
        roi = node.params.get("roi")
        self.assertIsInstance(roi, list)
        self.assertEqual(len(roi), 4)
        self.assertGreater(roi[2], 1)
        self.assertGreater(roi[3], 1)
        self.assertIsNone(widget.engine.error(op_id))
        result = widget.engine.result(op_id)
        self.assertIsNotNone(result)
        self.assertEqual(widget.preview.image_size(), (result.primary.width, result.primary.height))

    def test_export_current_node_result(self) -> None:
        from app.core.imageio import load_image

        widget = self._open()
        out = _TMP / "integration_export.png"
        self.addCleanup(lambda: out.unlink(missing_ok=True))
        self.assertTrue(widget.export_current_image(out))
        self.assertTrue(out.exists())
        loaded = load_image(out)
        self.assertEqual(loaded.array.shape[2], 3)

    def test_missing_input_marks_node_error(self) -> None:
        widget = self._open()
        op_id = widget.graph_view.add_operator_at("filter.gaussian_blur")
        widget._on_node_selected(op_id)
        self._wait_idle(widget)
        error = widget.engine.error(op_id)
        self.assertIsNotNone(error)
        self.assertIn("缺少输入", error.message)
        self.assertIsNone(widget.engine.result(op_id))

    def test_library_info_panel_and_compare_modes(self) -> None:
        widget = self._open()
        self.assertTrue(widget.library.show_operator_info("basic.grayscale"))
        _APP.processEvents()
        self.assertTrue(widget.library.info_panel_visible())
        geometry = widget.library.info_panel.geometry()
        self.assertGreater(geometry.width(), 200)
        self.assertGreater(geometry.y(), 0)
        widget.library._hide_info_panel()
        self.assertFalse(widget.library.info_panel_visible())

        widget.set_compare_mode("side_by_side")
        widget.preview.grab()
        widget.set_compare_mode("swipe")
        widget.preview.grab()
        widget.set_compare_mode("single")
        self.assertEqual(widget.compare_mode, "single")

    def test_preview_only_recomputes_current_node_until_save(self) -> None:
        widget = self._open()
        graph = widget.workspace.graph
        source = graph.source_nodes()[0]
        gray_id = widget.graph_view.add_operator_at("basic.grayscale")
        blur_id = widget.graph_view.add_operator_at("filter.gaussian_blur")
        gray = graph.nodes[gray_id]
        graph.add_connection(source.id, "image", gray.id, "image")
        graph.add_connection(gray.id, "image", blur_id, "image")
        widget.graph_view.auto_layout()
        widget._on_node_selected(blur_id)
        self._wait_idle(widget)
        before = widget.engine.result(blur_id)
        self.assertIsNotNone(before)
        before_array = before.primary.array.copy()

        widget.open_parameters(gray_id)
        combo = widget.param_panel.field_widget("mode")
        combo.setCurrentIndex(1)  # weighted -> average
        QTest.qWait(300)
        self._wait_idle(widget)
        self.assertEqual(gray.params["mode"], "average")
        # 只重算当前节点：下游 blur 的快照必须保持不变
        after_preview = widget.engine.result(blur_id)
        self.assertIs(after_preview, before)
        np.testing.assert_array_equal(after_preview.primary.array, before_array)

        widget.param_panel.save_button.click()
        QTest.qWait(200)
        self._wait_idle(widget)
        after_save = widget.engine.result(blur_id)
        self.assertIsNotNone(after_save)
        self.assertFalse(np.array_equal(after_save.primary.array, before_array))

    def test_high_bit_depth_and_float_sources(self) -> None:
        gray16 = (np.arange(64 * 96, dtype=np.uint16).reshape(64, 96) * 13).astype(np.uint16)
        path16 = _TMP / "integration_16.png"
        save_image(path16, ImageData(gray16))
        self.addCleanup(lambda: path16.unlink(missing_ok=True))
        self.win.open_image_path(str(path16))
        widget = self.win._active_workspace()
        self._wait_idle(widget)
        source_id = widget._current_node_id
        result = widget.engine.result(source_id)
        self.assertIsNotNone(result)
        self.assertEqual(result.primary.array.dtype, np.uint16)
        self.assertIsNotNone(widget.preview.image_size())

        float_img = (np.linspace(0, 1, 80 * 120, dtype=np.float32).reshape(80, 120))
        pathf = _TMP / "integration_float.npy"
        np.save(pathf, float_img)
        self.addCleanup(lambda: pathf.unlink(missing_ok=True))
        self.win.open_image_path(str(pathf))
        widget = self.win._active_workspace()
        self._wait_idle(widget)
        result = widget.engine.result(widget._current_node_id)
        self.assertIsNotNone(result)
        self.assertEqual(result.primary.array.dtype, np.float32)
        self.assertIsNotNone(widget.preview.image_size())

    def test_auto_layout_and_undo_redo_actions(self) -> None:
        widget = self._open()
        graph = widget.workspace.graph
        source = graph.source_nodes()[0]
        blur = widget.graph_view.add_operator_at("filter.gaussian_blur")
        graph.add_connection(source.id, "image", blur, "image")
        widget.graph_view.auto_layout()
        positions = [node.pos for node in graph.nodes.values()]
        self.assertTrue(all(np.isfinite(x) and np.isfinite(y) for x, y in positions))
        self.assertGreaterEqual(len({round(y, 1) for _x, y in positions}), 2)
        self.assertTrue(self.win.act_undo.isEnabled())
        self.win.undo()
        self._wait_idle(widget)
        self.assertTrue(self.win.act_redo.isEnabled())


if __name__ == "__main__":
    unittest.main()
