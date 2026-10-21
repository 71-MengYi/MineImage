"""异步执行引擎与算子扫描的单元测试。

运行方式（仓库根目录）::

    python -m unittest discover -s tests -p "test_*.py" -v

测试全部使用 ``QT_QPA_PLATFORM=offscreen``，只创建 QCoreApplication，不启动
GUI 事件循环（不使用 ``app.exec()``）。
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtWidgets import QApplication

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import AppSettings  # noqa: E402
from app.core.constants import CATEGORY_ORDER  # noqa: E402
from app.core.model import WorkspaceModel  # noqa: E402
from app.core.types import ImageData  # noqa: E402
from app.engine.engine import ExecutionEngine, NodeResult  # noqa: E402
from app.operators.loader import OperatorRegistry  # noqa: E402

FIXTURES = ROOT / "tests" / "fixtures"
OPS_FIXTURES = FIXTURES / "ops"


def _qapp() -> QApplication:
    # 使用 QApplication（而不仅是 QCoreApplication）：全量 unittest discovery
    # 时同一个进程里还有 QWidget 测试，QCoreApplication 会阻止它们创建控件。
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def _image(height: int = 24, width: int = 32, channels: int = 3, dtype=np.uint8) -> np.ndarray:
    rng = np.random.default_rng(20240601)
    if channels == 1:
        array = rng.integers(0, 256, (height, width), dtype=np.uint8)
    else:
        array = rng.integers(0, 256, (height, width, channels), dtype=np.uint8)
    if dtype == np.uint16:
        return (array.astype(np.uint16) * 257).astype(np.uint16)
    if dtype == np.float32:
        return (array.astype(np.float32) / 255.0).astype(np.float32)
    return array


class EngineTestCase(unittest.TestCase):
    """提供隔离的工作区 / 引擎 / 事件收集工具。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = _qapp()

    def setUp(self) -> None:
        # 使用仓库内预置的 fixtures 目录：运行期不再创建目录（沙箱可能禁止）。
        self.settings = AppSettings(FIXTURES / "settings.ini")
        self.registry = OperatorRegistry.build()
        self.workspace = WorkspaceModel("测试工作区")
        self.engine = ExecutionEngine(self.workspace, self.registry, self.settings)
        self.addCleanup(self.engine.shutdown)

        self.chain_events: list[tuple[str, bool]] = []
        self.finished: list[NodeResult] = []
        self.failures: list[tuple[str, str]] = []
        self.started: list[str] = []
        self.busy_events: list[bool] = []
        self.engine.chain_finished.connect(lambda node_id, ok: self.chain_events.append((node_id, ok)))
        self.engine.node_finished.connect(self.finished.append)
        self.engine.node_failed.connect(lambda node_id, msg: self.failures.append((node_id, msg)))
        self.engine.node_started.connect(self.started.append)
        self.engine.busy_changed.connect(self.busy_events.append)

    # ------------------------------------------------------------------ 工具
    def pump(self, predicate=None, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.app.processEvents()
            if predicate is not None and predicate():
                return True
            time.sleep(0.005)
        self.app.processEvents()
        return predicate() if predicate is not None else True

    def wait_chain(self, target_id: str, timeout: float = 10.0) -> bool:
        reached = self.pump(
            lambda: any(node_id == target_id for node_id, _ok in self.chain_events),
            timeout,
        )
        self.assertTrue(reached, f"等待 {target_id} 的 chain_finished 超时；事件={self.chain_events}")
        return next(ok for node_id, ok in reversed(self.chain_events) if node_id == target_id)

    def add_source(self, array: np.ndarray, name: str = "源图像"):
        image = ImageData(array, name)
        return self.workspace.add_source_image(image)

    def add_op(self, op_id: str, title: str, params: dict | None = None):
        return self.workspace.graph.create_operator_node(op_id, title, params or {})

    def connect(self, src, dst, src_port: str = "image", dst_port: str = "image") -> None:
        self.workspace.graph.add_connection(src.id, src_port, dst.id, dst_port)

    def build_chain(self):
        """source -> 灰度化 -> 高斯模糊。"""
        source = self.add_source(_image())
        gray = self.add_op("basic.grayscale", "灰度化", {"mode": "weighted"})
        blur = self.add_op(
            "filter.gaussian_blur",
            "高斯模糊",
            {"ksize": 5, "sigma": 0.0, "border": "replicate"},
        )
        self.connect(source, gray)
        self.connect(gray, blur)
        return source, gray, blur


class ExecutionEngineTests(EngineTestCase):
    # ------------------------------------------------------------------ 基本链路
    def test_source_gray_blur_chain(self) -> None:
        source, gray, blur = self.build_chain()
        self.engine.preview_node(blur.id)
        self.assertTrue(self.wait_chain(blur.id))

        blur_result = self.engine.result(blur.id)
        gray_result = self.engine.result(gray.id)
        source_result = self.engine.result(source.id)
        self.assertIsNotNone(blur_result)
        self.assertIsNotNone(gray_result)
        self.assertIsNotNone(source_result)
        self.assertEqual(blur_result.primary.array.shape, (_image().shape[0], _image().shape[1]))
        self.assertEqual(gray_result.primary.array.shape[:2], blur_result.primary.array.shape[:2])
        self.assertGreater(blur_result.elapsed_ms, 0.0)
        self.assertEqual(blur_result.revision, blur.revision)
        self.assertFalse(self.engine.busy)
        self.assertTrue(self.busy_events)
        self.assertTrue(self.busy_events[0])
        self.assertFalse(self.busy_events[-1])
        self.assertTrue(all(ok for _node, ok in self.chain_events))

    def test_upstream_recursion_computes_whole_chain(self) -> None:
        source, gray, blur = self.build_chain()
        self.engine.refresh_node(blur.id)
        self.assertTrue(self.wait_chain(blur.id))
        self.assertIsNotNone(self.engine.result(gray.id))
        self.assertIsNotNone(self.engine.result(source.id))
        self.assertIn(gray.id, self.started)
        self.assertEqual(self.started[-1], blur.id)
        self.assertLess(self.started.index(gray.id), self.started.index(blur.id))

    # ------------------------------------------------------------------ 缓存
    def test_snapshot_reuse_is_synchronous_and_does_not_rerun(self) -> None:
        _source, gray, blur = self.build_chain()
        self.engine.preview_node(blur.id)
        self.assertTrue(self.wait_chain(blur.id))

        cached = self.engine.result(gray.id)
        self.finished.clear()
        self.chain_events.clear()
        self.started.clear()
        self.engine.preview_node(gray.id)

        self.assertEqual(len(self.finished), 1)
        self.assertIs(self.finished[0], cached)
        self.assertEqual(self.chain_events, [])
        self.assertEqual(self.started, [])
        self.assertFalse(self.engine.busy)

    def test_preview_cached_node_while_busy_does_not_cancel_running_job(self) -> None:
        slow = self._build_slow_node(delay=0.3)
        source_id = self.workspace.graph.predecessors(slow.id)[0]
        self.engine.preview_node(slow.id)
        self.assertTrue(self.pump(lambda: slow.id in self.started, timeout=5.0))
        self.assertIsNotNone(self.engine.result(source_id))
        self.assertTrue(self.engine.busy)

        self.finished.clear()
        self.started.clear()
        self.chain_events.clear()
        # 点击已缓存的上游节点：立即回放，但不应取消正在跑的下游任务。
        self.engine.preview_node(source_id)
        self.assertEqual([result.node_id for result in self.finished], [source_id])
        self.assertTrue(self.engine.busy)
        self.assertEqual(self.started, [])

        self.assertTrue(self.wait_chain(slow.id, timeout=15.0))
        self.assertIsNotNone(self.engine.result(slow.id))
        self.assertFalse(self.engine.busy)

    def test_downstream_recompute_reuses_cached_ancestors(self) -> None:
        source, gray, blur = self.build_chain()
        self.engine.preview_node(blur.id)
        self.assertTrue(self.wait_chain(blur.id))
        cached_source = self.engine.result(source.id)
        cached_gray = self.engine.result(gray.id)
        old_blur = self.engine.result(blur.id)

        self.engine.invalidate(blur.id, downstream=False)
        self.started.clear()
        self.chain_events.clear()
        self.engine.preview_node(blur.id)
        self.assertTrue(self.wait_chain(blur.id))

        # 只重算 blur，上游快照直接复用。
        self.assertEqual(self.started, [blur.id])
        self.assertIs(self.engine.result(source.id), cached_source)
        self.assertIs(self.engine.result(gray.id), cached_gray)
        self.assertIsNot(self.engine.result(blur.id), old_blur)

    def test_rapid_successive_preview_requests_deliver_only_latest(self) -> None:
        slow = self._build_slow_node(delay=0.15)
        self.chain_events.clear()
        self.finished.clear()
        for _ in range(5):
            self.engine.preview_node(slow.id)
        self.assertTrue(self.wait_chain(slow.id, timeout=15.0))

        delivered = [result for result in self.finished if result.node_id == slow.id]
        self.assertEqual(len(delivered), 1)
        self.assertEqual(delivered[0].revision, slow.revision)
        self.assertEqual(self.engine.result(slow.id).revision, slow.revision)
        self.assertFalse(self.engine.busy)

    def test_rapid_param_changes_only_latest_revision_is_delivered(self) -> None:
        slow = self._build_slow_node(delay=0.2)
        self.finished.clear()
        for value in range(5):
            self.workspace.graph.set_node_params(slow.id, {"delay": 0.2 + value * 0.05})
            self.engine.invalidate(slow.id, downstream=False)
            self.engine.preview_node(slow.id)
        final_revision = slow.revision

        self.assertTrue(
            self.pump(
                lambda: self.engine.result(slow.id) is not None and not self.engine.busy,
                timeout=20.0,
            )
        )
        self.assertEqual(self.engine.result(slow.id).revision, final_revision)
        delivered = [result for result in self.finished if result.node_id == slow.id]
        self.assertTrue(delivered)
        self.assertEqual({result.revision for result in delivered}, {final_revision})

    def test_preview_only_recomputes_node_commit_recomputes_downstream(self) -> None:
        _source, gray, blur = self.build_chain()
        self.engine.preview_node(blur.id)
        self.assertTrue(self.wait_chain(blur.id))
        old_gray = self.engine.result(gray.id)
        old_blur = self.engine.result(blur.id)

        # 预览：只更新灰度节点，下游保留上一次快照。
        self.workspace.graph.set_node_params(gray.id, {"mode": "average"})
        self.engine.invalidate(gray.id, downstream=False)
        self.chain_events.clear()
        self.engine.preview_node(gray.id)
        self.assertTrue(self.wait_chain(gray.id))
        new_gray = self.engine.result(gray.id)
        self.assertIsNot(new_gray, old_gray)
        self.assertEqual(new_gray.revision, gray.revision)
        self.assertIs(self.engine.result(blur.id), old_blur)

        # 保存：当前节点与下游子树一起重算。
        self.workspace.graph.set_node_params(gray.id, {"mode": "max"})
        self.finished.clear()
        self.chain_events.clear()
        self.engine.commit_node(gray.id)
        self.assertTrue(self.wait_chain(gray.id))
        self.assertIsNot(self.engine.result(gray.id), new_gray)
        recomputed = self.engine.result(blur.id)
        self.assertIsNot(recomputed, old_blur)
        self.assertEqual(recomputed.revision, blur.revision)
        blur_revisions = [result.revision for result in self.finished if result.node_id == blur.id]
        self.assertEqual(blur_revisions, [blur.revision])

    def test_invalidate_downstream_clears_descendant_results(self) -> None:
        _source, gray, blur = self.build_chain()
        self.engine.preview_node(blur.id)
        self.assertTrue(self.wait_chain(blur.id))
        self.engine.invalidate(gray.id, downstream=True)
        self.assertIsNone(self.engine.result(gray.id))
        self.assertIsNone(self.engine.result(blur.id))
        self.assertFalse(self.engine.busy)

    def test_forget_nodes_and_set_registry(self) -> None:
        _source, gray, blur = self.build_chain()
        self.engine.preview_node(blur.id)
        self.assertTrue(self.wait_chain(blur.id))
        self.assertEqual(self.engine.result_revision(gray.id), gray.revision)

        self.engine.forget_nodes([gray.id])
        self.assertIsNone(self.engine.result(gray.id))
        self.assertIsNone(self.engine.result_revision(gray.id))
        self.assertIsNotNone(self.engine.result(blur.id))

        self.engine.set_registry(OperatorRegistry.build())
        self.assertIsNone(self.engine.result(blur.id))
        self.assertIsNone(self.engine.error(blur.id))
        self.assertFalse(self.engine.busy)

    # ------------------------------------------------------------------ 错误
    def test_invalid_params_are_reported(self) -> None:
        source = self.add_source(_image())
        gray = self.add_op("basic.grayscale", "灰度化", {"mode": "not-a-mode"})
        self.connect(source, gray)
        self.engine.preview_node(gray.id)
        self.assertFalse(self.wait_chain(gray.id))
        self.assertIsNone(self.engine.result(gray.id))
        error = self.engine.error(gray.id)
        self.assertIsNotNone(error)
        self.assertIn("参数错误", error.message)
        message = dict(self.failures)[gray.id]
        self.assertIn("参数错误", message)

    def test_missing_input_is_reported(self) -> None:
        blur = self.add_op("filter.gaussian_blur", "高斯模糊")
        self.engine.preview_node(blur.id)
        self.assertFalse(self.wait_chain(blur.id))
        error = self.engine.error(blur.id)
        self.assertIsNotNone(error)
        self.assertIn("缺少输入", error.message)
        self.assertIn("输入", error.message)

    def test_operator_error_propagates_and_downstream_is_marked_failed(self) -> None:
        source = self.add_source(_image(dtype=np.uint16))
        median = self.add_op("filter.median_blur", "中值模糊", {"ksize": 7})
        gray = self.add_op("basic.grayscale", "灰度化")
        self.connect(source, median)
        self.connect(median, gray)
        self.engine.preview_node(gray.id)
        self.assertFalse(self.wait_chain(gray.id))

        first = self.engine.error(median.id)
        self.assertIsNotNone(first)
        self.assertIn("仅支持 uint8", first.message)
        second = self.engine.error(gray.id)
        self.assertIsNotNone(second)
        self.assertIn("上游节点执行失败", second.message)
        self.assertIn(median.id, self.started)
        self.assertNotIn(gray.id, self.started)
    def test_operator_exception_is_captured_with_details(self) -> None:
        registry = OperatorRegistry.build([OPS_FIXTURES])
        self.engine.set_registry(registry)
        source = self.add_source(_image())
        boom = self.add_op("test.boom", "异常测试算子")
        self.connect(source, boom)
        self.engine.preview_node(boom.id)
        self.assertFalse(self.wait_chain(boom.id))
        error = self.engine.error(boom.id)
        self.assertIsNotNone(error)
        self.assertIn("算子执行失败", error.message)
        self.assertIn("boom-from-operator", error.message)
        self.assertIn("RuntimeError", error.details)

    def test_missing_source_image_is_reported(self) -> None:
        source = self.workspace.graph.create_source_node("missing-image-id", "坏源图")
        self.engine.preview_node(source.id)
        self.assertFalse(self.wait_chain(source.id))
        error = self.engine.error(source.id)
        self.assertIsNotNone(error)
        self.assertIn("源图像数据缺失", error.message)

    def test_unknown_operator_is_reported(self) -> None:
        source = self.add_source(_image())
        unknown = self.add_op("does.not.exist", "未知算子")
        self.connect(source, unknown)
        self.engine.preview_node(unknown.id)
        self.assertFalse(self.wait_chain(unknown.id))
        self.assertIn("找不到算子", self.engine.error(unknown.id).message)

    # ------------------------------------------------------------------ 多端口
    def test_channel_split_and_merge_multi_port_roundtrip(self) -> None:
        source_array = _image()
        source = self.add_source(source_array)
        split = self.add_op("channel.split", "通道拆分")
        merge = self.add_op("channel.merge", "通道合并")
        self.connect(source, split)
        self.connect(split, merge, "b", "b")
        self.connect(split, merge, "g", "g")
        self.connect(split, merge, "r", "r")

        self.engine.preview_node(merge.id)
        self.assertTrue(self.wait_chain(merge.id))

        split_result = self.engine.result(split.id)
        self.assertIsNotNone(split_result)
        self.assertEqual(set(split_result.outputs), {"b", "g", "r"})
        for output in split_result.outputs.values():
            self.assertEqual(output.array.shape, source_array.shape[:2])

        merged = self.engine.result(merge.id).primary
        np.testing.assert_array_equal(merged.array, source_array)

    def test_partial_channel_merge_reports_missing_port(self) -> None:
        source_array = _image()
        source = self.add_source(source_array)
        split = self.add_op("channel.split", "通道拆分")
        merge = self.add_op("channel.merge", "通道合并")
        self.connect(source, split)
        self.connect(split, merge, "b", "b")
        self.connect(split, merge, "g", "g")
        self.engine.preview_node(merge.id)
        self.assertFalse(self.wait_chain(merge.id))
        message = self.engine.error(merge.id).message
        self.assertIn("缺少输入", message)
        self.assertIn("红色通道", message)

    # ------------------------------------------------------------------ 取消 / 过期
    def _slow_registry(self) -> OperatorRegistry:
        return OperatorRegistry.build([OPS_FIXTURES])

    def _build_slow_node(self, delay: float = 0.3):
        self.engine.set_registry(self._slow_registry())
        source = self.add_source(_image())
        slow = self.add_op("test.slow", "慢速测试算子", {"delay": delay})
        self.connect(source, slow)
        return slow

    def test_cancel_and_stale_supersession(self) -> None:
        slow = self._build_slow_node(delay=0.3)
        self.engine.preview_node(slow.id)
        self.assertTrue(self.pump(lambda: slow.id in self.started, timeout=5.0))
        self.assertTrue(self.engine.busy)

        # 参数变更 + 保存：新 generation 必须让第一次计算的结果失效。
        self.workspace.graph.set_node_params(slow.id, {"delay": 0.2})
        self.finished.clear()
        self.chain_events.clear()
        revision = slow.revision
        self.engine.commit_node(slow.id)
        self.assertTrue(self.wait_chain(slow.id, timeout=15.0))

        delivered = [result for result in self.finished if result.node_id == slow.id]
        self.assertEqual([result.revision for result in delivered], [revision])
        self.assertEqual(self.engine.result(slow.id).revision, revision)
        self.assertEqual([ok for _node, ok in self.chain_events], [True])
        self.assertFalse(self.engine.busy)

    def test_invalidate_all_drops_inflight_results(self) -> None:
        slow = self._build_slow_node(delay=0.3)
        self.engine.preview_node(slow.id)
        self.assertTrue(self.pump(lambda: slow.id in self.started, timeout=5.0))
        self.engine.invalidate_all()
        self.assertFalse(self.engine.busy)
        self.finished.clear()

        # 让旧任务跑完；它的结果必须被 generation 检查丢弃。
        self.pump(lambda: False, timeout=0.8)
        self.assertEqual(self.finished, [])
        self.assertIsNone(self.engine.result(slow.id))
        self.assertEqual(self.chain_events, [])

    def test_shutdown_is_idempotent_and_stops_delivery(self) -> None:
        slow = self._build_slow_node(delay=0.2)
        self.engine.preview_node(slow.id)
        self.assertTrue(self.pump(lambda: slow.id in self.started, timeout=5.0))
        self.engine.shutdown()
        self.assertFalse(self.engine.busy)
        self.finished.clear()

        self.engine.preview_node(slow.id)  # 关闭后不再接受新请求
        self.pump(lambda: False, timeout=0.5)
        self.assertEqual(self.finished, [])
        self.assertFalse(self.engine.busy)
        self.engine.shutdown()  # 幂等


class RegistryTests(EngineTestCase):
    def test_builtin_registry_contains_all_categories(self) -> None:
        registry = OperatorRegistry.build()
        self.assertGreaterEqual(len(registry), 22)
        self.assertEqual(registry.errors, [])
        tabs = registry.tabs()
        for category in (
            "基础调整",
            "滤波与平滑",
            "阈值与分割",
            "边缘检测",
            "形态学",
            "几何变换",
            "通道与色彩",
            "交互处理",
        ):
            self.assertIn(category, tabs, f"缺少分类 {category}")
            self.assertGreater(len(tabs[category]), 0)
        self.assertTrue(set(tabs).issubset(set(CATEGORY_ORDER)))
        for entry in registry.all_entries():
            self.assertTrue(entry.builtin)
            self.assertTrue(entry.spec.name)
            self.assertTrue(entry.spec.description)
            self.assertTrue(entry.spec.category)

    def test_scanning_user_directory_with_broken_module(self) -> None:
        # tests/fixtures/mixed_ops：一个合法算子 + 语法错误 + 导入失败 + 缺少 OPERATOR。
        registry = OperatorRegistry.build([FIXTURES / "mixed_ops"])
        self.assertIn("user.double", registry)
        self.assertIn("basic.grayscale", registry)
        self.assertGreaterEqual(len(registry.errors), 2)
        messages = [issue.display() for issue in registry.issues]
        self.assertTrue(any("broken_syntax.py" in message for message in messages))
        self.assertTrue(any("broken_import.py" in message for message in messages))
        self.assertTrue(any("no_operator.py" in message for message in messages))
        # 坏模块不影响内置算子。
        self.assertGreaterEqual(len(registry), 23)

    def test_registry_override_user_operator_id(self) -> None:
        registry = OperatorRegistry.build([FIXTURES / "override_ops"])
        self.assertEqual(registry.entry("basic.grayscale").spec.name, "自定义灰度化")
        self.assertFalse(registry.entry("basic.grayscale").builtin)
        self.assertTrue(any("重复" in issue.message for issue in registry.issues))


if __name__ == "__main__":
    unittest.main(verbosity=2)
