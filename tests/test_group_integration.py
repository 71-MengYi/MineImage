"""算子组端到端集成测试：画布建模 -> 引擎执行 -> 保存/复用自定义算子。

运行方式（仓库根目录）::

    $env:PYTHONPATH = "."
    python -m unittest tests.test_group_integration -v
"""

from __future__ import annotations

import os
import shutil
import time
import unittest
import uuid
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QPointF
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import QApplication

from app.config import AppSettings
from app.core.model import WorkspaceModel
from app.core.types import ImageData
from app.engine.engine import ExecutionEngine
from app.graph.view import GraphView
from app.operators.loader import OperatorRegistry

_ROOT = Path(__file__).resolve().parents[1]
_TEMP_ROOT = _ROOT / ".test_tmp" / "group-integration"
_APP: QApplication | None = None


def setUpModule() -> None:
    global _APP
    _APP = QApplication.instance() or QApplication([])


def tearDownModule() -> None:
    global _APP
    if _APP is not None:
        _APP.processEvents()
        _APP = None


def _image() -> ImageData:
    rng = np.random.default_rng(20241022)
    return ImageData(rng.integers(0, 256, (32, 40, 3), dtype=np.uint8), "源图像")


class GroupEndToEndTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = _TEMP_ROOT / uuid.uuid4().hex[:8]
        self.temp.mkdir(parents=True, exist_ok=True)
        self.settings = AppSettings(self.temp / "settings.ini")
        self.registry = OperatorRegistry.build()
        self.workspace = WorkspaceModel("算子组端到端")
        self.undo = QUndoStack()
        self.view = GraphView(self.workspace, self.registry, self.undo, self.settings)
        self.view.resize(1000, 700)
        self.source = self.workspace.add_source_image(_image(), None, pos=(0.0, 0.0))
        self.engine = ExecutionEngine(self.workspace, self.registry, self.settings, workers=2)
        self.chain_events: list[tuple[str, bool]] = []
        self.finished = []
        self.failures: list[tuple[str, str]] = []
        self.engine.chain_finished.connect(lambda nid, ok: self.chain_events.append((nid, ok)))
        self.engine.node_finished.connect(self.finished.append)
        self.engine.node_failed.connect(lambda nid, msg: self.failures.append((nid, msg)))

    def tearDown(self) -> None:
        self.engine.shutdown()
        self.view.close()
        self.view.deleteLater()
        if _APP is not None:
            _APP.processEvents()
        shutil.rmtree(self.temp, ignore_errors=True)

    # ------------------------------------------------------------------ 工具
    def pump(self, predicate, timeout: float = 10.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            _APP.processEvents()
            if predicate():
                return True
            time.sleep(0.005)
        _APP.processEvents()
        return predicate()

    def group_rect(self, group_id: str):
        item = self.view._nodes[group_id]
        return item.pos(), item.size()

    def add_inner_operator(self, group_id: str, op_id: str = "basic.grayscale") -> str:
        pos, size = self.group_rect(group_id)
        node_id = self.view.add_operator_at(
            op_id,
            QPointF(pos.x() + size.width() / 2.0, pos.y() + size.height() / 2.0),
        )
        self.assertIsNotNone(node_id)
        node = self.workspace.graph.nodes[node_id]
        self.assertEqual(node.parent_group, group_id)
        return node_id

    # ------------------------------------------------------------------ 用例
    def test_group_end_to_end_save_and_reuse_operator(self) -> None:
        group_id = self.view.add_group_at(QPointF(360.0, 340.0), title="端到端流程")
        group = self.workspace.graph.nodes[group_id]
        self.assertEqual([port.key for port in group.group.inputs], ["input_1"])
        self.assertTrue(group.group.inputs[0].pending)
        self.assertTrue(group.group.outputs[0].pending)

        inner_id = self.add_inner_operator(group_id, "basic.grayscale")
        # 外部源 -> 算子组输入：类型确定并自动追加待定输入端口
        self.view.connect_ports(self.source.id, "image", group_id, "input_1")
        self.assertEqual(group.group.inputs[0].data_type.value, "image")
        self.assertEqual(len(group.group.inputs), 2)
        self.assertTrue(group.group.inputs[1].pending)
        # 组输入(内侧) -> 组内节点；组内节点 -> 组输出
        self.view.connect_ports(group_id, "input_1", inner_id, "image")
        self.view.connect_ports(inner_id, "image", group_id, "output_1")
        self.assertEqual(group.group.outputs[0].data_type.value, "image")
        self.assertEqual(len(group.group.outputs), 2)
        output_id = self.view.add_output_node_at(group_id, "output_1", QPointF(820.0, 620.0))
        self.assertIsNotNone(output_id)

        # 引擎执行：算子组输出图像；组内节点也有结果
        self.engine.preview_node(group_id)
        reached = self.pump(lambda: any(nid == group_id for nid, _ok in self.chain_events))
        self.assertTrue(reached, f"等待算子组执行超时：{self.chain_events}")
        result = self.engine.result(group_id)
        self.assertIsNotNone(result)
        self.assertIsNotNone(result.outputs.get("output_1"))
        self.assertEqual(result.outputs["output_1"].array.shape[:2], (32, 40))
        self.assertIsNotNone(self.engine.result(inner_id))

        # 保存为新算子（对话框 / 算子目录都重定向到临时目录）
        operator_root = self.temp / "operators"
        signals: list[bool] = []
        self.view.operators_changed.connect(lambda: signals.append(True))
        with (
            patch("app.graph.view.QInputDialog.getText", return_value=("端到端流程", True)),
            patch("app.operators.group.default_user_operator_dir", return_value=operator_root),
        ):
            self.view.save_group_as_operator(group_id)
        self.assertTrue(signals, "保存后应发出 operators_changed 信号")
        generated = list((operator_root / "custom").glob("*.py"))
        self.assertEqual(len(generated), 1, f"未生成算子文件：{operator_root}")
        new_registry = OperatorRegistry.build([operator_root])
        op_id = next(op for op in new_registry.entries if op.startswith("group."))
        self.assertTrue(op_id.startswith("group."))
        self.assertEqual(new_registry.spec(op_id).category[0], "用户自定义算子")
        self.assertEqual(next(iter(new_registry.tabs())), "用户自定义算子")

        # 刷新算子库后从库中拖入同一个自定义算子 -> 展开成算子组
        self.view.set_registry(new_registry)
        self.registry = new_registry
        new_group_id = self.view.add_operator_at(op_id, QPointF(900.0, 200.0))
        self.assertIsNotNone(new_group_id)
        new_group = self.workspace.graph.nodes[new_group_id]
        self.assertTrue(new_group.is_group)
        self.assertEqual(new_group.group.operator_id, op_id)
        children = self.workspace.graph.direct_children(new_group_id)
        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].op_id, "basic.grayscale")
        # 撤销 / 重做整次展开
        before = len(self.workspace.graph.nodes)
        self.undo.undo()
        self.assertNotIn(new_group_id, self.workspace.graph.nodes)
        self.assertEqual(len(self.workspace.graph.nodes), before - 2)
        self.undo.redo()
        self.assertIn(new_group_id, self.workspace.graph.nodes)

    def test_delete_group_subtree_and_undo(self) -> None:
        group_id = self.view.add_group_at(QPointF(300.0, 300.0), title="待删除算子组")
        inner_id = self.add_inner_operator(group_id)
        nested_id = self.view.add_group_at(
            QPointF(340.0, 340.0), parent_group=group_id, title="嵌套算子组"
        )
        self.view.set_selected_node(group_id)
        self.view.delete_selected()
        for node_id in (group_id, inner_id, nested_id):
            self.assertNotIn(node_id, self.workspace.graph.nodes)
        self.undo.undo()
        for node_id in (group_id, inner_id, nested_id):
            self.assertIn(node_id, self.workspace.graph.nodes)
        self.assertEqual(self.workspace.graph.nodes[inner_id].parent_group, group_id)
        self.assertEqual(self.workspace.graph.nodes[nested_id].parent_group, group_id)

    def test_cross_group_connection_rejected(self) -> None:
        messages: list[str] = []
        self.view.status_message.connect(messages.append)
        group_a = self.view.add_group_at(QPointF(200.0, 300.0), title="A")
        group_b = self.view.add_group_at(QPointF(800.0, 300.0), title="B")
        node_a = self.add_inner_operator(group_a)
        node_b = self.add_inner_operator(group_b)
        self.view.connect_ports(node_a, "image", node_b, "image")
        self.assertIsNone(self.workspace.graph.input_connection(node_b, "image"))
        self.assertTrue(
            any("算子组" in message for message in messages),
            f"跨组连线应被拒绝并给出提示，实际消息：{messages}",
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
