"""算子组节点（Group Node）的执行引擎测试。

覆盖：
* 快照阶段 resolve_source 跨组边界解析、算子组输出端口伪输入；
* 默认待定端口在外连后自动确定类型并追加新的待定端口；
* 算子组执行、组内节点 node_finished、输出类型规范化；
* 嵌套算子组级联；
* 缺失依赖时组输出失败但其它分支继续；
* 预览 / 下游更新的缓存语义；无内部来源时允许空输出。

全部使用 offscreen Qt（QApplication + processEvents），不启动事件循环。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# 通过模块引用复用 EngineTestCase，避免 unittest 把导入的测试类重复收集。
import tests.test_engine as engine_tests  # noqa: E402
from app.core.data_types import DataType, NumberValue  # noqa: E402
from app.core.group import GROUP_OP_ID, GroupModel, GroupPort  # noqa: E402
from app.core.types import ImageData  # noqa: E402


class GroupEngineTestCase(engine_tests.EngineTestCase):
    """算子组测试专用工具（在 EngineTestCase 的隔离工作区 / 引擎之上）。"""

    # ------------------------------------------------------------------ 工具
    def group_model(
        self,
        input_types=(DataType.IMAGE,),
        output_types=(DataType.IMAGE,),
    ) -> GroupModel:
        return GroupModel(
            inputs=[
                GroupPort(f"input_{index}", f"输入 {index}", data_type)
                for index, data_type in enumerate(input_types, start=1)
            ],
            outputs=[
                GroupPort(f"output_{index}", f"输出 {index}", data_type)
                for index, data_type in enumerate(output_types, start=1)
            ],
        )

    def add_group(
        self,
        title: str = "算子组",
        *,
        input_types=(DataType.IMAGE,),
        output_types=(DataType.IMAGE,),
        parent_group: str | None = None,
    ):
        return self.workspace.graph.create_group_node(
            title,
            group=self.group_model(input_types, output_types),
            parent_group=parent_group,
        )

    def add_inner(self, group, op_id: str, title: str, params: dict | None = None):
        node = self.add_op(op_id, title, params)
        self.workspace.graph.set_node_parent(node.id, group.id)
        return node

    def install_type_resolver(self) -> None:
        """模拟 GraphView 注入的端口类型解析：普通节点查注册表，源图固定图像。"""

        graph = self.workspace.graph

        def resolve(node_id: str, port_key: str, is_input: bool):
            node = graph.nodes.get(node_id)
            if node is None:
                return None
            if node.is_source:
                return DataType.IMAGE if port_key == "image" else None
            if node.is_group and node.group is not None:
                port = node.group.port(port_key, is_input)
                if port is None and not is_input:
                    # 组输入端口也可以作为组内节点的数据源。
                    port = node.group.port(port_key, True)
                return None if port is None else port.data_type
            spec = self.registry.spec(node.op_id)
            if spec is None:
                return None
            ports = spec.inputs if is_input else spec.outputs
            for port in ports:
                if port.key == port_key:
                    return None if port.any_type else port.data_type
            return None

        graph.set_port_type_resolver(resolve)

    # -------------------------------------------------------------- 常用链路
    def build_group_chain(self):
        """source -> group(input_1) -> gray -> group(output_1) -> invert。"""

        source = self.add_source(engine_tests._image())
        group = self.add_group("灰度组")
        gray = self.add_inner(group, "basic.grayscale", "灰度化", {"mode": "weighted"})
        invert = self.add_op("basic.invert", "反相")
        self.connect(source, group, "image", "input_1")
        self.connect(group, gray, "input_1", "image")
        self.connect(gray, group, "image", "output_1")
        self.connect(group, invert, "output_1", "image")
        return source, group, gray, invert


class GroupSnapshotTests(GroupEngineTestCase):
    def test_snapshot_resolves_group_boundaries_to_real_nodes(self) -> None:
        source, group, gray, invert = self.build_group_chain()

        snapshot = self.engine._capture_graph()

        # 组内节点读取组输入：解析到组外真实源节点。
        self.assertEqual(snapshot.inputs[(gray.id, "image")], (source.id, "image"))
        # 外部消费者读取组输出：解析到组内真实来源节点（不是算子组节点）。
        self.assertEqual(snapshot.inputs[(invert.id, "image")], (gray.id, "image"))
        # 组输出端口作为算子组节点的伪输入参与 deps / 拓扑排序。
        self.assertEqual(
            snapshot.inputs[(group.id, "@out:output_1")],
            (gray.id, "image"),
        )
        # 组内节点、算子组节点都作为独立节点出现在扁平快照中。
        self.assertEqual(snapshot.nodes[gray.id].parent_group, group.id)
        self.assertTrue(snapshot.nodes[group.id].is_group)
        self.assertEqual(snapshot.nodes[group.id].op_id, GROUP_OP_ID)
        # 快照持有 GroupModel 副本，后台线程不会读到界面线程修改。
        self.assertIsNot(snapshot.nodes[group.id].group, group.group)
        self.assertEqual(
            [port.key for port in snapshot.nodes[group.id].group.outputs],
            ["output_1"],
        )
        # 解析结果只存在于快照中，不能写回模型连线。
        raw = self.workspace.graph.input_connection(invert.id, "image")
        self.assertIsNotNone(raw)
        self.assertEqual(raw.src_node, group.id)
        self.assertEqual(raw.src_port, "output_1")


class GroupPortBindingTests(GroupEngineTestCase):
    def test_pending_group_ports_bind_and_append_pending_port(self) -> None:
        self.install_type_resolver()
        group = self.workspace.graph.create_group_node("算子组")
        source = self.add_source(engine_tests._image())

        # 默认各 1 个待定端口。
        self.assertEqual(len(group.group.inputs), 1)
        self.assertEqual(len(group.group.outputs), 1)
        self.assertTrue(group.group.inputs[0].pending)

        # 外连到组输入：端口确定为图像，并自动追加新的待定输入端口。
        self.connect(source, group, "image", "input_1")
        self.assertIs(group.group.inputs[0].data_type, DataType.IMAGE)
        self.assertEqual(len(group.group.inputs), 2)
        self.assertTrue(group.group.inputs[1].pending)
        self.assertTrue(group.group.inputs[1].port_spec().any_type)

        # 组内节点输出连到组输出：端口确定为图像，并自动追加新的待定输出端口。
        gray = self.add_inner(group, "basic.grayscale", "灰度化")
        self.connect(group, gray, "input_1", "image")
        self.connect(gray, group, "image", "output_1")
        self.assertIs(group.group.outputs[0].data_type, DataType.IMAGE)
        self.assertEqual(len(group.group.outputs), 2)
        self.assertTrue(group.group.outputs[1].pending)
        self.assertEqual(
            group.group.outputs[1].port_spec().type_text(), "任意数据"
        )


class GroupExecutionTests(GroupEngineTestCase):
    def test_group_executes_and_reports_inner_node_finished(self) -> None:
        _source, group, gray, invert = self.build_group_chain()

        self.engine.preview_node(group.id)
        self.assertTrue(self.wait_chain(group.id))

        gray_result = self.engine.result(gray.id)
        group_result = self.engine.result(group.id)
        invert_result = self.engine.result(invert.id)
        self.assertIsNotNone(gray_result)
        self.assertIsNotNone(group_result)
        self.assertIsNotNone(invert_result)

        # 组输出正确传递组内节点结果，并按 GroupPort 标签记录。
        group_image = group_result.outputs["output_1"]
        self.assertIsInstance(group_image, ImageData)
        self.assertEqual(group_result.port_labels, {"output_1": "输出 1"})
        np.testing.assert_array_equal(group_image.array, gray_result.outputs["image"].array)

        # 下游消费者拿到的是解析后的真实节点数据（反相）。
        self.assertEqual(invert_result.primary.width, gray_result.primary.width)
        np.testing.assert_array_equal(
            invert_result.primary.array,
            (255 - gray_result.primary.array).astype(np.uint8),
        )

        # 组内节点与算子组节点都正常发出 node_started / node_finished。
        self.assertIn(gray.id, self.started)
        self.assertIn(group.id, self.started)
        finished_ids = [result.node_id for result in self.finished]
        self.assertIn(gray.id, finished_ids)
        self.assertIn(group.id, finished_ids)
        self.assertIn(invert.id, finished_ids)
        self.assertFalse(self.engine.busy)

    def test_group_without_internal_source_allows_empty_outputs(self) -> None:
        group = self.add_group("空算子组", output_types=(None,))
        self.engine.preview_node(group.id)
        self.assertTrue(self.wait_chain(group.id))

        result = self.engine.result(group.id)
        self.assertIsNotNone(result)
        self.assertEqual(result.outputs, {})
        self.assertIsNone(result.primary)
        self.assertEqual(result.summary_text(), "算子没有输出")

    def test_nested_groups_cascade(self) -> None:
        source = self.add_source(engine_tests._image())
        outer = self.add_group("外层算子组")
        inner = self.add_group("内层算子组", parent_group=outer.id)
        gray = self.add_inner(inner, "basic.grayscale", "灰度化", {"mode": "average"})
        invert = self.add_op("basic.invert", "反相")

        self.connect(source, outer, "image", "input_1")
        self.connect(outer, inner, "input_1", "input_1")
        self.connect(inner, gray, "input_1", "image")
        self.connect(gray, inner, "image", "output_1")
        self.connect(inner, outer, "output_1", "output_1")
        self.connect(outer, invert, "output_1", "image")

        self.engine.preview_node(outer.id)
        self.assertTrue(self.wait_chain(outer.id))

        gray_result = self.engine.result(gray.id)
        inner_result = self.engine.result(inner.id)
        outer_result = self.engine.result(outer.id)
        invert_result = self.engine.result(invert.id)
        self.assertIsNotNone(gray_result)
        self.assertIsNotNone(inner_result)
        self.assertIsNotNone(outer_result)
        self.assertIsNotNone(invert_result)

        np.testing.assert_array_equal(
            inner_result.outputs["output_1"].array, gray_result.outputs["image"].array
        )
        np.testing.assert_array_equal(
            outer_result.outputs["output_1"].array, gray_result.outputs["image"].array
        )
        # 内外层算子组都解析到同一个真实来源节点；二者都要等灰度化完成后才开始。
        self.assertLess(self.started.index(gray.id), self.started.index(inner.id))
        self.assertLess(self.started.index(gray.id), self.started.index(outer.id))

    def test_typed_multi_outputs_are_coerced(self) -> None:
        source = self.add_source(engine_tests._image())
        group = self.add_group(
            "统计算子组",
            input_types=(DataType.IMAGE,),
            output_types=(DataType.IMAGE, DataType.NUMBER),
        )
        stats = self.add_inner(group, "analysis.mean_std", "均值统计")

        self.connect(source, group, "image", "input_1")
        self.connect(group, stats, "input_1", "image")
        self.connect(stats, group, "image", "output_1")
        self.connect(stats, group, "mean", "output_2")

        self.engine.preview_node(group.id)
        self.assertTrue(self.wait_chain(group.id))

        stats_result = self.engine.result(stats.id)
        group_result = self.engine.result(group.id)
        self.assertIsNotNone(stats_result)
        self.assertIsNotNone(group_result)

        self.assertIsInstance(group_result.outputs["output_1"], ImageData)
        mean_value = group_result.outputs["output_2"]
        self.assertIsInstance(mean_value, NumberValue)
        self.assertAlmostEqual(mean_value.value, stats_result.outputs["mean"].value)
        self.assertEqual(
            group_result.port_labels,
            {"output_1": "输出 1", "output_2": "输出 2"},
        )
        self.assertEqual(group_result.output_type_summary(), "图像、数值")

    def test_pending_output_accepts_any_data_type(self) -> None:
        source = self.add_source(engine_tests._image())
        group = self.add_group(
            "待定输出算子组",
            input_types=(DataType.IMAGE,),
            output_types=(None,),
        )
        stats = self.add_inner(group, "analysis.mean_std", "均值统计")
        self.connect(source, group, "image", "input_1")
        self.connect(group, stats, "input_1", "image")
        self.connect(stats, group, "mean", "output_1")

        # 未注入类型解析器：端口保持待定，引擎按 any_type 原样透传。
        self.assertTrue(group.group.outputs[0].pending)
        self.engine.preview_node(group.id)
        self.assertTrue(self.wait_chain(group.id))

        group_result = self.engine.result(group.id)
        self.assertIsNotNone(group_result)
        self.assertIsInstance(group_result.outputs["output_1"], NumberValue)
        self.assertTrue(group.group.outputs[0].pending)


class GroupFailureTests(GroupEngineTestCase):
    def test_missing_inner_dependency_fails_group_but_not_other_branch(self) -> None:
        source = self.add_source(engine_tests._image())
        group = self.add_group(
            "含缺失依赖的算子组",
            input_types=(DataType.IMAGE,),
            output_types=(DataType.IMAGE, DataType.IMAGE),
        )
        missing = self.add_inner(group, "does.not.exist", "缺失算子")
        gray = self.add_inner(group, "basic.grayscale", "正常分支")
        out_good = self.add_op("basic.invert", "正常下游")

        self.connect(source, group, "image", "input_1")
        self.connect(group, missing, "input_1", "image")
        self.connect(missing, group, "image", "output_1")
        self.connect(group, gray, "input_1", "image")
        self.connect(gray, group, "image", "output_2")
        self.connect(group, out_good, "output_2", "image")

        self.engine.preview_node(group.id)
        self.assertFalse(self.wait_chain(group.id))

        # 缺失依赖节点失败，算子组输出失败（带端口标签），但进程不崩溃。
        missing_error = self.engine.error(missing.id)
        self.assertIsNotNone(missing_error)
        self.assertIn("找不到算子", missing_error.message)
        group_error = self.engine.error(group.id)
        self.assertIsNotNone(group_error)
        self.assertIn("上游节点执行失败", group_error.message)
        self.assertIn("输出 1", group_error.message)
        self.assertIsNone(self.engine.result(group.id))

        # 其它分支照常计算并交付结果。
        self.assertIsNotNone(self.engine.result(gray.id))
        self.assertIsNotNone(self.engine.result(out_good.id))
        finished_ids = [result.node_id for result in self.finished]
        self.assertIn(gray.id, finished_ids)
        self.assertIn(out_good.id, finished_ids)

        # 之后单独重算正常分支仍然成功（失败没有被缓存成永久状态）。
        self.chain_events.clear()
        self.engine.commit_node(out_good.id)
        self.assertTrue(self.wait_chain(out_good.id))


class GroupCacheTests(GroupEngineTestCase):
    def test_group_result_replays_from_snapshot_cache(self) -> None:
        _source, group, _gray, _invert = self.build_group_chain()
        self.engine.preview_node(group.id)
        self.assertTrue(self.wait_chain(group.id))
        cached = self.engine.result(group.id)

        self.finished.clear()
        self.started.clear()
        self.chain_events.clear()
        self.engine.preview_node(group.id)

        self.assertEqual(len(self.finished), 1)
        self.assertIs(self.finished[0], cached)
        self.assertEqual(self.started, [])
        self.assertEqual(self.chain_events, [])
        self.assertFalse(self.engine.busy)

    def test_preview_source_updates_group_and_outer_consumer(self) -> None:
        source, group, gray, invert = self.build_group_chain()

        # 预览源图像：组内节点、算子组、组外下游都应通过解析后的依赖 /
        # 结构边被纳入同一次计算。
        self.engine.preview_node(source.id)
        self.assertTrue(self.wait_chain(source.id))

        self.assertIsNotNone(self.engine.result(gray.id))
        self.assertIsNotNone(self.engine.result(group.id))
        self.assertIsNotNone(self.engine.result(invert.id))
        self.assertIn(group.id, self.started)
        self.assertIn(invert.id, self.started)

    def test_update_nodes_with_group_and_inner_roots_runs_each_once(self) -> None:
        _source, group, gray, invert = self.build_group_chain()
        self.engine.preview_node(group.id)
        self.assertTrue(self.wait_chain(group.id))
        old_group = self.engine.result(group.id)
        old_invert = self.engine.result(invert.id)

        self.workspace.graph.set_node_params(gray.id, {"mode": "max"})
        self.finished.clear()
        self.started.clear()
        self.chain_events.clear()
        # 同时把组内节点与算子组作为根：共用一次快照，节点只执行一次。
        self.engine.update_nodes([gray.id, group.id, gray.id])
        target_ids = {gray.id, group.id}
        self.assertTrue(
            self.pump(
                lambda: target_ids.issubset({node_id for node_id, _ok in self.chain_events}),
                timeout=10.0,
            ),
            f"未收到全部 chain_finished：{self.chain_events}",
        )
        self.assertEqual(
            sorted(node_id for node_id, _ok in self.chain_events),
            sorted(target_ids),
        )
        self.assertTrue(all(ok for _node_id, ok in self.chain_events))
        self.assertEqual(self.started.count(gray.id), 1)
        self.assertEqual(self.started.count(group.id), 1)
        self.assertIsNot(self.engine.result(group.id), old_group)
        self.assertIsNot(self.engine.result(invert.id), old_invert)
        self.assertEqual(self.engine.result(gray.id).revision, gray.revision)

    def test_downstream_updates_after_inner_node_recompute(self) -> None:
        _source, group, gray, invert = self.build_group_chain()
        self.engine.preview_node(group.id)
        self.assertTrue(self.wait_chain(group.id))
        old_gray = self.engine.result(gray.id)
        old_group = self.engine.result(group.id)
        old_invert = self.engine.result(invert.id)

        # 修改组内节点参数后预览该节点：算子组节点与组外下游都要一起更新。
        self.workspace.graph.set_node_params(gray.id, {"mode": "max"})
        self.finished.clear()
        self.chain_events.clear()
        self.engine.preview_node(gray.id)
        self.assertTrue(self.wait_chain(gray.id))

        new_gray = self.engine.result(gray.id)
        new_group = self.engine.result(group.id)
        new_invert = self.engine.result(invert.id)
        self.assertIsNot(new_gray, old_gray)
        self.assertIsNot(new_group, old_group)
        self.assertIsNot(new_invert, old_invert)
        self.assertEqual(new_gray.revision, gray.revision)
        np.testing.assert_array_equal(
            new_group.outputs["output_1"].array, new_gray.outputs["image"].array
        )
        # 组内节点、算子组节点、外部下游都交付了 node_finished。
        delivered = [result.node_id for result in self.finished]
        self.assertIn(gray.id, delivered)
        self.assertIn(group.id, delivered)
        self.assertIn(invert.id, delivered)

        # 预览算子组本身也要把组输出连接的外部消费者一起更新。
        self.workspace.graph.set_node_params(gray.id, {"mode": "average"})
        previous_group = self.engine.result(group.id)
        previous_invert = self.engine.result(invert.id)
        self.engine.invalidate(gray.id, downstream=True)
        self.chain_events.clear()
        self.engine.preview_node(group.id)
        self.assertTrue(self.wait_chain(group.id))
        self.assertIsNot(self.engine.result(group.id), previous_group)
        self.assertIsNot(self.engine.result(invert.id), previous_invert)
        np.testing.assert_array_equal(
            self.engine.result(invert.id).primary.array,
            (255 - self.engine.result(gray.id).primary.array).astype(np.uint8),
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
