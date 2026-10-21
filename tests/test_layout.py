"""自动布局算法测试：网格对齐、兄弟子树顺序、与插入顺序无关。

运行方式（项目根目录）::

    $env:PYTHONPATH = "."
    python -m unittest discover -s tests -p "test_layout.py" -v
"""

from __future__ import annotations

import unittest

from app.core.model import ConnectionModel, GraphError, GraphModel, NodeModel
from app.graph.layout import DEFAULT_NODE_SIZE, GRID, compute_layout


def build_graph(
    nodes: list[tuple[str, tuple[float, float], tuple[float, float] | None]],
    edges: list[tuple[str, str]],
    *,
    insertion: list[int] | None = None,
    connection_order: list[int] | None = None,
) -> GraphModel:
    """按指定顺序把节点 / 连线塞进模型，模拟“后创建的节点先布局”的场景。"""

    graph = GraphModel()
    node_order = insertion if insertion is not None else list(range(len(nodes)))
    for index in node_order:
        node_id, pos, _size = nodes[index]
        graph.nodes[node_id] = NodeModel(
            id=node_id,
            op_id="test.node",
            title=node_id,
            pos=pos,
        )
    edge_order = connection_order if connection_order is not None else list(range(len(edges)))
    for index in edge_order:
        src, dst = edges[index]
        connection_id = f"c_{src}_{dst}_{index}"
        graph.connections[connection_id] = ConnectionModel(
            id=connection_id,
            src_node=src,
            src_port="image",
            dst_node=dst,
            dst_port="image",
        )
    return graph


def sizes_of(
    nodes: list[tuple[str, tuple[float, float], tuple[float, float] | None]],
) -> dict[str, tuple[float, float]]:
    return {node_id: size for node_id, _pos, size in nodes if size is not None}


def center_x(
    node_id: str,
    positions: dict[str, tuple[float, float]],
    sizes: dict[str, tuple[float, float]],
) -> float:
    width = sizes.get(node_id, DEFAULT_NODE_SIZE)[0]
    return positions[node_id][0] + width / 2.0


def _depths(graph: GraphModel) -> dict[str, int]:
    depth = {node_id: 0 for node_id in graph.nodes}
    for _ in range(len(graph.nodes) + 1):
        for connection in graph.connections.values():
            depth[connection.dst_node] = max(
                depth[connection.dst_node],
                depth[connection.src_node] + 1,
            )
    return depth


class LayoutAlgorithmTestCase(unittest.TestCase):
    def assert_grid_aligned(self, positions: dict[str, tuple[float, float]]) -> None:
        for node_id, (x, y) in positions.items():
            self.assertAlmostEqual(
                x / GRID,
                round(x / GRID),
                places=9,
                msg=f"{node_id} 的 x={x} 未对齐网格",
            )
            self.assertAlmostEqual(
                y / GRID,
                round(y / GRID),
                places=9,
                msg=f"{node_id} 的 y={y} 未对齐网格",
            )

    def assert_no_overlap(
        self,
        positions: dict[str, tuple[float, float]],
        sizes: dict[str, tuple[float, float]],
    ) -> None:
        items = list(positions.items())
        for index, (node_id, (x, y)) in enumerate(items):
            width, height = sizes.get(node_id, DEFAULT_NODE_SIZE)
            for other_id, (other_x, other_y) in items[index + 1 :]:
                other_width, other_height = sizes.get(other_id, DEFAULT_NODE_SIZE)
                overlap = (
                    x < other_x + other_width
                    and other_x < x + width
                    and y < other_y + other_height
                    and other_y < y + height
                )
                self.assertFalse(overlap, f"节点重叠：{node_id} / {other_id}")

    def assert_no_layer_crossings(
        self,
        graph: GraphModel,
        positions: dict[str, tuple[float, float]],
    ) -> None:
        """同一对相邻层之间的连线不应交叉。"""

        depth = _depths(graph)
        connections = list(graph.connections.values())
        for index, first in enumerate(connections):
            for second in connections[index + 1 :]:
                if first.src_node == second.src_node or first.dst_node == second.dst_node:
                    continue
                if depth[first.src_node] != depth[second.src_node]:
                    continue
                if depth[first.dst_node] != depth[second.dst_node]:
                    continue
                source_order = (
                    positions[first.src_node][0] - positions[second.src_node][0]
                )
                target_order = (
                    positions[first.dst_node][0] - positions[second.dst_node][0]
                )
                self.assertGreaterEqual(
                    source_order * target_order,
                    0.0,
                    f"连线交叉：{first.id} / {second.id}",
                )

    def assert_subtrees_ordered(self, graph: GraphModel, positions: dict[str, tuple[float, float]]) -> None:
        """每个父节点下，左侧子节点的整棵子树都应排在右侧子节点的子树左边。"""

        depth = _depths(graph)
        for parent in graph.nodes:
            children = graph.successors(parent)
            pairs = [(children[i], children[j]) for i in range(len(children)) for j in range(i + 1, len(children))]
            for left, right in pairs:
                if positions[left][0] > positions[right][0]:
                    continue
                left_group = graph.descendants(left) + [left]
                right_group = graph.descendants(right) + [right]
                for layer in set(depth[node_id] for node_id in graph.nodes):
                    left_band = [node_id for node_id in left_group if depth[node_id] == layer]
                    right_band = [node_id for node_id in right_group if depth[node_id] == layer]
                    if not left_band or not right_band:
                        continue
                    self.assertLessEqual(
                        max(positions[node_id][0] for node_id in left_band),
                        min(positions[node_id][0] for node_id in right_band),
                        f"{parent} 的子树交错：{left} / {right}（层 {layer}）",
                    )

    # ------------------------------------------------------------------ 基本
    def test_layout_grid_matches_canvas_grid(self) -> None:
        from app.graph.scene import GraphScene

        # 自动布局吸附网格应与画布背景网格保持一致，避免两者漂移。
        self.assertEqual(GRID, GraphScene.GRID)

    def test_layout_aligned_to_grid(self) -> None:
        nodes = [
            ("root", (0.0, 0.0), (300.0, 160.0)),
            ("left", (-500.0, 200.0), (222.0, 135.0)),
            ("right", (500.0, 200.0), (400.0, 220.0)),
            ("merge", (50.0, 400.0), (222.0, 135.0)),
            ("tail", (80.0, 600.0), (260.0, 180.0)),
        ]
        edges = [
            ("root", "left"),
            ("root", "right"),
            ("left", "merge"),
            ("right", "merge"),
            ("merge", "tail"),
        ]
        sizes = sizes_of(nodes)
        positions = compute_layout(build_graph(nodes, edges), sizes)

        self.assertEqual(set(positions), {node_id for node_id, _pos, _size in nodes})
        self.assert_grid_aligned(positions)
        self.assert_no_overlap(positions, sizes)
        # 纵向分层仍然保持自顶向下。
        self.assertLess(positions["root"][1], positions["left"][1])
        self.assertLess(positions["left"][1], positions["merge"][1])
        self.assertLess(positions["merge"][1], positions["tail"][1])
        self.assertTrue(all(x >= 0.0 and y >= 0.0 for x, y in positions.values()))

    def test_roots_keep_current_left_to_right_order(self) -> None:
        # 节点添加顺序与当前左右位置相反，布局应以当前 x 为准。
        nodes = [
            ("right_root", (800.0, 0.0), None),
            ("left_root", (0.0, 0.0), None),
        ]
        positions = compute_layout(build_graph(nodes, []))
        self.assertLess(positions["left_root"][0], positions["right_root"][0])

    def test_roots_in_same_grid_cell_keep_raw_x_order(self) -> None:
        # 85 与 95 会吸附到同一网格，排序必须先看原始 x，不能退化成 id 顺序。
        nodes = [
            ("node_a", (95.0, 0.0), None),
            ("node_b", (85.0, 0.0), None),
        ]
        positions = compute_layout(build_graph(nodes, []))
        self.assertLess(positions["node_b"][0], positions["node_a"][0])
        self.assert_grid_aligned(positions)

    def test_parent_centered_over_children_within_grid(self) -> None:
        nodes = [
            ("root", (0.0, 0.0), None),
            ("a", (-400.0, 200.0), None),
            ("b", (400.0, 200.0), None),
        ]
        edges = [("root", "a"), ("root", "b")]
        sizes = sizes_of(nodes)
        positions = compute_layout(build_graph(nodes, edges), sizes)
        parent_center = center_x("root", positions, sizes)
        children_center = (center_x("a", positions, sizes) + center_x("b", positions, sizes)) / 2.0
        self.assertAlmostEqual(parent_center, children_center, delta=GRID)

    # ------------------------------------------------------ 兄弟子树的左右顺序
    def test_sibling_subtrees_follow_left_to_right_order(self) -> None:
        # 右侧兄弟先创建、连线也先创建；布局后它的子树必须整体排在左侧兄弟右边。
        nodes = [
            ("left_root", (0.0, 0.0), None),
            ("right_root", (900.0, 0.0), None),
            ("left_leaf", (-200.0, 200.0), None),
            ("left_branch", (150.0, 200.0), None),
            ("right_branch", (700.0, 200.0), None),
            ("right_leaf", (1150.0, 200.0), None),
            ("left_tail", (150.0, 400.0), None),
            ("right_tail", (700.0, 400.0), None),
        ]
        edges = [
            ("left_root", "left_leaf"),
            ("left_root", "left_branch"),
            ("left_branch", "left_tail"),
            ("right_root", "right_branch"),
            ("right_root", "right_leaf"),
            ("right_branch", "right_tail"),
        ]
        # 让右侧子树先出现在模型中，复现“先布局右侧节点子节点”的问题。
        insertion = [1, 5, 4, 7, 0, 2, 3, 6]
        connection_order = [3, 4, 5, 0, 1, 2]
        graph = build_graph(
            nodes,
            edges,
            insertion=insertion,
            connection_order=connection_order,
        )
        positions = compute_layout(graph)

        self.assert_grid_aligned(positions)
        self.assert_no_layer_crossings(graph, positions)
        self.assert_subtrees_ordered(graph, positions)

    def test_layout_does_not_depend_on_insertion_order(self) -> None:
        nodes = [
            ("root", (10.0, 0.0), (222.0, 135.0)),
            ("left", (-320.0, 200.0), (222.0, 135.0)),
            ("right", (420.0, 200.0), (260.0, 180.0)),
            ("left_leaf", (-420.0, 400.0), (222.0, 135.0)),
            ("right_leaf", (520.0, 400.0), (222.0, 135.0)),
        ]
        edges = [
            ("root", "left"),
            ("root", "right"),
            ("left", "left_leaf"),
            ("right", "right_leaf"),
        ]
        sizes = sizes_of(nodes)
        baseline = compute_layout(build_graph(nodes, edges), sizes)

        orders = [
            [4, 3, 2, 1, 0],
            [2, 4, 0, 3, 1],
            [1, 3, 0, 2, 4],
        ]
        connection_orders = [
            [3, 2, 1, 0],
            [0, 3, 1, 2],
            [2, 0, 3, 1],
        ]
        for insertion, connection_order in zip(orders, connection_orders):
            positions = compute_layout(
                build_graph(
                    nodes,
                    edges,
                    insertion=insertion,
                    connection_order=connection_order,
                ),
                sizes,
            )
            self.assertEqual(positions, baseline)

    # ---------------------------------------------------------------- 子集布局
    def test_layout_supports_node_subset(self) -> None:
        nodes = [
            ("outer", (0.0, -400.0), None),
            ("root", (0.0, 0.0), None),
            ("child", (0.0, 200.0), None),
            ("other", (900.0, 0.0), None),
        ]
        edges = [("root", "child"), ("root", "other"), ("outer", "root")]
        graph = build_graph(nodes, edges)
        positions = compute_layout(graph, node_ids=["root", "child"])
        self.assertEqual(set(positions), {"root", "child"})
        self.assertLess(positions["root"][1], positions["child"][1])
        self.assert_grid_aligned(positions)

    # ------------------------------------------------------------------ 异常
    def test_layout_raises_on_cycle(self) -> None:
        graph = build_graph(
            [("a", (0.0, 0.0), None), ("b", (0.0, 200.0), None)],
            [],
        )
        graph.connections["c_ab"] = ConnectionModel(
            id="c_ab", src_node="a", src_port="image", dst_node="b", dst_port="image"
        )
        graph.connections["c_ba"] = ConnectionModel(
            id="c_ba", src_node="b", src_port="image", dst_node="a", dst_port="image"
        )
        with self.assertRaises(GraphError):
            compute_layout(graph)


if __name__ == "__main__":
    unittest.main()
