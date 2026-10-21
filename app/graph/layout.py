"""纵向 DAG 自动布局。

节点深度 = 从根节点出发的最长路径长度；同层节点横向均匀分布，并尽量把父节点
居中到子节点上方。布局结果以节点左上角场景坐标返回，保证坐标有限且互不重叠。
"""

from __future__ import annotations

import math
from collections import deque
from typing import Mapping

from app.core.model import GraphError, GraphModel

DEFAULT_NODE_SIZE = (222.0, 130.0)
H_GAP = 80.0
V_GAP = 110.0


def _size_for(sizes: Mapping[str, tuple[float, float]], node_id: str) -> tuple[float, float]:
    width, height = sizes.get(node_id, DEFAULT_NODE_SIZE)
    width = float(width)
    height = float(height)
    if not math.isfinite(width) or width <= 1.0:
        width = DEFAULT_NODE_SIZE[0]
    if not math.isfinite(height) or height <= 1.0:
        height = DEFAULT_NODE_SIZE[1]
    return width, height


def _layered_order(graph: GraphModel, node_ids: list[str]) -> list[str]:
    """Kahn 拓扑排序（按唯一后继计算入度）。

    ``GraphModel.successors`` 会对重复边去重，因此不能直接用模型自带的
    ``topological_order``（同一对节点的两条连线会被误判为环）。这里用去重后的
    邻接表重新计算，兼容“一个节点的多个输出连到另一个节点多个输入”的情况。
    """

    successors = {node_id: list(graph.successors(node_id)) for node_id in node_ids}
    indegree = {node_id: 0 for node_id in node_ids}
    for node_id in node_ids:
        for child in successors[node_id]:
            indegree[child] += 1
    queue = deque(node_id for node_id in node_ids if indegree[node_id] == 0)
    order: list[str] = []
    while queue:
        node_id = queue.popleft()
        order.append(node_id)
        for child in successors[node_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(order) != len(node_ids):
        raise GraphError("图中存在环路，无法布局")
    return order


def compute_layout(
    graph: GraphModel,
    sizes: Mapping[str, tuple[float, float]] | None = None,
    *,
    h_gap: float = H_GAP,
    v_gap: float = V_GAP,
) -> dict[str, tuple[float, float]]:
    """计算布局坐标；图中有环时抛出 :class:`GraphError`。"""

    sizes = sizes or {}
    node_ids = list(graph.nodes.keys())
    if not node_ids:
        return {}

    order = _layered_order(graph, node_ids)

    depth: dict[str, int] = {}
    for node_id in order:
        parents = [parent for parent in graph.predecessors(node_id) if parent in depth]
        depth[node_id] = 0 if not parents else max(depth[parent] for parent in parents) + 1

    layers: dict[int, list[str]] = {}
    for node_id in order:
        layers.setdefault(depth[node_id], []).append(node_id)
    layer_keys = sorted(layers)
    index_of = {node_id: index for index, node_id in enumerate(node_ids)}

    layer_heights: dict[int, float] = {}
    for layer_key in layer_keys:
        layer_heights[layer_key] = max(_size_for(sizes, node_id)[1] for node_id in layers[layer_key])

    layer_y: dict[int, float] = {}
    cursor_y = 0.0
    for layer_key in layer_keys:
        layer_y[layer_key] = cursor_y
        cursor_y += layer_heights[layer_key] + v_gap

    # 初始横向位置：每层居中等距排列。
    centers: dict[str, float] = {}
    for layer_key in layer_keys:
        layer = layers[layer_key]
        total_width = sum(_size_for(sizes, node_id)[0] for node_id in layer) + h_gap * (len(layer) - 1)
        cursor_x = -total_width / 2.0
        for node_id in layer:
            width, _height = _size_for(sizes, node_id)
            centers[node_id] = cursor_x + width / 2.0
            cursor_x += width + h_gap

    # 自底向上让父节点对齐到子节点中心。
    for layer_key in reversed(layer_keys):
        layer = layers[layer_key]
        desired: dict[str, float] = {}
        has_children: dict[str, bool] = {}
        for node_id in layer:
            children = [child for child in graph.successors(node_id) if depth.get(child, -1) > layer_key]
            has_children[node_id] = bool(children)
            if children:
                desired[node_id] = sum(centers[child] for child in children) / len(children)
            else:
                desired[node_id] = centers[node_id]
        ordered = sorted(layer, key=lambda node_id: (desired[node_id], index_of[node_id]))
        packed: dict[str, float] = {}
        cursor_x: float | None = None
        for node_id in ordered:
            width, _height = _size_for(sizes, node_id)
            left = desired[node_id] - width / 2.0
            if cursor_x is not None:
                left = max(left, cursor_x + h_gap)
            packed[node_id] = left + width / 2.0
            cursor_x = left + width
        if len(ordered) > 1:
            # 只让“有子节点”的父节点决定整层偏移；孤立叶子不应把父节点推离
            # 子节点中心。若整层都是叶子，则以整层均值居中即可。
            focus = [node_id for node_id in ordered if has_children[node_id]] or ordered
            mean_desired = sum(desired[node_id] for node_id in focus) / len(focus)
            mean_packed = sum(packed[node_id] for node_id in focus) / len(focus)
            shift = mean_desired - mean_packed
            for node_id in packed:
                packed[node_id] += shift
        centers.update(packed)

    positions: dict[str, tuple[float, float]] = {}
    for node_id in node_ids:
        layer_key = depth[node_id]
        width, height = _size_for(sizes, node_id)
        x = centers[node_id] - width / 2.0
        y = layer_y[layer_key] + (layer_heights[layer_key] - height) / 2.0
        if not math.isfinite(x):
            x = 0.0
        if not math.isfinite(y):
            y = 0.0
        positions[node_id] = (x, y)

    # 平移到非负坐标，避免图整体跑到很远的负方向。
    if positions:
        min_x = min(position[0] for position in positions.values())
        min_y = min(position[1] for position in positions.values())
        if min_x != 0.0 or min_y != 0.0:
            positions = {
                node_id: (x - min_x, y - min_y) for node_id, (x, y) in positions.items()
            }
    return positions


__all__ = ["compute_layout"]
