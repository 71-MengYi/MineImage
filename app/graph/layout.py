"""纵向 DAG 自动布局。

节点深度 = 从根节点出发的最长路径长度。布局时先按节点当前的左右位置确定
根节点顺序，再让子节点严格跟随父节点在上一层的左右顺序，逐层从左到右放置，
最后把坐标对齐到 GRID 网格。这样兄弟节点的子树不会因为节点添加 / 连线顺序
不同而交叉。布局结果以节点左上角场景坐标返回，保证坐标有限且互不重叠。
"""

from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Mapping

from app.core.model import GraphError, GraphModel

DEFAULT_NODE_SIZE = (222.0, 130.0)
H_GAP = 80.0
V_GAP = 110.0
#: 自动布局网格，与画布背景网格（GraphScene.GRID）保持一致。
GRID = 28.0


def _size_for(sizes: Mapping[str, tuple[float, float]], node_id: str) -> tuple[float, float]:
    width, height = sizes.get(node_id, DEFAULT_NODE_SIZE)
    width = float(width)
    height = float(height)
    if not math.isfinite(width) or width <= 1.0:
        width = DEFAULT_NODE_SIZE[0]
    if not math.isfinite(height) or height <= 1.0:
        height = DEFAULT_NODE_SIZE[1]
    return width, height


def _snap(value: float, grid: float) -> float:
    """把坐标吸附到网格；半格向上取整，保证结果确定且可复现。"""

    if not math.isfinite(value) or grid <= 0.0:
        return value
    return math.floor(value / grid + 0.5) * grid


def _layered_order(graph: GraphModel, node_ids: list[str]) -> list[str]:
    """Kahn 拓扑排序（按唯一后继计算入度）。

    GraphModel.successors 会对重复边去重，因此不能直接用模型自带的
    topological_order（同一对节点的两条连线会被误判为环）。这里用去重后的
    邻接表重新计算，兼容“一个节点的多个输出连到另一个节点多个输入”的情况。
    """

    selected = set(node_ids)
    successors = {
        node_id: [child for child in graph.successors(node_id) if child in selected]
        for node_id in node_ids
    }
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
    node_ids: Iterable[str] | None = None,
    grid: float = GRID,
) -> dict[str, tuple[float, float]]:
    """计算布局坐标；图中有环时抛出 GraphError。

    node_ids 不为空时只布局这些节点（画布用它只排布顶层节点，算子组内部
    节点保持相对位置不变）。指向集合之外的连线会被忽略。

    层内顺序与节点添加 / 连线顺序无关：根节点按当前左→右位置排序，之后每层
    按父节点在上一层的顺序分组，再按当前左→右位置排列。放置阶段只做“从左到
    右、遇到重叠向右让位”，不会重新排序，因此兄弟节点的子树不会交叉。

    grid > 0 时把节点左上角坐标吸附到网格，避免布局结果出现零散小数。
    """

    sizes = sizes or {}
    node_ids = (
        [node_id for node_id in node_ids if node_id in graph.nodes]
        if node_ids is not None
        else list(graph.nodes.keys())
    )
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
    node_sizes = {node_id: _size_for(sizes, node_id) for node_id in node_ids}
    layer_heights: dict[int, float] = {
        layer_key: max(node_sizes[node_id][1] for node_id in layers[layer_key])
        for layer_key in layer_keys
    }

    def seed_x(node_id: str) -> float:
        """节点当前横向位置。

        排序用原始 x 而不是吸附后的 x：两个位置很接近的节点可能落在同一格内，
        先吸附会让它们的左右顺序退化成 id 顺序。网格吸附只用于最终坐标。
        """

        node = graph.nodes.get(node_id)
        raw = float(node.pos[0]) if node is not None else 0.0
        return raw if math.isfinite(raw) else 0.0

    # 1) 层内顺序：从左到右。根节点看当前 x；其余节点先按父节点顺序分组。
    layer_order: dict[int, list[str]] = {}
    previous_key: int | None = None
    for layer_key in layer_keys:
        layer = list(layers[layer_key])
        if previous_key is None:
            layer.sort(key=lambda node_id: (seed_x(node_id), node_id))
        else:
            position = {
                node_id: index for index, node_id in enumerate(layer_order[previous_key])
            }

            def sort_key(
                node_id: str,
                position: dict[str, int] = position,
            ) -> tuple[float, float, str]:
                parent_indexes = sorted(
                    position[parent]
                    for parent in graph.predecessors(node_id)
                    if parent in position
                )
                if parent_indexes:
                    group = sum(parent_indexes) / len(parent_indexes)
                else:
                    # 父节点不在本层（例如只布局顶层节点时），排到最右侧。
                    group = float(len(position))
                return (group, seed_x(node_id), node_id)

            layer.sort(key=sort_key)
        layer_order[layer_key] = layer
        previous_key = layer_key

    # 2) 横向放置：先自上而下让子节点落在父节点下方。
    centers: dict[str, float] = {}
    for layer_key in layer_keys:
        cursor_right: float | None = None
        for node_id in layer_order[layer_key]:
            width = node_sizes[node_id][0]
            parents = sorted(
                parent for parent in graph.predecessors(node_id) if parent in centers
            )
            if parents:
                center = sum(centers[parent] for parent in parents) / len(parents)
            elif cursor_right is None:
                center = width / 2.0
            else:
                center = cursor_right + h_gap + width / 2.0
            if cursor_right is not None:
                center = max(center, cursor_right + h_gap + width / 2.0)
            centers[node_id] = center
            cursor_right = center + width / 2.0

    # 3) 再自下而上把父节点居中到子节点上方；顺序固定，不重新排序。
    for layer_key in reversed(layer_keys[:-1]):
        layer = layer_order[layer_key]
        desired: dict[str, float] = {}
        has_children: dict[str, bool] = {}
        for node_id in layer:
            children = sorted(
                child
                for child in graph.successors(node_id)
                if child in centers and depth.get(child, -1) > layer_key
            )
            has_children[node_id] = bool(children)
            if children:
                desired[node_id] = sum(centers[child] for child in children) / len(children)
            else:
                desired[node_id] = centers[node_id]
        packed: dict[str, float] = {}
        cursor_right = None
        for node_id in layer:
            width = node_sizes[node_id][0]
            center = desired[node_id]
            if cursor_right is not None:
                center = max(center, cursor_right + h_gap + width / 2.0)
            packed[node_id] = center
            cursor_right = center + width / 2.0
        if len(layer) > 1:
            # 只让“有子节点”的父节点决定整层偏移；孤立叶子不应把父节点推离
            # 子节点中心。若整层都是叶子，则以整层均值居中即可。
            focus = [node_id for node_id in layer if has_children[node_id]] or layer
            mean_desired = sum(desired[node_id] for node_id in focus) / len(focus)
            mean_packed = sum(packed[node_id] for node_id in focus) / len(focus)
            shift = mean_desired - mean_packed
            for node_id in packed:
                packed[node_id] += shift
        centers.update(packed)

    # 4) 纵向分层：每层顶边对齐网格，同层节点顶边对齐。
    layer_tops: dict[int, float] = {}
    cursor_y = 0.0
    for layer_key in layer_keys:
        layer_tops[layer_key] = _snap(cursor_y, grid)
        cursor_y = layer_tops[layer_key] + layer_heights[layer_key] + v_gap

    positions: dict[str, tuple[float, float]] = {}
    for node_id in node_ids:
        layer_key = depth[node_id]
        width, _height = node_sizes[node_id]
        x = _snap(centers[node_id] - width / 2.0, grid)
        y = layer_tops[layer_key]
        if not math.isfinite(x):
            x = 0.0
        if not math.isfinite(y):
            y = 0.0
        positions[node_id] = (x, y)

    # 平移到非负坐标，避免图整体跑到很远的负方向。整体平移量是网格的整数倍，
    # 平移后仍然对齐网格。
    if positions:
        min_x = min(position[0] for position in positions.values())
        min_y = min(position[1] for position in positions.values())
        if min_x != 0.0 or min_y != 0.0:
            positions = {
                node_id: (x - min_x, y - min_y) for node_id, (x, y) in positions.items()
            }
    return positions


__all__ = ["GRID", "compute_layout"]
