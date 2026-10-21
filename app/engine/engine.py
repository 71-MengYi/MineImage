"""异步 DAG 执行引擎。

设计要点
--------
* 每次请求在主线程抓取一份不可变的图快照（含算子注册表引用），后台线程
  完全不接触正在变化的模型；
* 一次请求内部用 concurrent.futures.ThreadPoolExecutor 并行执行互不依赖的
  节点：任务按拓扑序提交，兄弟分支（同一节点的多个下游）真正同时运行，
  有依赖的节点在任务内部等待上游 future 完成；
* 节点数据更新后会递归更新全部下游：preview_node / commit_node /
  update_nodes 计算目标节点的同时计算它的所有子孙；refresh_node 仍只重算
  单个节点（以及它尚未缓存的上游）；
* 通过 generation + revision 双重检查丢弃过期结果：新请求会让旧 generation
  的所有排队信号失效，节点 revision 变化后旧结果也不会写入缓存；旧的慢算子
  无法中断时允许新请求并行启动，旧结果最终被 generation 检查丢弃；
* 节点级结果缓存（快照）：点击节点立刻拿到上次结果，参数变更会重算当前节点
  与下游，保证画布显示的数据始终与参数一致；
* 算子异常被捕获并记录，节点标红，不中断其它分支；
* 默认并行度取 AppSettings.execution_workers（默认 4，范围 1..16），构造时
  也可用 workers= 显式指定；
* shutdown() 会先使 generation 失效再等待后台线程，避免退出后还有结果回写
  到已销毁的界面。

线程模型
--------
同一个请求内的节点任务由多个后台线程并发执行，因此 Operator.process() 必须
线程安全：不得依赖算子实例上的可变状态，也不得原地修改输入图像（传入的
ImageData 可能被多个兄弟分支同时读取）。历史上“算子只在单一后台线程里串行
执行”的假设已经移除。

引擎自身的结果缓存只在主线程的槽函数里写入 _results / _errors；后台线程只
通过 Qt 信号（自动排队到主线程）报告进度，并用一把只保护 cache /
error_nodes / failed 的锁同步线程间的求值状态。
"""

from __future__ import annotations

import copy
import logging
import threading
import time
import traceback
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from app.config import AppSettings
from app.core.data_types import data_type_of, summarize_outputs
from app.core.model import WorkspaceModel
from app.core.types import ImageData
from app.operators.loader import OperatorRegistry
from app.operators.spec import (
    OperatorError,
    ParamValidationError,
    normalize_params,
    output_to_dict,
    validate_inputs,
)

LOG = logging.getLogger(__name__)


@dataclass
class NodeResult:
    node_id: str
    outputs: dict[str, Any]
    elapsed_ms: float
    revision: int
    preview: bool = True
    port_labels: dict[str, str] = field(default_factory=dict)

    @property
    def primary(self) -> ImageData | None:
        """第一个图像输出；纯数值 / 结构化数据的节点返回 None。"""

        for value in self.outputs.values():
            if isinstance(value, ImageData):
                return value
        return None

    @property
    def preview_image(self) -> ImageData | None:
        """用于主预览 / 缩略图的图像：只取图像输出，结构化数据不再渲染。"""

        return self.primary

    def output_type_text(self, key: str) -> str:
        value = self.outputs.get(key)
        data_type = data_type_of(value)
        return data_type.label if data_type is not None else ""

    def summary_lines(self) -> list[str]:
        return summarize_outputs(self.outputs, self.port_labels)

    def summary_text(self) -> str:
        lines = self.summary_lines()
        if not lines:
            return "算子没有输出"
        return "；".join(lines)

    def output_type_summary(self) -> str:
        """按输出数据类型汇总，例如 图像、连通域、数值×2。"""

        counts: dict[str, int] = {}
        for value in self.outputs.values():
            data_type = data_type_of(value)
            label = data_type.label if data_type is not None else "未知数据"
            counts[label] = counts.get(label, 0) + 1
        return "、".join(
            f"{label}×{count}" if count > 1 else label for label, count in counts.items()
        )


@dataclass
class NodeError:
    node_id: str
    message: str
    details: str = ""
    revision: int = 0


@dataclass
class _NodeSnap:
    id: str
    op_id: str
    title: str
    params: dict[str, Any]
    source_image_id: str | None
    revision: int


@dataclass
class _GraphSnapshot:
    nodes: dict[str, _NodeSnap]
    inputs: dict[tuple[str, str], tuple[str, str]]
    children: dict[str, list[str]]
    images: dict[str, ImageData]
    registry: OperatorRegistry
    seed: dict[str, NodeResult] = field(default_factory=dict)
    compute_downstream: bool = False


class _CancelToken:
    def __init__(self) -> None:
        self._event = threading.Event()

    def cancel(self) -> None:
        self._event.set()

    @property
    def cancelled(self) -> bool:
        return self._event.is_set()


class _EvalJob(QRunnable):
    """一次求值请求：在 QThreadPool 线程里驱动内部的多线程节点执行器。"""

    def __init__(
        self,
        engine: "ExecutionEngine",
        roots: tuple[str, ...],
        generation: int,
        snapshot: _GraphSnapshot,
        token: _CancelToken,
        max_workers: int,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.roots = roots
        self.generation = generation
        self.snapshot = snapshot
        self.token = token
        self.max_workers = max_workers
        self.setAutoDelete(True)

    def run(self) -> None:  # pragma: no cover - 后台线程
        self.engine._run_job(
            self.roots,
            self.generation,
            self.snapshot,
            self.token,
            self.max_workers,
        )


class ExecutionEngine(QObject):
    """工作区级别的执行引擎。每个工作区（tab）持有一个实例。"""

    node_started = Signal(str)
    node_finished = Signal(object)  # NodeResult
    node_failed = Signal(str, str)  # node id, message
    chain_finished = Signal(str, bool)  # target id, success
    busy_changed = Signal(bool)

    # 内部跨线程信号（接收者都是本对象，自动排队到主线程）
    _started = Signal(int, str)
    _computed = Signal(int, object)
    _failed = Signal(int, str, int, str, str)  # generation, node, revision, message, details
    _finished = Signal(int, object, bool)  # generation, roots, success

    def __init__(
        self,
        workspace: WorkspaceModel,
        registry: OperatorRegistry,
        settings: AppSettings,
        parent: QObject | None = None,
        *,
        workers: int | None = None,
    ) -> None:
        super().__init__(parent)
        self.workspace = workspace
        self.registry = registry
        self.settings = settings
        if workers is None:
            workers = settings.execution_workers
        try:
            self._workers = max(1, int(workers))
        except (TypeError, ValueError):
            self._workers = 1
        self._results: dict[str, NodeResult] = {}
        self._errors: dict[str, NodeError] = {}
        self._generation = 0
        self._token: _CancelToken | None = None
        self._busy = False
        self._shutting_down = False
        self._pool = QThreadPool(self)
        # 允许两个请求同时在跑：新请求不必等待旧请求里无法中断的慢算子结束，
        # 旧 generation 的排队结果会由 _on_* 的 generation 检查丢弃。
        self._pool.setMaxThreadCount(2)

        self._started.connect(self._on_started)
        self._computed.connect(self._on_computed)
        self._failed.connect(self._on_failed)
        self._finished.connect(self._on_chain_finished)

    # ------------------------------------------------------------------ 查询
    def result(self, node_id: str) -> NodeResult | None:
        return self._results.get(node_id)

    def error(self, node_id: str) -> NodeError | None:
        return self._errors.get(node_id)

    @property
    def busy(self) -> bool:
        return self._busy

    def result_revision(self, node_id: str) -> int | None:
        result = self._results.get(node_id)
        return result.revision if result else None

    # ------------------------------------------------------------------ 请求
    def preview_node(self, node_id: str) -> None:
        """选中 / 预览节点：重算该节点并递归更新全部下游。

        已有同一 revision 的快照时同步回放，且绝不取消正在运行的后台任务；
        同一 revision 的失败也直接回放错误，避免重复跑图。需要真正计算时先让
        该节点与全部下游缓存失效，再启动一次“下游递归”请求。
        """
        if self._shutting_down:
            return
        node = self.workspace.graph.nodes.get(node_id)
        if node is None:
            return
        cached = self._results.get(node_id)
        if cached is not None and cached.revision == node.revision:
            # 只是查看已有快照，不是新的计算请求：保留后台任务与其 generation。
            self.node_finished.emit(cached)
            return
        cached_error = self._errors.get(node_id)
        if cached_error is not None and cached_error.revision == node.revision:
            self.node_failed.emit(node_id, cached_error.message)
            return
        self._clear_cached(node_id, downstream=True)
        self._start((node_id,), compute_downstream=True)

    def commit_node(self, node_id: str) -> None:
        """保存：使该节点及全部下游失效，并重算整棵下游子树。"""
        if self._shutting_down:
            return
        if node_id not in self.workspace.graph.nodes:
            return
        self._clear_cached(node_id, downstream=True)
        self._start((node_id,), compute_downstream=True)

    def refresh_node(self, node_id: str) -> None:
        """强制重算单个节点（不重算下游，上游按需补齐）。"""
        if self._shutting_down:
            return
        if node_id not in self.workspace.graph.nodes:
            return
        self._clear_cached(node_id, downstream=False)
        self._start((node_id,), compute_downstream=False)

    def update_nodes(self, node_ids: Iterable[str]) -> None:
        """一次请求重算多个根节点及其全部下游。

        不存在的节点会被忽略；重复根会去重。整批根节点共用一次快照与同一个
        并行执行器，chain_finished 对每个根各发一次（成功与否取整次请求）。
        """
        if self._shutting_down:
            return
        roots: list[str] = []
        seen: set[str] = set()
        for node_id in node_ids:
            if node_id in seen or node_id not in self.workspace.graph.nodes:
                continue
            seen.add(node_id)
            roots.append(node_id)
        if not roots:
            return
        for root in roots:
            self._clear_cached(root, downstream=True)
        self._start(tuple(roots), compute_downstream=True)

    def _start(self, roots: Sequence[str], *, compute_downstream: bool = False) -> None:
        if self._shutting_down:
            return
        roots = tuple(dict.fromkeys(roots))
        if not roots:
            return
        if any(root not in self.workspace.graph.nodes for root in roots):
            roots = tuple(root for root in roots if root in self.workspace.graph.nodes)
            if not roots:
                return
        # 先让 generation 失效，再取消旧任务：即使旧任务正在执行且无法立刻
        # 停止，它后续发出的排队信号也会在主线程被丢弃。
        self._generation += 1
        generation = self._generation
        self._cancel_current()
        token = _CancelToken()
        try:
            snapshot = self._capture_graph()
        except Exception as exc:  # noqa: BLE001 - 快照失败不能把引擎留在“忙”状态
            self._set_busy(False)
            for root in roots:
                self.record_immediate_failure(
                    root,
                    f"无法创建执行快照：{exc.__class__.__name__}: {exc}",
                    traceback.format_exc(),
                )
            return
        snapshot.compute_downstream = compute_downstream
        self._token = token
        self._set_busy(True)
        job = _EvalJob(self, roots, generation, snapshot, token, self._workers)
        self._pool.start(job)

    def _clear_cached(self, node_id: str, *, downstream: bool) -> None:
        """只清空缓存，不影响正在执行的任务（由下一个请求统一取消）。"""
        targets = [node_id]
        if downstream:
            targets.extend(self.workspace.graph.descendants(node_id))
        for target in targets:
            self._results.pop(target, None)
            self._errors.pop(target, None)

    def _abandon_current(self) -> None:
        """放弃当前请求，并让它的 queued 结果全部失效。"""
        self._generation += 1
        self._cancel_current()
        self._set_busy(False)

    def _cancel_current(self) -> None:
        """取消令牌并清空队列中尚未开始的任务；不会主动增加 generation。"""
        if self._token is not None:
            self._token.cancel()
            self._token = None
        self._pool.clear()

    def _set_busy(self, value: bool) -> None:
        if self._busy == value:
            return
        self._busy = value
        self.busy_changed.emit(value)

    def invalidate(self, node_id: str, *, downstream: bool = True) -> None:
        graph = self.workspace.graph
        targets = [node_id]
        if downstream:
            targets.extend(graph.descendants(node_id))
        for target in targets:
            self._results.pop(target, None)
            self._errors.pop(target, None)
        if self._busy:
            # 正在执行/排队的任务可能引用旧参数；让它的结果全部失效。
            self._abandon_current()

    def invalidate_all(self) -> None:
        self._generation += 1
        self._cancel_current()
        self._results.clear()
        self._errors.clear()
        self._set_busy(False)

    def forget_nodes(self, node_ids: Iterable[str]) -> None:
        for node_id in node_ids:
            self._results.pop(node_id, None)
            self._errors.pop(node_id, None)

    def set_workers(self, count: int) -> None:
        """调整后续请求的并行 worker 数；不会中断正在执行的请求。"""

        try:
            self._workers = max(1, min(16, int(count)))
        except (TypeError, ValueError):
            self._workers = 1

    def set_registry(self, registry: OperatorRegistry) -> None:
        self.registry = registry
        self.invalidate_all()

    def shutdown(self) -> None:
        """停止后台计算；可重复调用。"""
        if self._shutting_down:
            return
        self._shutting_down = True
        self._generation += 1
        self._cancel_current()
        self._results.clear()
        self._errors.clear()
        self._set_busy(False)
        # 后台算子不可中断，这里给一个上界避免界面退出被无限阻塞。
        # 正常情况下单帧算子远快于此；超时后 QThreadPool 由 Qt 在析构时等待。
        self._pool.waitForDone(15000)

    # ------------------------------------------------------------------ 快照
    def _capture_graph(self) -> _GraphSnapshot:
        graph = self.workspace.graph
        nodes: dict[str, _NodeSnap] = {}
        for node in graph.nodes.values():
            nodes[node.id] = _NodeSnap(
                id=node.id,
                op_id=node.op_id,
                title=node.title,
                params=copy.deepcopy(node.params),
                source_image_id=node.source_image_id,
                revision=node.revision,
            )
        inputs: dict[tuple[str, str], tuple[str, str]] = {}
        children: dict[str, list[str]] = {node_id: [] for node_id in nodes}
        for connection in graph.connections.values():
            if connection.src_node not in nodes or connection.dst_node not in nodes:
                continue
            inputs[(connection.dst_node, connection.dst_port)] = (
                connection.src_node,
                connection.src_port,
            )
            children.setdefault(connection.src_node, []).append(connection.dst_node)
        seed = {
            node_id: result
            for node_id, result in self._results.items()
            if node_id in nodes and nodes[node_id].revision == result.revision
        }
        return _GraphSnapshot(
            nodes=nodes,
            inputs=inputs,
            children=children,
            images=dict(self.workspace.images),
            registry=self.registry,
            seed=seed,
        )

    # ------------------------------------------------------------- 后台执行
    def _run_job(
        self,
        roots: Sequence[str],
        generation: int,
        snapshot: _GraphSnapshot,
        token: _CancelToken,
        max_workers: int,
    ) -> None:
        state = _EvalState(self, snapshot, generation, token, max_workers)
        try:
            state.compute(roots)
        except Exception as exc:  # noqa: BLE001 - 防止后台线程崩溃
            fallback = roots[0] if roots else ""
            state.record_failure(fallback, f"执行失败：{exc}", traceback.format_exc())
        finally:
            self._finished.emit(generation, list(roots), state.success)

    def record_immediate_failure(self, node_id: str, message: str, details: str = "") -> None:
        """主线程侧的失败通知（例如快照创建失败）。"""
        node = self.workspace.graph.nodes.get(node_id)
        revision = node.revision if node is not None else 0
        LOG.error(
            "节点执行失败 [%s] %s：%s",
            node_id,
            node.title if node is not None else "?",
            message,
            extra={"log_detail": details},
        )
        self._errors[node_id] = NodeError(node_id, message, details, revision)
        self._results.pop(node_id, None)
        self.node_failed.emit(node_id, message)
        self.chain_finished.emit(node_id, False)

    # ------------------------------------------------------------------ 槽
    def _on_started(self, generation: int, node_id: str) -> None:
        if self._shutting_down or generation != self._generation:
            return
        if node_id not in self.workspace.graph.nodes:
            return
        self.node_started.emit(node_id)

    def _on_computed(self, generation: int, result: NodeResult) -> None:
        if self._shutting_down or generation != self._generation:
            return
        node = self.workspace.graph.nodes.get(result.node_id)
        if node is None or node.revision != result.revision:
            return
        self._results[result.node_id] = result
        self._errors.pop(result.node_id, None)
        self.node_finished.emit(result)

    def _on_failed(self, generation: int, node_id: str, revision: int, message: str, details: str) -> None:
        if self._shutting_down or generation != self._generation:
            return
        node = self.workspace.graph.nodes.get(node_id)
        if node is None or node.revision != revision:
            return
        self._results.pop(node_id, None)
        self._errors[node_id] = NodeError(node_id, message, details, revision)
        self.node_failed.emit(node_id, message)

    def _on_chain_finished(self, generation: int, roots: Sequence[str], success: bool) -> None:
        if self._shutting_down or generation != self._generation:
            return
        self._set_busy(False)
        for root in roots:
            self.chain_finished.emit(root, success)


class _EvalState:
    """一次请求内部的执行状态。

    实例会被同一请求内的多个 worker 线程并发访问，因此 cache / error_nodes /
    failed 统一由 self._lock 保护。锁只覆盖字典与集合操作，绝不在持锁期间执行
    算子。
    """

    def __init__(
        self,
        engine: ExecutionEngine,
        snapshot: _GraphSnapshot,
        generation: int,
        token: _CancelToken,
        max_workers: int,
    ) -> None:
        self.engine = engine
        self.snapshot = snapshot
        self.generation = generation
        self.token = token
        self.max_workers = max(1, int(max_workers))
        self._lock = threading.Lock()
        self.cache: dict[str, NodeResult] = dict(snapshot.seed)
        self.error_nodes: set[str] = set()
        self.failed = False
        self._preds, self._succs = self._build_adjacency()

    # ------------------------------------------------------------- 状态访问
    @property
    def success(self) -> bool:
        with self._lock:
            return not self.failed

    def _cached(self, node_id: str) -> NodeResult | None:
        with self._lock:
            return self.cache.get(node_id)

    def _cached_failure(self, node_id: str) -> bool:
        with self._lock:
            return node_id in self.error_nodes

    def _store_result(self, result: NodeResult) -> None:
        with self._lock:
            self.cache[result.node_id] = result
            self.error_nodes.discard(result.node_id)

    def _mark_failure(self, node_id: str) -> bool:
        with self._lock:
            if node_id in self.error_nodes:
                return False
            self.error_nodes.add(node_id)
            self.failed = True
            return True

    # --------------------------------------------------------------- 图收集
    def _build_adjacency(self) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
        """从快照 inputs 构建去重后的前驱/后继邻接表。

        同一对节点之间可能存在多条端口连线（例如通道拆分 / 合并），这里按
        “唯一节点对”去重，避免拓扑排序把并行端口误判成多条依赖边。
        """
        nodes = self.snapshot.nodes
        preds: dict[str, list[str]] = {node_id: [] for node_id in nodes}
        succs: dict[str, list[str]] = {node_id: [] for node_id in nodes}
        seen: set[tuple[str, str]] = set()
        for (dst_node, _dst_port), (src_node, _src_port) in self.snapshot.inputs.items():
            if src_node not in nodes or dst_node not in nodes:
                continue
            edge = (src_node, dst_node)
            if edge in seen:
                continue
            seen.add(edge)
            preds[dst_node].append(src_node)
            succs[src_node].append(dst_node)
        return preds, succs

    def _closure(self, starts: Iterable[str], adjacency: dict[str, list[str]]) -> set[str]:
        seen: set[str] = set()
        queue: deque[str] = deque(
            node_id for node_id in starts if node_id in self.snapshot.nodes
        )
        while queue:
            node_id = queue.popleft()
            if node_id in seen:
                continue
            seen.add(node_id)
            for neighbour in adjacency.get(node_id, ()):
                if neighbour not in seen:
                    queue.append(neighbour)
        return seen

    def _collect_nodes(self, roots: Sequence[str]) -> set[str]:
        """计算集合：根的祖先闭包 ∪ 根（∪ 下游闭包及其祖先）。"""
        selected = self._closure(roots, self._preds)
        if self.snapshot.compute_downstream:
            downstream = self._closure(roots, self._succs)
            selected |= downstream
            # 下游节点可能还有根之外的上游（例如两条分支汇合的 merge）：这些
            # 外部输入必须一起求值，否则汇合节点拿不到输入。该扩展只会补上
            # 计算所必需的祖先，已有快照的节点会直接命中缓存。
            selected |= self._closure(downstream, self._preds)
        return {node_id for node_id in selected if node_id in self.snapshot.nodes}

    def _topological_order(self, selected: set[str]) -> list[str]:
        """Kahn 拓扑排序；同一对节点的多条端口连线只算一条边。"""
        indegree: dict[str, int] = {}
        for node_id in selected:
            indegree[node_id] = sum(
                1 for src in self._preds.get(node_id, ()) if src in selected
            )
        queue: deque[str] = deque(
            node_id
            for node_id in self.snapshot.nodes
            if node_id in selected and indegree[node_id] == 0
        )
        order: list[str] = []
        while queue:
            node_id = queue.popleft()
            order.append(node_id)
            for dst_node in self._succs.get(node_id, ()):
                if dst_node not in indegree:
                    continue
                indegree[dst_node] -= 1
                if indegree[dst_node] == 0:
                    queue.append(dst_node)
        if len(order) != len(selected):
            # 模型层已经禁止回环；这里只做防御，保证任务仍会被提交而不死锁。
            ordered = set(order)
            remaining = [
                node_id
                for node_id in self.snapshot.nodes
                if node_id in selected and node_id not in ordered
            ]
            LOG.warning(
                "执行快照存在环：%d 个节点无法排序，已追加到末尾",
                len(remaining),
            )
            order.extend(remaining)
        return order

    # ------------------------------------------------------------------ 执行
    def compute(self, roots: Sequence[str]) -> None:
        """并行求值：按拓扑序提交任务，依赖节点等待上游 future。"""
        selected = self._collect_nodes(roots)
        order = self._topological_order(selected)
        if not order:
            return
        futures: dict[str, Future[Any]] = {}
        max_workers = min(self.max_workers, len(order))
        with ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="mineimage-eval",
        ) as executor:
            for node_id in order:
                # 拓扑序保证依赖节点一定先提交，因此这里能拿到全部上游 future；
                # 任务内部不再读写共享的 futures 字典，避免额外加锁。
                deps = {
                    src_node: futures[src_node]
                    for src_node in self._preds.get(node_id, ())
                    if src_node in futures
                }
                futures[node_id] = executor.submit(self._run_node, node_id, deps)
            for future in futures.values():
                try:
                    future.result()
                except Exception:  # noqa: BLE001 - _run_node 已兜底，这里只等待完成
                    LOG.exception("节点任务异常退出")
        with self._lock:
            self.failed = self.failed or bool(self.error_nodes)

    def _run_node(
        self,
        node_id: str,
        deps: dict[str, Future[Any]],
    ) -> NodeResult | None:
        """单个节点任务：任何异常都在任务内部消化，不抛给等待方。"""
        try:
            return self._evaluate_node(node_id, deps)
        except Exception as exc:  # noqa: BLE001 - 后台任务不能把异常抛给等待方
            self.record_failure(node_id, f"执行失败：{exc}", traceback.format_exc())
            return None

    def _evaluate_node(
        self,
        node_id: str,
        deps: dict[str, Future[Any]],
    ) -> NodeResult | None:
        if self.token.cancelled:
            return None
        cached = self._cached(node_id)
        if cached is not None:
            # seed 里已有 fresh 结果：不重算、不发 _started。
            return cached
        if self._cached_failure(node_id):
            return None
        snap = self.snapshot.nodes.get(node_id)
        if snap is None:
            return None
        if snap.op_id == "__source__":
            return self._compute_source(snap)
        return self._compute_operator(snap, deps)

    def _wait_upstream(
        self,
        src_node: str,
        deps: dict[str, Future[Any]],
    ) -> NodeResult | None:
        future = deps.get(src_node)
        if future is not None:
            try:
                return future.result()
            except Exception:  # noqa: BLE001 - 上游任务已兜底，仍防御一次
                LOG.exception("上游节点 %s 任务异常", src_node)
                return None
        # 依赖不在本次计算集合里（理论上已被祖先闭包补齐）：只能复用已有快照。
        return self._cached(src_node)

    def _compute_source(self, snap: _NodeSnap) -> NodeResult | None:
        self.engine._started.emit(self.generation, snap.id)
        started = time.perf_counter()
        image = self.snapshot.images.get(snap.source_image_id or "")
        if image is None:
            self.record_failure(snap.id, "源图像数据缺失，请重新载入图像")
            return None
        wrapped = ImageData(image.array, snap.title or image.name, image.color_space)
        elapsed = (time.perf_counter() - started) * 1000.0
        result = NodeResult(
            snap.id,
            {"image": wrapped},
            elapsed,
            snap.revision,
            port_labels={"image": "图像"},
        )
        self._store_result(result)
        self.engine._computed.emit(self.generation, result)
        return result

    def _compute_operator(
        self,
        snap: _NodeSnap,
        deps: dict[str, Future[Any]],
    ) -> NodeResult | None:
        entry = self.snapshot.registry.entry(snap.op_id)
        if entry is None:
            self.record_failure(snap.id, f"找不到算子“{snap.op_id}”，请刷新算子库")
            return None
        spec = entry.spec
        inputs: dict[str, Any | None] = {}
        upstream_results: dict[str, NodeResult | None] = {}
        for port in spec.inputs:
            source = self.snapshot.inputs.get((snap.id, port.key))
            if source is None:
                inputs[port.key] = None
                continue
            src_node, src_port = source
            if src_node not in upstream_results:
                # 同一 src 的多个端口只等待一次 future。
                upstream_results[src_node] = self._wait_upstream(src_node, deps)
            upstream = upstream_results[src_node]
            if upstream is None:
                self.record_failure(
                    snap.id,
                    f"上游节点执行失败（端口“{port.label}”），无法计算当前节点",
                )
                return None
            inputs[port.key] = upstream.outputs.get(src_port)
            if inputs[port.key] is None:
                self.record_failure(
                    snap.id,
                    f"上游节点未提供输出端口“{src_port}”的数据",
                )
                return None
        if self.token.cancelled:
            return None
        started = time.perf_counter()
        # 上游就绪后才算真正开始执行本节点，UI 的“运行中”高亮更准确。
        self.engine._started.emit(self.generation, snap.id)
        try:
            params = normalize_params(spec, snap.params)
            validate_inputs(spec, inputs)
            entry.operator.validate(inputs, params)
            call_kwargs: dict[str, Any] = dict(inputs)
            call_kwargs.update(params)
            output = entry.operator.process(**call_kwargs)
            outputs = output_to_dict(spec, output)
            if not outputs:
                raise OperatorError("算子没有产生任何输出")
            for key, value in outputs.items():
                if isinstance(value, ImageData) and value.name in ("", "image"):
                    value.name = f"{snap.title}:{key}"
        except ParamValidationError as exc:
            self.record_failure(snap.id, f"参数错误：{exc.message}", traceback.format_exc())
            return None
        except OperatorError as exc:
            self.record_failure(snap.id, str(exc), traceback.format_exc())
            return None
        except Exception as exc:  # noqa: BLE001 - 算子内部异常统一转成节点错误
            self.record_failure(
                snap.id,
                f"算子执行失败：{exc.__class__.__name__}: {exc}",
                traceback.format_exc(),
            )
            return None
        elapsed = (time.perf_counter() - started) * 1000.0
        labels = {port.key: port.label for port in spec.outputs if port.key in outputs}
        result = NodeResult(snap.id, outputs, elapsed, snap.revision, port_labels=labels)
        self._store_result(result)
        self.engine._computed.emit(self.generation, result)
        return result

    def record_failure(self, node_id: str, message: str, details: str = "") -> None:
        if not self._mark_failure(node_id):
            return
        snap = self.snapshot.nodes.get(node_id)
        revision = snap.revision if snap is not None else 0
        LOG.error(
            "节点执行失败 [%s] %s (%s)：%s",
            node_id,
            snap.title if snap is not None else "?",
            snap.op_id if snap is not None else "?",
            message,
            extra={"log_detail": details},
        )
        self.engine._failed.emit(self.generation, node_id, revision, message, details)
