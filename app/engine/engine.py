"""异步 DAG 执行引擎。

设计要点
--------
* 所有计算在单一后台线程池中执行，UI 线程只接收结果信号；
* 每次请求在主线程抓取一份不可变的图快照（含算子注册表引用），后台线程
  完全不接触正在变化的模型；
* 通过 generation + revision 双重检查丢弃过期结果：新请求会让旧 generation
  的所有排队信号失效，节点 revision 变化后旧结果也不会写入缓存；
* 节点级结果缓存（快照）：点击节点立刻拿到上次结果，参数变更只重算当前节点，
  点击“保存”后才重算下游；
* 算子异常被捕获并记录，节点标红，不中断其它分支；
* ``shutdown()`` 会先使 generation 失效再等待后台线程，避免退出后还有结果
  回写到已销毁的界面。
"""

from __future__ import annotations

import copy
import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Any, Iterable

from PySide6.QtCore import QObject, QRunnable, QThreadPool, Signal

from app.config import AppSettings
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


@dataclass
class NodeResult:
    node_id: str
    outputs: dict[str, ImageData]
    elapsed_ms: float
    revision: int
    preview: bool = True

    @property
    def primary(self) -> ImageData | None:
        if not self.outputs:
            return None
        return next(iter(self.outputs.values()))


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
    def __init__(
        self,
        engine: "ExecutionEngine",
        target_id: str,
        generation: int,
        snapshot: _GraphSnapshot,
        token: _CancelToken,
    ) -> None:
        super().__init__()
        self.engine = engine
        self.target_id = target_id
        self.generation = generation
        self.snapshot = snapshot
        self.token = token
        self.setAutoDelete(True)

    def run(self) -> None:  # pragma: no cover - 后台线程
        self.engine._run_job(self.target_id, self.generation, self.snapshot, self.token)


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
    _finished = Signal(int, str, bool)

    def __init__(
        self,
        workspace: WorkspaceModel,
        registry: OperatorRegistry,
        settings: AppSettings,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self.workspace = workspace
        self.registry = registry
        self.settings = settings
        self._results: dict[str, NodeResult] = {}
        self._errors: dict[str, NodeError] = {}
        self._generation = 0
        self._token: _CancelToken | None = None
        self._busy = False
        self._shutting_down = False
        self._pool = QThreadPool(self)
        # 单后台线程：保证同一时刻只有一个计算在跑，并且结果顺序可控。
        self._pool.setMaxThreadCount(1)

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
        """点击节点：有快照时立即回放，不打断正在运行的计算；否则异步计算上游。"""
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
            # 同一 revision 下重复点击失败节点：直接回放错误，避免重复跑图。
            self.node_failed.emit(node_id, cached_error.message)
            return
        self._start(node_id)

    def commit_node(self, node_id: str) -> None:
        """保存：使该节点及全部下游失效，并重算整棵下游子树。"""
        if self._shutting_down:
            return
        if node_id not in self.workspace.graph.nodes:
            return
        self.invalidate(node_id, downstream=True)
        self._start(node_id, compute_downstream=True)

    def refresh_node(self, node_id: str) -> None:
        """强制重算单个节点（不重算下游）。"""
        if self._shutting_down:
            return
        if node_id not in self.workspace.graph.nodes:
            return
        self.invalidate(node_id, downstream=False)
        self._start(node_id)

    def _start(self, target_id: str, *, compute_downstream: bool = False) -> None:
        if self._shutting_down:
            return
        if target_id not in self.workspace.graph.nodes:
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
            self.record_immediate_failure(
                target_id,
                f"无法创建执行快照：{exc.__class__.__name__}: {exc}",
                traceback.format_exc(),
            )
            return
        snapshot.compute_downstream = compute_downstream
        self._token = token
        self._set_busy(True)
        job = _EvalJob(self, target_id, generation, snapshot, token)
        self._pool.start(job)

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
        target_id: str,
        generation: int,
        snapshot: _GraphSnapshot,
        token: _CancelToken,
    ) -> None:
        state = _EvalState(self, snapshot, generation, token)
        try:
            if snapshot.compute_downstream:
                state.compute_subtree(target_id)
            else:
                state.compute_node(target_id)
        except Exception as exc:  # noqa: BLE001 - 防止后台线程崩溃
            state.record_failure(target_id, f"执行失败：{exc}", traceback.format_exc())
        finally:
            self._finished.emit(generation, target_id, not state.failed)

    def record_immediate_failure(self, node_id: str, message: str, details: str = "") -> None:
        """主线程侧的失败通知（例如快照创建失败）。"""
        node = self.workspace.graph.nodes.get(node_id)
        revision = node.revision if node is not None else 0
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

    def _on_chain_finished(self, generation: int, target_id: str, success: bool) -> None:
        if self._shutting_down or generation != self._generation:
            return
        self._set_busy(False)
        self.chain_finished.emit(target_id, success)


class _EvalState:
    """一次请求内部的执行状态（仅后台线程访问）。"""

    def __init__(
        self,
        engine: ExecutionEngine,
        snapshot: _GraphSnapshot,
        generation: int,
        token: _CancelToken,
    ) -> None:
        self.engine = engine
        self.snapshot = snapshot
        self.generation = generation
        self.token = token
        self.cache: dict[str, NodeResult] = dict(snapshot.seed)
        self.failed = False
        self.error_nodes: set[str] = set()

    # ------------------------------------------------------------------ 执行
    def compute_node(self, node_id: str) -> NodeResult | None:
        if self.token.cancelled:
            return None
        if node_id in self.cache:
            return self.cache[node_id]
        if node_id in self.error_nodes:
            return None
        snap = self.snapshot.nodes.get(node_id)
        if snap is None:
            return None

        if snap.op_id == "__source__":
            result = self._compute_source(snap)
        else:
            result = self._compute_operator(snap)
        if result is not None:
            self.cache[node_id] = result
            self.engine._computed.emit(self.generation, result)
        return result

    def compute_subtree(self, target_id: str) -> None:
        start = self.compute_node(target_id)
        if start is None:
            self.failed = True
        # 下游按 BFS 顺序逐个计算；每个节点所需的其它上游由 compute_node
        # 递归补齐（DAG 已在模型层保证无环），重复访问会被 cache 命中。
        queue = list(self.snapshot.children.get(target_id, []))
        visited: set[str] = set()
        while queue and not self.token.cancelled:
            node_id = queue.pop(0)
            if node_id in visited:
                continue
            visited.add(node_id)
            self.compute_node(node_id)
            queue.extend(self.snapshot.children.get(node_id, []))
        if self.token.cancelled:
            return
        self.failed = self.failed or bool(self.error_nodes)

    def _compute_source(self, snap: _NodeSnap) -> NodeResult | None:
        self.engine._started.emit(self.generation, snap.id)
        started = time.perf_counter()
        image = self.snapshot.images.get(snap.source_image_id or "")
        if image is None:
            self.record_failure(snap.id, "源图像数据缺失，请重新载入图像")
            return None
        wrapped = ImageData(image.array, snap.title or image.name, image.color_space)
        elapsed = (time.perf_counter() - started) * 1000.0
        return NodeResult(snap.id, {"image": wrapped}, elapsed, snap.revision)

    def _compute_operator(self, snap: _NodeSnap) -> NodeResult | None:
        if self.token.cancelled:
            return None
        started = time.perf_counter()
        entry = self.snapshot.registry.entry(snap.op_id)
        if entry is None:
            self.record_failure(snap.id, f"找不到算子“{snap.op_id}”，请刷新算子库")
            return None
        spec = entry.spec
        inputs: dict[str, ImageData | None] = {}
        for port in spec.inputs:
            source = self.snapshot.inputs.get((snap.id, port.key))
            if source is None:
                inputs[port.key] = None
                continue
            src_node, src_port = source
            upstream = self.compute_node(src_node)
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
            for key, image in outputs.items():
                if image.name in ("", "image"):
                    image.name = f"{snap.title}:{key}"
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
        return NodeResult(snap.id, outputs, elapsed, snap.revision)

    def record_failure(self, node_id: str, message: str, details: str = "") -> None:
        if node_id in self.error_nodes:
            return
        self.failed = True
        self.error_nodes.add(node_id)
        snap = self.snapshot.nodes.get(node_id)
        revision = snap.revision if snap is not None else 0
        self.engine._failed.emit(self.generation, node_id, revision, message, details)
