"""工作区页面：算子库 + 流程画布/参数面板 + 预览 + 引擎的组装。"""

from __future__ import annotations

import copy
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QFrame,
    QMessageBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from app.config import AppSettings
from app.core.imageio import ImageIOError, load_image, save_image
from app.core.model import NodeModel, WorkspaceModel
from app.core.project import save_project
from app.engine.engine import ExecutionEngine, NodeResult
from app.graph.commands import MoveNodeCommand, SetNodeParamsCommand
from app.graph.view import GraphView
from app.operators.loader import OperatorRegistry
from app.ui.library import OperatorLibrary
from app.ui.params import ParamPanel
from app.ui.preview import PreviewWidget

#: 三栏默认宽度比例 算子库 : 画布 : 显示 = 1:2:4。
DEFAULT_SPLITTER_SIZES = (200, 400, 800)
#: 旧版本默认比例（1:1:3 / 1:1:4）；保存的宽度受窗口大小影响，按比例识别。
LEGACY_SPLITTER_RATIOS = ((1, 1, 3), (1, 1, 4))
#: 识别旧默认比例时允许的相对误差（窗口缩放与最小宽度会改变实际数值）。
LEGACY_RATIO_TOLERANCE = 0.22


def _is_legacy_splitter_sizes(sizes: tuple[int, ...]) -> bool:
    """判断保存的三栏宽度是否仍是旧版本的默认比例（视为“未自定义”）。"""

    if len(sizes) != 3 or min(sizes) <= 0:
        return False
    first = float(sizes[0])
    for first_ratio, second_ratio, third_ratio in LEGACY_SPLITTER_RATIOS:
        expected_second = first * second_ratio / first_ratio
        expected_third = first * third_ratio / first_ratio
        if (
            abs(sizes[1] - expected_second) <= expected_second * LEGACY_RATIO_TOLERANCE
            and abs(sizes[2] - expected_third) <= expected_third * LEGACY_RATIO_TOLERANCE
        ):
            return True
    return False


class WorkspaceWidget(QWidget):
    """一个工程标签页的完整交互区域。"""

    status_info = Signal(dict)
    status_message = Signal(str)
    title_changed = Signal(str)
    modified_changed = Signal(bool)

    def __init__(
        self,
        workspace: WorkspaceModel,
        registry: OperatorRegistry,
        settings: AppSettings,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.workspace = workspace
        self.registry = registry
        self.settings = settings
        self.project_path: Path | None = None
        self._dirty = False
        self._current_node_id: str | None = None
        self._last_elapsed_ms: float | None = None
        self._pixel_coord = "--"
        self._pixel_value = "--"
        self._compare_mode = "single"
        self._active_pick: tuple[str, str, str] | None = None
        self._param_node_id: str | None = None
        self._param_session_old: dict = {}
        self._current_image = None
        self._last_pixel_sample = None
        self._last_undo_index = 0
        # 默认比例 1:2:4；旧版本默认比例（1:1:3 / 1:1:4）也视为“未自定义”，
        # 首次显示时校正，避免已有配置一直保留旧比例。
        saved_sizes = tuple(self.settings.workspace_splitter_sizes)
        self._splitter_ratio_pending = (
            saved_sizes == DEFAULT_SPLITTER_SIZES or _is_legacy_splitter_sizes(saved_sizes)
        )

        self._build_ui()
        self._connect_signals()
        self._sync_settings()
        self.workspace.name_changed.connect(lambda _name: self._update_title())

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(5)

        self.library = OperatorLibrary(self.splitter)
        self.library.setMinimumWidth(220)
        self.library.set_registry(self.registry)

        self.center = QWidget(self.splitter)
        center_layout = QVBoxLayout(self.center)
        center_layout.setContentsMargins(0, 0, 0, 0)
        center_layout.setSpacing(0)
        self.center_stack = QStackedWidget(self.center)
        self.graph_view = GraphView(
            self.workspace,
            self.registry,
            self._make_undo_stack(),
            self.settings,
            self.center_stack,
        )
        self.param_panel = ParamPanel(self.center_stack)
        self.center_stack.addWidget(self.graph_view)
        self.center_stack.addWidget(self.param_panel)
        center_layout.addWidget(self.center_stack)

        self.preview = PreviewWidget(self.splitter)

        self.splitter.addWidget(self.library)
        self.splitter.addWidget(self.center)
        self.splitter.addWidget(self.preview)
        self.splitter.setStretchFactor(0, 1)
        self.splitter.setStretchFactor(1, 2)
        self.splitter.setStretchFactor(2, 4)
        sizes = self.settings.workspace_splitter_sizes
        if len(sizes) != 3 or sum(sizes) <= 0:
            sizes = list(DEFAULT_SPLITTER_SIZES)
        self.splitter.setSizes(sizes)
        layout.addWidget(self.splitter)

        self.engine = ExecutionEngine(self.workspace, self.registry, self.settings, self)
        self.center_stack.setCurrentWidget(self.graph_view)

        # 分隔线视觉强调一下中间/右侧边界（不改变布局比例）
        line = QFrame(self.preview)
        line.setObjectName("previewDivider")
        line.setFixedWidth(0)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt 命名
        super().showEvent(event)
        if self._splitter_ratio_pending:
            self._splitter_ratio_pending = False
            QTimer.singleShot(0, self._apply_default_splitter_ratio)

    def _apply_default_splitter_ratio(self) -> None:
        """首次显示时按 1:2:4 设置三栏宽度（算子库最小宽度可能压缩预览比例）。"""
        available = self.splitter.width() - self.splitter.handleWidth() * 2
        if available <= 0:
            return
        unit = max(1, available // 7)
        first = max(unit, self.library.minimumWidth())
        second = first * 2
        third = max(1, available - first - second)
        self.splitter.setSizes([first, second, third])

    def _make_undo_stack(self):
        from PySide6.QtGui import QUndoStack

        stack = QUndoStack(self)
        stack.setUndoLimit(self.settings.undo_limit)
        self.undo_stack = stack
        return stack

    # ------------------------------------------------------------------ 连接
    def _connect_signals(self) -> None:
        self.library.refresh_requested.connect(lambda: self.status_message.emit("请从“算子”菜单刷新算子库"))
        self.graph_view.selection_changed.connect(self._on_node_selected)
        self.graph_view.node_activated.connect(self.open_parameters)
        self.graph_view.node_created.connect(self._on_node_created)
        self.graph_view.node_removed.connect(self._on_node_removed)
        self.graph_view.connection_changed.connect(self._on_connection_changed)
        self.graph_view.images_dropped.connect(self._on_images_dropped)
        self.graph_view.status_message.connect(self.status_message.emit)

        self.param_panel.preview_requested.connect(self._on_param_preview)
        self.param_panel.save_requested.connect(self._on_param_save)
        self.param_panel.back_requested.connect(self.back_to_graph)
        self.param_panel.pick_requested.connect(self._on_pick_requested)
        if hasattr(self.param_panel, "status_message"):
            self.param_panel.status_message.connect(self.status_message.emit)

        self.preview.pixel_info.connect(self._on_pixel_info)
        self.preview.zoom_changed.connect(lambda _z: self._emit_status())
        self.preview.pick_finished.connect(self._on_pick_finished)
        self.preview.pick_cancelled.connect(self._on_pick_cancelled)

        self.engine.node_started.connect(self._on_node_started)
        self.engine.node_finished.connect(self._on_node_finished)
        self.engine.node_failed.connect(self._on_node_failed)
        self.engine.chain_finished.connect(self._on_chain_finished)
        self.engine.busy_changed.connect(self._on_busy_changed)

        self.undo_stack.indexChanged.connect(self._on_undo_index_changed)

    # ------------------------------------------------------------------ 设置
    def _sync_settings(self) -> None:
        self.undo_stack.setUndoLimit(self.settings.undo_limit)
        self.graph_view.set_thumbnails_enabled(self.settings.show_node_thumbnails)
        self.preview.set_preview_quality(
            self.settings.downsample_preview,
            self.settings.downsample_max_dim,
        )
        if hasattr(self.preview, "set_auto_fit_on_open"):
            self.preview.set_auto_fit_on_open(self.settings.auto_fit_on_open)

    def apply_settings(self) -> None:
        self._sync_settings()
        self._on_pixel_info(self._last_pixel_sample)

    def set_registry(self, registry: OperatorRegistry) -> None:
        self.registry = registry
        self.library.set_registry(registry)
        self.graph_view.set_registry(registry)
        self.engine.set_registry(registry)
        if self._param_node_id:
            node = self.workspace.graph.nodes.get(self._param_node_id)
            entry = registry.entry(node.op_id) if node else None
            if entry is None or node is None:
                self.back_to_graph()
        if self._current_node_id and self._current_node_id in self.workspace.graph.nodes:
            self.engine.preview_node(self._current_node_id)

    # ------------------------------------------------------------------ 图像
    def add_image(self, path: str | Path) -> bool:
        try:
            image = load_image(path)
        except ImageIOError as exc:
            QMessageBox.critical(self, "无法打开图像", str(exc))
            self.status_message.emit(str(exc))
            return False
        except Exception as exc:  # noqa: BLE001
            QMessageBox.critical(self, "无法打开图像", f"{exc.__class__.__name__}: {exc}")
            return False
        position = self.workspace.next_source_position()
        node = self.workspace.add_source_image(image, str(path), pos=position)
        self.graph_view.set_selected_node(node.id)
        self._on_node_selected(node.id)
        self.graph_view.auto_layout()
        self.graph_view.fit_graph()
        self._mark_dirty()
        self.status_message.emit(f"已添加图像：{Path(path).name}")
        return True

    def _on_images_dropped(self, paths: list[str]) -> None:
        for path in paths:
            self.add_image(path)

    # ------------------------------------------------------------------ 选择
    def select_node_and_preview(self, node_id: str) -> None:
        """程序化选中并预览一个节点（用于打开工程后定位到终止节点）。"""
        if node_id not in self.workspace.graph.nodes:
            return
        self.graph_view.set_selected_node(node_id)
        self._on_node_selected(node_id)

    def _on_node_selected(self, node_id: str) -> None:
        if not node_id:
            self._current_node_id = None
            self._current_image = None
            self._pixel_coord = "--"
            self._pixel_value = "--"
            self.preview.set_image(None, "", "未选择节点")
            self._emit_status()
            return
        node = self.workspace.graph.nodes.get(node_id)
        if node is None:
            return
        if self._param_node_id and self._param_node_id != node_id:
            # 在参数编辑过程中切换到其它节点：先把当前编辑会话落成一步撤销。
            self._push_param_session()
            self._param_node_id = None
            self._param_session_old = {}
        self.center_stack.setCurrentWidget(self.graph_view)
        self._current_node_id = node_id
        self._last_elapsed_ms = None
        self._current_image = None
        self._pixel_coord = "--"
        self._pixel_value = "--"
        self._refresh_original(node)
        result = self.engine.result(node_id)
        if result is not None and result.revision == node.revision:
            self._show_result(result, node)
        else:
            self.preview.set_image(None, node.display_title(), "正在计算…")
            self.engine.preview_node(node_id)
        self._emit_status()

    def _refresh_original(self, node: NodeModel) -> None:
        source = self._find_source_ancestor(node)
        if source is None:
            self.preview.set_original_image(None, "原始图像")
            return
        image = self.workspace.image_for_node(source)
        self.preview.set_original_image(image, source.display_title())

    def _find_source_ancestor(self, node: NodeModel) -> NodeModel | None:
        if node.is_source:
            return node
        queue = list(self.workspace.graph.predecessors(node.id))
        visited: set[str] = set()
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            candidate = self.workspace.graph.nodes.get(current)
            if candidate is None:
                continue
            if candidate.is_source:
                return candidate
            queue.extend(self.workspace.graph.predecessors(current))
        sources = self.workspace.graph.source_nodes()
        return sources[0] if sources else None

    # ------------------------------------------------------------------ 算子
    def _on_node_created(self, node_id: str) -> None:
        self._mark_dirty()
        node = self.workspace.graph.nodes.get(node_id)
        if node:
            self.status_message.emit(f"已添加算子：{node.display_title()}，请连接输入后预览")
        self.graph_view.fit_graph()
        self._emit_status()

    def _on_node_removed(self, node_id: str) -> None:
        self.engine.forget_nodes([node_id])
        self.engine.invalidate_all()
        if self._current_node_id == node_id:
            self._current_node_id = None
            self.preview.set_image(None, "", "节点已删除")
        if self._param_node_id == node_id:
            self._param_node_id = None
            self._param_session_old = {}
            self.center_stack.setCurrentWidget(self.graph_view)
        self._mark_dirty()
        self._emit_status()

    def _on_connection_changed(self, dst_node_id: str) -> None:
        self.engine.invalidate(dst_node_id, downstream=True)
        self._mark_dirty()
        if self._current_node_id:
            current = self.workspace.graph.nodes.get(self._current_node_id)
            if current is not None:
                self.engine.preview_node(current.id)
        self._emit_status()

    # ------------------------------------------------------------------ 参数
    def open_parameters(self, node_id: str) -> None:
        node = self.workspace.graph.nodes.get(node_id)
        if node is None or node.is_source:
            self.status_message.emit("源图像节点没有可调参数")
            return
        entry = self.registry.entry(node.op_id)
        if entry is None:
            QMessageBox.warning(self, "算子不存在", f"找不到算子：{node.op_id}\n请刷新算子库。")
            return
        self._current_node_id = node_id
        self._param_node_id = node_id
        self._param_session_old = copy.deepcopy(node.params)
        self.param_panel.load_node(node, entry.operator)
        self.center_stack.setCurrentWidget(self.param_panel)
        self.preview.cancel_pick()
        self._refresh_original(node)
        cached = self.engine.result(node_id)
        if cached is not None and cached.revision == node.revision:
            self._show_result(cached, node)
        else:
            self.preview.set_image(None, node.display_title(), "正在计算…")
            self.engine.preview_node(node_id)
        self._emit_status()

    def back_to_graph(self) -> None:
        if self._param_node_id is None:
            self.center_stack.setCurrentWidget(self.graph_view)
            return
        self._push_param_session()
        self._param_node_id = None
        self._param_session_old = {}
        self.preview.cancel_pick()
        self.center_stack.setCurrentWidget(self.graph_view)
        self.graph_view.setFocus()
        self._emit_status()

    def _on_param_preview(self, node_id: str, params: dict) -> None:
        node = self.workspace.graph.nodes.get(node_id)
        if node is None:
            return
        if self._param_node_id == node_id and node.params != params:
            self.workspace.graph.set_node_params(node_id, params)
            self.engine.invalidate(node_id, downstream=False)
            self.engine.preview_node(node_id)
            self._mark_dirty()
            self._emit_status()

    def _on_param_save(self, node_id: str, params: dict) -> None:
        node = self.workspace.graph.nodes.get(node_id)
        if node is None:
            return
        if node.params != params:
            self.workspace.graph.set_node_params(node_id, params)
        self._push_param_session()
        self.engine.commit_node(node_id)
        self._param_session_old = copy.deepcopy(node.params)
        self._mark_dirty()
        self.status_message.emit("参数已保存，正在更新下游…")

    def _push_param_session(self) -> None:
        node_id = self._param_node_id
        if not node_id:
            return
        node = self.workspace.graph.nodes.get(node_id)
        if node is None:
            return
        new_params = copy.deepcopy(node.params)
        old_params = copy.deepcopy(self._param_session_old)
        if new_params == old_params:
            return
        # 命令的首次 redo 会重新写入新参数；先恢复旧值，保证撤销语义正确。
        self.workspace.graph.set_node_params(node_id, old_params)
        command = SetNodeParamsCommand(self.workspace.graph, node_id, old_params, new_params)
        self.undo_stack.push(command)
        self._param_session_old = copy.deepcopy(new_params)
        self._mark_dirty()

    def _on_undo_index_changed(self, _index: int) -> None:
        previous = self._last_undo_index
        self._last_undo_index = self.undo_stack.index()
        self._mark_dirty()
        # 只改变节点位置（拖拽 / 自动布局）不影响计算结果，不要清空快照或重算。
        if self._is_layout_only_change(previous, self._last_undo_index):
            self._emit_status()
            return
        # 撤销/重做可能改变参数或连线；清空所有快照并重新预览当前节点。
        self.engine.invalidate_all()
        if self._current_node_id and self._current_node_id in self.workspace.graph.nodes:
            self.engine.preview_node(self._current_node_id)
        self._emit_status()

    def _is_layout_only_change(self, previous: int, current: int) -> bool:
        """判断本次 undo/redo 是否只改变节点位置（不触发重算）。"""

        if current == previous:
            return False
        try:
            command = (
                self.undo_stack.command(current)
                if current < previous
                else self.undo_stack.command(current - 1)
            )
        except (IndexError, RuntimeError):
            return False
        return self._is_layout_only_command(command)

    def _is_layout_only_command(self, command) -> bool:
        if command is None:
            return False
        if isinstance(command, MoveNodeCommand):
            return True
        try:
            child_count = command.childCount()
        except (AttributeError, RuntimeError):
            return False
        if child_count <= 0:
            return False
        return all(
            self._is_layout_only_command(command.child(index))
            for index in range(child_count)
        )

    # ------------------------------------------------------------------ 拾取
    def _on_pick_requested(self, node_id: str, param_key: str, tool: str) -> None:
        node = self.workspace.graph.nodes.get(node_id)
        entry = self.registry.entry(node.op_id) if node else None
        if node is None or entry is None:
            return
        try:
            param = entry.spec.param(param_key)
        except KeyError:
            return
        count = param.count or 1
        initial = node.params.get(param_key)
        result = self.engine.result(node_id)
        if result is None:
            # 交互参数常常在首次执行前导致当前节点报错（例如 ROI 尚未框选）。
            # 此时用最近的上游输出（或源图）作为拾取底图，否则无法开始框选。
            fallback = self._fallback_input_image(node)
            if fallback is None:
                self.status_message.emit("当前节点没有可显示的上游图像，无法拾取")
                return
            self.preview.set_image(fallback, f"{node.display_title()} · 输入预览", "请在此图上完成拾取")
        self._active_pick = (node_id, param_key, tool)
        self.preview.begin_pick(tool, count, initial)
        self.status_message.emit("在右侧图像上拾取，ESC 取消")

    def _fallback_input_image(self, node: NodeModel):
        for connection in self.workspace.graph.incoming(node.id):
            result = self.engine.result(connection.src_node)
            if result is not None and result.primary is not None:
                return result.primary
        source = self._find_source_ancestor(node)
        if source is not None:
            return self.workspace.image_for_node(source)
        return None

    def _on_pick_finished(self, tool: str, value: object) -> None:
        if not self._active_pick:
            return
        _node_id, param_key, active_tool = self._active_pick
        self._active_pick = None
        if tool != active_tool:
            return
        self.param_panel.set_pick_result(param_key, value)
        self.status_message.emit("已应用拾取结果")

    def _on_pick_cancelled(self, _tool: str) -> None:
        self._active_pick = None
        self.status_message.emit("已取消拾取")

    # ------------------------------------------------------------------ 引擎
    def _on_node_started(self, node_id: str) -> None:
        node = self.workspace.graph.nodes.get(node_id)
        if node:
            self.status_message.emit(f"正在计算：{node.display_title()}")

    def _on_node_finished(self, result: NodeResult) -> None:
        node = self.workspace.graph.nodes.get(result.node_id)
        if node is None:
            return
        self.graph_view.notify_result(node.id, result)
        if node.id == self._current_node_id:
            self._show_result(result, node)
        self._emit_status()

    def _show_result(self, result: NodeResult, node: NodeModel) -> None:
        self._last_elapsed_ms = result.elapsed_ms
        image = result.preview_image
        summary = result.summary_text()
        if image is None:
            self._current_image = None
            self.preview.set_image(None, node.display_title(), summary or "算子没有输出图像")
            return
        self._current_image = image
        meta = f"{result.elapsed_ms:.1f} ms · {image.width}×{image.height} · {image.color_space} · {image.dtype_name}"
        if len(result.outputs) > 1:
            meta += f" · 输出 {len(result.outputs)}：{result.output_type_summary()}"
        if result.primary is None:
            meta += " · 结构化数据预览"
        self.preview.set_image(image, node.display_title(), meta)

    def _on_node_failed(self, node_id: str, message: str) -> None:
        self.graph_view.notify_error(node_id, message)
        if node_id == self._current_node_id:
            node = self.workspace.graph.nodes.get(node_id)
            title = node.display_title() if node else ""
            self._current_image = None
            self.preview.set_image(None, title, f"计算失败：{message}")
            self._last_elapsed_ms = None
        self.status_message.emit(message)
        self._emit_status()

    def _on_chain_finished(self, target: str, success: bool) -> None:
        if success:
            node = self.workspace.graph.nodes.get(target)
            if node is not None:
                self.status_message.emit(f"计算完成：{node.display_title()}")
        self._emit_status()

    def _on_busy_changed(self, busy: bool) -> None:
        if busy:
            self.status_message.emit("后台计算中…")
        self._emit_status()

    # ------------------------------------------------------------------ 状态
    def _on_pixel_info(self, sample) -> None:
        self._last_pixel_sample = sample
        if sample is None or not getattr(sample, "in_bounds", False):
            self._pixel_coord = "--"
            self._pixel_value = "--"
        else:
            self._pixel_coord = f"{sample.row}, {sample.col}"
            if self.settings.cursor_pixel_info:
                self._pixel_value = f"{sample.value_text()} · {sample.dtype_name}"
            else:
                self._pixel_value = "--"
        self._emit_status()

    def _emit_status(self) -> None:
        node = self.workspace.graph.nodes.get(self._current_node_id or "")
        image = self._current_image
        size = "--"
        if image is not None:
            size = f"{image.width}×{image.height}"
        zoom = f"{self.preview.zoom_percent():.0f}%"
        info = {
            "node": node.display_title() if node else "--",
            "size": size,
            "zoom": zoom,
            "coord": self._pixel_coord,
            "pixel": self._pixel_value,
            "elapsed": f"{self._last_elapsed_ms:.1f} ms" if self._last_elapsed_ms is not None else "--",
            "busy": self.engine.busy,
            "mode": self._compare_mode,
        }
        self.status_info.emit(info)

    def refresh_status(self) -> None:
        self._emit_status()

    # ------------------------------------------------------------------ 工程
    @property
    def dirty(self) -> bool:
        return self._dirty

    def mark_clean(self) -> None:
        self._dirty = False
        self.undo_stack.setClean()
        self.modified_changed.emit(False)
        self._update_title()

    def _mark_dirty(self) -> None:
        if not self._dirty:
            self._dirty = True
            self.modified_changed.emit(True)
            self._update_title()

    def tab_title(self) -> str:
        return f"{self.workspace.name}{' *' if self._dirty else ''}"

    def _update_title(self) -> None:
        self.title_changed.emit(self.tab_title())

    def save_to(self, path: str | Path) -> Path:
        target = save_project(path, self.workspace)
        self.project_path = target
        if self.workspace.name.startswith("未命名"):
            self.workspace.name = target.stem
        self.mark_clean()
        self.status_message.emit(f"已保存工程：{target.name}")
        return target

    def set_compare_mode(self, mode: str) -> None:
        if mode not in ("single", "side_by_side", "swipe"):
            return
        self._compare_mode = mode
        self.preview.set_compare_mode(mode)
        self._emit_status()

    @property
    def compare_mode(self) -> str:
        return self._compare_mode

    def fit_preview(self) -> None:
        self.preview.fit_to_window()

    def reset_preview_zoom(self) -> None:
        self.preview.reset_zoom()

    def splitter_sizes(self) -> list[int]:
        return self.splitter.sizes()

    def export_current_image(self, path: str | Path) -> bool:
        """把当前节点最新快照导出为文件（保留原始 dtype）。"""
        node_id = self._current_node_id
        result = self.engine.result(node_id) if node_id else None
        if result is None or result.primary is None:
            if result is not None and result.outputs:
                self.status_message.emit("当前节点没有图像输出，无法导出为图像文件")
            else:
                self.status_message.emit("当前节点没有可导出的结果")
            return False
        try:
            save_image(path, result.primary)
        except (ImageIOError, OSError) as exc:
            QMessageBox.critical(self, "导出失败", str(exc))
            return False
        self.status_message.emit(f"已导出：{Path(path).name}")
        return True

    def shutdown(self) -> None:
        self.engine.shutdown()
