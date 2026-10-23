"""工作区页面：算子库 + 流程画布/参数面板 + 预览 + 引擎的组装。"""

from __future__ import annotations

import copy
import logging
import time
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
from app.core.constants import IMAGE_DISPLAY_MODES
from app.core.imageio import ImageIOError, load_image, save_image
from app.core.matrixio import is_matrix_file
from app.core.model import NodeModel, WorkspaceModel
from app.core.project import save_project
from app.core.types import ImageData, image_values
from app.engine.engine import ExecutionEngine, NodeResult
from app.graph.commands import MoveNodeCommand, SetNodeParamsCommand
from app.graph.view import GraphView
from app.operators.loader import OperatorRegistry
from app.ui.display_modes import display_mode_title
from app.ui.library import OperatorLibrary
from app.ui.params import ParamPanel
from app.ui.preview import PreviewWidget

#: 三栏默认宽度比例 算子库 : 画布 : 显示 = 1:2:4。
DEFAULT_SPLITTER_SIZES = (200, 400, 800)
#: 旧版本默认比例（1:1:3 / 1:1:4）；保存的宽度受窗口大小影响，按比例识别。
LEGACY_SPLITTER_RATIOS = ((1, 1, 3), (1, 1, 4))
#: 识别旧默认比例时允许的相对误差（窗口缩放与最小宽度会改变实际数值）。
LEGACY_RATIO_TOLERANCE = 0.22
#: 连线变化后合并触发的延迟（毫秒）：一次撤销宏 / 粘贴可能产生多条连线，
#: 合并成一次引擎更新，避免互相取消。
CONNECTION_UPDATE_DELAY_MS = 30
#: 状态栏「耗时」在节点运行中的刷新间隔（毫秒）。
ELAPSED_TICK_MS = 100

LOG = logging.getLogger(__name__)


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
    #: 画布保存了新的自定义算子 / 需要重新扫描算子目录
    operator_refresh_requested = Signal()

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
        #: 状态栏「耗时」的显示状态：idle / running / done。
        self._elapsed_state = "idle"
        self._run_started_at: float | None = None
        self._pixel_coord = "--"
        self._pixel_value = "--"
        self._compare_mode = "single"
        self._active_pick: tuple[str, str, str] | None = None
        self._param_node_id: str | None = None
        self._param_session_old: dict = {}
        self._current_image = None
        #: 当前节点的全部图像结果（单图只有一个元素，图像矩阵按端口顺序展开）。
        self._current_images: list[ImageData] = []
        self._current_image_index = 0
        self._last_pixel_sample = None
        self._last_undo_index = 0
        # 连线变化后待更新的目标节点：合并同一次操作里的多条连线事件。
        self._pending_connection_roots: set[str] = set()
        self._connection_update_timer = QTimer(self)
        self._connection_update_timer.setSingleShot(True)
        self._connection_update_timer.setInterval(CONNECTION_UPDATE_DELAY_MS)
        self._connection_update_timer.timeout.connect(self._flush_connection_updates)
        # 当前节点运行中时，按 tick 刷新状态栏「耗时」。
        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(ELAPSED_TICK_MS)
        self._elapsed_timer.timeout.connect(self._emit_status)
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

        self.library = OperatorLibrary(self.splitter, settings=self.settings)
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
        self.preview.set_display_mode(self.workspace.display_mode)

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
        self.library.status_message.connect(self.status_message.emit)
        self.graph_view.selection_changed.connect(self._on_node_selected)
        image_selected = getattr(self.graph_view, "node_image_selected", None)
        if image_selected is not None:
            image_selected.connect(self._on_node_image_selected)
        self.graph_view.node_activated.connect(self.open_parameters)
        self.graph_view.node_created.connect(self._on_node_created)
        self.graph_view.node_removed.connect(self._on_node_removed)
        self.graph_view.connection_changed.connect(self._on_connection_changed)
        self.graph_view.images_dropped.connect(self._on_images_dropped)
        self.graph_view.status_message.connect(self.status_message.emit)
        operators_changed = getattr(self.graph_view, "operators_changed", None)
        if operators_changed is not None:
            operators_changed.connect(self.operator_refresh_requested.emit)

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
        self.engine.request_started.connect(self._on_engine_request_started)

        self.undo_stack.indexChanged.connect(self._on_undo_index_changed)

    # ------------------------------------------------------------------ 设置
    def _sync_settings(self) -> None:
        self.undo_stack.setUndoLimit(self.settings.undo_limit)
        self.engine.set_workers(self.settings.execution_workers)
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

    def set_registry(self, registry: OperatorRegistry, *, preview: bool = True) -> None:
        """替换算子注册表并刷新依赖它的界面。

        ``preview=False`` 用于引擎在计算前发现文件变化后静默重载：只更新
        library / graph / engine 的注册表，不在这里再次触发 ``preview_node``，
        避免与引擎当前请求重入；当前请求会继续使用新注册表执行。
        """
        self.registry = registry
        self.library.set_registry(registry)
        self.graph_view.set_registry(registry)
        self.engine.set_registry(registry)
        if self._param_node_id:
            node = self.workspace.graph.nodes.get(self._param_node_id)
            entry = registry.entry(node.op_id) if node else None
            if entry is None or node is None:
                self.back_to_graph()
        if (
            preview
            and self._current_node_id
            and self._current_node_id in self.workspace.graph.nodes
        ):
            self.engine.preview_node(self._current_node_id)

    # ------------------------------------------------------------------ 图像
    def _load_image_file(self, path: str | Path, *, report: bool = True) -> ImageData | None:
        """读取一张图像（含 CSV 等矩阵文件）；失败时记录日志并按需弹出错误框。"""

        is_matrix = is_matrix_file(path)
        kind = "矩阵" if is_matrix else "图像"
        try:
            return load_image(path)
        except ImageIOError as exc:
            LOG.error("无法打开%s %s：%s", kind, path, exc)
            if report:
                QMessageBox.critical(self, f"无法打开{kind}", str(exc))
            self.status_message.emit(str(exc))
        except Exception as exc:  # noqa: BLE001
            LOG.exception("无法打开%s %s", kind, path)
            if report:
                QMessageBox.critical(self, f"无法打开{kind}", f"{exc.__class__.__name__}: {exc}")
        return None

    def add_image(self, path: str | Path) -> bool:
        """追加一张源图像节点（保留“一个文件一个源节点”的旧行为）。"""

        is_matrix = is_matrix_file(path)
        image = self._load_image_file(path)
        if image is None:
            return False
        position = self.workspace.next_source_position()
        node = self.workspace.add_source_image(image, str(path), pos=position)
        self.graph_view.set_selected_node(node.id)
        self._on_node_selected(node.id)
        self.graph_view.auto_layout()
        self.graph_view.fit_graph()
        self._mark_dirty()
        if is_matrix:
            self.status_message.emit(
                f"已添加矩阵：{Path(path).name}"
                f"（{image.height} 行 × {image.width} 列 · {image.dtype_name}）"
            )
        else:
            self.status_message.emit(f"已添加图像：{Path(path).name}")
        return True

    def add_image_matrix(self, paths: list[str | Path]) -> bool:
        """把一次选择的多张图像放进同一个源节点（图像矩阵）。

        通过“文件 → 打开图像”多选打开时使用；单张图像也走同一入口，保证
        源节点结构一致。读取失败的文件会记录日志并汇总提示，其余图像仍会
        正常打开。
        """

        path_list = [Path(path) for path in paths if str(path)]
        if not path_list:
            return False
        images: list[ImageData] = []
        loaded_paths: list[str] = []
        failures: list[str] = []
        for path in path_list:
            image = self._load_image_file(path, report=False)
            if image is None:
                failures.append(path.name)
                continue
            images.append(image)
            loaded_paths.append(str(path))
        if failures:
            LOG.error("以下图像无法打开：%s", "、".join(failures))
            QMessageBox.critical(
                self,
                "部分图像无法打开",
                "以下文件读取失败，已跳过：\n" + "\n".join(failures),
            )
        if not images:
            self.status_message.emit("没有可打开的图像")
            return False
        position = self.workspace.next_source_position()
        node = self.workspace.add_source_images(images, loaded_paths, pos=position)
        self.graph_view.set_selected_node(node.id)
        self._on_node_selected(node.id)
        self.graph_view.auto_layout()
        self.graph_view.fit_graph()
        self._mark_dirty()
        if len(images) == 1:
            kind = "矩阵" if is_matrix_file(loaded_paths[0]) else "图像"
            self.status_message.emit(f"已打开{kind}：{Path(loaded_paths[0]).name}")
        else:
            self.status_message.emit(
                f"已打开 {len(images)} 个文件：{Path(loaded_paths[0]).name} 等"
            )
        return True

    def _on_images_dropped(self, paths: list[str]) -> None:
        for path in paths:
            self.add_image(path)

    # ------------------------------------------------------------------ 选择
    def select_node_and_preview(self, node_id: str) -> None:
        """程序化选中并预览一个节点（有有效快照直接回放，否则后台重算）。"""
        if node_id not in self.workspace.graph.nodes:
            return
        self.graph_view.set_selected_node(node_id)
        self._on_node_selected(node_id)

    def precompute_project(self, preview_node_id: str | None = None) -> None:
        """打开工程后完整预计算：定位预览节点，再计算每个子树的所有节点。

        引擎用一次请求覆盖整张图（根为全部节点），并按广度优先的拓扑层序
        执行。因此这里不再单独调用 ``preview_node``，避免两次请求互相取消；
        预览节点会在引擎回填 ``node_finished`` 时自动刷新，未选中的节点与
        缩略图也会在后台算完后直接显示，不需要再手动点击。
        """

        if not self.workspace.graph.nodes:
            return
        if preview_node_id and preview_node_id in self.workspace.graph.nodes:
            self.graph_view.set_selected_node(preview_node_id)
            # 预计算请求会覆盖整张图，选中节点只需等待结果回填。
            self._on_node_selected(preview_node_id, request_preview=False)
        self.engine.precompute_all()
        self.status_message.emit(
            f"正在按广度优先顺序预计算全部 {len(self.workspace.graph.nodes)} 个节点…"
        )

    def _on_node_selected(self, node_id: str, *, request_preview: bool = True) -> None:
        if not node_id:
            self._stop_elapsed_clock()
            self._elapsed_state = "idle"
            self._current_node_id = None
            self._current_image = None
            self._current_images = []
            self._current_image_index = 0
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
        self._stop_elapsed_clock()
        self._elapsed_state = "idle"
        self._last_elapsed_ms = None
        self._current_image = None
        self._current_images = []
        self._current_image_index = 0
        self._pixel_coord = "--"
        self._pixel_value = "--"
        self._refresh_original(node)
        result = self.engine.result(node_id)
        if result is not None and result.revision == node.revision:
            self._show_result(result, node)
        else:
            self.preview.set_image(None, node.display_title(), "正在计算…")
            if request_preview:
                self.engine.preview_node(node_id)
        self._emit_status()

    def _refresh_original(self, node: NodeModel) -> None:
        source = self._find_source_ancestor(node)
        if source is None:
            self.preview.set_original_image(None, "原始图像")
            return
        images = self.workspace.images_for_node(source)
        if not images:
            self.preview.set_original_image(None, source.display_title())
            return
        # 图像矩阵在对比模式下使用与当前结果相同的序号：结果节点看自己的选中序号，
        # 源节点看源节点网格的序号；越界时回退第一张。
        current_id = self._current_node_id
        if current_id and current_id != source.id:
            index = self._selected_image_index(current_id)
        else:
            index = self._selected_image_index(source.id)
        if index >= len(images):
            index = 0
        self.preview.set_original_image(images[index], source.display_title())

    def _find_source_ancestor(self, node: NodeModel) -> NodeModel | None:
        if node.is_source:
            return node
        graph = self.workspace.graph
        queue: list[str] = []
        # 组内节点的直接上游可能是算子组边界端口：先解析成真实上游节点。
        for connection in graph.incoming(node.id):
            resolved = graph.resolve_source(connection.src_node, connection.src_port)
            if resolved is not None:
                queue.append(resolved[0])
        queue.extend(graph.predecessors(node.id))
        visited: set[str] = set()
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            visited.add(current)
            candidate = graph.nodes.get(current)
            if candidate is None:
                continue
            if candidate.is_source:
                return candidate
            if candidate.is_group:
                continue  # 算子组本身不是源图像，继续找它外部的上游
            queue.extend(graph.predecessors(current))
        sources = graph.source_nodes()
        return sources[0] if sources else None

    def _result_images(self, result: NodeResult | None) -> list[ImageData]:
        """按输出端口顺序取结果的图像列表；图像矩阵会展开成多张。"""

        if result is None:
            return []
        getter = getattr(result, "primary_images", None)
        if callable(getter):
            try:
                images = list(getter())
            except Exception:  # noqa: BLE001 - 旧结果对象没有该属性时回退
                images = []
            if images:
                return images
        for value in (getattr(result, "outputs", {}) or {}).values():
            images = list(image_values(value))
            if images:
                return images
        return []

    def _selected_image_index(self, node_id: str | None) -> int:
        """节点卡片上当前选中的图像序号（单图 / 无结果时为 0）。"""

        if not node_id:
            return 0
        getter = getattr(self.graph_view, "selected_image_index", None)
        if callable(getter):
            try:
                return max(0, int(getter(node_id)))
            except Exception:  # noqa: BLE001
                return 0
        return 0

    def _on_node_image_selected(self, node_id: str, index: int) -> None:
        """点击节点网格中的某张图：只切换预览，不改变画布节点选择。"""

        node = self.workspace.graph.nodes.get(node_id)
        if node is None:
            return
        self._current_node_id = node_id
        self._last_elapsed_ms = None
        self._current_images = []
        self._current_image_index = max(0, int(index))
        self._refresh_original(node)
        result = self.engine.result(node_id)
        if result is not None and result.revision == node.revision:
            self._show_result(result, node)
        else:
            self.engine.preview_node(node_id)
            self.preview.set_image(None, node.display_title(), "正在计算…")
        self._emit_status()

    # ------------------------------------------------------------------ 算子
    def _on_node_created(self, node_id: str) -> None:
        self._mark_dirty()
        node = self.workspace.graph.nodes.get(node_id)
        if node:
            self.status_message.emit(f"已添加算子：{node.display_title()}，连接输入后会自动计算")
        self.graph_view.fit_graph()
        self._emit_status()

    def _on_node_removed(self, node_id: str) -> None:
        self.engine.forget_nodes([node_id])
        self.engine.invalidate_all()
        if self._current_node_id == node_id:
            self._current_node_id = None
            self._current_image = None
            self._current_images = []
            self._current_image_index = 0
            self.preview.set_image(None, "", "节点已删除")
        if self._param_node_id == node_id:
            self._param_node_id = None
            self._param_session_old = {}
            self.param_panel.set_result(None)
            self.center_stack.setCurrentWidget(self.graph_view)
        self._mark_dirty()
        self._emit_status()

    def _on_connection_changed(self, dst_node_id: str) -> None:
        """连线增删后立即安排目标节点及其下游重算。

        一次操作（粘贴、替换连线、删除子树）可能连续产生多条连线信号，
        这里先让旧快照失效，再在 :data:`CONNECTION_UPDATE_DELAY_MS` 后合并成
        一次引擎请求，避免后一次请求取消前一次尚未完成的计算。
        """

        self._mark_dirty()
        if dst_node_id in self.workspace.graph.nodes:
            self.engine.invalidate(dst_node_id, downstream=True)
            self._pending_connection_roots.add(dst_node_id)
            self._connection_update_timer.start()
        self._emit_status()

    def _flush_connection_updates(self) -> None:
        """把合并后的连线变化交给引擎：目标节点与全部下游一起重算。"""

        roots = [
            node_id
            for node_id in self._pending_connection_roots
            if node_id in self.workspace.graph.nodes
        ]
        self._pending_connection_roots.clear()
        if not roots:
            return
        self.engine.update_nodes(roots)
        self.status_message.emit("连线已更新，正在重新计算下游…")
        self._emit_status()

    # ------------------------------------------------------------------ 参数
    def open_parameters(self, node_id: str) -> None:
        node = self.workspace.graph.nodes.get(node_id)
        if node is None or node.is_source:
            self.status_message.emit("源图像节点没有可调参数")
            return
        if node.is_group:
            self.status_message.emit("算子组本身没有参数；双击组内节点可编辑算子参数")
            return
        entry = self.registry.entry(node.op_id)
        if entry is None:
            QMessageBox.warning(self, "算子不存在", f"找不到算子：{node.op_id}\n请刷新算子库。")
            return
        self._current_node_id = node_id
        self._stop_elapsed_clock()
        self._elapsed_state = "idle"
        self._last_elapsed_ms = None
        self._param_node_id = node_id
        self._param_session_old = copy.deepcopy(node.params)
        self.param_panel.load_node(node, entry.operator)
        self.center_stack.setCurrentWidget(self.param_panel)
        self.preview.cancel_pick()
        self._refresh_original(node)
        cached = self.engine.result(node_id)
        if cached is not None and cached.revision == node.revision:
            self.param_panel.set_result(cached)
            self._show_result(cached, node)
        else:
            self.param_panel.set_result(None)
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
            # 参数 revision 变化后 preview_node 会让该节点与全部下游快照失效，
            # 并递归重算下游，保证画布上的数据始终与参数一致。
            self.workspace.graph.set_node_params(node_id, params)
            self._stop_elapsed_clock()
            self._elapsed_state = "idle"
            self._last_elapsed_ms = None
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
        self._stop_elapsed_clock()
        self._elapsed_state = "idle"
        self._last_elapsed_ms = None
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
        # 撤销/重做可能改变参数或连线；清空所有快照后重新计算当前节点。
        self.engine.invalidate_all()
        self._stop_elapsed_clock()
        self._elapsed_state = "idle"
        self._last_elapsed_ms = None
        if self._pending_connection_roots:
            # 连线变化已经安排了延迟更新：把当前节点并入同一批请求，
            # 避免两次请求互相取消。
            if self._current_node_id and self._current_node_id in self.workspace.graph.nodes:
                self._pending_connection_roots.add(self._current_node_id)
            self._connection_update_timer.start()
        elif self._current_node_id and self._current_node_id in self.workspace.graph.nodes:
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
        if self.preview.display_mode() == "mesh":
            self.status_message.emit("三维网格（mesh）模式下不支持拾取，请先切换到 Imshow / Imagesc")
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
            images = self._result_images(result)
            if images:
                index = self._selected_image_index(connection.src_node)
                return images[min(index, len(images) - 1)]
        source = self._find_source_ancestor(node)
        if source is not None:
            images = self.workspace.images_for_node(source)
            if images:
                index = self._selected_image_index(source.id)
                return images[min(index, len(images) - 1)]
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
        # 画布卡片对每个节点独立计时；状态栏只跟随当前选中的节点。
        self.graph_view.notify_started(node_id)
        if node_id == self._current_node_id:
            self._start_elapsed_clock()
            self._emit_status()

    def _on_node_finished(self, result: NodeResult) -> None:
        node = self.workspace.graph.nodes.get(result.node_id)
        if node is None:
            return
        self.graph_view.notify_result(node.id, result)
        if node.id == self._current_node_id:
            self._stop_elapsed_clock()
            self._elapsed_state = "done"
            self._show_result(result, node)
        if node.id == self._param_node_id:
            self.param_panel.set_result(result)
        self._emit_status()

    def _show_result(self, result: NodeResult, node: NodeModel) -> None:
        self._last_elapsed_ms = result.elapsed_ms
        images = self._result_images(result)
        summary = result.summary_text()
        if not images:
            self._current_image = None
            self._current_images = []
            self._current_image_index = 0
            self.preview.set_image(None, node.display_title(), summary or "算子没有输出图像")
            return
        index = self._selected_image_index(node.id)
        index = max(0, min(index, len(images) - 1))
        image = images[index]
        self._current_images = images
        self._current_image_index = index
        self._current_image = image
        # 预览右下角的信息标签已并入状态栏：尺寸 / 缩放 / 耗时由独立字段展示，
        # 这里只把色彩空间、数据类型和多输出汇总等补充信息交给状态栏的「节点」字段。
        detail = f"{image.color_space} · {image.dtype_name}"
        if len(images) > 1:
            detail = f"图像矩阵 {len(images)} 张 · 第 {index + 1}/{len(images)} 张 · {detail}"
        if len(result.outputs) > 1:
            detail += f" · 输出 {len(result.outputs)}：{result.output_type_summary()}"
        self.preview.set_image(image, node.display_title(), detail)

    def _on_node_failed(self, node_id: str, message: str) -> None:
        self.graph_view.notify_error(node_id, message)
        if node_id == self._current_node_id:
            node = self.workspace.graph.nodes.get(node_id)
            title = node.display_title() if node else ""
            self._current_image = None
            self._current_images = []
            self._current_image_index = 0
            self.preview.set_image(None, title, f"计算失败：{message}")
            self._last_elapsed_ms = None
            self._stop_elapsed_clock()
            self._elapsed_state = "idle"
        if node_id == self._param_node_id:
            self.param_panel.set_result(None)
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
            self.status_message.emit("后台并行计算中…")
        else:
            # 正常结束时运行中集合已经清空；这里兜底清理被取消的请求。
            self.graph_view.clear_running()
            if self._elapsed_state == "running":
                # 请求被取消（例如快速切换节点）时可能收不到 finished / failed。
                self._stop_elapsed_clock()
                self._elapsed_state = "idle"
        self._emit_status()

    def _on_engine_request_started(self, _generation: int) -> None:
        """新请求会取代旧请求：清掉旧请求遗留的“运行中”计时。"""

        self.graph_view.clear_running()
        if self._elapsed_state == "running":
            self._stop_elapsed_clock()
            self._elapsed_state = "idle"

    # ------------------------------------------------------------------ 耗时
    def _running_elapsed_text(self) -> str | None:
        """运行中的实时耗时文本；数字随 tick 增加。"""

        if self._run_started_at is None:
            return None
        seconds = max(0.0, time.monotonic() - self._run_started_at)
        if seconds < 1.0:
            return f"{seconds * 1000.0:.0f} ms"
        return f"{seconds:.1f} s"

    def _start_elapsed_clock(self) -> None:
        self._run_started_at = time.monotonic()
        self._elapsed_state = "running"
        if not self._elapsed_timer.isActive():
            self._elapsed_timer.start()

    def _stop_elapsed_clock(self) -> None:
        self._elapsed_timer.stop()
        self._run_started_at = None

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
        elapsed = self._running_elapsed_text()
        if elapsed is None:
            elapsed = (
                f"{self._last_elapsed_ms:.1f} ms"
                if self._last_elapsed_ms is not None
                else "--"
            )
        node_title = node.display_title() if node else ""
        detail = self.preview.info_text()
        node_text = " · ".join(part for part in (node_title, detail) if part) or "--"
        info = {
            "node": node_text,
            "size": size,
            "zoom": zoom,
            "coord": self._pixel_coord,
            "pixel": self._pixel_value,
            "elapsed": elapsed,
            "elapsed_state": self._elapsed_state,
            "busy": self.engine.busy,
            "mode": self._compare_mode,
            "display": self.workspace.display_mode,
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

    def set_display_mode(self, mode: str) -> None:
        """切换当前工作区的预览显示模式（Imshow / Imagesc / Mesh）。

        模式保存在 :class:`WorkspaceModel` 上，切换节点或对比布局都不会改变；
        修改会标记工作区为已修改，以便随 ``.mi`` 工程一起保存。
        """
        normalized = str(mode or "").strip().lower()
        if normalized not in IMAGE_DISPLAY_MODES:
            return
        if normalized == self.workspace.display_mode:
            self.preview.set_display_mode(normalized)
            return
        self.workspace.display_mode = normalized
        self.preview.set_display_mode(normalized)
        self._mark_dirty()
        self.status_message.emit(f"显示模式：{display_mode_title(normalized)}")
        self._emit_status()

    @property
    def display_mode(self) -> str:
        return self.workspace.display_mode

    def fit_preview(self) -> None:
        self.preview.fit_to_window()

    def reset_preview_zoom(self) -> None:
        self.preview.reset_zoom()

    def splitter_sizes(self) -> list[int]:
        return self.splitter.sizes()

    def export_current_image(self, path: str | Path) -> bool:
        """把当前节点最新快照导出为文件（图像矩阵导出当前选中的一张）。"""
        node_id = self._current_node_id
        result = self.engine.result(node_id) if node_id else None
        images = self._result_images(result)
        if not images:
            if result is not None and result.outputs:
                self.status_message.emit("当前节点没有图像输出，无法导出为文件")
            else:
                self.status_message.emit("当前节点没有可导出的结果")
            return False
        index = self._selected_image_index(node_id)
        index = max(0, min(index, len(images) - 1))
        image = images[index]
        try:
            save_image(path, image)
        except (ImageIOError, OSError) as exc:
            LOG.error("导出图像失败 %s：%s", path, exc)
            QMessageBox.critical(self, "导出失败", str(exc))
            return False
        suffix = f"（第 {index + 1}/{len(images)} 张）" if len(images) > 1 else ""
        self.status_message.emit(f"已导出：{Path(path).name}{suffix}")
        return True

    def shutdown(self) -> None:
        self._connection_update_timer.stop()
        self._pending_connection_roots.clear()
        self.engine.shutdown()
