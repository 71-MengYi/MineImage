"""主窗口：多工作区标签、菜单、工具栏、全局状态栏。"""

from __future__ import annotations

import logging
import platform
from pathlib import Path

from PySide6.QtCore import QByteArray, QSize, Qt, QTimer, QUrl
from PySide6.QtGui import QAction, QActionGroup, QCloseEvent, QDesktopServices, QKeySequence
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QStackedWidget,
    QStyle,
    QTabWidget,
    QTextEdit,
    QDialog,
    QDialogButtonBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from app import APP_DISPLAY_NAME, __version__
from app.config import (
    AppSettings,
    application_root,
    default_log_db_path,
    default_user_operator_dir,
    resource_root,
)
from app.core.constants import (
    DEFAULT_IMAGE_DISPLAY_MODE,
    IMAGE_FILTER,
    PROJECT_FILTER,
)
from app.core.logstore import LogStore
from app.core.model import WorkspaceModel
from app.core.project import ProjectFormatError, load_project
from app.operators.loader import OperatorRegistry
from app.ui.display_modes import DISPLAY_MODE_SPECS, display_mode_icon
from app.ui.log_dialog import LogDialog
from app.ui.settings_dialog import SettingsDialog
from app.ui.workspace import WorkspaceWidget

LOG = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    def __init__(
        self,
        settings: AppSettings | None = None,
        log_store: LogStore | None = None,
    ) -> None:
        super().__init__()
        self.settings = settings or AppSettings()
        # 由 app.main 注入与日志处理器共用的仓库；未注入时在首次打开日志时创建。
        self.log_store = log_store
        self._log_dialog: LogDialog | None = None
        self.registry = self._build_registry()
        self._workspace_counter = 0
        self._workspaces: list[WorkspaceWidget] = []
        self._status_labels: dict[str, QLabel] = {}
        self._watch_snapshot: dict[str, tuple[int, int]] = {}
        self._watch_timer = QTimer(self)
        self._watch_timer.setInterval(3000)
        self._watch_timer.timeout.connect(self._check_operator_watch)

        self.setWindowTitle(APP_DISPLAY_NAME)
        self.resize(1500, 920)
        self.setMinimumSize(1180, 720)

        self._create_actions()
        self._create_central()
        self._create_menus()
        self._create_toolbar()
        self._create_statusbar()
        self._restore_window_state()
        self._update_central()
        self._refresh_edit_actions()
        self._setup_operator_watcher()

        QTimer.singleShot(100, self._startup_status)

    # ------------------------------------------------------------------ 初始化
    def _build_registry(self) -> OperatorRegistry:
        user_dir = default_user_operator_dir()
        try:
            user_dir.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        dirs: list[str] = []
        for raw in [str(user_dir), *self.settings.extra_operator_dirs]:
            path = str(Path(raw).expanduser())
            if path not in dirs:
                dirs.append(path)
        registry = OperatorRegistry.build(
            dirs,
            include_builtin=self.settings.builtin_operators_enabled,
        )
        for issue in registry.issues:
            LOG.warning(
                "算子加载问题 [%s] %s：%s",
                issue.level,
                issue.path,
                issue.message,
                extra={"log_detail": issue.traceback or ""},
            )
        return registry

    def _startup_status(self) -> None:
        if self.tabs.count():
            return
        count = len(self.registry)
        errors = len(self.registry.errors)
        if errors:
            self.statusBar().showMessage(f"已加载 {count} 个算子，{errors} 个加载失败（算子 → 查看加载日志）", 8000)
        else:
            self.statusBar().showMessage(f"已加载 {count} 个算子 · 打开图像开始实验", 6000)

    # ------------------------------------------------------------------ 动作
    def _icon(self, pixmap: QStyle.StandardPixmap):
        return self.style().standardIcon(pixmap)

    def _action(
        self,
        text: str,
        slot=None,
        shortcut: str | QKeySequence | None = None,
        icon: QStyle.StandardPixmap | None = None,
        checkable: bool = False,
    ) -> QAction:
        action = QAction(text, self)
        if icon is not None:
            action.setIcon(self._icon(icon))
        if shortcut:
            action.setShortcut(shortcut)
        if checkable:
            action.setCheckable(True)
        if slot is not None:
            action.triggered.connect(slot)
        return action

    def _create_actions(self) -> None:
        self.act_new = self._action("新建工作区", self.new_workspace, "Ctrl+N")
        self.act_open = self._action(
            "打开图像…", self.open_image, "Ctrl+O", QStyle.StandardPixmap.SP_DialogOpenButton
        )
        self.act_add_image = self._action(
            "添加图像…", self.add_image, "Ctrl+Shift+O", QStyle.StandardPixmap.SP_FileDialogNewFolder
        )
        self.act_open_project = self._action(
            "打开工程…", self.open_project, "Ctrl+Alt+O", QStyle.StandardPixmap.SP_DirOpenIcon
        )
        self.act_save = self._action(
            "保存工程", self.save_workspace, "Ctrl+S", QStyle.StandardPixmap.SP_DialogSaveButton
        )
        self.act_save_as = self._action("工程另存为…", self.save_workspace_as, "Ctrl+Shift+S")
        self.act_export = self._action(
            "导出当前节点图像…", self.export_current_image, "Ctrl+E",
            QStyle.StandardPixmap.SP_DialogSaveButton,
        )
        self.act_close_tab = self._action("关闭当前标签", self.close_current_tab, "Ctrl+W")
        self.act_quit = self._action("退出", self.close, "Ctrl+Q")

        self.act_undo = self._action(
            "撤销", self.undo, "Ctrl+Z", QStyle.StandardPixmap.SP_ArrowBack
        )
        self.act_redo = self._action(
            "重做", self.redo, "Ctrl+Y", QStyle.StandardPixmap.SP_ArrowForward
        )
        self.act_delete = self._action("删除选中", self.delete_selected, "Delete")
        self.act_add_group = self._action("添加算子组", self.add_group, "Ctrl+Shift+F")

        self.act_compare_single = self._action("单视图", lambda: self._set_compare_mode("single"), "Ctrl+1", checkable=True)
        self.act_compare_side = self._action(
            "左右对比", lambda: self._set_compare_mode("side_by_side"), "Ctrl+2", checkable=True
        )
        self.act_compare_swipe = self._action(
            "滑动对比", lambda: self._set_compare_mode("swipe"), "Ctrl+3", checkable=True
        )
        self.compare_group = QActionGroup(self)
        self.compare_group.setExclusive(True)
        for action in (self.act_compare_single, self.act_compare_side, self.act_compare_swipe):
            self.compare_group.addAction(action)
        self.act_compare_single.setChecked(True)

        # 显示模式：Imshow / Imagesc / Mesh，按工作区保存；工具栏右侧为图标按钮组。
        self.display_group = QActionGroup(self)
        self.display_group.setExclusive(True)
        self.display_actions: dict[str, QAction] = {}
        for mode, title, tooltip in DISPLAY_MODE_SPECS:
            action = QAction(title, self)
            action.setIcon(display_mode_icon(mode))
            action.setToolTip(tooltip)
            action.setStatusTip(tooltip)
            action.setCheckable(True)
            action.setData(mode)
            action.triggered.connect(
                lambda _checked=False, target=mode: self._set_display_mode(target)
            )
            self.display_group.addAction(action)
            self.display_actions[mode] = action
        self.act_display_imshow = self.display_actions["imshow"]
        self.act_display_imagesc = self.display_actions["imagesc"]
        self.act_display_mesh = self.display_actions["mesh"]
        self.act_display_imshow.setChecked(True)

        self.act_fit = self._action("适应窗口", self.fit_preview, "Ctrl+9", QStyle.StandardPixmap.SP_TitleBarMaxButton)
        self.act_actual = self._action("100% 显示", self.reset_preview_zoom, "Ctrl+0")
        self.act_thumbnails = self._action(
            "显示节点缩略图", self._toggle_thumbnails, checkable=True
        )
        self.act_thumbnails.setChecked(self.settings.show_node_thumbnails)
        self.act_auto_layout = self._action("自动布局", self.auto_layout, "Ctrl+L")

        self.act_refresh_ops = self._action(
            "刷新算子库", self.refresh_operators, "F5", QStyle.StandardPixmap.SP_BrowserReload
        )
        self.act_open_operator_dir = self._action("打开用户算子目录", self.open_user_operator_dir)
        self.act_load_log = self._action("查看加载日志…", self.show_load_log)
        self.act_logs = self._action(
            "日志…",
            self.show_logs,
            "Ctrl+Shift+L",
            QStyle.StandardPixmap.SP_FileDialogDetailedView,
        )

        self.act_settings = self._action("首选项…", self.open_settings, "Ctrl+,")
        self.act_operator_guide = self._action("算子编写指南", self.open_operator_guide)
        self.act_about = self._action("关于 MineImage", self.show_about)

        self.act_close_tab.setEnabled(False)
        self.act_save.setEnabled(False)
        self.act_save_as.setEnabled(False)
        self.act_export.setEnabled(False)

    # ------------------------------------------------------------------ 界面
    def _create_central(self) -> None:
        self.central_stack = QStackedWidget(self)
        self.welcome = self._create_welcome()
        self.tabs = QTabWidget(self)
        self.tabs.setTabsClosable(True)
        self.tabs.setMovable(True)
        self.tabs.setDocumentMode(True)
        self.tabs.setElideMode(Qt.TextElideMode.ElideRight)
        self.tabs.tabCloseRequested.connect(self._on_tab_close_requested)
        self.tabs.currentChanged.connect(self._on_current_tab_changed)

        self.central_stack.addWidget(self.welcome)
        self.central_stack.addWidget(self.tabs)
        self.setCentralWidget(self.central_stack)

    def _create_welcome(self) -> QWidget:
        page = QWidget(self)
        page.setObjectName("welcomePage")
        outer = QVBoxLayout(page)
        outer.setContentsMargins(60, 60, 60, 60)
        outer.addStretch(1)

        card = QFrame(page)
        card.setObjectName("welcomeCard")
        card.setMaximumWidth(760)
        card.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(44, 40, 44, 40)
        layout.setSpacing(14)

        title = QLabel("MineImage", card)
        title.setObjectName("welcomeTitle")
        subtitle = QLabel("节点式图像处理实验工作台", card)
        subtitle.setObjectName("welcomeSubtitle")
        description = QLabel(
            "把算子拖入画布、连线成流程，随时查看每个中间节点的结果。\n"
            "算子目录中的 Python 文件会被自动发现，无需改动主程序。",
            card,
        )
        description.setWordWrap(True)
        description.setObjectName("welcomeText")

        buttons = QHBoxLayout()
        buttons.setSpacing(10)
        open_btn = QPushButton("打开图像…", card)
        open_btn.setObjectName("primaryButton")
        open_btn.clicked.connect(self.open_image)
        project_btn = QPushButton("打开工程…", card)
        project_btn.clicked.connect(self.open_project)
        new_btn = QPushButton("新建工作区", card)
        new_btn.clicked.connect(self.new_workspace)
        for button in (open_btn, project_btn, new_btn):
            button.setMinimumHeight(34)
            buttons.addWidget(button)
        buttons.addStretch(1)

        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(description)
        layout.addSpacing(8)
        layout.addLayout(buttons)

        row = QHBoxLayout()
        row.addStretch(1)
        row.addWidget(card)
        row.addStretch(1)
        outer.addLayout(row)
        outer.addStretch(2)
        return page

    def _create_menus(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("文件(&F)")
        file_menu.addAction(self.act_new)
        file_menu.addAction(self.act_open)
        file_menu.addAction(self.act_add_image)
        file_menu.addAction(self.act_open_project)
        self.recent_menu = file_menu.addMenu("最近打开")
        self.recent_menu.aboutToShow.connect(self._populate_recent_menu)
        file_menu.addSeparator()
        file_menu.addAction(self.act_save)
        file_menu.addAction(self.act_save_as)
        file_menu.addAction(self.act_export)
        file_menu.addSeparator()
        file_menu.addAction(self.act_close_tab)
        file_menu.addAction(self.act_quit)

        edit_menu = menu.addMenu("编辑(&E)")
        edit_menu.addAction(self.act_undo)
        edit_menu.addAction(self.act_redo)
        edit_menu.addSeparator()
        edit_menu.addAction(self.act_delete)
        edit_menu.addSeparator()
        edit_menu.addAction(self.act_add_group)

        view_menu = menu.addMenu("视图(&V)")
        view_menu.addAction(self.act_compare_single)
        view_menu.addAction(self.act_compare_side)
        view_menu.addAction(self.act_compare_swipe)
        view_menu.addSeparator()
        display_menu = view_menu.addMenu("显示模式")
        for mode, _title, _tooltip in DISPLAY_MODE_SPECS:
            display_menu.addAction(self.display_actions[mode])
        view_menu.addSeparator()
        view_menu.addAction(self.act_fit)
        view_menu.addAction(self.act_actual)
        view_menu.addSeparator()
        view_menu.addAction(self.act_thumbnails)
        view_menu.addAction(self.act_auto_layout)

        op_menu = menu.addMenu("算子(&O)")
        op_menu.addAction(self.act_refresh_ops)
        op_menu.addAction(self.act_open_operator_dir)
        op_menu.addAction(self.act_load_log)

        settings_menu = menu.addMenu("设置(&S)")
        settings_menu.addAction(self.act_settings)

        help_menu = menu.addMenu("帮助(&H)")
        help_menu.addAction(self.act_operator_guide)
        help_menu.addSeparator()
        help_menu.addAction(self.act_logs)
        help_menu.addAction(self.act_about)

    def _create_toolbar(self) -> None:
        toolbar = self.addToolBar("主工具栏")
        toolbar.setObjectName("mainToolBar")
        toolbar.setMovable(False)
        toolbar.setIconSize(QSize(18, 18))
        toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        for action in (
            self.act_open,
            self.act_add_image,
            self.act_save,
            None,
            self.act_undo,
            self.act_redo,
            None,
            self.act_fit,
            self.act_actual,
        ):
            if action is None:
                toolbar.addSeparator()
            else:
                toolbar.addAction(action)

        # 右侧：显示模式按钮组（图标按钮，按工作区独立保存）。
        spacer = QWidget(toolbar)
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        spacer.setMinimumWidth(12)
        spacer.setAccessibleName("工具栏弹性占位")
        toolbar.addWidget(spacer)
        toolbar.addSeparator()
        hint = QLabel("显示模式", toolbar)
        hint.setObjectName("toolbarHintLabel")
        hint.setToolTip("图像显示模式：Imshow / Imagesc / Mesh（每个工作区独立保存）")
        toolbar.addWidget(hint)
        for mode, title, tooltip in DISPLAY_MODE_SPECS:
            action = self.display_actions[mode]
            toolbar.addAction(action)
            button = toolbar.widgetForAction(action)
            if isinstance(button, QToolButton):
                button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonIconOnly)
                button.setObjectName("displayModeButton")
                button.setAccessibleName(f"{title} 显示模式")
                button.setToolTip(tooltip)

    def _create_statusbar(self) -> None:
        bar = self.statusBar()
        bar.setSizeGripEnabled(True)
        self.status_message_label = QLabel("就绪", self)
        self.status_message_label.setObjectName("statusMessage")
        self.status_message_label.setMinimumWidth(220)
        bar.addWidget(self.status_message_label, 1)

        specs = [
            ("node", "节点"),
            ("size", "图像"),
            ("zoom", "缩放"),
            ("elapsed", "耗时"),
            ("coord", "行列"),
            ("pixel", "像素"),
        ]
        for index, (key, title) in enumerate(specs):
            if index:
                line = QFrame(self)
                line.setFrameShape(QFrame.Shape.VLine)
                line.setObjectName("statusSeparator")
                bar.addPermanentWidget(line)
            label = QLabel(self)
            label.setObjectName(f"status_{key}")
            label.setProperty("statusKey", key)
            if key == "pixel":
                label.setMinimumWidth(280)
            elif key == "node":
                label.setMinimumWidth(130)
            bar.addPermanentWidget(label)
            self._status_labels[key] = label
            self._set_status_field(key, "--")
        self._reset_status()

    # ------------------------------------------------------------------ 工作区
    def _add_workspace(self, workspace: WorkspaceModel, project_path: Path | None = None) -> WorkspaceWidget:
        widget = WorkspaceWidget(workspace, self.registry, self.settings)
        if project_path is not None:
            widget.project_path = Path(project_path)
        index = self.tabs.addTab(widget, widget.tab_title())
        self.tabs.setTabToolTip(index, str(project_path) if project_path else workspace.name)

        widget.title_changed.connect(lambda _title, w=widget: self._on_workspace_title_changed(w))
        widget.modified_changed.connect(lambda _dirty, w=widget: self._on_workspace_modified(w))
        widget.status_info.connect(lambda info, w=widget: self._on_status_info(w, info))
        widget.status_message.connect(lambda msg, w=widget: self._on_workspace_message(w, msg))
        widget.library.refresh_requested.connect(self.refresh_operators)
        widget.operator_refresh_requested.connect(self.refresh_operators)
        widget.undo_stack.canUndoChanged.connect(self._refresh_edit_actions)
        widget.undo_stack.canRedoChanged.connect(self._refresh_edit_actions)
        widget.graph_view.node_created.connect(lambda _id: self._refresh_edit_actions())

        self._workspaces.append(widget)
        self.tabs.setCurrentWidget(widget)
        self._sync_display_mode_actions(widget)
        self._update_central()
        return widget

    def _active_workspace(self) -> WorkspaceWidget | None:
        widget = self.tabs.currentWidget()
        return widget if isinstance(widget, WorkspaceWidget) else None

    def _update_central(self) -> None:
        has_tabs = self.tabs.count() > 0
        self.central_stack.setCurrentWidget(self.tabs if has_tabs else self.welcome)
        self.act_close_tab.setEnabled(has_tabs)
        self.act_save.setEnabled(has_tabs)
        self.act_save_as.setEnabled(has_tabs)
        self.act_export.setEnabled(has_tabs)
        for action in self.display_actions.values():
            action.setEnabled(has_tabs)
        self._update_window_title()

    def _update_window_title(self) -> None:
        widget = self._active_workspace()
        if widget is None:
            self.setWindowTitle(APP_DISPLAY_NAME)
        else:
            self.setWindowTitle(f"{widget.workspace.name}{' *' if widget.dirty else ''} — {APP_DISPLAY_NAME}")

    def _on_workspace_title_changed(self, widget: WorkspaceWidget) -> None:
        index = self.tabs.indexOf(widget)
        if index >= 0:
            self.tabs.setTabText(index, widget.tab_title())
        if widget is self._active_workspace():
            self._update_window_title()

    def _on_workspace_modified(self, widget: WorkspaceWidget) -> None:
        self._on_workspace_title_changed(widget)
        self._refresh_edit_actions()

    def _on_workspace_message(self, widget: WorkspaceWidget, message: str) -> None:
        if widget is self._active_workspace():
            self.status_message_label.setText(message)
            self.statusBar().showMessage(message, 5000)

    def _on_tab_close_requested(self, index: int) -> None:
        widget = self.tabs.widget(index)
        if isinstance(widget, WorkspaceWidget):
            self._maybe_close_workspace(widget)

    def _on_current_tab_changed(self, _index: int) -> None:
        widget = self._active_workspace()
        if widget is None:
            self._reset_status()
            self._sync_display_mode_actions(None)
            self._update_window_title()
            self._refresh_edit_actions()
            return
        action = {
            "single": self.act_compare_single,
            "side_by_side": self.act_compare_side,
            "swipe": self.act_compare_swipe,
        }.get(widget.compare_mode, self.act_compare_single)
        action.setChecked(True)
        self._sync_display_mode_actions(widget)
        widget.refresh_status()
        self._update_window_title()
        self._refresh_edit_actions()

    def _toggle_thumbnails(self, checked: bool) -> None:
        self.settings.show_node_thumbnails = checked
        self.settings.sync()
        for widget in self._workspaces:
            widget.graph_view.set_thumbnails_enabled(checked)

    def _set_compare_mode(self, mode: str) -> None:
        widget = self._active_workspace()
        if widget is not None:
            widget.set_compare_mode(mode)

    def _set_display_mode(self, mode: str) -> None:
        widget = self._active_workspace()
        if widget is None:
            self._sync_display_mode_actions(None)
            return
        widget.set_display_mode(mode)

    def _sync_display_mode_actions(self, widget: WorkspaceWidget | None) -> None:
        mode = widget.display_mode if widget is not None else DEFAULT_IMAGE_DISPLAY_MODE
        action = self.display_actions.get(mode, self.act_display_imshow)
        action.setChecked(True)

    # ------------------------------------------------------------------ 文件操作
    def new_workspace(self) -> WorkspaceWidget:
        self._workspace_counter += 1
        workspace = WorkspaceModel(f"未命名工作区 {self._workspace_counter}")
        widget = self._add_workspace(workspace)
        widget.status_message.emit("空白工作区：打开/添加图像，或从左侧拖入算子")
        return widget

    def _populate_recent_menu(self) -> None:
        self.recent_menu.clear()
        items = self.settings.recent_files
        if not items:
            action = self.recent_menu.addAction("（暂无）")
            action.setEnabled(False)
            return
        for raw in items:
            path = Path(raw)
            action = self.recent_menu.addAction(path.name)
            action.setToolTip(str(path))
            action.triggered.connect(lambda _checked=False, p=raw: self._open_recent(p))

    def _open_recent(self, path_text: str) -> None:
        path = Path(path_text)
        if not path.exists():
            QMessageBox.warning(self, "文件不存在", f"最近文件已不存在：\n{path}")
            return
        if path.suffix.lower() == ".mi":
            self.open_project_path(path)
        else:
            self.open_image_path(path)

    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开图像", "", IMAGE_FILTER)
        if not path:
            return
        self.open_image_path(path)

    def open_image_path(self, path: str | Path) -> None:
        workspace = WorkspaceModel(Path(path).stem)
        widget = self._add_workspace(workspace)
        self.tabs.setCurrentWidget(widget)
        if not widget.add_image(str(path)):
            index = self.tabs.indexOf(widget)
            if index >= 0:
                self.tabs.removeTab(index)
            if widget in self._workspaces:
                self._workspaces.remove(widget)
            widget.shutdown()
            widget.deleteLater()
            self._update_central()
            return
        self.settings.add_recent_file(str(path))
        self.status_message_label.setText(f"已打开图像：{Path(path).name}")
        self._update_window_title()

    def add_image(self) -> None:
        widget = self._active_workspace()
        if widget is None:
            widget = self.new_workspace()
        path, _ = QFileDialog.getOpenFileName(self, "添加图像到当前流程", "", IMAGE_FILTER)
        if not path:
            return
        if widget.add_image(path):
            self.settings.add_recent_file(path)
            self._update_window_title()

    def open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "打开 MineImage 工程", "", PROJECT_FILTER)
        if not path:
            return
        self.open_project_path(path)

    def open_project_path(self, path: str | Path) -> None:
        try:
            workspace = load_project(path)
        except ProjectFormatError as exc:
            LOG.error("无法打开工程 %s：%s", path, exc)
            QMessageBox.critical(self, "无法打开工程", str(exc))
            return
        except Exception as exc:  # noqa: BLE001
            LOG.exception("无法打开工程 %s", path)
            QMessageBox.critical(self, "无法打开工程", f"{exc.__class__.__name__}: {exc}")
            return
        widget = self._add_workspace(workspace, Path(path))
        widget.mark_clean()
        self.settings.add_recent_file(str(path))
        self.status_message_label.setText(f"已打开工程：{Path(path).name}")
        self.tabs.setCurrentWidget(widget)
        # 打开工程后自动定位到一个终止节点，避免右侧预览停留在空白状态。
        sinks = [
            node
            for node in workspace.graph.nodes.values()
            if not workspace.graph.outgoing(node.id)
        ]
        if sinks:
            widget.select_node_and_preview(sinks[-1].id)
        # 标签页/分隔条布局完成后才能得到正确 viewport，延迟一帧再 fit。
        QTimer.singleShot(0, widget.graph_view.fit_graph)
        self._update_window_title()

    def save_workspace(self) -> bool:
        widget = self._active_workspace()
        if widget is None:
            return False
        return self._save_workspace(widget, save_as=False)

    def save_workspace_as(self) -> bool:
        widget = self._active_workspace()
        if widget is None:
            return False
        return self._save_workspace(widget, save_as=True)

    def _save_workspace(self, widget: WorkspaceWidget, save_as: bool) -> bool:
        target: Path | None = widget.project_path
        if target is None or save_as:
            default_dir = target.parent if target else Path.home() / "Documents"
            default_name = (target.name if target else f"{widget.workspace.name}.mi")
            chosen, _ = QFileDialog.getSaveFileName(
                self,
                "保存 MineImage 工程",
                str(default_dir / default_name),
                PROJECT_FILTER,
            )
            if not chosen:
                return False
            target = Path(chosen)
        try:
            saved = widget.save_to(target)
        except Exception as exc:  # noqa: BLE001
            LOG.exception("保存工程失败：%s", target)
            QMessageBox.critical(self, "保存失败", str(exc))
            return False
        self.settings.add_recent_file(str(saved))
        self.settings.sync()
        self._on_workspace_title_changed(widget)
        self._update_window_title()
        self.statusBar().showMessage(f"已保存：{saved.name}", 5000)
        return True

    def close_current_tab(self) -> None:
        index = self.tabs.currentIndex()
        if index >= 0:
            self._on_tab_close_requested(index)

    def export_current_image(self) -> None:
        widget = self._active_workspace()
        if widget is None:
            return
        node = widget.workspace.graph.nodes.get(widget._current_node_id or "")
        stem = node.display_title() if node else widget.workspace.name
        default_path = Path.home() / "Pictures" / f"{stem}.png"
        default_path.parent.mkdir(parents=True, exist_ok=True)
        chosen, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "导出当前节点图像",
            str(default_path),
            "PNG 图像 (*.png);;TIFF 图像 (*.tif *.tiff);;NumPy 数组 (*.npy)",
        )
        if not chosen:
            return
        if widget.export_current_image(chosen):
            self.statusBar().showMessage(f"已导出：{Path(chosen).name}", 5000)

    def _maybe_close_workspace(self, widget: WorkspaceWidget) -> bool:
        if widget.dirty:
            result = QMessageBox.warning(
                self,
                "未保存的修改",
                f"“{widget.workspace.name}”有未保存的修改，是否保存？",
                QMessageBox.StandardButton.Save
                | QMessageBox.StandardButton.Discard
                | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Save,
            )
            if result == QMessageBox.StandardButton.Cancel:
                return False
            if result == QMessageBox.StandardButton.Save and not self._save_workspace(widget, save_as=False):
                return False
        index = self.tabs.indexOf(widget)
        try:
            self.settings.workspace_splitter_sizes = widget.splitter_sizes()
        except Exception:  # noqa: BLE001
            pass
        widget.shutdown()
        if index >= 0:
            self.tabs.removeTab(index)
        if widget in self._workspaces:
            self._workspaces.remove(widget)
        widget.deleteLater()
        self._update_central()
        return True

    # ------------------------------------------------------------------ 编辑
    def _refresh_edit_actions(self) -> None:
        widget = self._active_workspace()
        can_undo = widget is not None and widget.undo_stack.canUndo()
        can_redo = widget is not None and widget.undo_stack.canRedo()
        self.act_undo.setEnabled(can_undo)
        self.act_redo.setEnabled(can_redo)
        self.act_delete.setEnabled(widget is not None)
        if widget is not None:
            self.act_undo.setText(f"撤销 {widget.undo_stack.undoText()}".strip())
            self.act_redo.setText(f"重做 {widget.undo_stack.redoText()}".strip())

    def undo(self) -> None:
        widget = self._active_workspace()
        if widget is not None:
            widget.undo_stack.undo()
            self._refresh_edit_actions()

    def redo(self) -> None:
        widget = self._active_workspace()
        if widget is not None:
            widget.undo_stack.redo()
            self._refresh_edit_actions()

    def delete_selected(self) -> None:
        widget = self._active_workspace()
        if widget is not None:
            widget.graph_view.delete_selected()

    def add_group(self) -> None:
        """在当前画布视口中心添加一个算子组。"""
        widget = self._active_workspace()
        if widget is not None:
            widget.graph_view.add_group_at()

    def auto_layout(self) -> None:
        widget = self._active_workspace()
        if widget is not None:
            widget.graph_view.auto_layout()

    def fit_preview(self) -> None:
        widget = self._active_workspace()
        if widget is not None:
            widget.fit_preview()

    def reset_preview_zoom(self) -> None:
        widget = self._active_workspace()
        if widget is not None:
            widget.reset_preview_zoom()

    # ------------------------------------------------------------------ 设置
    def open_settings(self) -> None:
        dialog = SettingsDialog(self.settings, self)
        dialog.applied.connect(self._apply_settings)
        dialog.exec()
        dialog.deleteLater()

    def _apply_settings(self) -> None:
        self.settings.sync()
        self.act_thumbnails.setChecked(self.settings.show_node_thumbnails)
        for widget in self._workspaces:
            widget.apply_settings()
        self.refresh_operators()
        self._setup_operator_watcher()

    # ------------------------------------------------------------------ 算子
    def refresh_operators(self) -> None:
        self.statusBar().showMessage("正在扫描算子目录…")
        QApplication.setOverrideCursor(Qt.CursorShape.WaitCursor)
        try:
            new_registry = self._build_registry()
        finally:
            QApplication.restoreOverrideCursor()
        self.registry = new_registry
        for widget in self._workspaces:
            widget.set_registry(new_registry)
        self._watch_snapshot = self._operator_snapshot()
        count = len(new_registry)
        errors = len(new_registry.errors)
        message = f"算子库已刷新：{count} 个算子"
        if errors:
            message += f"，{errors} 个加载失败"
        self.statusBar().showMessage(message, 6000)
        self.status_message_label.setText(message)

    # ---------------------------------------------------------- 算子热加载轮询
    def _operator_snapshot(self) -> dict[str, tuple[int, int]]:
        """记录扫描目录内 .py 文件的 (mtime_ns, size)，用于可选的自动刷新。"""
        snapshot: dict[str, tuple[int, int]] = {}
        for root in self.registry.scan_roots:
            try:
                files = root.rglob("*.py")
                for path in files:
                    if "__pycache__" in path.parts or path.name.startswith("."):
                        continue
                    try:
                        stat = path.stat()
                    except OSError:
                        continue
                    snapshot[str(path)] = (stat.st_mtime_ns, stat.st_size)
            except OSError:
                continue
        return snapshot

    def _setup_operator_watcher(self) -> None:
        self._watch_snapshot = self._operator_snapshot()
        if self.settings.watch_operator_dirs:
            self._watch_timer.start()
        else:
            self._watch_timer.stop()

    def _check_operator_watch(self) -> None:
        if not self.settings.watch_operator_dirs:
            self._watch_timer.stop()
            return
        current = self._operator_snapshot()
        if current != self._watch_snapshot:
            self._watch_snapshot = current
            self.statusBar().showMessage("检测到算子文件变化，正在重新扫描…", 3000)
            self.refresh_operators()

    def open_user_operator_dir(self) -> None:
        path = default_user_operator_dir()
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(path)))

    def show_load_log(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("算子加载日志")
        dialog.resize(840, 560)
        layout = QVBoxLayout(dialog)
        if self.registry.issues:
            lines = []
            for issue in self.registry.issues:
                lines.append(f"[{issue.level.upper()}] {issue.path}")
                lines.append(f"    {issue.message}")
                if issue.traceback:
                    lines.append(issue.traceback.rstrip())
                lines.append("")
            text = "\n".join(lines)
        else:
            text = "没有加载问题。\n\n扫描目录：\n" + "\n".join(str(path) for path in self.registry.scan_roots)
        editor = QTextEdit(dialog)
        editor.setReadOnly(True)
        editor.setPlainText(text)
        layout.addWidget(editor)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
        buttons.rejected.connect(dialog.reject)
        open_btn = buttons.addButton("打开用户算子目录", QDialogButtonBox.ButtonRole.ActionRole)
        open_btn.clicked.connect(self.open_user_operator_dir)
        layout.addWidget(buttons)
        dialog.exec()
        dialog.deleteLater()

    # ------------------------------------------------------------------ 日志
    def show_logs(self) -> None:
        """打开/前置 SQLite 应用日志查看器（帮助 → 日志）。

        使用非模态窗口：用户可以一边操作主界面、复现问题，一边观察日志
        自动刷新；再次触发菜单时只前置已有窗口。
        """
        try:
            if self.log_store is None:
                self.log_store = LogStore(default_log_db_path())
            dialog = self._log_dialog
            if dialog is None or not dialog.isVisible():
                if dialog is not None:
                    dialog.deleteLater()
                dialog = LogDialog(self.log_store, self)
                dialog.setModal(False)
                self._log_dialog = dialog
            dialog.show()
            dialog.raise_()
            dialog.activateWindow()
        except Exception as exc:  # noqa: BLE001
            LOG.exception("无法打开应用日志")
            QMessageBox.critical(self, "无法打开日志", f"{exc.__class__.__name__}: {exc}")

    # ------------------------------------------------------------------ 帮助
    def open_operator_guide(self) -> None:
        candidates = [
            application_root() / "docs" / "operator_guide.md",
            resource_root() / "docs" / "operator_guide.md",
        ]
        guide = next((path for path in candidates if path.exists()), None)
        if guide is not None:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(guide)))
        else:
            QMessageBox.information(self, "算子编写指南", f"文档尚未生成：\n{candidates[0]}")

    def show_about(self) -> None:
        from PySide6 import __version__ as pyside_version
        from PySide6.QtCore import qVersion

        QMessageBox.about(
            self,
            "关于 MineImage",
            f"<h3>MineImage {__version__}</h3>"
            "<p>节点式图像处理实验工作台。</p>"
            f"<p>Python {platform.python_version()} · PySide6 {pyside_version} · Qt {qVersion()}</p>"
            f"<p>用户算子目录：<code>{default_user_operator_dir()}</code></p>",
        )

    # ------------------------------------------------------------------ 状态栏
    def _set_status_field(self, key: str, value: str) -> None:
        label = self._status_labels.get(key)
        if label is None:
            return
        prefix = {
            "node": "节点",
            "size": "图像",
            "zoom": "缩放",
            "elapsed": "耗时",
            "coord": "行,列",
            "pixel": "像素",
        }.get(key, key)
        label.setText(f"{prefix}: {value}")

    def _reset_status(self) -> None:
        for key in ("node", "size", "zoom", "elapsed", "coord", "pixel"):
            self._set_status_field(key, "--")
        self.status_message_label.setText("就绪")

    def _on_status_info(self, widget: WorkspaceWidget, info: dict) -> None:
        if widget is not self._active_workspace():
            return
        self._set_status_field("node", info.get("node", "--"))
        self._set_status_field("size", info.get("size", "--"))
        self._set_status_field("zoom", info.get("zoom", "--"))
        self._set_status_field("elapsed", info.get("elapsed", "--"))
        self._set_status_field("coord", info.get("coord", "--"))
        self._set_status_field("pixel", info.get("pixel", "--"))

    # ------------------------------------------------------------------ 窗口
    def _restore_window_state(self) -> None:
        geometry = self.settings.window_geometry
        if geometry:
            try:
                self.restoreGeometry(QByteArray(geometry))
            except Exception:  # noqa: BLE001
                pass
        state = self.settings.window_state
        if state:
            try:
                self.restoreState(QByteArray(state))
            except Exception:  # noqa: BLE001
                pass

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802
        for widget in list(self._workspaces):
            if not self._maybe_close_workspace(widget):
                event.ignore()
                return
        # 非模态日志窗口是主窗口的子窗口；显式关闭，避免主窗口退出后
        # 它仍作为可见顶层窗口阻止 QApplication 退出。
        if self._log_dialog is not None:
            self._log_dialog.close()
            self._log_dialog.deleteLater()
            self._log_dialog = None
        self.settings.window_geometry = bytes(self.saveGeometry())
        self.settings.window_state = bytes(self.saveState())
        self.settings.sync()
        event.accept()
