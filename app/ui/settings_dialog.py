"""设置对话框：常规 / 预览 / 模型 / 算子目录。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.config import AppSettings
from app.core import models
from app.ui.model_download import (
    DownloadSignals,
    ModelDownloadTask,
    start_model_download,
)


class SettingsDialog(QDialog):
    """应用设置对话框；点击「确定」时写入并持久化。"""

    applied = Signal()

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("设置")
        self.setModal(True)
        self.setMinimumSize(680, 560)
        self.resize(720, 620)
        self._build_ui()
        self._load_values()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title = QLabel("设置", self)
        title.setObjectName("sectionTitle")
        layout.addWidget(title)

        self.tabs = QTabWidget(self)
        self.tabs.addTab(self._build_general_tab(), "常规")
        self.tabs.addTab(self._build_preview_tab(), "预览")
        self.tabs.addTab(self._build_models_tab(), "模型")
        self.tabs.addTab(self._build_operators_tab(), "算子目录")
        layout.addWidget(self.tabs, 1)

        self.button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        ok_button = self.button_box.button(QDialogButtonBox.StandardButton.Ok)
        ok_button.setText("确定")
        ok_button.setObjectName("primaryButton")
        cancel_button = self.button_box.button(QDialogButtonBox.StandardButton.Cancel)
        cancel_button.setText("取消")
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        layout.addWidget(self.button_box)

    def _build_general_tab(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(10)

        group = QGroupBox("画布与撤销", page)
        form = QFormLayout(group)
        form.setContentsMargins(12, 16, 12, 10)
        form.setSpacing(10)
        self.undo_spin = QSpinBox(group)
        self.undo_spin.setRange(1, 10000)
        self.undo_spin.setSingleStep(10)
        self.undo_spin.setSuffix(" 步")
        self.undo_spin.setToolTip("撤销栈最多保留的步数（至少 1）。")
        self.thumbnails_check = QCheckBox("在节点上显示结果缩略图", group)
        self.thumbnails_check.setToolTip("关闭可提升大图场景下的画布性能。")
        self.auto_fit_check = QCheckBox("打开预览时自动适应窗口", group)
        form.addRow("撤销步数", self.undo_spin)
        form.addRow("节点缩略图", self.thumbnails_check)
        form.addRow("预览", self.auto_fit_check)
        layout.addWidget(group)

        hint = QLabel("修改撤销步数只会影响之后的撤销操作，不会清空当前历史。", page)
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _build_preview_tab(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(10)

        group = QGroupBox("显示质量", page)
        form = QFormLayout(group)
        form.setContentsMargins(12, 16, 12, 10)
        form.setSpacing(10)
        self.downsample_check = QCheckBox("启用预览降采样（只影响显示，不改变计算像素）", group)
        self.max_dim_spin = QSpinBox(group)
        self.max_dim_spin.setRange(256, 16384)
        self.max_dim_spin.setSingleStep(256)
        self.max_dim_spin.setSuffix(" px")
        self.max_dim_spin.setToolTip("降采样后图像最长边的像素数（至少 256）。")
        self.cursor_info_check = QCheckBox("在状态栏显示光标处像素值", group)
        form.addRow(self.downsample_check)
        form.addRow("最大显示边长", self.max_dim_spin)
        form.addRow("像素信息", self.cursor_info_check)
        self.downsample_check.toggled.connect(self.max_dim_spin.setEnabled)
        layout.addWidget(group)

        hint = QLabel(
            "降采样仅作用于右侧预览控件的显示缓冲；节点缓存与保存结果始终使用原始分辨率。",
            page,
        )
        hint.setObjectName("hintLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _build_models_tab(self) -> QWidget:
        # 模型较多 / 窗口较小时可滚动，不影响其它标签页的尺寸。
        container = QWidget(self)
        outer = QVBoxLayout(container)
        outer.setContentsMargins(0, 0, 0, 0)
        scroll = QScrollArea(container)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        page = QWidget(scroll)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(8)

        intro = QLabel(
            "OpenCV 官方 ONNX 模型托管在 OpenCV Zoo。可点击下方的下载地址手动下载，"
            "或直接点击「下载」在后台自动下载；下载期间界面仍可正常操作。",
            page,
        )
        intro.setObjectName("hintLabel")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        dir_group = QGroupBox("模型目录", page)
        dir_layout = QVBoxLayout(dir_group)
        dir_layout.setContentsMargins(12, 16, 12, 10)
        dir_layout.setSpacing(6)
        dir_row = QHBoxLayout()
        dir_row.setSpacing(6)
        dir_caption = QLabel("缓存目录：", dir_group)
        self.model_dir_edit = QLineEdit(dir_group)
        self.model_dir_edit.setReadOnly(True)
        self.model_dir_edit.setText(str(models.model_dir()))
        self.model_dir_edit.setToolTip(str(models.model_dir()))
        self.open_model_dir_button = QPushButton("打开模型目录", dir_group)
        self.open_model_dir_button.setObjectName("subtleButton")
        self.open_model_dir_button.setToolTip("在文件管理器中打开模型缓存目录（不存在时自动创建）。")
        self.open_model_dir_button.clicked.connect(self._on_open_model_dir)
        dir_row.addWidget(dir_caption, 0)
        dir_row.addWidget(self.model_dir_edit, 1)
        dir_row.addWidget(self.open_model_dir_button, 0)
        dir_layout.addLayout(dir_row)
        dir_hint = QLabel(
            "下载的模型文件保存在该目录；也可用环境变量 MINEIMAGE_MODEL_DIR 指定其它位置。",
            dir_group,
        )
        dir_hint.setObjectName("hintLabel")
        dir_hint.setWordWrap(True)
        dir_layout.addWidget(dir_hint)
        layout.addWidget(dir_group)

        zoo_row = QHBoxLayout()
        zoo_row.setSpacing(6)
        self.open_zoo_button = QPushButton("打开 OpenCV Zoo", page)
        self.open_zoo_button.setObjectName("subtleButton")
        self.open_zoo_button.setToolTip(models.OPENCV_ZOO_URL)
        self.open_zoo_button.clicked.connect(self._on_open_zoo)
        zoo_row.addWidget(self.open_zoo_button, 0)
        zoo_hint = QLabel("模型来源、许可证与使用说明见 OpenCV Zoo 仓库。", page)
        zoo_hint.setObjectName("hintLabel")
        zoo_row.addWidget(zoo_hint, 1)
        layout.addLayout(zoo_row)

        # 模型状态 / 地址 / 下载按钮；下载信号与任务都由本对话框持有。
        self.model_status_labels: dict[str, QLabel] = {}
        self.model_url_labels: dict[str, QLabel] = {}
        self.model_download_buttons: dict[str, QPushButton] = {}
        self.model_progress_bars: dict[str, QProgressBar] = {}
        self._download_signals: dict[str, DownloadSignals] = {}
        self._model_tasks: dict[str, ModelDownloadTask] = {}
        for spec in models.MODELS.values():
            layout.addWidget(self._build_model_group(spec, page))
        layout.addStretch(1)

        scroll.setWidget(page)
        outer.addWidget(scroll)
        return container

    def _build_model_group(self, spec: models.ModelSpec, parent: QWidget) -> QGroupBox:
        """一个内置模型：名称、文件名、官方地址、状态与下载按钮。"""
        group = QGroupBox(spec.name, parent)
        layout = QVBoxLayout(group)
        layout.setContentsMargins(12, 16, 12, 10)
        layout.setSpacing(6)

        description = QLabel(spec.description, group)
        description.setObjectName("hintLabel")
        description.setWordWrap(True)
        layout.addWidget(description)

        file_row = QHBoxLayout()
        file_row.setSpacing(6)
        file_caption = QLabel("文件名：", group)
        file_label = QLabel(spec.filename, group)
        file_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        file_label.setToolTip(spec.filename)
        file_row.addWidget(file_caption, 0)
        file_row.addWidget(file_label, 0)
        file_row.addStretch(1)
        layout.addLayout(file_row)

        url_row = QHBoxLayout()
        url_row.setSpacing(6)
        url_caption = QLabel("下载地址：", group)
        url_label = QLabel(
            f'<a href="{spec.url}">{spec.url}</a>',
            group,
        )
        url_label.setTextFormat(Qt.TextFormat.RichText)
        url_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextBrowserInteraction
        )
        url_label.setOpenExternalLinks(True)
        url_label.setWordWrap(True)
        url_label.setToolTip(spec.url)
        url_row.addWidget(url_caption, 0)
        url_row.addWidget(url_label, 1)
        layout.addLayout(url_row)

        status_row = QHBoxLayout()
        status_row.setSpacing(6)
        status_caption = QLabel("状态：", group)
        status_label = QLabel(models.model_status_text(spec), group)
        status_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        status_label.setWordWrap(True)
        progress = QProgressBar(group)
        progress.setRange(0, 100)
        progress.setValue(0)
        progress.setTextVisible(False)
        progress.setFixedHeight(6)
        progress.setMaximumWidth(160)
        progress.setVisible(False)
        download_button = QPushButton("下载", group)
        download_button.setToolTip(
            f"在后台下载 {spec.filename}，保存到模型目录。"
        )
        download_button.clicked.connect(
            lambda _checked=False, key=spec.key: self._on_download_model(key)
        )
        status_row.addWidget(status_caption, 0)
        status_row.addWidget(status_label, 1)
        status_row.addWidget(progress, 0)
        status_row.addWidget(download_button, 0)
        layout.addLayout(status_row)

        signals = DownloadSignals(group)
        signals.progress.connect(
            lambda done, total, key=spec.key: self._on_model_progress(
                key, done, total
            )
        )
        signals.succeeded.connect(
            lambda path, key=spec.key: self._on_model_succeeded(key, path)
        )
        signals.failed.connect(
            lambda message, key=spec.key: self._on_model_failed(key, message)
        )
        signals.finished.connect(
            lambda key=spec.key: self._on_model_finished(key)
        )

        self.model_status_labels[spec.key] = status_label
        self.model_url_labels[spec.key] = url_label
        self.model_download_buttons[spec.key] = download_button
        self.model_progress_bars[spec.key] = progress
        self._download_signals[spec.key] = signals
        return group

    def _build_operators_tab(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 10, 8, 8)
        layout.setSpacing(8)

        group = QGroupBox("用户算子目录", page)
        group_layout = QVBoxLayout(group)
        group_layout.setContentsMargins(12, 16, 12, 10)
        group_layout.setSpacing(8)
        self.dirs_list = QListWidget(group)
        self.dirs_list.setMinimumHeight(120)
        self.dirs_list.setToolTip("这些目录会被扫描，模块级 OPERATOR 即算子入口。")
        group_layout.addWidget(self.dirs_list)
        buttons = QHBoxLayout()
        buttons.setSpacing(6)
        self.add_dir_button = QPushButton("添加目录…", group)
        self.add_dir_button.clicked.connect(self._on_add_dir)
        self.remove_dir_button = QPushButton("移除选中", group)
        self.remove_dir_button.clicked.connect(self._on_remove_dir)
        buttons.addWidget(self.add_dir_button)
        buttons.addWidget(self.remove_dir_button)
        buttons.addStretch(1)
        group_layout.addLayout(buttons)
        layout.addWidget(group)

        options = QGroupBox("扫描选项", page)
        options_layout = QVBoxLayout(options)
        options_layout.setContentsMargins(12, 16, 12, 10)
        options_layout.setSpacing(8)
        self.builtin_check = QCheckBox("启用内置算子目录", options)
        self.watch_check = QCheckBox("监视算子目录变化（自动重新扫描）", options)
        options_layout.addWidget(self.builtin_check)
        options_layout.addWidget(self.watch_check)
        layout.addWidget(options)

        rescan_hint = QLabel(
            "修改后请回到主界面，点击左侧算子库的「刷新」（或按 F5）重新扫描。",
            page,
        )
        rescan_hint.setObjectName("hintLabel")
        rescan_hint.setWordWrap(True)
        layout.addWidget(rescan_hint)

        path_row = QHBoxLayout()
        path_row.setSpacing(6)
        path_label = QLabel("配置文件：", page)
        self.ini_path_edit = QLineEdit(page)
        self.ini_path_edit.setReadOnly(True)
        self.ini_path_edit.setText(str(self.settings.path))
        self.ini_path_edit.setToolTip(str(self.settings.path))
        self.open_folder_button = QPushButton("打开文件夹", page)
        self.open_folder_button.setObjectName("subtleButton")
        self.open_folder_button.clicked.connect(self._on_open_folder)
        path_row.addWidget(path_label, 0)
        path_row.addWidget(self.ini_path_edit, 1)
        path_row.addWidget(self.open_folder_button, 0)
        layout.addLayout(path_row)
        layout.addStretch(1)
        return page

    # ------------------------------------------------------------------ 数据
    def _load_values(self) -> None:
        self.undo_spin.setValue(max(1, int(self.settings.undo_limit)))
        self.thumbnails_check.setChecked(bool(self.settings.show_node_thumbnails))
        self.auto_fit_check.setChecked(bool(self.settings.auto_fit_on_open))
        self.downsample_check.setChecked(bool(self.settings.downsample_preview))
        self.max_dim_spin.setValue(max(256, int(self.settings.downsample_max_dim)))
        self.max_dim_spin.setEnabled(self.downsample_check.isChecked())
        self.cursor_info_check.setChecked(bool(self.settings.cursor_pixel_info))
        self.builtin_check.setChecked(bool(self.settings.builtin_operators_enabled))
        self.watch_check.setChecked(bool(self.settings.watch_operator_dirs))
        self.dirs_list.clear()
        for path in self.settings.extra_operator_dirs:
            self._add_dir_item(path)

    def extra_dirs(self) -> list[str]:
        result: list[str] = []
        for index in range(self.dirs_list.count()):
            item = self.dirs_list.item(index)
            value = item.data(Qt.ItemDataRole.UserRole) or item.text()
            text = str(value).strip()
            if text and text not in result:
                result.append(text)
        return result

    def _add_dir_item(self, path: str) -> None:
        text = str(Path(path).expanduser())
        item = QListWidgetItem(text)
        item.setData(Qt.ItemDataRole.UserRole, text)
        item.setToolTip(text)
        self.dirs_list.addItem(item)

    def _on_add_dir(self) -> None:
        start = str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "选择算子目录", start)
        if not path:
            return
        normalized = str(Path(path))
        for index in range(self.dirs_list.count()):
            if str(self.dirs_list.item(index).data(Qt.ItemDataRole.UserRole)) == normalized:
                return
        self._add_dir_item(normalized)

    def _on_remove_dir(self) -> None:
        for item in self.dirs_list.selectedItems():
            self.dirs_list.takeItem(self.dirs_list.row(item))

    def _on_open_folder(self) -> None:
        folder = Path(self.settings.path).parent
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # ------------------------------------------------------------------ 模型
    def refresh_model_status(self) -> None:
        """重新读取所有内置模型的本地状态（下载成功或手动放入文件后调用）。"""

        for key in self.model_status_labels:
            self.model_status_labels[key].setText(models.model_status_text(key))

    def _on_download_model(self, key: str) -> None:
        if key in self._model_tasks:
            return
        button = self.model_download_buttons.get(key)
        status = self.model_status_labels.get(key)
        progress = self.model_progress_bars.get(key)
        signals = self._download_signals.get(key)
        if button is None or status is None or signals is None:
            return

        button.setEnabled(False)
        status.setToolTip("")
        status.setText("正在下载…")
        if progress is not None:
            # 总量未知时用忙碌状态，收到 total 后切换为百分比。
            progress.setRange(0, 0)
            progress.setVisible(True)
        try:
            task = start_model_download(key, signals)
        except Exception as exc:  # noqa: BLE001 - 启动失败时恢复按钮，避免卡在禁用态
            button.setEnabled(True)
            if progress is not None:
                progress.setVisible(False)
            self._on_model_failed(key, f"无法启动下载：{exc}")
            return
        # 保持任务引用，避免 QRunnable 在后台线程运行期间被回收。
        self._model_tasks[key] = task

    def _on_model_progress(self, key: str, downloaded: int, total: int) -> None:
        status = self.model_status_labels.get(key)
        if status is None:
            return
        downloaded = max(0, int(downloaded))
        total = max(0, int(total))
        if total > 0:
            percent = min(100, downloaded * 100 // total)
            status.setText(
                f"正在下载 {percent}% · "
                f"{models.format_size(downloaded)}/{models.format_size(total)}"
            )
            progress = self.model_progress_bars.get(key)
            if progress is not None:
                progress.setRange(0, 100)
                progress.setValue(percent)
        else:
            status.setText(f"正在下载 · 已下载 {models.format_size(downloaded)}")

    def _on_model_succeeded(self, key: str, path: str) -> None:
        status = self.model_status_labels.get(key)
        if status is None:
            return
        # model_status_text 会重新读取文件大小，显示「已就绪 · 大小」。
        status.setText(models.model_status_text(key))
        status.setToolTip(f"已保存到：{path}")
        progress = self.model_progress_bars.get(key)
        if progress is not None:
            progress.setRange(0, 100)
            progress.setValue(100)

    def _on_model_failed(self, key: str, message: str) -> None:
        status = self.model_status_labels.get(key)
        if status is None:
            return
        # 状态行只放简短中文提示，完整错误放 tooltip，避免长文本撑破布局 / 弹窗。
        detail = str(message or "").strip()
        brief = detail.splitlines()[0] if detail else "下载失败"
        if not brief.startswith(("下载失败", "模型下载失败", "无法启动下载")):
            brief = f"下载失败：{brief}"
        if len(brief) > 96:
            brief = f"{brief[:95]}…"
        status.setText(brief)
        status.setToolTip(detail or brief)
        progress = self.model_progress_bars.get(key)
        if progress is not None:
            progress.setRange(0, 100)
            progress.setValue(0)
            progress.setVisible(False)

    def _on_model_finished(self, key: str) -> None:
        # 成功 / 失败 / 取消都会走到这里：恢复按钮并丢弃任务引用。
        self._model_tasks.pop(key, None)
        button = self.model_download_buttons.get(key)
        if button is not None:
            button.setEnabled(True)
        status = self.model_status_labels.get(key)
        if status is not None and status.text().startswith("正在下载"):
            # 没有收到 succeeded / failed 的异常结束（例如取消）时回到真实状态。
            status.setText(models.model_status_text(key))
        progress = self.model_progress_bars.get(key)
        if progress is not None:
            progress.setRange(0, 100)
            progress.setValue(0)
            progress.setVisible(False)

    def _on_open_model_dir(self) -> None:
        folder = models.model_dir()
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    def _on_open_zoo(self) -> None:
        QDesktopServices.openUrl(QUrl(models.OPENCV_ZOO_URL))

    def _persist(self) -> None:
        self.settings.undo_limit = max(1, int(self.undo_spin.value()))
        self.settings.show_node_thumbnails = self.thumbnails_check.isChecked()
        self.settings.auto_fit_on_open = self.auto_fit_check.isChecked()
        self.settings.downsample_preview = self.downsample_check.isChecked()
        self.settings.downsample_max_dim = max(256, int(self.max_dim_spin.value()))
        self.settings.cursor_pixel_info = self.cursor_info_check.isChecked()
        self.settings.extra_operator_dirs = self.extra_dirs()
        self.settings.builtin_operators_enabled = self.builtin_check.isChecked()
        self.settings.watch_operator_dirs = self.watch_check.isChecked()
        self.settings.sync()

    # ------------------------------------------------------------------ 确认/取消
    def accept(self) -> None:
        self._persist()
        self.applied.emit()
        super().accept()

    def reject(self) -> None:
        super().reject()
