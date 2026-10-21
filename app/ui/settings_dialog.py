"""设置对话框：常规 / 预览 / 算子目录。"""

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
    QPushButton,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.config import AppSettings


class SettingsDialog(QDialog):
    """应用设置对话框；点击「确定」时写入并持久化。"""

    applied = Signal()

    def __init__(self, settings: AppSettings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.settings = settings
        self.setWindowTitle("设置")
        self.setModal(True)
        self.setMinimumSize(600, 460)
        self.resize(640, 520)
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
