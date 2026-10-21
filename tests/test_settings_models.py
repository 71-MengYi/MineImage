"""设置对话框「模型」页测试。

不访问网络、不点击真实下载按钮：只验证界面元素、状态刷新与就地显示的进度 /
错误文本。临时目录放在工作区 tests/.tmp 下（系统 Temp 可能被沙箱拒绝）。
"""

from __future__ import annotations

import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt  # noqa: E402
from PySide6.QtWidgets import (  # noqa: E402
    QApplication,
    QGroupBox,
    QLabel,
    QLineEdit,
    QPushButton,
    QTabWidget,
    QWidget,
)

from app.config import AppSettings  # noqa: E402
from app.core import models  # noqa: E402
from app.ui.settings_dialog import SettingsDialog  # noqa: E402
from app.ui.theme import apply_theme  # noqa: E402

_APP: QApplication | None = None
_TMP_BASE = ROOT / "tests" / ".tmp"


def setUpModule() -> None:
    global _APP
    _APP = QApplication.instance() or QApplication([])
    apply_theme(_APP)


def tearDownModule() -> None:
    global _APP
    _APP = None


class SettingsModelsTabTests(unittest.TestCase):
    def setUp(self) -> None:
        _TMP_BASE.mkdir(parents=True, exist_ok=True)
        self.tmp = _TMP_BASE / f"settings-models-{uuid.uuid4().hex[:10]}"
        self.tmp.mkdir()
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

        self.ini_path = self.tmp / "config" / "MineImage.ini"
        self.models_dir = self.tmp / "models"
        env_patch = mock.patch.dict(
            os.environ, {"MINEIMAGE_MODEL_DIR": str(self.models_dir)}
        )
        env_patch.start()
        self.addCleanup(env_patch.stop)

    # ------------------------------------------------------------ 工具
    def _make_dialog(self) -> SettingsDialog:
        dialog = SettingsDialog(AppSettings(self.ini_path))
        self.addCleanup(self._dispose_dialog, dialog)
        return dialog

    @staticmethod
    def _dispose_dialog(dialog: SettingsDialog) -> None:
        dialog.close()
        dialog.deleteLater()
        if _APP is not None:
            _APP.processEvents()

    def _model_page(self, dialog: SettingsDialog) -> QWidget:
        tabs = dialog.findChild(QTabWidget)
        self.assertIsNotNone(tabs)
        assert tabs is not None
        for index in range(tabs.count()):
            if tabs.tabText(index) == "模型":
                page = tabs.widget(index)
                self.assertIsNotNone(page)
                assert page is not None
                return page
        self.fail("设置对话框缺少标题为「模型」的标签页")

    @staticmethod
    def _texts(widget: QWidget) -> list[str]:
        return [label.text() for label in widget.findChildren(QLabel)]

    # ------------------------------------------------------------ 界面结构
    def test_models_tab_lists_builtin_models_and_links(self) -> None:
        dialog = self._make_dialog()
        page = self._model_page(dialog)

        titles = [dialog.tabs.tabText(index) for index in range(dialog.tabs.count())]
        self.assertEqual(titles, ["常规", "预览", "模型", "算子目录"])
        self.assertGreaterEqual(dialog.minimumWidth(), 680)
        self.assertGreaterEqual(dialog.minimumHeight(), 560)

        texts = self._texts(page)
        joined = "\n".join(texts)
        # 顶部说明与 OpenCV Zoo 来源。
        self.assertIn("OpenCV Zoo", joined)

        # 每个模型使用一个分组框，标题即模型名称。
        group_titles = [group.title() for group in page.findChildren(QGroupBox)]
        for spec in models.MODELS.values():
            self.assertIn(spec.name, group_titles)
            self.assertIn(spec.filename, joined)
            self.assertIn(spec.url, joined)
            # 状态初始为未下载（模型目录里还没有文件）。
            self.assertEqual(dialog.model_status_labels[spec.key].text(), "未下载")

            url_labels = [label for label in page.findChildren(QLabel) if spec.url in label.text()]
            self.assertTrue(url_labels, f"缺少 {spec.key} 的下载地址")
            for label in url_labels:
                self.assertIn('<a href="', label.text())
                self.assertTrue(label.openExternalLinks())
                flags = label.textInteractionFlags()
                self.assertTrue(
                    bool(flags & Qt.TextInteractionFlag.TextSelectableByMouse)
                )

        # 每个模型一个「下载」按钮，另有打开目录 / OpenCV Zoo 按钮。
        buttons = page.findChildren(QPushButton)
        button_texts = [button.text() for button in buttons]
        self.assertEqual(button_texts.count("下载"), len(models.MODELS))
        self.assertIn("打开模型目录", button_texts)
        self.assertIn("打开 OpenCV Zoo", button_texts)
        for spec in models.MODELS.values():
            self.assertTrue(dialog.model_download_buttons[spec.key].isEnabled())

        # 模型目录以只读文本形式完整展示，且与实际缓存目录一致。
        self.assertTrue(dialog.model_dir_edit.isReadOnly())
        self.assertEqual(dialog.model_dir_edit.text(), str(self.models_dir))
        self.assertIn(str(self.models_dir), [edit.text() for edit in page.findChildren(QLineEdit)])

    # ------------------------------------------------------------ 状态刷新
    def test_ready_model_status_after_restart_and_refresh(self) -> None:
        dialog = self._make_dialog()
        key = "face_detection_yunet"
        self.assertEqual(dialog.model_status_labels[key].text(), "未下载")

        # 写入一个大于 min_size 的假模型文件（不访问网络）。
        self.models_dir.mkdir(parents=True, exist_ok=True)
        fake_model = self.models_dir / "face_detection_yunet_2023mar.onnx"
        fake_model.write_bytes(b"x" * 120_000)

        # 重新构造对话框后应显示已就绪。
        dialog2 = self._make_dialog()
        status2 = dialog2.model_status_labels[key].text()
        self.assertIn("已就绪", status2)
        self.assertIn(models.format_size(120_000), status2)
        self.assertEqual(
            dialog2.model_status_labels["face_recognition_sface"].text(), "未下载"
        )
        self.assertTrue(dialog2.model_download_buttons[key].isEnabled())

        # 已打开的对话框可手动刷新状态（下载成功回调也走这里）。
        dialog.refresh_model_status()
        self.assertIn("已就绪", dialog.model_status_labels[key].text())

    # ------------------------------------------------------------ 就地反馈
    def test_progress_and_failure_are_shown_inline(self) -> None:
        dialog = self._make_dialog()
        key = "face_detection_yunet"
        status = dialog.model_status_labels[key]

        dialog._on_model_progress(key, 1_048_576, 2_097_152)
        self.assertIn("正在下载 50%", status.text())
        self.assertIn("1.0 MB/2.0 MB", status.text())
        self.assertEqual(dialog.model_progress_bars[key].value(), 50)

        # 失败：状态行只显示简短中文错误，完整信息进 tooltip，不弹窗。
        long_error = (
            "模型下载失败：YuNet 人脸检测模型。下载失败：HTTP 404 Not Found\n"
            f"手动下载地址：{models.MODELS[key].url}\n"
            f"下载后放到：{self.models_dir}"
        )
        dialog._on_model_failed(key, long_error)
        self.assertIn("下载失败", status.text())
        self.assertNotIn("\n", status.text())
        self.assertEqual(status.toolTip(), long_error)

        # finished 恢复按钮，便于再次尝试。
        dialog._on_model_finished(key)
        self.assertTrue(dialog.model_download_buttons[key].isEnabled())
        self.assertNotIn(key, dialog._model_tasks)


if __name__ == "__main__":
    unittest.main(verbosity=2)
