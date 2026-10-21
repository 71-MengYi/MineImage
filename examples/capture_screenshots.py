"""离屏生成 docs/images 下的界面截图（真实中文字体）。

用法（在仓库根目录）::

    $env:QT_QPA_PLATFORM = "offscreen"
    python examples/capture_screenshots.py

脚本只读取应用、不修改 app/ 代码，也不会调用 ``app.exec()``；全部用
``QApplication.processEvents`` / ``QTest.qWait`` 推进。结束时把工作区标记为
干净再析构，避免触发“是否保存”的模态对话框。
"""

from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import QPointF  # noqa: E402
from PySide6.QtGui import QFont, QFontDatabase  # noqa: E402
from PySide6.QtTest import QTest  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402

from app.config import AppSettings  # noqa: E402
from app.core.logstore import LogStore  # noqa: E402
from app.ui.log_dialog import LogDialog  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.theme import apply_theme  # noqa: E402

OUT_DIR = ROOT / "docs" / "images"
INI_PATH = ROOT / ".test_tmp" / "capture_screenshots.ini"
LOG_DB_PATH = ROOT / ".test_tmp" / "capture_logs.db"
SAMPLE_IMAGE = ROOT / "sample_images" / "sample_color.png"

WINDOW_SIZE = (1500, 920)


def load_chinese_font(app: QApplication) -> None:
    """加载 Windows 自带微软雅黑，保证离屏渲染中文不是方框。"""
    candidates = [
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/msyhbd.ttc",
        "C:/Windows/Fonts/msyhl.ttc",
        "C:/Windows/Fonts/simhei.ttf",
    ]
    loaded = []
    for path in candidates:
        if Path(path).exists():
            font_id = QFontDatabase.addApplicationFont(path)
            if font_id >= 0:
                families = QFontDatabase.applicationFontFamilies(font_id)
                loaded.extend(families)
    if loaded:
        print(f"已加载字体：{', '.join(sorted(set(loaded)))}")
    else:
        print("警告：未找到中文字体，截图中的中文可能显示为方框。")
    font = QFont("Microsoft YaHei", 9)
    font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(font)


def ensure_sample_image() -> Path:
    if SAMPLE_IMAGE.exists() and SAMPLE_IMAGE.stat().st_size > 0:
        return SAMPLE_IMAGE
    from scripts.make_samples import make_color_sample
    import cv2

    SAMPLE_IMAGE.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SAMPLE_IMAGE), make_color_sample())
    return SAMPLE_IMAGE


def wait_idle(app: QApplication, widget, timeout: float = 20.0) -> None:
    """等待后台引擎空闲；不使用 app.exec()。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if widget.engine.busy:
            QTest.qWait(20)
            continue
        app.processEvents()
        QTest.qWait(80)
        app.processEvents()
        if not widget.engine.busy:
            return
    raise TimeoutError("后台计算在限定时间内没有结束")


def capture(app: QApplication, window: MainWindow, name: str) -> Path:
    for _ in range(4):
        app.processEvents()
        QTest.qWait(120)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUT_DIR / name
    pixmap = window.grab()
    if not pixmap.save(str(target), "PNG"):
        raise RuntimeError(f"截图保存失败：{target}")
    size_kib = target.stat().st_size / 1024
    print(f"  {target.relative_to(ROOT)}  {pixmap.width()}×{pixmap.height()}  {size_kib:.1f} KiB")
    return target


def build_chain(window: MainWindow, widget) -> tuple[str, str, str]:
    """原始图像 -> 灰度化 -> 高斯模糊，并自动布局。"""
    graph = widget.workspace.graph
    source = graph.source_nodes()[0]
    grayscale = widget.graph_view.add_operator_at("basic.grayscale", QPointF(360, 40))
    blur = widget.graph_view.add_operator_at("filter.gaussian_blur", QPointF(720, 40))
    assert grayscale and blur
    graph.add_connection(source.id, "image", grayscale, "image")
    graph.add_connection(grayscale, "image", blur, "image")
    widget.graph_view.auto_layout()
    widget.graph_view.fit_graph()
    return source.id, grayscale, blur


def capture_log_dialog(app: QApplication) -> Path:
    """生成应用日志查看器截图（示例日志写入临时 SQLite 数据库）。"""

    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(LOG_DB_PATH) + suffix)
        candidate.unlink(missing_ok=True)
    store = LogStore(LOG_DB_PATH)
    now = datetime.now()
    samples = [
        ("INFO", "MineImage 0.1.0 启动，Python 3.14.3，设置 D:\\Code\\MineImage\\config\\MineImage.ini",
         "mineimage", "", "main.py:114", "MainThread"),
        ("INFO", "算子库：128 个算子，0 个加载问题", "mineimage", "", "main.py:122", "MainThread"),
        ("WARNING", "算子加载问题 [warning] D:\\Code\\MineImage\\operators\\demo_bad.py：模块缺少 OPERATOR 变量",
         "app.ui.main_window",
         "Traceback (most recent call last):\n"
         '  File "operators/demo_bad.py", line 4, in <module>\n'
         "    OPERATOR_NAME = 1\n"
         "AttributeError: 'int' object has no attribute 'id'",
         "main_window.py:98", "MainThread"),
        ("INFO", "已打开图像：sample.png", "app.ui.workspace", "", "workspace.py:254", "MainThread"),
        ("ERROR", "节点执行失败 [n3] 高斯模糊 (filter.gaussian_blur)：算子执行失败：ValueError: ksize must be odd",
         "app.engine.engine",
         "Traceback (most recent call last):\n"
         '  File "app/engine/engine.py", line 550, in _execute\n'
         "    output = entry.operator.process(**call_kwargs)\n"
         '  File "app/operators/builtin/gaussian_blur.py", line 41, in process\n'
         "    return cv2.GaussianBlur(image, (k, k), sigma)\n"
         "ValueError: ksize must be odd",
         "engine.py:563", "Dummy-3"),
        ("ERROR", "无法打开工程 D:\\images\\broken.mi：工程文件缺少 project.json",
         "app.ui.main_window", "ProjectFormatError: 工程文件缺少 project.json",
         "main_window.py:529", "MainThread"),
        ("WARNING", "参数校验失败：阈值必须在 0–255 之间（当前 300）", "app.ui.params", "", "params.py:512", "MainThread"),
        ("ERROR", "导出图像失败 D:\\out\\result.png：[Errno 13] Permission denied", "app.ui.workspace",
         "PermissionError: [Errno 13] Permission denied: 'D:\\out\\result.png'",
         "workspace.py:691", "MainThread"),
    ]
    detail_error_id = ""
    for index, (level, message, logger, detail, source, thread) in enumerate(reversed(samples)):
        entry = store.append(
            level,
            message,
            logger=logger,
            detail=detail,
            source=source,
            thread=thread,
            created_at=now - timedelta(minutes=len(samples) - index),
        )
        if "ksize must be odd" in message:
            detail_error_id = entry.entry_id
    store.append(
        "CRITICAL",
        "未捕获异常：ZeroDivisionError: division by zero",
        logger="mineimage",
        detail=(
            "Traceback (most recent call last):\n"
            '  File "app/ui/main_window.py", line 500, in open_project_path\n'
            "    workspace = load_project(path)\n"
            "ZeroDivisionError: division by zero"
        ),
        source="main_window.py:500",
        thread="MainThread",
        created_at=now,
    )

    dialog = LogDialog(store)
    dialog.resize(1180, 760)
    dialog.show()
    app.processEvents()
    row = dialog.model.row_for_entry_id(detail_error_id)
    if row >= 0:
        dialog.table_view.selectRow(row)
        dialog.table_view.setCurrentIndex(dialog.model.index(row, 0))
    target = capture(app, dialog, "app_logs.png")
    dialog.close()
    dialog.deleteLater()
    app.processEvents()
    for suffix in ("", "-wal", "-shm"):
        Path(str(LOG_DB_PATH) + suffix).unlink(missing_ok=True)
    return target


def main() -> int:
    ensure_sample_image()
    app = QApplication.instance() or QApplication([])
    load_chinese_font(app)
    apply_theme(app)

    INI_PATH.parent.mkdir(parents=True, exist_ok=True)
    if INI_PATH.exists():
        INI_PATH.unlink()
    settings = AppSettings(INI_PATH)
    settings.show_node_thumbnails = True
    settings.sync()

    window = MainWindow(settings)
    window.resize(*WINDOW_SIZE)
    window.show()
    app.processEvents()

    window.open_image_path(str(SAMPLE_IMAGE))
    widget = window._active_workspace()
    assert widget is not None, "打开示例图像失败"
    wait_idle(app, widget)

    print("生成截图：")
    source_id, grayscale_id, blur_id = build_chain(window, widget)

    # 1) 主工作区：选中高斯模糊节点，右侧显示该节点结果
    widget._on_node_selected(blur_id)
    wait_idle(app, widget)
    capture(app, window, "main_workspace.png")

    # 2) 参数面板：双击节点后进入的参数编辑页（用高斯模糊展示多参数）
    widget.open_parameters(blur_id)
    wait_idle(app, widget)
    capture(app, window, "parameter_panel.png")

    # 3) 左右对比
    widget.back_to_graph()
    widget._on_node_selected(blur_id)
    wait_idle(app, widget)
    widget.set_compare_mode("side_by_side")
    app.processEvents()
    QTest.qWait(200)
    capture(app, window, "compare_side_by_side.png")

    # 4) 滑动对比
    widget.set_compare_mode("swipe")
    app.processEvents()
    QTest.qWait(200)
    capture(app, window, "compare_swipe.png")

    # 5) 算子库说明浮层
    widget.set_compare_mode("single")
    shown = widget.library.show_operator_info("basic.grayscale")
    app.processEvents()
    if not shown:
        raise RuntimeError("未能显示 basic.grayscale 的说明浮层")
    capture(app, window, "operator_info.png")
    widget.library._hide_info_panel()

    # 6) 应用日志查看器（帮助 → 日志）
    capture_log_dialog(app)

    # 干净退出：避免关闭时的模态保存提示
    for ws in list(window._workspaces):
        ws.engine.shutdown()
        ws.mark_clean()
    window.hide()
    app.processEvents()
    print("完成。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
