"""生成 examples/demo.mi 演示工程（自包含：源图像内嵌，无外部依赖）。

用法（在仓库根目录）::

    python examples/make_demo_project.py

生成流程会真实启动一次 offscreen 的 ``MainWindow``，用 ``GraphView.auto_layout()``
按当前样式计算节点位置后保存，因此打开工程时的布局与界面一致。工程内容::

    原始图像 ──> 灰度化 ──> 高斯模糊

这是最小、最稳的三节点链路：在默认 1500×920 窗口 + 1:1:4 三栏下，打开工程后
三个节点都完整可见。多输入/多输出端口的演示请见
``examples/custom_operators/highlight.py``（双输入/双输出）。

保存后会重新打开工程，校验节点/连线/内嵌图像数量，并在
``.test_tmp/demo_open.png`` 留一张 1500×920 的验证截图（该目录已被 gitignore）。
"""

from __future__ import annotations

import os
import sys
import time
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
from app.core.project import load_project  # noqa: E402
from app.ui.main_window import MainWindow  # noqa: E402
from app.ui.theme import apply_theme  # noqa: E402

SAMPLE = ROOT / "sample_images" / "sample_color.png"
TARGET = ROOT / "examples" / "demo.mi"
VERIFY_SHOT = ROOT / ".test_tmp" / "demo_open.png"
INI = ROOT / ".test_tmp" / "demo_build.ini"


def _load_font(app: QApplication) -> None:
    for path in ("C:/Windows/Fonts/msyh.ttc", "C:/Windows/Fonts/msyhbd.ttc"):
        if Path(path).exists():
            QFontDatabase.addApplicationFont(path)
    app.setFont(QFont("Microsoft YaHei", 9))


def _wait_idle(app: QApplication, widget, timeout: float = 20.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if widget.engine.busy:
            QTest.qWait(20)
            continue
        app.processEvents()
        QTest.qWait(80)
        if not widget.engine.busy:
            return
    raise TimeoutError("后台计算在限定时间内没有结束")


def _visible_nodes(window: MainWindow, widget) -> dict[str, bool]:
    """检查每个节点在图元包围盒层面是否落在 viewport 内。"""

    view = widget.graph_view
    viewport = view.viewport().rect()
    result: dict[str, bool] = {}
    for node_id, item in view._nodes.items():  # noqa: SLF001 - QA 脚本
        rect = view.mapFromScene(item.sceneBoundingRect()).boundingRect()
        result[node_id] = viewport.contains(rect)
    return result


def main() -> int:
    if not SAMPLE.exists():
        raise SystemExit(
            f"缺少示例图像：{SAMPLE}\n请先运行 python scripts/make_samples.py"
        )
    app = QApplication.instance() or QApplication([])
    _load_font(app)
    apply_theme(app)
    INI.parent.mkdir(parents=True, exist_ok=True)
    if INI.exists():
        INI.unlink()

    window = MainWindow(AppSettings(INI))
    window.resize(1500, 920)
    window.show()
    app.processEvents()

    window.open_image_path(str(SAMPLE))
    widget = window._active_workspace()
    assert widget is not None, "打开示例图像失败"
    _wait_idle(app, widget)
    graph = widget.workspace.graph
    source = graph.source_nodes()[0]

    def add(op_id: str, x: float, y: float) -> str:
        node_id = widget.graph_view.add_operator_at(op_id, QPointF(x, y))
        assert node_id is not None, f"添加算子失败：{op_id}"
        return node_id

    grayscale = add("basic.grayscale", 360, 20)
    blur = add("filter.gaussian_blur", 720, 20)
    graph.add_connection(source.id, "image", grayscale, "image")
    graph.add_connection(grayscale, "image", blur, "image")

    # 用真实画布按当前节点尺寸自动布局，再适应视图（视图变换不写入工程）。
    widget.workspace.name = "演示流程"
    widget.graph_view.auto_layout()
    widget.graph_view.fit_graph()
    app.processEvents()
    QTest.qWait(200)

    widget.save_to(TARGET)
    sizes = widget.splitter.sizes()
    print(f"已生成：{TARGET.relative_to(ROOT)}  ({TARGET.stat().st_size / 1024:.1f} KiB)")
    print(f"三栏宽度：{sizes}（1:1:4）")

    # 回读校验 + 打开后的可视性检查
    loaded = load_project(TARGET)
    assert len(loaded.images) == 1, f"内嵌图像数量异常：{len(loaded.images)}"
    assert len(loaded.graph.nodes) == 3, f"节点数量异常：{len(loaded.graph.nodes)}"
    assert len(loaded.graph.connections) == 2, f"连线数量异常：{len(loaded.graph.connections)}"

    window.open_project_path(str(TARGET))
    opened = window._active_workspace()
    assert opened is not None
    _wait_idle(app, opened)
    app.processEvents()
    QTest.qWait(300)
    visibility = _visible_nodes(window, opened)
    shot = VERIFY_SHOT
    shot.parent.mkdir(parents=True, exist_ok=True)
    window.grab().save(str(shot), "PNG")
    print(f"打开工程节点可见性：{visibility}")
    print(f"验证截图：{shot.relative_to(ROOT)}")
    assert all(visibility.values()), f"有节点未完整显示：{visibility}"
    assert opened.preview.image_size() is not None, "打开工程后预览为空"
    print(
        f"回读校验通过：{len(loaded.graph.nodes)} 个节点 / "
        f"{len(loaded.graph.connections)} 条连线 / {len(loaded.images)} 张内嵌图像"
    )
    print("节点位置：", {node.op_id: node.pos for node in loaded.graph.nodes.values()})

    for workspace in list(window._workspaces):
        workspace.engine.shutdown()
        workspace.mark_clean()
    window.hide()
    app.processEvents()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
