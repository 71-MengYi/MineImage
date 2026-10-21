"""生成 docs/opencv_catalog.md：OpenCV 5.0 算子目录与覆盖清单。

用法：
    python scripts/gen_opencv_catalog_doc.py

脚本读取当前注册表，因此新增 / 删除目录算子后重新运行即可同步文档。
"""

from __future__ import annotations

import sys
from collections import Counter, OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.constants import CATEGORY_ORDER  # noqa: E402
from app.operators.loader import OperatorRegistry  # noqa: E402

OUT = ROOT / "docs" / "opencv_catalog.md"

MODULES = [
    (
        "core",
        "算术 / 位运算 / 归一化 / LUT / DFT / 积分图 / 图像矩 / 统计量",
        "basic.*、analysis.*（fft、integral、moments）、misc.*",
        "✅ 主要",
    ),
    (
        "imgproc",
        "滤波、形态学、几何变换、阈值、分割、边缘、直方图、轮廓、模板匹配、绘图",
        "filter.*、morph.*、geom.*、threshold.*、segment.*、edge.*、histogram.*、draw.*",
        "✅ 主要",
    ),
    (
        "imgcodecs",
        "图像编解码",
        "由 app/core/imageio.py 与工程文件负责，不作为节点",
        "➖ 非节点",
    ),
    (
        "highgui / videoio",
        "窗口、摄像头、视频读写",
        "GUI 已由 PySide6 承担；暂未提供摄像头节点",
        "➖ 非节点",
    ),
    (
        "video",
        "光流、背景建模、均值漂移 / CamShift、ECC 对齐、目标跟踪",
        "video.*（10 个）",
        "✅ 主要",
    ),
    (
        "calib3d",
        "棋盘格、畸变校正、立体匹配、单应、全景拼接",
        "calib.*（6 个）",
        "✅ 主要",
    ),
    (
        "features2d",
        "SIFT / ORB / AKAZE / BRISK / KAZE / MSER / FAST / SimpleBlob、角点、匹配",
        "keypoint.*（12 个）、feature.match、analysis.match_shapes",
        "✅ 主要",
    ),
    (
        "objdetect",
        "Haar 人脸、HOG 行人、二维码、条码",
        "detect.*（4 个）",
        "✅ 主要",
    ),
    (
        "dnn",
        "Blob 预处理、通用模型推理",
        "dnn.blob、dnn.infer（需要模型文件）",
        "🟡 部分",
    ),
    (
        "ml",
        "传统机器学习",
        "segment.kmeans（K 均值分割）",
        "🟡 部分",
    ),
    (
        "flann",
        "近似最近邻匹配",
        "feature.match 的 FLANN 匹配器",
        "✅ 主要",
    ),
    (
        "photo",
        "去噪、修复、无缝克隆、风格化、HDR、色调映射",
        "enhance.*（17 个）",
        "✅ 主要",
    ),
    (
        "stitching",
        "全景拼接",
        "calib.panorama",
        "✅ 单节点",
    ),
    (
        "gapi",
        "Graph API",
        "暂未封装；可在后续用图执行器接入",
        "🧭 后续",
    ),
    (
        "3d / calib / stereo（5.0 新增模块）",
        "多相机标定、点云 / 网格、稠密 RGB-D、立体深度",
        "calib.stereo_disparity、calib.stereo_depth（基于 4.x/5.0 通用 StereoMatcher）",
        "🧭 后续（本机 4.12 无 3d/stereo 模块）",
    ),
]


def build_inventory(registry: OperatorRegistry):
    entries = [
        entry
        for entry in registry.all_entries()
        if entry.path.parent.name == "catalog" and entry.path.name != "__init__.py"
    ]
    grouped: "OrderedDict[str, list]" = OrderedDict()
    for category in CATEGORY_ORDER:
        matched = [entry for entry in entries if entry.spec.category[0] == category]
        if matched:
            grouped[category] = sorted(matched, key=lambda item: (item.spec.category[1:], item.id))
    return entries, grouped


def short(text: str, limit: int = 44) -> str:
    cleaned = " ".join(str(text).split())
    return cleaned if len(cleaned) <= limit else cleaned[: limit - 1] + "…"


def main() -> None:
    registry = OperatorRegistry.build()
    catalog, grouped = build_inventory(registry)
    total = len(registry)
    counts = Counter(entry.spec.category[0] for entry in catalog)

    lines: list[str] = []
    lines.append("# OpenCV 5.0 算子目录")
    lines.append("")
    lines.append(
        "MineImage 的算子库按 OpenCV 模块组织，目标是让常见图像处理流程可以完全"
        "用节点搭出来：从读图、滤波、分割、特征、增强到检测、标定与 DNN 推理。"
        "本页由 `scripts/gen_opencv_catalog_doc.py` 从注册表自动生成。"
    )
    lines.append("")
    lines.append("![OpenCV 算子目录](images/opencv_catalog_library.png)")
    lines.append("")
    lines.append("## 1. 规模")
    lines.append("")
    lines.append(f"* 内置算子总数：**{total}** 个（目录算子 {len(catalog)} 个 + 既有算子 {total - len(catalog)} 个）")
    lines.append(f"* 一级分类：**{len(grouped)}** 个，顺序由 `app/core/constants.py` 的 `CATEGORY_ORDER` 决定")
    lines.append("* 目录算子统一位于 `app/operators/builtin/catalog/`，一个模块文件可导出多个算子")
    lines.append("* 每个目录算子在 L/BGR/BGRA × uint8/uint16/float32 共 7 种代表性输入上做默认参数冒烟")
    lines.append("")
    lines.append("## 2. OpenCV 5.0 模块覆盖")
    lines.append("")
    lines.append("| OpenCV 模块 | 覆盖的能力 | 对应节点 | 状态 |")
    lines.append("| --- | --- | --- | --- |")
    for name, capability, nodes, status in MODULES:
        lines.append(f"| `{name}` | {capability} | {nodes} | {status} |")
    lines.append("")
    lines.append(
        "> 说明：OpenCV 5.0 的公共 API 数量以千计，其中大量是底层数据结构、"
        "I/O、GPU / HAL、机器学习训练接口，并不适合作为图像节点。本项目采用"
        "「语义节点 + 通用调用兜底」的策略：常见图像处理语义逐一封装为节点；"
        "其余函数可通过 `misc.cv2_call` 按函数名直接调用（JSON 参数，不执行任意代码），"
        "`misc.cv2_info` 用于确认当前 OpenCV 版本与模块可用性。"
    )
    lines.append("")
    lines.append("### 2.1 OpenCV 5.0 迁移说明")
    lines.append("")
    lines.append("* 本机开发环境实测为 opencv-python 4.12.0；目录算子只使用 4.x 与 5.0 都保留的稳定 API，")
    lines.append("  切换到 5.0 后无需改代码。")
    lines.append("* OpenCV 5.0 新增 `3d` / `calib` / `stereo` 模块，并重构了 DNN 引擎；")
    lines.append("  本目录用 `calib.stereo_*` 覆盖立体匹配，`dnn.infer` 覆盖 Net 推理，")
    lines.append("  5.0 专有 API 可在 `misc.cv2_call` 中直接调用。")
    lines.append("* OpenCV 5.0 支持关键字参数调用；`misc.cv2_call` 同时支持位置参数（JSON 数组）")
    lines.append("  与关键字参数（JSON 对象），方便跟随 5.0 的调用习惯。")
    lines.append("")
    lines.append("## 3. 分类清单")
    lines.append("")
    lines.append("下表只统计目录算子；加上既有 27 个算子后的总数见第 1 节。")
    lines.append("")
    lines.append("| 一级分类 | 算子数 |")
    lines.append("| --- | --- |")
    for category in grouped:
        lines.append(f"| {category} | {len(grouped[category])} |")
    lines.append("")
    lines.append("## 4. 算子明细")
    for category, entries in grouped.items():
        lines.append("")
        lines.append(f"### {category}（{len(entries)}）")
        lines.append("")
        lines.append("| id | 名称 | 子分类 | 说明 |")
        lines.append("| --- | --- | --- | --- |")
        for entry in entries:
            subgroup = " / ".join(entry.spec.category[1:]) or "—"
            lines.append(
                f"| `{entry.id}` | {entry.spec.name} | {subgroup} | {short(entry.spec.description)} |"
            )
    lines.append("")
    lines.append("## 5. 扩展方式")
    lines.append("")
    lines.append("目录模块使用批量导出约定，一个文件可以放同一 OpenCV 模块的多个算子：")
    lines.append("")
    lines.append("```python")
    lines.append("from app.operators.spec import Operator, OperatorSpec")
    lines.append("")
    lines.append("class MyOp(Operator):")
    lines.append('    spec = OperatorSpec(id="filter.my_op", name="我的算子", category=("滤波与平滑",))')
    lines.append("    def process(self, image=None, **_):")
    lines.append("        ...")
    lines.append("")
    lines.append("OPERATORS = (MyOp, AnotherOp)")
    lines.append("```")
    lines.append("")
    lines.append("* `OPERATOR`（单算子）与 `OPERATORS`（多算子）都支持，可同时存在；")
    lines.append("* 共享工具在 `app/operators/builtin/catalog/_cv.py`；")
    lines.append("* 新增目录算子后把 id 补进 `tests/test_catalog_all.py` 的 `REQUIRED_IDS`，并运行：")
    lines.append("")
    lines.append("```powershell")
    lines.append('$env:QT_QPA_PLATFORM = "offscreen"')
    lines.append('python -m unittest discover -s tests -p "test_catalog_*.py" -v')
    lines.append("```")
    lines.append("")
    lines.append("## 6. 验收状态")
    lines.append("")
    lines.append("* `OperatorRegistry.build()`：目录模块无加载错误、无 id 冲突；")
    lines.append("* `tests/test_catalog_all.py`：元数据、默认参数冒烟、输入不可变、匹配行为；")
    lines.append("* 各分组测试：`tests/test_catalog_filter_geom.py`、`test_catalog_segment_feature.py`、")
    lines.append("  `test_catalog_arithmetic_color.py`、`test_catalog_enhance_draw.py`、")
    lines.append("  `test_catalog_video_analysis.py`。")
    lines.append("")

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print(f"wrote {OUT} ({len(catalog)} catalog operators, {total} total)")


if __name__ == "__main__":
    main()
