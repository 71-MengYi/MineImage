"""OpenCV 目录算子：图像分割与轮廓几何。

包含分水岭、GrabCut、漫水填充、距离变换、连通域、K 均值分割，以及
轮廓 / 凸包 / 多边形逼近 / 最小外接矩形 / 椭圆拟合等轮廓几何算子。
所有算子都通过 :mod:`app.operators.builtin.catalog._cv` 读取输入，返回新数组。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.core.data_types import (
    ComponentsData,
    ContoursData,
    DataType,
    NumberValue,
)
from app.operators.builtin.catalog import _cv
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec

_INPUT = PortSpec(
    "image",
    "输入图像",
    accepts=("L", "BGR", "BGRA"),
    description="灰度、BGR 或 BGRA 图像；彩色图像会先转为灰度。",
)

_RETRIEVALS = {
    "external": cv2.RETR_EXTERNAL,
    "list": cv2.RETR_LIST,
    "tree": cv2.RETR_TREE,
}


def _binary_flag(mode: str) -> int:
    return cv2.THRESH_BINARY_INV if mode == "binary_inv" else cv2.THRESH_BINARY


def _find_contours(
    image,
    threshold: int,
    binary_mode: str,
    retrieval: str,
    method: str,
    min_area: float,
):
    """统一提取轮廓：返回 ``(灰度图, 二值图, 过滤后的轮廓列表)``。"""
    gray = _cv.source_gray8(image, "输入图像")
    _value, binary = cv2.threshold(
        gray, int(threshold), 255, _binary_flag(binary_mode)
    )
    contour_mode = _RETRIEVALS.get(retrieval, cv2.RETR_EXTERNAL)
    approx_mode = cv2.CHAIN_APPROX_SIMPLE if method == "simple" else cv2.CHAIN_APPROX_NONE
    contours, _hierarchy = cv2.findContours(binary, contour_mode, approx_mode)
    kept = [contour for contour in contours if cv2.contourArea(contour) >= float(min_area)]
    return gray, binary, kept


def _contour_params(*extra: ParamSpec) -> tuple[ParamSpec, ...]:
    """轮廓类算子共用的参数组。"""
    return (
        ParamSpec(
            "threshold",
            "阈值",
            "int",
            default=128,
            min=0,
            max=255,
            step=1,
            tooltip="0-255 通用刻度；内部先转 8 位灰度再按该阈值二值化。",
        ),
        ParamSpec(
            "binary_mode",
            "二值模式",
            "enum",
            choices=(("binary", "二值"), ("binary_inv", "反二值")),
            default="binary",
            tooltip="二值：亮于阈值为前景；反二值：暗于阈值为前景。",
        ),
        ParamSpec(
            "retrieval",
            "轮廓模式",
            "enum",
            choices=(("external", "仅外轮廓"), ("list", "全部轮廓"), ("tree", "树状结构")),
            default="external",
        ),
        ParamSpec(
            "method",
            "近似方式",
            "enum",
            choices=(("simple", "精简"), ("none", "完整")),
            default="simple",
            tooltip="精简只保留拐点，完整保留边界上的所有像素。",
        ),
        ParamSpec(
            "min_area",
            "最小面积",
            "float",
            default=10.0,
            min=0.0,
            max=1000000.0,
            step=1.0,
            tooltip="面积小于该值的轮廓会被过滤。",
        ),
    ) + extra


def _contours_output(contours, width: int, height: int, label: str = "轮廓") -> ContoursData:
    return ContoursData(
        contours=tuple(contour.reshape(-1, 2) for contour in contours),
        image_size=(width, height),
        label=label,
    )


# ---------------------------------------------------------------------------
# 分水岭
# ---------------------------------------------------------------------------
class Watershed(Operator):
    """基于距离变换种子的分水岭分割。"""

    spec = OperatorSpec(
        id="segment.watershed",
        name="分水岭分割",
        category=("阈值与分割", "分割"),
        description=(
            "先把图像二值化，再用距离变换的局部峰值作为前景种子，最后用分水岭算法"
            "把粘连目标分开。可连接可选的标记图直接指定种子；输出区域标注图、"
            "区域标签与区域数量。未连接标记图时会自动生成种子。"
        ),
        keywords=("watershed", "分水岭", "分割", "种子", "距离变换", "粘连"),
        inputs=(
            _INPUT,
            PortSpec(
                "markers",
                "标记图（可选）",
                accepts=("L",),
                optional=True,
                description="与输入同尺寸的整数标记图（非零为种子）；未连接时自动生成。",
            ),
        ),
        outputs=(
            PortSpec("image", "区域标注图", accepts=("BGR",)),
            PortSpec("regions", "区域标签", data_type=DataType.COMPONENTS),
            PortSpec("count", "区域数", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "threshold",
                "前景阈值",
                "int",
                default=128,
                min=0,
                max=255,
                step=1,
                tooltip="0-255 通用刻度；内部先转 8 位灰度再二值化。",
            ),
            ParamSpec(
                "peak_ratio",
                "种子峰值比例",
                "float",
                default=0.5,
                min=0.01,
                max=1.0,
                step=0.05,
                tooltip="距离变换大于「最大值 × 比例」的像素作为前景种子。",
            ),
            ParamSpec(
                "min_area",
                "最小区域面积",
                "float",
                default=10.0,
                min=0.0,
                max=1000000.0,
                step=1.0,
                tooltip="面积小于该值的区域会被过滤。",
            ),
            ParamSpec(
                "connectivity",
                "连通方式",
                "enum",
                choices=((4, "4 连通"), (8, "8 连通")),
                default=8,
            ),
            ParamSpec(
                "boundary",
                "绘制边界",
                "bool",
                default=True,
                tooltip="在区域标注图上用彩色线条画出分水岭边界。",
            ),
            ParamSpec("color", "边界颜色", "color", default="#ff3b30"),
        ),
    )

    def process(
        self,
        image=None,
        markers=None,
        threshold: int = 128,
        peak_ratio: float = 0.5,
        min_area: float = 10.0,
        connectivity: int = 8,
        boundary: bool = True,
        color: str = "#ff3b30",
        **_: Any,
    ):
        bgr = _cv.source_bgr8(image, "输入图像")
        height, width = bgr.shape[:2]

        ws_markers: np.ndarray | None = None
        if markers is not None:
            marker_image = _cv.source_gray8(markers, "标记图")
            _cv.same_size(bgr, marker_image, "标记图")
            candidate = marker_image.astype(np.int32)
            if int(candidate.max()) > 1:
                ws_markers = candidate

        if ws_markers is None:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            _value, binary = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
            distance = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
            peak = float(distance.max()) if distance.size else 0.0
            if peak > 0:
                sure_fg = (distance >= peak * float(peak_ratio)).astype(np.uint8)
            else:
                sure_fg = np.zeros_like(binary)
            seed_count, seed_labels = cv2.connectedComponents(
                sure_fg, connectivity=int(connectivity)
            )
            ws_markers = seed_labels.astype(np.int32) + 1
            unknown = cv2.subtract(binary, sure_fg * 255)
            ws_markers[unknown > 0] = 0
            if seed_count <= 1:
                # 没有可靠前景时，用中心区域作为唯一种子，保证默认参数可运行。
                center_x, center_y, center_w, center_h = _cv.center_rect(width, height, 0.25)
                ws_markers[:, :] = 1
                ws_markers[
                    center_y : center_y + center_h, center_x : center_x + center_w
                ] = 2

        ws_markers = np.ascontiguousarray(ws_markers, dtype=np.int32)
        cv2.watershed(bgr, ws_markers)

        region_labels = np.where(ws_markers > 1, ws_markers - 1, 0).astype(np.int32)
        filtered = np.zeros_like(region_labels)
        next_label = 1
        max_label = int(region_labels.max()) if region_labels.size else 0
        for index in range(1, max_label + 1):
            region = region_labels == index
            if int(region.sum()) >= int(min_area):
                filtered[region] = next_label
                next_label += 1

        components = ComponentsData.from_labels(filtered, label="分水岭区域")
        canvas = components.render_preview().array
        if boundary:
            canvas[ws_markers == -1] = _cv.draw_color(canvas, color)
        return {
            "image": canvas,
            "regions": components,
            "count": NumberValue(value=components.count, label="区域数", unit="个"),
        }


# ---------------------------------------------------------------------------
# GrabCut
# ---------------------------------------------------------------------------
class GrabCut(Operator):
    """GrabCut 交互式前景分割。"""

    spec = OperatorSpec(
        id="segment.grabcut",
        name="GrabCut 前景分割",
        category=("阈值与分割", "分割"),
        description=(
            "用 GrabCut 图割算法分离前景与背景。默认在图像中心取矩形作为初始前景区域；"
            "也可以连接可选的初始蒙版（非零视为可能前景）做蒙版初始化。"
            "输出前景高亮图、前景蒙版与前景占比。"
        ),
        keywords=("grabcut", "前景", "分割", "抠图", "图割", "mask"),
        inputs=(
            _INPUT,
            PortSpec(
                "mask",
                "初始蒙版（可选）",
                accepts=("L",),
                optional=True,
                description="非零像素视为可能前景；未连接时使用中心矩形初始化。",
            ),
        ),
        outputs=(
            PortSpec("image", "前景高亮图", accepts=("BGR",)),
            PortSpec("mask", "前景蒙版", accepts=("L",)),
            PortSpec("foreground_ratio", "前景占比", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "roi",
                "初始矩形",
                "roi",
                default=None,
                placeholder="[x, y, w, h]",
                tooltip="未连接初始蒙版时使用的初始前景矩形；留空则取图像中心 50% 区域。",
            ),
            ParamSpec(
                "iterations",
                "迭代次数",
                "int",
                default=3,
                min=1,
                max=10,
                step=1,
                tooltip="迭代次数越多结果越稳定，耗时也越长。",
            ),
            ParamSpec("outline", "绘制边界", "bool", default=True),
            ParamSpec("color", "前景颜色", "color", default="#00c853"),
        ),
    )

    def process(
        self,
        image=None,
        mask=None,
        roi=None,
        iterations: int = 3,
        outline: bool = True,
        color: str = "#00c853",
        **_: Any,
    ):
        bgr = _cv.source_bgr8(image, "输入图像")
        height, width = bgr.shape[:2]
        gc_mask = np.full((height, width), cv2.GC_BGD, np.uint8)

        initialized_with_mask = False
        if mask is not None:
            initial = _cv.source_gray8(mask, "初始蒙版")
            _cv.same_size(bgr, initial, "初始蒙版")
            if bool((initial > 0).any()):
                gc_mask[initial > 0] = cv2.GC_PR_FGD
                initialized_with_mask = True

        background_model = np.zeros((1, 65), np.float64)
        foreground_model = np.zeros((1, 65), np.float64)
        if initialized_with_mask:
            cv2.grabCut(
                bgr,
                gc_mask,
                None,
                background_model,
                foreground_model,
                int(iterations),
                cv2.GC_INIT_WITH_MASK,
            )
        else:
            x, y, rect_w, rect_h = _cv.resolve_rect(roi, width, height, fraction=0.5)
            gc_mask[:] = cv2.GC_BGD
            cv2.grabCut(
                bgr,
                gc_mask,
                (x, y, rect_w, rect_h),
                background_model,
                foreground_model,
                int(iterations),
                cv2.GC_INIT_WITH_RECT,
            )

        foreground = np.where(
            (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        canvas = bgr.copy()
        color_value = _cv.draw_color(canvas, color)
        tinted = canvas.copy()
        tinted[foreground > 0] = color_value
        canvas = cv2.addWeighted(canvas, 0.55, tinted, 0.45, 0.0)
        if outline:
            edge = cv2.morphologyEx(
                foreground, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)
            )
            canvas[edge > 0] = color_value

        ratio = float((foreground > 0).mean() * 100.0) if foreground.size else 0.0
        return {
            "image": canvas,
            "mask": foreground,
            "foreground_ratio": NumberValue(value=ratio, label="前景占比", unit="%"),
        }


# ---------------------------------------------------------------------------
# 漫水填充
# ---------------------------------------------------------------------------
class FloodFill(Operator):
    """漫水填充（种子点缺省取图像中心）。"""

    spec = OperatorSpec(
        id="segment.floodfill",
        name="漫水填充",
        category=("阈值与分割", "分割"),
        description=(
            "从种子点开始把颜色相近的连通区域填充成指定颜色。种子点留空时使用图像"
            "中心点；可连接可选的障碍蒙版（非零像素不参与填充）。"
            "输出填充结果、填充蒙版与填充面积。"
        ),
        keywords=("floodfill", "漫水填充", "填充", "种子", "分割", "油漆桶"),
        inputs=(
            _INPUT,
            PortSpec(
                "mask",
                "障碍蒙版（可选）",
                accepts=("L",),
                optional=True,
                description="非零像素作为障碍，不会被填充；未连接时不受限制。",
            ),
        ),
        outputs=(
            PortSpec("image", "填充结果", accepts=("L", "BGR", "BGRA")),
            PortSpec("mask", "填充蒙版", accepts=("L",)),
            PortSpec("area", "填充面积", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "seed",
                "种子点",
                "points",
                default=[],
                count=0,
                tooltip="可拾取多个种子点；留空时使用图像中心点。",
            ),
            ParamSpec(
                "tolerance",
                "容差",
                "int",
                default=10,
                min=0,
                max=255,
                step=1,
                tooltip="与种子点颜色差在容差内的像素会被填充（0-255 通用刻度）。",
            ),
            ParamSpec(
                "connectivity",
                "连通方式",
                "enum",
                choices=((4, "4 连通"), (8, "8 连通")),
                default=8,
            ),
            ParamSpec(
                "mode",
                "输出方式",
                "enum",
                choices=(("fill", "填充颜色"), ("mask", "仅输出蒙版")),
                default="fill",
            ),
            ParamSpec("color", "填充颜色", "color", default="#ff6d00"),
        ),
    )

    def process(
        self,
        image=None,
        mask=None,
        seed=None,
        tolerance: int = 10,
        connectivity: int = 8,
        mode: str = "fill",
        color: str = "#ff6d00",
        **_: Any,
    ):
        original = _cv.source(image, "输入图像")
        height, width = original.shape[:2]
        # cv2.floodFill 只支持 1 或 3 通道，这里统一用 3 通道 BGR 工作图；
        # 填充结果再按 mask 写回原始 dtype / 通道数（BGRA 也能正常处理）。
        work = np.ascontiguousarray(_cv.source_bgr8(image, "输入图像")).copy()
        fill_mask = np.zeros((height + 2, width + 2), np.uint8)

        barriers = None
        if mask is not None:
            barrier_image = _cv.source_gray8(mask, "障碍蒙版")
            _cv.same_size(original, barrier_image, "障碍蒙版")
            barriers = barrier_image > 0
            fill_mask[1:-1, 1:-1][barriers] = 255

        seeds = _cv.resolve_points(seed, width, height, count=0)
        if not seeds:
            seeds = [(width // 2, height // 2)]

        fill_value = _cv.draw_color(work, color)
        tolerance_value = int(tolerance)
        for seed_x, seed_y in seeds:
            x = min(max(int(seed_x), 0), width - 1)
            y = min(max(int(seed_y), 0), height - 1)
            flags = int(connectivity) | (255 << 8)
            if mode == "mask":
                flags |= cv2.FLOODFILL_MASK_ONLY
            cv2.floodFill(
                work,
                fill_mask,
                (x, y),
                fill_value,
                (tolerance_value, tolerance_value, tolerance_value),
                (tolerance_value, tolerance_value, tolerance_value),
                flags,
            )

        filled = fill_mask[1:-1, 1:-1] > 0
        if barriers is not None:
            filled = np.logical_and(filled, np.logical_not(barriers))

        result = original.copy()
        if mode == "fill" and filled.any():
            result[filled] = _cv.draw_color(result, color)
        mask_output = (filled.astype(np.uint8) * 255)
        return {
            "image": result,
            "mask": mask_output,
            "area": NumberValue(value=int(filled.sum()), label="填充面积", unit="像素"),
        }


# ---------------------------------------------------------------------------
# 距离变换
# ---------------------------------------------------------------------------
class DistanceTransform(Operator):
    """二值图距离变换。"""

    spec = OperatorSpec(
        id="segment.distance",
        name="距离变换",
        category=("阈值与分割", "分割"),
        description=(
            "先把图像二值化，再计算每个前景像素到最近背景像素的距离（L1 / L2 / 棋盘）。"
            "输出的灰度图会把距离归一化到 0-255 方便预览，同时输出最大距离。"
        ),
        keywords=("distance", "距离变换", "dt", "分割", "距离"),
        inputs=(_INPUT,),
        outputs=(
            PortSpec("image", "距离图", accepts=("L",)),
            PortSpec("max_distance", "最大距离", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "threshold",
                "前景阈值",
                "int",
                default=128,
                min=0,
                max=255,
                step=1,
                tooltip="0-255 通用刻度；内部先转 8 位灰度再二值化。",
            ),
            ParamSpec(
                "distance_type",
                "距离类型",
                "enum",
                choices=(("L1", "曼哈顿距离"), ("L2", "欧氏距离"), ("C", "棋盘距离")),
                default="L2",
            ),
            ParamSpec(
                "mask_size",
                "掩膜尺寸",
                "enum",
                choices=((3, "3×3"), (5, "5×5")),
                default=5,
                tooltip="L2 距离使用的近似掩膜尺寸。",
            ),
            ParamSpec(
                "invert",
                "反转前景",
                "bool",
                default=False,
                tooltip="开启后计算背景像素到最近前景像素的距离。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        threshold: int = 128,
        distance_type: str = "L2",
        mask_size: int = 5,
        invert: bool = False,
        **_: Any,
    ):
        gray = _cv.source_gray8(image, "输入图像")
        _value, binary = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)
        if invert:
            binary = cv2.bitwise_not(binary)
        flags = {
            "L1": cv2.DIST_L1,
            "L2": cv2.DIST_L2,
            "C": cv2.DIST_C,
        }
        if distance_type not in flags:
            raise OperatorError(f"不支持的距离类型：{distance_type}")
        distance = cv2.distanceTransform(binary, flags[distance_type], int(mask_size))
        peak = float(distance.max()) if distance.size else 0.0
        if peak > 0:
            canvas = np.clip(distance / peak * 255.0, 0, 255).astype(np.uint8)
        else:
            canvas = np.zeros(gray.shape, dtype=np.uint8)
        return {
            "image": canvas,
            "max_distance": NumberValue(value=peak, label="最大距离", unit="像素"),
        }


# ---------------------------------------------------------------------------
# 连通域
# ---------------------------------------------------------------------------
class Components(Operator):
    """连通域标记与统计。"""

    spec = OperatorSpec(
        id="segment.components",
        name="连通域标记",
        category=("阈值与分割", "分割"),
        description=(
            "先二值化，再用 connectedComponentsWithStats 标记连通域，并按最小面积过滤。"
            "输出彩色标签图、连通域数据（标签图 / 面积 / 质心）、数量与最大面积。"
        ),
        keywords=("components", "连通域", "标记", "label", "面积", "分割"),
        inputs=(_INPUT,),
        outputs=(
            PortSpec("image", "彩色标签图", accepts=("BGR",)),
            PortSpec("components", "连通域", data_type=DataType.COMPONENTS),
            PortSpec("count", "数量", data_type=DataType.NUMBER),
            PortSpec("largest_area", "最大面积", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "threshold",
                "阈值",
                "int",
                default=128,
                min=0,
                max=255,
                step=1,
                tooltip="0-255 通用刻度；内部先转 8 位灰度再二值化。",
            ),
            ParamSpec(
                "binary_mode",
                "二值模式",
                "enum",
                choices=(("binary", "二值"), ("binary_inv", "反二值")),
                default="binary",
            ),
            ParamSpec(
                "min_area",
                "最小面积",
                "int",
                default=10,
                min=0,
                max=1000000,
                step=1,
                tooltip="面积小于该值的连通域会被删除。",
            ),
            ParamSpec(
                "connectivity",
                "连通方式",
                "enum",
                choices=((4, "4 连通"), (8, "8 连通")),
                default=8,
            ),
        ),
    )

    def process(
        self,
        image=None,
        threshold: int = 128,
        binary_mode: str = "binary",
        min_area: int = 10,
        connectivity: int = 8,
        **_: Any,
    ):
        gray = _cv.source_gray8(image, "输入图像")
        _value, binary = cv2.threshold(
            gray, int(threshold), 255, _binary_flag(binary_mode)
        )
        num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            binary, connectivity=int(connectivity)
        )
        filtered = np.zeros_like(labels, dtype=np.int32)
        next_label = 1
        for index in range(1, num_labels):
            if int(stats[index, cv2.CC_STAT_AREA]) >= int(min_area):
                filtered[labels == index] = next_label
                next_label += 1

        components = ComponentsData.from_labels(filtered)
        return {
            "image": components.render_preview().array,
            "components": components,
            "count": NumberValue(value=components.count, label="连通域数量", unit="个"),
            "largest_area": NumberValue(
                value=components.largest_area, label="最大面积", unit="像素"
            ),
        }


# ---------------------------------------------------------------------------
# K 均值分割
# ---------------------------------------------------------------------------
class KMeansSegment(Operator):
    """K 均值颜色聚类分割。"""

    spec = OperatorSpec(
        id="segment.kmeans",
        name="K 均值分割",
        category=("阈值与分割", "分割"),
        description=(
            "把每个像素的 BGR 颜色作为三维特征做 K 均值聚类，用聚类中心替换像素颜色，"
            "得到色块化的分割结果。输出量化后的彩色图、聚类数与紧凑度。"
        ),
        keywords=("kmeans", "K均值", "聚类", "颜色量化", "分割"),
        inputs=(_INPUT,),
        outputs=(
            PortSpec("image", "色块分割图", accepts=("BGR",)),
            PortSpec("clusters", "聚类数", data_type=DataType.NUMBER),
            PortSpec("compactness", "紧凑度", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "clusters",
                "聚类数",
                "int",
                default=4,
                min=2,
                max=16,
                step=1,
                tooltip="颜色聚类的类别数。",
            ),
            ParamSpec(
                "attempts",
                "尝试次数",
                "int",
                default=3,
                min=1,
                max=10,
                step=1,
                tooltip="用不同初始中心重复聚类的次数，取最好结果。",
            ),
            ParamSpec(
                "max_iter",
                "最大迭代",
                "int",
                default=20,
                min=1,
                max=100,
                step=1,
            ),
            ParamSpec(
                "epsilon",
                "收敛精度",
                "float",
                default=1.0,
                min=0.1,
                max=100.0,
                step=0.1,
            ),
        ),
    )

    def process(
        self,
        image=None,
        clusters: int = 4,
        attempts: int = 3,
        max_iter: int = 20,
        epsilon: float = 1.0,
        **_: Any,
    ):
        bgr = _cv.source_bgr8(image, "输入图像")
        height, width = bgr.shape[:2]
        samples = bgr.reshape(-1, 3).astype(np.float32)
        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            int(max_iter),
            float(epsilon),
        )
        compactness, labels, centers = cv2.kmeans(
            samples,
            int(clusters),
            None,
            criteria,
            int(attempts),
            cv2.KMEANS_PP_CENTERS,
        )
        center_colors = np.clip(np.rint(centers), 0, 255).astype(np.uint8)
        quantized = center_colors[labels.reshape(-1)].reshape(height, width, 3)
        return {
            "image": quantized,
            "clusters": NumberValue(value=float(clusters), label="聚类数", unit="类"),
            "compactness": NumberValue(
                value=float(compactness), label="紧凑度"
            ),
        }


# ---------------------------------------------------------------------------
# 轮廓提取
# ---------------------------------------------------------------------------
class Contours(Operator):
    """查找并绘制轮廓。"""

    spec = OperatorSpec(
        id="segment.contours",
        name="轮廓提取",
        category=("阈值与分割", "分割"),
        description=(
            "二值化后查找轮廓，按最小面积过滤，再把轮廓绘制到彩色图上。"
            "输出绘制结果、轮廓点集与轮廓数量。"
        ),
        keywords=("contours", "轮廓", "findcontours", "分割", "边缘"),
        inputs=(_INPUT,),
        outputs=(
            PortSpec("image", "轮廓标注图", accepts=("BGR",)),
            PortSpec("contours", "轮廓", data_type=DataType.CONTOURS),
            PortSpec("count", "轮廓数", data_type=DataType.NUMBER),
        ),
        params=_contour_params(
            ParamSpec("color", "绘制颜色", "color", default="#00ff66"),
            ParamSpec("thickness", "线宽", "int", default=2, min=1, max=10, step=1),
            ParamSpec("filled", "填充轮廓", "bool", default=False),
        ),
    )

    def process(
        self,
        image=None,
        threshold: int = 128,
        binary_mode: str = "binary",
        retrieval: str = "external",
        method: str = "simple",
        min_area: float = 10.0,
        color: str = "#00ff66",
        thickness: int = 2,
        filled: bool = False,
        **_: Any,
    ):
        _gray, _binary, contours = _find_contours(
            image, threshold, binary_mode, retrieval, method, min_area
        )
        canvas = _cv.source_bgr8(image, "输入图像")
        draw_value = _cv.draw_color(canvas, color)
        if contours:
            cv2.drawContours(
                canvas,
                contours,
                -1,
                draw_value,
                -1 if filled else int(thickness),
                lineType=_cv.draw_type(canvas),
            )
        height, width = canvas.shape[:2]
        return {
            "image": canvas,
            "contours": _contours_output(contours, width, height),
            "count": NumberValue(value=len(contours), label="轮廓数", unit="个"),
        }


class ConvexHull(Operator):
    """计算并绘制轮廓凸包。"""

    spec = OperatorSpec(
        id="segment.convex_hull",
        name="轮廓凸包",
        category=("阈值与分割", "分割"),
        description=(
            "提取轮廓后计算每个轮廓的凸包，并绘制到彩色图上。"
            "输出凸包标注图、凸包点集与数量。"
        ),
        keywords=("convex", "凸包", "hull", "轮廓", "分割"),
        inputs=(_INPUT,),
        outputs=(
            PortSpec("image", "凸包标注图", accepts=("BGR",)),
            PortSpec("contours", "凸包", data_type=DataType.CONTOURS),
            PortSpec("count", "凸包数", data_type=DataType.NUMBER),
        ),
        params=_contour_params(
            ParamSpec("color", "绘制颜色", "color", default="#ffd400"),
            ParamSpec("thickness", "线宽", "int", default=2, min=1, max=10, step=1),
        ),
    )

    def process(
        self,
        image=None,
        threshold: int = 128,
        binary_mode: str = "binary",
        retrieval: str = "external",
        method: str = "simple",
        min_area: float = 10.0,
        color: str = "#ffd400",
        thickness: int = 2,
        **_: Any,
    ):
        _gray, _binary, contours = _find_contours(
            image, threshold, binary_mode, retrieval, method, min_area
        )
        hulls = []
        for contour in contours:
            hull = cv2.convexHull(contour)
            if hull is not None and len(hull) >= 1:
                hulls.append(hull)

        canvas = _cv.source_bgr8(image, "输入图像")
        draw_value = _cv.draw_color(canvas, color)
        for hull in hulls:
            cv2.polylines(
                canvas,
                [hull.reshape(-1, 1, 2)],
                True,
                draw_value,
                int(thickness),
                lineType=_cv.draw_type(canvas),
            )
        height, width = canvas.shape[:2]
        return {
            "image": canvas,
            "contours": _contours_output(hulls, width, height, label="凸包"),
            "count": NumberValue(value=len(hulls), label="凸包数", unit="个"),
        }


class ApproxPoly(Operator):
    """多边形逼近（approxPolyDP）。"""

    spec = OperatorSpec(
        id="segment.approx_poly",
        name="多边形逼近",
        category=("阈值与分割", "分割"),
        description=(
            "用 Douglas-Peucker 算法把轮廓逼近成更少顶点的多边形。"
            "逼近精度按轮廓周长的百分比计算；输出逼近结果、多边形点集与数量。"
        ),
        keywords=("approx", "approxpoly", "多边形逼近", "多边形", "轮廓", "分割"),
        inputs=(_INPUT,),
        outputs=(
            PortSpec("image", "多边形标注图", accepts=("BGR",)),
            PortSpec("contours", "多边形", data_type=DataType.CONTOURS),
            PortSpec("count", "多边形数", data_type=DataType.NUMBER),
        ),
        params=_contour_params(
            ParamSpec(
                "epsilon",
                "逼近精度",
                "float",
                default=2.0,
                min=0.1,
                max=20.0,
                step=0.1,
                unit="%",
                tooltip="允许的最大偏差，按轮廓周长的百分比计算；越大顶点越少。",
            ),
            ParamSpec("color", "绘制颜色", "color", default="#3ddc97"),
            ParamSpec("thickness", "线宽", "int", default=2, min=1, max=10, step=1),
        ),
    )

    def process(
        self,
        image=None,
        threshold: int = 128,
        binary_mode: str = "binary",
        retrieval: str = "external",
        method: str = "simple",
        min_area: float = 10.0,
        epsilon: float = 2.0,
        color: str = "#3ddc97",
        thickness: int = 2,
        **_: Any,
    ):
        _gray, _binary, contours = _find_contours(
            image, threshold, binary_mode, retrieval, method, min_area
        )
        polygons = []
        for contour in contours:
            perimeter = cv2.arcLength(contour, True)
            if perimeter <= 0:
                continue
            approx = cv2.approxPolyDP(
                contour, float(epsilon) / 100.0 * perimeter, True
            )
            if approx is not None and len(approx) >= 1:
                polygons.append(approx)

        canvas = _cv.source_bgr8(image, "输入图像")
        draw_value = _cv.draw_color(canvas, color)
        for polygon in polygons:
            cv2.polylines(
                canvas,
                [polygon.reshape(-1, 1, 2)],
                True,
                draw_value,
                int(thickness),
                lineType=_cv.draw_type(canvas),
            )
        height, width = canvas.shape[:2]
        return {
            "image": canvas,
            "contours": _contours_output(polygons, width, height, label="多边形"),
            "count": NumberValue(value=len(polygons), label="多边形数", unit="个"),
        }


class MinAreaRect(Operator):
    """最小外接旋转矩形。"""

    spec = OperatorSpec(
        id="segment.min_area_rect",
        name="最小外接矩形",
        category=("阈值与分割", "分割"),
        description=(
            "为每个轮廓计算最小外接旋转矩形（可倾斜），并绘制到彩色图上。"
            "输出矩形标注图、矩形四角点集与数量。"
        ),
        keywords=("minarearect", "最小外接矩形", "旋转矩形", "轮廓", "分割"),
        inputs=(_INPUT,),
        outputs=(
            PortSpec("image", "矩形标注图", accepts=("BGR",)),
            PortSpec("rects", "矩形角点", data_type=DataType.CONTOURS),
            PortSpec("count", "矩形数", data_type=DataType.NUMBER),
        ),
        params=_contour_params(
            ParamSpec("color", "绘制颜色", "color", default="#4aa8ff"),
            ParamSpec("thickness", "线宽", "int", default=2, min=1, max=10, step=1),
        ),
    )

    def process(
        self,
        image=None,
        threshold: int = 128,
        binary_mode: str = "binary",
        retrieval: str = "external",
        method: str = "simple",
        min_area: float = 10.0,
        color: str = "#4aa8ff",
        thickness: int = 2,
        **_: Any,
    ):
        _gray, _binary, contours = _find_contours(
            image, threshold, binary_mode, retrieval, method, min_area
        )
        boxes = []
        for contour in contours:
            rect = cv2.minAreaRect(contour)
            if rect[1][0] <= 0 or rect[1][1] <= 0:
                continue
            box = cv2.boxPoints(rect)
            boxes.append(np.rint(box).astype(np.int32).reshape(4, 2))

        canvas = _cv.source_bgr8(image, "输入图像")
        draw_value = _cv.draw_color(canvas, color)
        for box in boxes:
            cv2.polylines(
                canvas,
                [box.reshape(-1, 1, 2)],
                True,
                draw_value,
                int(thickness),
                lineType=_cv.draw_type(canvas),
            )
        height, width = canvas.shape[:2]
        return {
            "image": canvas,
            "rects": ContoursData(
                contours=tuple(box.copy() for box in boxes),
                image_size=(width, height),
                label="矩形角点",
            ),
            "count": NumberValue(value=len(boxes), label="矩形数", unit="个"),
        }


class FitEllipse(Operator):
    """椭圆拟合。"""

    spec = OperatorSpec(
        id="segment.fit_ellipse",
        name="椭圆拟合",
        category=("阈值与分割", "分割"),
        description=(
            "用最小二乘法为每个轮廓（至少 5 个点）拟合椭圆并绘制。"
            "输出椭圆标注图、椭圆轮廓点集与数量；点数不足的轮廓会被跳过。"
        ),
        keywords=("fitellipse", "椭圆拟合", "椭圆", "轮廓", "分割"),
        inputs=(_INPUT,),
        outputs=(
            PortSpec("image", "椭圆标注图", accepts=("BGR",)),
            PortSpec("ellipses", "椭圆轮廓", data_type=DataType.CONTOURS),
            PortSpec("count", "椭圆数", data_type=DataType.NUMBER),
        ),
        params=_contour_params(
            ParamSpec("color", "绘制颜色", "color", default="#c77dff"),
            ParamSpec("thickness", "线宽", "int", default=2, min=1, max=10, step=1),
        ),
    )

    def process(
        self,
        image=None,
        threshold: int = 128,
        binary_mode: str = "binary",
        retrieval: str = "external",
        method: str = "simple",
        min_area: float = 10.0,
        color: str = "#c77dff",
        thickness: int = 2,
        **_: Any,
    ):
        _gray, _binary, contours = _find_contours(
            image, threshold, binary_mode, retrieval, method, min_area
        )
        ellipses = []
        outlines = []
        for contour in contours:
            if len(contour) < 5:
                continue
            try:
                ellipse = cv2.fitEllipse(contour)
            except cv2.error:
                # 共线 / 退化的轮廓无法拟合，直接跳过。
                continue
            (center, axes, angle) = ellipse
            if axes[0] <= 0 or axes[1] <= 0:
                continue
            ellipses.append(ellipse)
            polygon = cv2.ellipse2Poly(
                (int(round(center[0])), int(round(center[1]))),
                (max(1, int(round(axes[0] / 2.0))), max(1, int(round(axes[1] / 2.0)))),
                int(round(angle)),
                0,
                360,
                5,
            )
            outlines.append(polygon.reshape(-1, 2))

        canvas = _cv.source_bgr8(image, "输入图像")
        draw_value = _cv.draw_color(canvas, color)
        for ellipse in ellipses:
            cv2.ellipse(
                canvas,
                ellipse,
                draw_value,
                int(thickness),
                lineType=_cv.draw_type(canvas),
            )
        height, width = canvas.shape[:2]
        return {
            "image": canvas,
            "ellipses": ContoursData(
                contours=tuple(outlines), image_size=(width, height), label="椭圆轮廓"
            ),
            "count": NumberValue(value=len(ellipses), label="椭圆数", unit="个"),
        }


OPERATORS = (
    Watershed,
    GrabCut,
    FloodFill,
    DistanceTransform,
    Components,
    KMeansSegment,
    Contours,
    ConvexHull,
    ApproxPoly,
    MinAreaRect,
    FitEllipse,
)
