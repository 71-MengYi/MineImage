"""OpenCV 目录算子：视频与运动。

包含帧差、稠密/稀疏光流、背景建模、均值漂移 / CAMShift 跟踪、ECC 配准、
MIL 目标跟踪与运动能量图共 10 个算子。

约定
----
* 全部算子默认参数即可运行：``image2`` 未连接时把第一幅图水平平移（``np.roll``）
  当作第二帧，因此帧差 / 光流 / 视差等结果不会是空图；
* 视频类接口多数只支持 8 位图像，内部统一走 ``_cv.source_gray8`` 转换；
* 主输出始终是图像（预览依赖第一个图像输出），需要时额外输出 ``DataType.NUMBER``。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.core.data_types import DataType, NumberValue
from app.operators.builtin.catalog import _cv
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec

#: 第二帧未连接时的默认水平平移像素数。
DEFAULT_SHIFT = 3


# ---------------------------------------------------------------------------
# 共享工具
# ---------------------------------------------------------------------------
def _derived(base: np.ndarray, shift: int = DEFAULT_SHIFT) -> np.ndarray:
    """把第一幅图水平平移，作为“第二帧”的替代输入（返回新数组）。"""
    return np.roll(base, max(int(shift), 1), axis=1).copy()


def _second_gray(second, base: np.ndarray, shift: int = DEFAULT_SHIFT) -> np.ndarray:
    """第二幅图的灰度版本（保持 dtype）；未连接时由第一幅图平移派生。"""
    if second is None:
        return _derived(base, shift)
    other = _cv.source_gray(second, "第二幅图像")
    _cv.same_size(base, other, "第二幅图像")
    if other.dtype != base.dtype:
        # 两路输入 dtype 不一致时对齐到第一幅图，避免 OpenCV 混合 dtype 报英文错误。
        other = other.astype(base.dtype, copy=False)
    return other


def _second_gray8(second, base: np.ndarray, shift: int = DEFAULT_SHIFT) -> np.ndarray:
    """第二幅图的 8 位灰度版本；未连接时由第一幅图平移派生。"""
    if second is None:
        return _derived(base, shift)
    other = _cv.source_gray8(second, "第二幅图像")
    _cv.same_size(base, other, "第二幅图像")
    return other


def _second_bgr8(second, base: np.ndarray, shift: int = DEFAULT_SHIFT) -> np.ndarray:
    """第二幅图的 8 位 BGR 版本；未连接时由第一幅图平移派生。"""
    if second is None:
        return _derived(base, shift)
    other = _cv.source_bgr8(second, "第二幅图像")
    _cv.same_size(base, other, "第二幅图像")
    return other


def _flow_visual(flow: np.ndarray, mag_scale: float) -> np.ndarray:
    """把稠密光流 ``(H, W, 2)`` 渲染成 HSV 色轮 + 亮度表示的 BGR 图。"""
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
    hsv[..., 0] = np.clip(angle * (90.0 / np.pi), 0, 179).astype(np.uint8)
    hsv[..., 1] = 255
    hsv[..., 2] = np.clip(magnitude * float(mag_scale), 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def _clip_point(x: float, y: float, width: int, height: int) -> tuple[int, int]:
    return (
        int(min(max(round(float(x)), 0), max(width - 1, 0))),
        int(min(max(round(float(y)), 0), max(height - 1, 0))),
    )


def _backproject(bgr: np.ndarray) -> np.ndarray:
    """用 HSV 色调直方图计算反向投影，作为均值漂移 / CAMShift 的概率图。"""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    histogram = cv2.calcHist([hsv], [0], None, [180], [0, 180])
    cv2.normalize(histogram, histogram, 0, 255, cv2.NORM_MINMAX)
    return cv2.calcBackProject([hsv], [0], histogram, [0, 180], 1)


# ---------------------------------------------------------------------------
# 1. 帧差检测
# ---------------------------------------------------------------------------
class VideoFrameDiff(Operator):
    spec = OperatorSpec(
        id="video.frame_diff",
        name="帧差检测",
        category=("视频与运动", "运动检测"),
        description=(
            "对两帧图像做逐像素绝对差并放大显示，用于快速定位画面中发生变化的区域。"
            "未连接第二帧时自动把第一幅图水平平移若干像素作为参考帧，因此默认参数即可运行。"
            "输出灰度差异图与平均差异数值，支持 uint8 / uint16 / float32。"
        ),
        keywords=("帧差", "帧间差分", "运动检测", "frame diff", "motion"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec("image2", "第二帧（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "差异图", accepts=("L",)),
            PortSpec("mean_diff", "平均差异", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "gain",
                "差异增益",
                "float",
                default=3.0,
                min=0.1,
                max=16.0,
                step=0.1,
                tooltip="放大差异值，便于在预览中观察微小的帧间变化。",
            ),
        ),
    )

    def process(self, image=None, image2=None, gain: float = 3.0, **_: Any):
        base = _cv.source_gray(image, "输入图像")
        other = _second_gray(image2, base)
        diff = cv2.absdiff(base, other)
        amplified = diff.astype(np.float32) * float(gain)
        result = _cv.restore(amplified, base, "差异图")
        mean_diff = float(np.mean(diff.astype(np.float64))) if diff.size else 0.0
        return {
            "image": result,
            "mean_diff": NumberValue(value=mean_diff, label="平均差异"),
        }


# ---------------------------------------------------------------------------
# 2. Farneback 稠密光流
# ---------------------------------------------------------------------------
class VideoOpticalFlowFarneback(Operator):
    spec = OperatorSpec(
        id="video.optical_flow_farneback",
        name="稠密光流（Farneback）",
        category=("视频与运动", "运动检测"),
        description=(
            "用 Farneback 多项式展开法计算两帧之间的稠密光流，并渲染成 HSV 色轮图："
            "色相表示运动方向、亮度表示运动速度。未连接第二帧时自动平移第一幅图作为参考帧。"
            "输出光流可视化图与平均位移（像素）。"
        ),
        keywords=("光流", "稠密光流", "farneback", "optical flow", "运动"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec("image2", "第二帧（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "光流图", accepts=("BGR",)),
            PortSpec("mean_shift", "平均位移", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "winsize",
                "像素窗口",
                "int",
                default=15,
                min=3,
                max=51,
                step=2,
                odd=True,
                tooltip="每个像素邻域的窗口大小，越大越平滑。",
            ),
            ParamSpec("levels", "金字塔层数", "int", default=3, min=1, max=6, step=1),
            ParamSpec("iterations", "迭代次数", "int", default=3, min=1, max=10, step=1),
            ParamSpec(
                "mag_scale",
                "速度放大倍数",
                "float",
                default=8.0,
                min=1.0,
                max=40.0,
                step=1.0,
                group="显示",
            ),
        ),
    )

    def process(
        self,
        image=None,
        image2=None,
        winsize: int = 15,
        levels: int = 3,
        iterations: int = 3,
        mag_scale: float = 8.0,
        **_: Any,
    ):
        first = _cv.source_gray8(image, "输入图像")
        second = _second_gray8(image2, first)
        flow = cv2.calcOpticalFlowFarneback(
            first,
            second,
            None,
            0.5,
            int(levels),
            int(winsize),
            int(iterations),
            5,
            1.2,
            0,
        )
        magnitude, _angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        mean_shift = float(magnitude.mean()) if magnitude.size else 0.0
        return {
            "image": _flow_visual(flow, mag_scale),
            "mean_shift": NumberValue(value=mean_shift, label="平均位移", unit="px"),
        }


# ---------------------------------------------------------------------------
# 3. Lucas-Kanade 稀疏光流
# ---------------------------------------------------------------------------
class VideoOpticalFlowLk(Operator):
    spec = OperatorSpec(
        id="video.optical_flow_lk",
        name="稀疏光流（Lucas-Kanade）",
        category=("视频与运动", "运动检测"),
        description=(
            "先用 Shi-Tomasi 角点提取特征点，再用 Lucas-Kanade 金字塔光流在两帧之间跟踪，"
            "并在画面上画出运动矢量。特征点不足时自动改用均匀网格点，保证默认参数有可见输出。"
            "输出运动矢量图与成功跟踪的点数。"
        ),
        keywords=("光流", "稀疏光流", "lucas kanade", "lk", "角点追踪", "特征点"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec("image2", "第二帧（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "跟踪图", accepts=("BGR",)),
            PortSpec("tracked", "跟踪点数", data_type=DataType.NUMBER),
            PortSpec("mean_shift", "平均位移", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec("max_corners", "最大特征点数", "int", default=120, min=4, max=2000, step=1),
            ParamSpec(
                "quality",
                "角点质量",
                "float",
                default=0.01,
                min=0.001,
                max=1.0,
                step=0.001,
                tooltip="Shi-Tomasi 角点的最低质量阈值。",
            ),
            ParamSpec("min_distance", "最小间距", "float", default=3.0, min=1.0, max=50.0, step=0.5),
            ParamSpec(
                "winsize",
                "光流窗口",
                "int",
                default=15,
                min=3,
                max=51,
                step=2,
                odd=True,
            ),
            ParamSpec("max_level", "金字塔层数", "int", default=2, min=0, max=5, step=1),
            ParamSpec(
                "color",
                "矢量颜色",
                "color",
                default="#00d26a",
                group="显示",
            ),
        ),
    )

    def process(
        self,
        image=None,
        image2=None,
        max_corners: int = 120,
        quality: float = 0.01,
        min_distance: float = 3.0,
        winsize: int = 15,
        max_level: int = 2,
        color: str = "#00d26a",
        **_: Any,
    ):
        first = _cv.source_gray8(image, "输入图像")
        second = _second_gray8(image2, first)
        height, width = first.shape[:2]
        points = cv2.goodFeaturesToTrack(
            first,
            maxCorners=int(max_corners),
            qualityLevel=float(quality),
            minDistance=float(min_distance),
            blockSize=5,
        )
        if points is None or len(points) < 2:
            grid = _cv.resolve_points(None, width, height, count=12)
            points = np.asarray(grid, dtype=np.float32).reshape(-1, 1, 2)
        tracked, status, _error = cv2.calcOpticalFlowPyrLK(
            first,
            second,
            points,
            None,
            winSize=(int(winsize), int(winsize)),
            maxLevel=int(max_level),
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        canvas = _cv.source_bgr8(image, "输入图像")
        draw_color = _cv.draw_color(canvas, color)
        line_type = _cv.draw_type(canvas)
        shifts: list[float] = []
        if tracked is not None and status is not None:
            for old_point, new_point, ok in zip(points.reshape(-1, 2), tracked.reshape(-1, 2), status.ravel()):
                if not ok:
                    continue
                start = _clip_point(old_point[0], old_point[1], width, height)
                end = _clip_point(new_point[0], new_point[1], width, height)
                shifts.append(float(np.hypot(end[0] - start[0], end[1] - start[1])))
                cv2.arrowedLine(canvas, start, end, draw_color, 1, line_type, tipLength=0.3)
                cv2.circle(canvas, end, 2, draw_color, -1, line_type)
        mean_shift = float(np.mean(shifts)) if shifts else 0.0
        return {
            "image": canvas,
            "tracked": NumberValue(value=float(len(shifts)), label="跟踪点数", unit="个"),
            "mean_shift": NumberValue(value=mean_shift, label="平均位移", unit="px"),
        }


# ---------------------------------------------------------------------------
# 4/5. 背景建模（MOG2 / KNN）
# ---------------------------------------------------------------------------
class _BackgroundOperator(Operator):
    """背景减除算子的共享实现。"""

    def _build(self, history: int, threshold: float, detect_shadows: bool):
        raise NotImplementedError

    def process(
        self,
        image=None,
        image2=None,
        history: int = 100,
        var_threshold: float = 16.0,
        detect_shadows: bool = True,
        learning_rate: float = 0.05,
        **_: Any,
    ):
        first = _cv.source_gray8(image, "输入图像")
        second = _second_gray8(image2, first)
        model = self._build(int(history), float(var_threshold), bool(detect_shadows))
        model.apply(first, learningRate=float(learning_rate))
        mask = model.apply(second, learningRate=float(learning_rate))
        mask = cv2.medianBlur(mask, 5)
        ratio = float((mask > 0).sum()) / float(mask.size) * 100.0 if mask.size else 0.0
        return {
            "image": cv2.applyColorMap(mask, cv2.COLORMAP_HOT),
            "foreground": NumberValue(value=ratio, label="前景像素占比", unit="%"),
        }


class VideoBackgroundMog2(_BackgroundOperator):
    spec = OperatorSpec(
        id="video.background_mog2",
        name="MOG2 背景建模",
        category=("视频与运动", "背景建模"),
        description=(
            "用高斯混合模型（MOG2）建立背景并提取前景蒙版。第一帧用于建模，"
            "第二帧得到前景结果；未连接第二帧时自动平移第一幅图派生第二帧。"
            "输出热力图形式的前景蒙版与前景像素占比，默认参数即可运行。"
        ),
        keywords=("背景建模", "背景减除", "MOG2", "高斯混合", "foreground", "前景"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec("image2", "第二帧（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "前景蒙版", accepts=("BGR",)),
            PortSpec("foreground", "前景像素占比", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec("history", "历史帧数", "int", default=100, min=1, max=5000, step=1),
            ParamSpec("var_threshold", "方差阈值", "float", default=16.0, min=1.0, max=200.0, step=1.0),
            ParamSpec("detect_shadows", "检测阴影", "bool", default=True),
            ParamSpec(
                "learning_rate",
                "学习率",
                "float",
                default=0.05,
                min=0.0,
                max=1.0,
                step=0.01,
                tooltip="0 表示不更新背景，-1 表示自动学习率。",
            ),
        ),
    )

    def _build(self, history: int, threshold: float, detect_shadows: bool):
        return cv2.createBackgroundSubtractorMOG2(
            history=int(history),
            varThreshold=float(threshold),
            detectShadows=bool(detect_shadows),
        )


class VideoBackgroundKnn(_BackgroundOperator):
    spec = OperatorSpec(
        id="video.background_knn",
        name="KNN 背景建模",
        category=("视频与运动", "背景建模"),
        description=(
            "用 K 近邻（KNN）背景减除器建立背景并提取前景蒙版。第一帧用于建模，"
            "第二帧得到前景结果；未连接第二帧时自动平移第一幅图派生第二帧。"
            "输出热力图形式的前景蒙版与前景像素占比，默认参数即可运行。"
        ),
        keywords=("背景建模", "背景减除", "KNN", "近邻", "foreground", "前景"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec("image2", "第二帧（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "前景蒙版", accepts=("BGR",)),
            PortSpec("foreground", "前景像素占比", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec("history", "历史帧数", "int", default=100, min=1, max=5000, step=1),
            ParamSpec("var_threshold", "距离平方阈值", "float", default=400.0, min=10.0, max=5000.0, step=10.0),
            ParamSpec("detect_shadows", "检测阴影", "bool", default=True),
            ParamSpec(
                "learning_rate",
                "学习率",
                "float",
                default=0.05,
                min=0.0,
                max=1.0,
                step=0.01,
                tooltip="0 表示不更新背景，-1 表示自动学习率。",
            ),
        ),
    )

    def _build(self, history: int, threshold: float, detect_shadows: bool):
        return cv2.createBackgroundSubtractorKNN(
            history=int(history),
            dist2Threshold=float(threshold),
            detectShadows=bool(detect_shadows),
        )


# ---------------------------------------------------------------------------
# 6. 均值漂移
# ---------------------------------------------------------------------------
class VideoMeanShift(Operator):
    spec = OperatorSpec(
        id="video.mean_shift",
        name="均值漂移跟踪",
        category=("视频与运动", "目标跟踪"),
        description=(
            "用色调直方图反向投影 + 均值漂移（meanShift）迭代定位目标窗口。"
            "初始窗口取 ROI 参数，未指定时使用图像中心区域。输出画有跟踪窗口的图像"
            "与窗口中心位移（像素）。"
        ),
        keywords=("均值漂移", "mean shift", "目标跟踪", "反向投影", "tracking"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec("image2", "跟踪帧（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "跟踪窗口", accepts=("BGR",)),
            PortSpec("shift", "窗口位移", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "roi",
                "初始窗口",
                "roi",
                default=None,
                placeholder="[x, y, w, h]",
                tooltip="目标初始位置；不指定时使用图像中心区域。",
            ),
            ParamSpec("iterations", "最大迭代次数", "int", default=10, min=1, max=100, step=1),
            ParamSpec("color", "窗口颜色", "color", default="#00d26a", group="显示"),
        ),
    )

    def process(self, image=None, image2=None, roi=None, iterations: int = 10, color: str = "#00d26a", **_: Any):
        base = _cv.source_bgr8(image, "输入图像")
        frame = _second_bgr8(image2, base)
        height, width = base.shape[:2]
        x, y, rect_w, rect_h = _cv.resolve_rect(roi, width, height, fraction=0.3)
        probability = _backproject(frame)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iterations), 1.0)
        _iterations, window = cv2.meanShift(probability, (x, y, rect_w, rect_h), criteria)
        canvas = frame.copy()
        track_x, track_y, track_w, track_h = (int(value) for value in window)
        cv2.rectangle(
            canvas,
            (track_x, track_y),
            (track_x + track_w, track_y + track_h),
            _cv.draw_color(canvas, color),
            2,
            _cv.draw_type(canvas),
        )
        shift = float(np.hypot(track_x - x, track_y - y))
        return {
            "image": canvas,
            "shift": NumberValue(value=shift, label="窗口位移", unit="px"),
        }


# ---------------------------------------------------------------------------
# 7. CAMShift
# ---------------------------------------------------------------------------
class VideoCamShift(Operator):
    spec = OperatorSpec(
        id="video.camshift",
        name="CAMShift 跟踪",
        category=("视频与运动", "目标跟踪"),
        description=(
            "连续自适应均值漂移（CAMShift）在色调反向投影上迭代，得到带旋转角度的目标窗口。"
            "输出绘制了旋转矩形（椭圆）的图像、窗口位移与旋转角度。未指定 ROI 时使用中心区域。"
        ),
        keywords=("CAMShift", "连续自适应均值漂移", "目标跟踪", "旋转窗口", "tracking"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec("image2", "跟踪帧（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "跟踪窗口", accepts=("BGR",)),
            PortSpec("shift", "窗口位移", data_type=DataType.NUMBER),
            PortSpec("angle", "旋转角度", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "roi",
                "初始窗口",
                "roi",
                default=None,
                placeholder="[x, y, w, h]",
                tooltip="目标初始位置；不指定时使用图像中心区域。",
            ),
            ParamSpec("iterations", "最大迭代次数", "int", default=10, min=1, max=100, step=1),
            ParamSpec("color", "窗口颜色", "color", default="#00d26a", group="显示"),
        ),
    )

    def process(self, image=None, image2=None, roi=None, iterations: int = 10, color: str = "#00d26a", **_: Any):
        base = _cv.source_bgr8(image, "输入图像")
        frame = _second_bgr8(image2, base)
        height, width = base.shape[:2]
        x, y, rect_w, rect_h = _cv.resolve_rect(roi, width, height, fraction=0.3)
        probability = _backproject(frame)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, int(iterations), 1.0)
        rotated, _window = cv2.CamShift(probability, (x, y, rect_w, rect_h), criteria)
        canvas = frame.copy()
        box = cv2.boxPoints(rotated).astype(np.int32)
        cv2.polylines(canvas, [box], True, _cv.draw_color(canvas, color), 2, _cv.draw_type(canvas))
        center = rotated[0]
        shift = float(np.hypot(float(center[0]) - (x + rect_w / 2.0), float(center[1]) - (y + rect_h / 2.0)))
        return {
            "image": canvas,
            "shift": NumberValue(value=shift, label="窗口位移", unit="px"),
            "angle": NumberValue(value=float(rotated[2]), label="旋转角度", unit="°"),
        }


# ---------------------------------------------------------------------------
# 8. ECC 配准
# ---------------------------------------------------------------------------
class VideoEccAlign(Operator):
    spec = OperatorSpec(
        id="video.ecc_align",
        name="ECC 图像配准",
        category=("视频与运动", "配准对齐"),
        description=(
            "用增强相关系数（ECC）估计两幅图之间的几何变换，并对第一幅图做逆变换对齐。"
            "支持平移 / 欧氏 / 仿射 / 单应四种变换模型。未连接第二幅图时自动平移第一幅图作为目标。"
            "输出对齐后的图像与相关系数（越接近 1 越好）。"
        ),
        keywords=("配准", "对齐", "ECC", "增强相关系数", "registration", "align"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec("image2", "参考图像（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "对齐图"),
            PortSpec("correlation", "相关系数", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "motion",
                "变换模型",
                "enum",
                choices=(
                    ("translation", "平移"),
                    ("euclidean", "欧氏（平移 + 旋转）"),
                    ("affine", "仿射"),
                    ("homography", "单应"),
                ),
                default="translation",
            ),
            ParamSpec("iterations", "最大迭代次数", "int", default=200, min=10, max=5000, step=10),
        ),
    )

    def process(self, image=None, image2=None, motion: str = "translation", iterations: int = 200, **_: Any):
        base = _cv.source(image, "输入图像")
        other = _cv.optional_second(image2, _derived(base), "参考图像")
        height, width = base.shape[:2]
        template = _cv.source_gray8(base, "输入图像").astype(np.float32) / 255.0
        incoming = _cv.source_gray8(other, "参考图像").astype(np.float32) / 255.0

        # 先用相位相关给出平移初值：既能让 ECC 更快收敛，也保证纹理不足时有可用结果。
        try:
            (shift_x, shift_y), response = cv2.phaseCorrelate(template, incoming)
        except cv2.error:
            shift_x, shift_y, response = 0.0, 0.0, 0.0
        translation = np.array(
            [[1.0, 0.0, float(shift_x)], [0.0, 1.0, float(shift_y)]],
            dtype=np.float32,
        )

        if motion == "homography":
            motion_flag = cv2.MOTION_HOMOGRAPHY
            warp = np.eye(3, 3, dtype=np.float32)
            warp[:2, :] = translation
        else:
            motion_flag = {
                "translation": cv2.MOTION_TRANSLATION,
                "euclidean": cv2.MOTION_EUCLIDEAN,
                "affine": cv2.MOTION_AFFINE,
            }.get(str(motion), cv2.MOTION_TRANSLATION)
            warp = translation.copy()
        criteria = (
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
            int(iterations),
            1e-6,
        )
        try:
            correlation, warp = cv2.findTransformECC(
                template,
                incoming,
                warp,
                motion_flag,
                criteria,
                None,
                5,
            )
            if motion == "homography":
                aligned = cv2.warpPerspective(
                    base,
                    warp,
                    (width, height),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE,
                )
            else:
                aligned = cv2.warpAffine(
                    base,
                    warp,
                    (width, height),
                    flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                    borderMode=cv2.BORDER_REPLICATE,
                )
        except cv2.error:
            # 纹理不足（例如纯噪声）时 ECC 可能不收敛，退回相位相关平移结果。
            correlation = max(float(np.nan_to_num(response, nan=0.0)), 0.0)
            aligned = cv2.warpAffine(
                base,
                translation,
                (width, height),
                flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                borderMode=cv2.BORDER_REPLICATE,
            )
        return {
            "image": aligned,
            "correlation": NumberValue(value=float(correlation), label="相关系数"),
        }


# ---------------------------------------------------------------------------
# 9. MIL 目标跟踪
# ---------------------------------------------------------------------------
class VideoTracker(Operator):
    spec = OperatorSpec(
        id="video.tracker",
        name="目标跟踪（MIL）",
        category=("视频与运动", "目标跟踪"),
        description=(
            "用 OpenCV 的 MIL（多示例学习）跟踪器在第一帧初始化目标窗口，"
            "在第二帧上定位目标并画出跟踪框。未连接第二帧时自动平移第一幅图派生。"
            "输出跟踪结果图与跟踪状态（1 成功 / 0 丢失）。"
        ),
        keywords=("目标跟踪", "tracker", "MIL", "跟踪", "tracking"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec("image2", "跟踪帧（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "跟踪结果", accepts=("BGR",)),
            PortSpec("tracked", "跟踪状态", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "roi",
                "目标窗口",
                "roi",
                default=None,
                placeholder="[x, y, w, h]",
                tooltip="目标在第一帧中的位置；不指定时使用图像中心区域。",
            ),
            ParamSpec("color", "跟踪框颜色", "color", default="#00d26a", group="显示"),
        ),
    )

    def process(self, image=None, image2=None, roi=None, color: str = "#00d26a", **_: Any):
        first = _cv.source_bgr8(image, "输入图像")
        second = _second_bgr8(image2, first)
        height, width = first.shape[:2]
        x, y, rect_w, rect_h = _cv.resolve_rect(roi, width, height, fraction=0.3)
        tracker = cv2.TrackerMIL_create()
        try:
            tracker.init(first, (x, y, rect_w, rect_h))
        except cv2.error as exc:
            raise OperatorError("目标跟踪初始化失败：请缩小或重新框选目标窗口。") from exc
        ok, box = tracker.update(second)
        canvas = second.copy()
        box_x, box_y, box_w, box_h = (int(round(float(value))) for value in box)
        cv2.rectangle(
            canvas,
            (box_x, box_y),
            (box_x + box_w, box_y + box_h),
            _cv.draw_color(canvas, color),
            2,
            _cv.draw_type(canvas),
        )
        return {
            "image": canvas,
            "tracked": NumberValue(value=1.0 if ok else 0.0, label="跟踪状态"),
        }


# ---------------------------------------------------------------------------
# 10. 运动能量图
# ---------------------------------------------------------------------------
class VideoMotionEnergy(Operator):
    spec = OperatorSpec(
        id="video.motion_energy",
        name="运动能量图",
        category=("视频与运动", "运动检测"),
        description=(
            "对两帧做帧差、平滑并阈值化，得到运动区域的能量图，再渲染为热力图。"
            "未连接第二帧时自动平移第一幅图派生。输出热力运动图与平均运动能量。"
        ),
        keywords=("运动能量", "motion energy", "热力图", "帧差", "运动"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec("image2", "第二帧（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "运动热力图", accepts=("BGR",)),
            PortSpec("energy", "平均运动能量", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "threshold",
                "运动阈值",
                "int",
                default=20,
                min=0,
                max=255,
                step=1,
                tooltip="低于该差异值的像素视为静止。",
            ),
            ParamSpec("ksize", "平滑核", "int", default=5, min=1, max=31, step=2, odd=True),
        ),
    )

    def process(self, image=None, image2=None, threshold: int = 20, ksize: int = 5, **_: Any):
        first = _cv.source_gray8(image, "输入图像")
        second = _second_gray8(image2, first)
        diff = cv2.absdiff(first, second)
        if int(ksize) > 1:
            diff = cv2.GaussianBlur(diff, (int(ksize), int(ksize)), 0)
        _value, energy = cv2.threshold(diff, int(threshold), 255, cv2.THRESH_TOZERO)
        mean_energy = float(energy.mean()) if energy.size else 0.0
        return {
            "image": cv2.applyColorMap(energy, cv2.COLORMAP_JET),
            "energy": NumberValue(value=mean_energy, label="平均运动能量"),
        }


OPERATORS = (
    VideoFrameDiff,
    VideoOpticalFlowFarneback,
    VideoOpticalFlowLk,
    VideoBackgroundMog2,
    VideoBackgroundKnn,
    VideoMeanShift,
    VideoCamShift,
    VideoEccAlign,
    VideoTracker,
    VideoMotionEnergy,
)
