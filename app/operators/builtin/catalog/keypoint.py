"""OpenCV 目录算子：关键点检测与绘制。

覆盖 Harris / 最小特征值 / Shi-Tomasi / FAST / SIFT / ORB / AKAZE / BRISK /
KAZE / MSER / SimpleBlobDetector，以及统一的关键点绘制算子。

所有检测算子都：
* 输入统一转成 8 位灰度（这些接口只支持 8 位）；
* 支持可选检测蒙版（非零像素才参与检测）；
* 输出关键点标注图（BGR）、关键点集合（``DataType.EDGES``）与数量。
"""

from __future__ import annotations

from typing import Any, Callable

import cv2
import numpy as np

from app.core.data_types import ContoursData, DataType, EdgesData, NumberValue
from app.operators.builtin.catalog import _cv
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec

_INPUT = PortSpec(
    "image",
    "输入图像",
    accepts=("L", "BGR", "BGRA"),
    description="灰度、BGR 或 BGRA 图像；彩色图像会先转为灰度。",
)

_MASK = PortSpec(
    "mask",
    "检测蒙版（可选）",
    accepts=("L",),
    optional=True,
    description="只有非零像素参与关键点检测；未连接时对整幅图检测。",
)

_OUTPUTS = (
    PortSpec("image", "关键点标注图", accepts=("BGR",)),
    PortSpec("points", "关键点", data_type=DataType.EDGES),
    PortSpec("count", "关键点数", data_type=DataType.NUMBER),
)

_APERTURES = ((3, "3×3"), (5, "5×5"), (7, "7×7"))


# ---------------------------------------------------------------------------
# 通用辅助
# ---------------------------------------------------------------------------
def _gray(image, blur: int = 1) -> np.ndarray:
    gray = _cv.source_gray8(image, "输入图像")
    if int(blur) > 1:
        gray = cv2.GaussianBlur(gray, (int(blur), int(blur)), 0)
    return gray


def _mask_of(mask, reference: np.ndarray) -> np.ndarray | None:
    if mask is None:
        return None
    mask_array = _cv.source_gray8(mask, "检测蒙版")
    _cv.same_size(reference, mask_array, "检测蒙版")
    return mask_array


def _limit(keypoints, max_points: int):
    items = list(keypoints or [])
    if int(max_points) > 0 and len(items) > int(max_points):
        items = items[: int(max_points)]
    return items


def _filter_by_mask(keypoints, mask_array: np.ndarray | None):
    items = list(keypoints or [])
    if mask_array is None:
        return items
    height, width = mask_array.shape[:2]
    kept = []
    for keypoint in items:
        x, y = keypoint.pt
        column, row = int(round(x)), int(round(y))
        if 0 <= column < width and 0 <= row < height and mask_array[row, column] > 0:
            kept.append(keypoint)
    return kept


def _draw(
    image,
    keypoints,
    color: str,
    show_size: bool,
    scale: float = 1.0,
) -> np.ndarray:
    canvas = _cv.source_bgr8(image, "输入图像")
    items = list(keypoints or [])
    if not items:
        return canvas
    if abs(float(scale) - 1.0) > 1e-6:
        rescaled = []
        for keypoint in items:
            rescaled.append(
                cv2.KeyPoint(
                    float(keypoint.pt[0]),
                    float(keypoint.pt[1]),
                    max(1.0, float(keypoint.size) * float(scale)),
                    float(keypoint.angle),
                    float(keypoint.response),
                    int(keypoint.octave),
                    int(keypoint.class_id),
                )
            )
        items = rescaled
    flags = (
        cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
        if show_size
        else cv2.DRAW_MATCHES_FLAGS_DEFAULT
    )
    return cv2.drawKeypoints(
        canvas, items, None, _cv.draw_color(canvas, color), flags
    )


def _points_output(keypoints, width: int, height: int) -> EdgesData:
    segments = tuple(
        np.rint(np.asarray([[keypoint.pt[0], keypoint.pt[1]]], dtype=np.float64)).astype(
            np.int32
        )
        for keypoint in keypoints
    )
    return EdgesData(segments=segments, image_size=(width, height), label="关键点")


def _finish(image, keypoints, shape_source: np.ndarray, color: str, show_size: bool):
    height, width = shape_source.shape[:2]
    return {
        "image": _draw(image, keypoints, color, show_size),
        "points": _points_output(keypoints, width, height),
        "count": NumberValue(value=len(list(keypoints)), label="关键点数", unit="个"),
    }


def _keypoint_params(*detection: ParamSpec) -> tuple[ParamSpec, ...]:
    return detection + (
        ParamSpec("color", "绘制颜色", "color", default="#00ff66"),
        ParamSpec(
            "show_size",
            "显示尺度与方向",
            "bool",
            default=True,
            tooltip="关闭后只画等大的关键点圆点。",
        ),
        ParamSpec(
            "max_points",
            "最大关键点数",
            "int",
            default=0,
            min=0,
            max=5000,
            step=1,
            tooltip="限制输出的关键点数量，0 表示不限制。",
        ),
    )


def _run_detector(
    factory: Callable[[], Any],
    gray: np.ndarray,
    mask_array: np.ndarray | None,
    label: str,
):
    """创建并运行一个 OpenCV 关键点检测器，把英文异常转成中文提示。"""
    try:
        detector = factory()
        keypoints = detector.detect(gray, mask_array)
    except cv2.error as exc:
        raise OperatorError(
            f"{label}检测失败：当前参数组合不受当前 OpenCV 支持，请调整参数后重试。"
        ) from exc
    return list(keypoints) if keypoints else []


def _harris_keypoints(
    gray: np.ndarray, block_size: int, ksize: int, k: float, ratio: float
):
    response = cv2.cornerHarris(
        np.float32(gray), int(block_size), int(ksize), float(k)
    )
    normalized = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX)
    dilated = cv2.dilate(normalized, None)
    peak = float(normalized.max()) if normalized.size else 0.0
    if peak <= 0:
        return []
    rows, columns = np.nonzero(
        np.logical_and(normalized >= peak * float(ratio), normalized >= dilated - 1e-3)
    )
    return [cv2.KeyPoint(float(column), float(row), 7.0) for row, column in zip(rows, columns)]


# ---------------------------------------------------------------------------
# Harris 角点
# ---------------------------------------------------------------------------
class CornerHarris(Operator):
    """Harris 角点检测。"""

    spec = OperatorSpec(
        id="keypoint.corner_harris",
        name="Harris 角点",
        category=("特征与分析", "关键点"),
        description=(
            "用 Harris 响应函数检测角点：先计算响应图，再按响应峰值的比例做阈值，"
            "最后做 3×3 局部极大值抑制。输出角点标注图、角点集合与数量。"
        ),
        keywords=("harris", "角点", "corner", "关键点", "特征"),
        inputs=(_INPUT, _MASK),
        outputs=_OUTPUTS,
        params=_keypoint_params(
            ParamSpec(
                "block_size",
                "邻域大小",
                "int",
                default=5,
                min=3,
                max=31,
                step=2,
                odd=True,
            ),
            ParamSpec("ksize", "Sobel 孔径", "enum", choices=_APERTURES, default=3),
            ParamSpec(
                "k",
                "Harris 常数",
                "float",
                default=0.04,
                min=0.001,
                max=1.0,
                step=0.001,
            ),
            ParamSpec(
                "ratio",
                "响应阈值比例",
                "float",
                default=0.01,
                min=0.0001,
                max=1.0,
                step=0.001,
                tooltip="保留响应大于「最大响应 × 比例」的角点。",
            ),
            ParamSpec(
                "blur",
                "预模糊核",
                "int",
                default=3,
                min=1,
                max=31,
                step=2,
                odd=True,
                tooltip="检测前的高斯模糊核，必须为奇数；1 表示不模糊。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        mask=None,
        block_size: int = 5,
        ksize: int = 3,
        k: float = 0.04,
        ratio: float = 0.01,
        blur: int = 3,
        color: str = "#00ff66",
        show_size: bool = True,
        max_points: int = 0,
        **_: Any,
    ):
        gray = _gray(image, int(blur))
        mask_array = _mask_of(mask, gray)
        keypoints = _harris_keypoints(gray, block_size, ksize, k, ratio)
        keypoints = _limit(_filter_by_mask(keypoints, mask_array), max_points)
        return _finish(image, keypoints, gray, color, show_size)


# ---------------------------------------------------------------------------
# 最小特征值角点
# ---------------------------------------------------------------------------
class CornerMinEigen(Operator):
    """最小特征值角点检测。"""

    spec = OperatorSpec(
        id="keypoint.corner_min_eigen",
        name="最小特征值角点",
        category=("特征与分析", "关键点"),
        description=(
            "用结构张量的最小特征值（Shi-Tomasi 响应）检测角点：响应越大说明两个"
            "方向的梯度都强。输出角点标注图、角点集合与数量。"
        ),
        keywords=("mineigen", "最小特征值", "shi-tomasi", "角点", "关键点"),
        inputs=(_INPUT, _MASK),
        outputs=_OUTPUTS,
        params=_keypoint_params(
            ParamSpec(
                "block_size",
                "邻域大小",
                "int",
                default=5,
                min=3,
                max=31,
                step=2,
                odd=True,
            ),
            ParamSpec("ksize", "Sobel 孔径", "enum", choices=_APERTURES, default=3),
            ParamSpec(
                "ratio",
                "响应阈值比例",
                "float",
                default=0.01,
                min=0.0001,
                max=1.0,
                step=0.001,
            ),
            ParamSpec(
                "blur",
                "预模糊核",
                "int",
                default=3,
                min=1,
                max=31,
                step=2,
                odd=True,
            ),
        ),
    )

    def process(
        self,
        image=None,
        mask=None,
        block_size: int = 5,
        ksize: int = 3,
        ratio: float = 0.01,
        blur: int = 3,
        color: str = "#00ff66",
        show_size: bool = True,
        max_points: int = 0,
        **_: Any,
    ):
        gray = _gray(image, int(blur))
        mask_array = _mask_of(mask, gray)
        response = cv2.cornerMinEigenVal(
            np.float32(gray), int(block_size), int(ksize)
        )
        normalized = cv2.normalize(response, None, 0, 255, cv2.NORM_MINMAX)
        dilated = cv2.dilate(normalized, None)
        peak = float(normalized.max()) if normalized.size else 0.0
        keypoints = []
        if peak > 0:
            rows, columns = np.nonzero(
                np.logical_and(
                    normalized >= peak * float(ratio), normalized >= dilated - 1e-3
                )
            )
            keypoints = [
                cv2.KeyPoint(float(column), float(row), 7.0)
                for row, column in zip(rows, columns)
            ]
        keypoints = _limit(_filter_by_mask(keypoints, mask_array), max_points)
        return _finish(image, keypoints, gray, color, show_size)


# ---------------------------------------------------------------------------
# Shi-Tomasi 优质角点
# ---------------------------------------------------------------------------
class GoodFeatures(Operator):
    """Shi-Tomasi 优质角点（goodFeaturesToTrack）。"""

    spec = OperatorSpec(
        id="keypoint.good_features",
        name="优质角点",
        category=("特征与分析", "关键点"),
        description=(
            "用 goodFeaturesToTrack 检测最优质的角点：可以按质量阈值、点间距与数量"
            "筛选，也可切换到 Harris 评分。输出角点标注图、角点集合与数量。"
        ),
        keywords=("goodfeatures", "shi-tomasi", "优质角点", "角点", "关键点"),
        inputs=(_INPUT, _MASK),
        outputs=_OUTPUTS,
        params=_keypoint_params(
            ParamSpec(
                "max_features",
                "最大角点数",
                "int",
                default=200,
                min=0,
                max=5000,
                step=1,
                tooltip="0 表示不限制数量。",
            ),
            ParamSpec(
                "quality",
                "质量系数",
                "float",
                default=0.01,
                min=0.0001,
                max=1.0,
                step=0.001,
            ),
            ParamSpec(
                "min_distance",
                "最小间距",
                "float",
                default=10.0,
                min=1.0,
                max=500.0,
                step=1.0,
                unit="px",
            ),
            ParamSpec(
                "block_size",
                "邻域大小",
                "int",
                default=3,
                min=3,
                max=31,
                step=2,
                odd=True,
            ),
            ParamSpec(
                "use_harris",
                "使用 Harris 评分",
                "bool",
                default=False,
                tooltip="关闭时使用 Shi-Tomasi 最小特征值评分。",
            ),
            ParamSpec(
                "harris_k",
                "Harris 常数",
                "float",
                default=0.04,
                min=0.001,
                max=1.0,
                step=0.001,
                visible_when="use_harris",
                enabled_when="use_harris",
            ),
            ParamSpec(
                "blur",
                "预模糊核",
                "int",
                default=1,
                min=1,
                max=31,
                step=2,
                odd=True,
            ),
        ),
    )

    def process(
        self,
        image=None,
        mask=None,
        max_features: int = 200,
        quality: float = 0.01,
        min_distance: float = 10.0,
        block_size: int = 3,
        use_harris: bool = False,
        harris_k: float = 0.04,
        blur: int = 1,
        color: str = "#00ff66",
        show_size: bool = True,
        max_points: int = 0,
        **_: Any,
    ):
        gray = _gray(image, int(blur))
        mask_array = _mask_of(mask, gray)
        try:
            corners = cv2.goodFeaturesToTrack(
                gray,
                maxCorners=int(max_features),
                qualityLevel=float(quality),
                minDistance=float(min_distance),
                mask=mask_array,
                blockSize=int(block_size),
                useHarrisDetector=bool(use_harris),
                k=float(harris_k),
            )
        except cv2.error as exc:
            raise OperatorError(
                "优质角点检测失败：当前参数组合不受当前 OpenCV 支持，请调整参数后重试。"
            ) from exc
        keypoints = []
        if corners is not None:
            for corner in corners:
                keypoints.append(
                    cv2.KeyPoint(float(corner[0][0]), float(corner[0][1]), 7.0)
                )
        keypoints = _limit(_filter_by_mask(keypoints, mask_array), max_points)
        return _finish(image, keypoints, gray, color, show_size)


# ---------------------------------------------------------------------------
# FAST
# ---------------------------------------------------------------------------
class FastKeypoints(Operator):
    """FAST 角点检测。"""

    spec = OperatorSpec(
        id="keypoint.fast",
        name="FAST 角点",
        category=("特征与分析", "关键点"),
        description=(
            "用 FAST（Features from Accelerated Segment Test）快速检测角点，"
            "速度快、适合实时场景；可选非极大值抑制与 7×12 / 5×8 模板。"
        ),
        keywords=("fast", "角点", "corner", "关键点", "实时"),
        inputs=(_INPUT, _MASK),
        outputs=_OUTPUTS,
        params=_keypoint_params(
            ParamSpec(
                "threshold",
                "亮度阈值",
                "int",
                default=10,
                min=1,
                max=255,
                step=1,
                tooltip="候选点与周围圆环像素的亮度差阈值。",
            ),
            ParamSpec("nonmax", "非极大值抑制", "bool", default=True),
            ParamSpec(
                "fast_type",
                "检测模板",
                "enum",
                choices=(("9_16", "9/16 圆环"), ("7_12", "7/12 圆环"), ("5_8", "5/8 圆环")),
                default="9_16",
            ),
            ParamSpec(
                "blur",
                "预模糊核",
                "int",
                default=1,
                min=1,
                max=31,
                step=2,
                odd=True,
            ),
        ),
    )

    def process(
        self,
        image=None,
        mask=None,
        threshold: int = 10,
        nonmax: bool = True,
        fast_type: str = "9_16",
        blur: int = 1,
        color: str = "#00ff66",
        show_size: bool = True,
        max_points: int = 0,
        **_: Any,
    ):
        gray = _gray(image, int(blur))
        mask_array = _mask_of(mask, gray)
        type_flags = {
            "9_16": cv2.FAST_FEATURE_DETECTOR_TYPE_9_16,
            "7_12": cv2.FAST_FEATURE_DETECTOR_TYPE_7_12,
            "5_8": cv2.FAST_FEATURE_DETECTOR_TYPE_5_8,
        }
        if fast_type not in type_flags:
            raise OperatorError(f"不支持的 FAST 模板：{fast_type}")
        keypoints = _run_detector(
            lambda: cv2.FastFeatureDetector_create(
                int(threshold), bool(nonmax), type_flags[fast_type]
            ),
            gray,
            mask_array,
            "FAST 角点",
        )
        keypoints = _limit(_filter_by_mask(keypoints, mask_array), max_points)
        return _finish(image, keypoints, gray, color, show_size)


# ---------------------------------------------------------------------------
# SIFT
# ---------------------------------------------------------------------------
class SiftKeypoints(Operator):
    """SIFT 关键点检测。"""

    spec = OperatorSpec(
        id="keypoint.sift",
        name="SIFT 关键点",
        category=("特征与分析", "关键点"),
        description=(
            "用 SIFT（尺度不变特征变换）检测尺度与旋转不变的关键点，"
            "对缩放、旋转和光照变化都比较稳定。输出关键点标注图、关键点集合与数量。"
        ),
        keywords=("sift", "尺度不变", "关键点", "特征点", "keypoint"),
        inputs=(_INPUT, _MASK),
        outputs=_OUTPUTS,
        params=_keypoint_params(
            ParamSpec(
                "nfeatures",
                "特征点数量",
                "int",
                default=0,
                min=0,
                max=5000,
                step=1,
                tooltip="保留响应最高的前 N 个特征点，0 表示不限制。",
            ),
            ParamSpec(
                "n_octave_layers",
                "每层高斯数",
                "int",
                default=3,
                min=1,
                max=10,
                step=1,
            ),
            ParamSpec(
                "contrast",
                "对比度阈值",
                "float",
                default=0.04,
                min=0.001,
                max=0.5,
                step=0.001,
            ),
            ParamSpec(
                "edge",
                "边缘阈值",
                "float",
                default=10.0,
                min=1.0,
                max=100.0,
                step=1.0,
            ),
            ParamSpec(
                "sigma",
                "初始 σ",
                "float",
                default=1.6,
                min=0.5,
                max=10.0,
                step=0.1,
            ),
        ),
    )

    def process(
        self,
        image=None,
        mask=None,
        nfeatures: int = 0,
        n_octave_layers: int = 3,
        contrast: float = 0.04,
        edge: float = 10.0,
        sigma: float = 1.6,
        color: str = "#00ff66",
        show_size: bool = True,
        max_points: int = 0,
        **_: Any,
    ):
        gray = _gray(image)
        mask_array = _mask_of(mask, gray)
        keypoints = _run_detector(
            lambda: cv2.SIFT_create(
                nfeatures=int(nfeatures),
                nOctaveLayers=int(n_octave_layers),
                contrastThreshold=float(contrast),
                edgeThreshold=float(edge),
                sigma=float(sigma),
            ),
            gray,
            mask_array,
            "SIFT 关键点",
        )
        keypoints = _limit(_filter_by_mask(keypoints, mask_array), max_points)
        return _finish(image, keypoints, gray, color, show_size)


# ---------------------------------------------------------------------------
# ORB
# ---------------------------------------------------------------------------
class OrbKeypoints(Operator):
    """ORB 关键点检测。"""

    spec = OperatorSpec(
        id="keypoint.orb",
        name="ORB 关键点",
        category=("特征与分析", "关键点"),
        description=(
            "用 ORB（Oriented FAST and Rotated BRIEF）检测关键点，兼具 FAST 的速度"
            "与 BRIEF 的描述能力，适合实时匹配。输出关键点标注图、关键点集合与数量。"
        ),
        keywords=("orb", "关键点", "特征点", "fast", "brief"),
        inputs=(_INPUT, _MASK),
        outputs=_OUTPUTS,
        params=_keypoint_params(
            ParamSpec(
                "nfeatures",
                "特征点数量",
                "int",
                default=500,
                min=0,
                max=5000,
                step=1,
                tooltip="保留响应最高的前 N 个特征点，0 表示不限制。",
            ),
            ParamSpec(
                "scale",
                "缩放系数",
                "float",
                default=1.2,
                min=1.01,
                max=4.0,
                step=0.01,
            ),
            ParamSpec(
                "nlevels",
                "金字塔层数",
                "int",
                default=8,
                min=1,
                max=16,
                step=1,
            ),
            ParamSpec(
                "edge_threshold",
                "边界阈值",
                "int",
                default=31,
                min=1,
                max=100,
                step=1,
            ),
            ParamSpec(
                "wta_k",
                "描述子点数",
                "enum",
                choices=((2, "2 点"), (3, "3 点"), (4, "4 点")),
                default=2,
            ),
            ParamSpec(
                "score",
                "评分方式",
                "enum",
                choices=(("harris", "Harris"), ("fast", "FAST")),
                default="harris",
            ),
            ParamSpec(
                "patch_size",
                "描述子块大小",
                "int",
                default=31,
                min=1,
                max=100,
                step=1,
            ),
            ParamSpec(
                "fast_threshold",
                "FAST 阈值",
                "int",
                default=20,
                min=1,
                max=255,
                step=1,
            ),
        ),
    )

    def process(
        self,
        image=None,
        mask=None,
        nfeatures: int = 500,
        scale: float = 1.2,
        nlevels: int = 8,
        edge_threshold: int = 31,
        wta_k: int = 2,
        score: str = "harris",
        patch_size: int = 31,
        fast_threshold: int = 20,
        color: str = "#00ff66",
        show_size: bool = True,
        max_points: int = 0,
        **_: Any,
    ):
        gray = _gray(image)
        mask_array = _mask_of(mask, gray)
        score_flag = (
            cv2.ORB_HARRIS_SCORE if score == "harris" else cv2.ORB_FAST_SCORE
        )
        keypoints = _run_detector(
            lambda: cv2.ORB_create(
                nfeatures=int(nfeatures),
                scaleFactor=float(scale),
                nlevels=int(nlevels),
                edgeThreshold=int(edge_threshold),
                firstLevel=0,
                WTA_K=int(wta_k),
                scoreType=score_flag,
                patchSize=int(patch_size),
                fastThreshold=int(fast_threshold),
            ),
            gray,
            mask_array,
            "ORB 关键点",
        )
        keypoints = _limit(_filter_by_mask(keypoints, mask_array), max_points)
        return _finish(image, keypoints, gray, color, show_size)


# ---------------------------------------------------------------------------
# AKAZE
# ---------------------------------------------------------------------------
class AkazeKeypoints(Operator):
    """AKAZE 关键点检测。"""

    spec = OperatorSpec(
        id="keypoint.akaze",
        name="AKAZE 关键点",
        category=("特征与分析", "关键点"),
        description=(
            "用 AKAZE（加速的 KAZE）检测非线性尺度空间关键点，"
            "在保持稳定性的同时速度接近 ORB。输出关键点标注图、关键点集合与数量。"
        ),
        keywords=("akaze", "关键点", "特征点", "非线性尺度"),
        inputs=(_INPUT, _MASK),
        outputs=_OUTPUTS,
        params=_keypoint_params(
            ParamSpec(
                "descriptor",
                "描述子类型",
                "enum",
                choices=(("mlb", "MLDB"), ("upright", "MLDB 无旋转")),
                default="mlb",
            ),
            ParamSpec(
                "descriptor_size",
                "描述子长度",
                "int",
                default=0,
                min=0,
                max=512,
                step=1,
                tooltip="0 表示使用默认长度。",
            ),
            ParamSpec(
                "descriptor_channels",
                "描述子通道数",
                "int",
                default=3,
                min=1,
                max=3,
                step=1,
            ),
            ParamSpec(
                "threshold",
                "检测阈值",
                "float",
                default=0.001,
                min=0.0001,
                max=1.0,
                step=0.0001,
            ),
            ParamSpec(
                "n_octaves",
                "金字塔层数",
                "int",
                default=4,
                min=1,
                max=10,
                step=1,
            ),
            ParamSpec(
                "n_octave_layers",
                "每层子层数",
                "int",
                default=4,
                min=1,
                max=10,
                step=1,
            ),
        ),
    )

    def process(
        self,
        image=None,
        mask=None,
        descriptor: str = "mlb",
        descriptor_size: int = 0,
        descriptor_channels: int = 3,
        threshold: float = 0.001,
        n_octaves: int = 4,
        n_octave_layers: int = 4,
        color: str = "#00ff66",
        show_size: bool = True,
        max_points: int = 0,
        **_: Any,
    ):
        gray = _gray(image)
        mask_array = _mask_of(mask, gray)
        descriptor_flag = (
            cv2.AKAZE_DESCRIPTOR_MLDB
            if descriptor == "mlb"
            else cv2.AKAZE_DESCRIPTOR_MLDB_UPRIGHT
        )
        keypoints = _run_detector(
            lambda: cv2.AKAZE_create(
                descriptor_type=descriptor_flag,
                descriptor_size=int(descriptor_size),
                descriptor_channels=int(descriptor_channels),
                threshold=float(threshold),
                nOctaves=int(n_octaves),
                nOctaveLayers=int(n_octave_layers),
            ),
            gray,
            mask_array,
            "AKAZE 关键点",
        )
        keypoints = _limit(_filter_by_mask(keypoints, mask_array), max_points)
        return _finish(image, keypoints, gray, color, show_size)


# ---------------------------------------------------------------------------
# BRISK
# ---------------------------------------------------------------------------
class BriskKeypoints(Operator):
    """BRISK 关键点检测。"""

    spec = OperatorSpec(
        id="keypoint.brisk",
        name="BRISK 关键点",
        category=("特征与分析", "关键点"),
        description=(
            "用 BRISK（Binary Robust Invariant Scalable Keypoints）检测关键点，"
            "使用二进制描述子，匹配速度快。输出关键点标注图、关键点集合与数量。"
        ),
        keywords=("brisk", "关键点", "特征点", "二进制描述子"),
        inputs=(_INPUT, _MASK),
        outputs=_OUTPUTS,
        params=_keypoint_params(
            ParamSpec(
                "threshold",
                "检测阈值",
                "int",
                default=30,
                min=0,
                max=255,
                step=1,
            ),
            ParamSpec(
                "octaves",
                "金字塔层数",
                "int",
                default=3,
                min=0,
                max=10,
                step=1,
            ),
            ParamSpec(
                "pattern_scale",
                "采样模式尺度",
                "float",
                default=1.0,
                min=0.1,
                max=10.0,
                step=0.1,
            ),
        ),
    )

    def process(
        self,
        image=None,
        mask=None,
        threshold: int = 30,
        octaves: int = 3,
        pattern_scale: float = 1.0,
        color: str = "#00ff66",
        show_size: bool = True,
        max_points: int = 0,
        **_: Any,
    ):
        gray = _gray(image)
        mask_array = _mask_of(mask, gray)
        keypoints = _run_detector(
            lambda: cv2.BRISK_create(
                thresh=int(threshold),
                octaves=int(octaves),
                patternScale=float(pattern_scale),
            ),
            gray,
            mask_array,
            "BRISK 关键点",
        )
        keypoints = _limit(_filter_by_mask(keypoints, mask_array), max_points)
        return _finish(image, keypoints, gray, color, show_size)


# ---------------------------------------------------------------------------
# KAZE
# ---------------------------------------------------------------------------
class KazeKeypoints(Operator):
    """KAZE 关键点检测。"""

    spec = OperatorSpec(
        id="keypoint.kaze",
        name="KAZE 关键点",
        category=("特征与分析", "关键点"),
        description=(
            "用 KAZE 在非线性尺度空间中检测关键点，对模糊和缩放更稳健，"
            "但速度比 AKAZE 慢。输出关键点标注图、关键点集合与数量。"
        ),
        keywords=("kaze", "关键点", "特征点", "非线性尺度"),
        inputs=(_INPUT, _MASK),
        outputs=_OUTPUTS,
        params=_keypoint_params(
            ParamSpec(
                "threshold",
                "检测阈值",
                "float",
                default=0.001,
                min=0.0001,
                max=1.0,
                step=0.0001,
            ),
            ParamSpec(
                "n_octaves",
                "金字塔层数",
                "int",
                default=4,
                min=1,
                max=10,
                step=1,
            ),
            ParamSpec(
                "n_octave_layers",
                "每层子层数",
                "int",
                default=4,
                min=1,
                max=10,
                step=1,
            ),
            ParamSpec(
                "diffusivity",
                "扩散方式",
                "enum",
                choices=(
                    ("pm_g1", "PM G1"),
                    ("pm_g2", "PM G2"),
                    ("weickert", "Weickert"),
                    ("charard", "Charbonnier"),
                ),
                default="pm_g2",
            ),
        ),
    )

    def process(
        self,
        image=None,
        mask=None,
        threshold: float = 0.001,
        n_octaves: int = 4,
        n_octave_layers: int = 4,
        diffusivity: str = "pm_g2",
        color: str = "#00ff66",
        show_size: bool = True,
        max_points: int = 0,
        **_: Any,
    ):
        gray = _gray(image)
        mask_array = _mask_of(mask, gray)
        diffusivity_flags = {
            "pm_g1": cv2.KAZE_DIFF_PM_G1,
            "pm_g2": cv2.KAZE_DIFF_PM_G2,
            "weickert": cv2.KAZE_DIFF_WEICKERT,
            "charard": cv2.KAZE_DIFF_CHARBONNIER,
        }
        if diffusivity not in diffusivity_flags:
            raise OperatorError(f"不支持的扩散方式：{diffusivity}")
        keypoints = _run_detector(
            lambda: cv2.KAZE_create(
                extended=False,
                upright=False,
                threshold=float(threshold),
                nOctaves=int(n_octaves),
                nOctaveLayers=int(n_octave_layers),
                diffusivity=diffusivity_flags[diffusivity],
            ),
            gray,
            mask_array,
            "KAZE 关键点",
        )
        keypoints = _limit(_filter_by_mask(keypoints, mask_array), max_points)
        return _finish(image, keypoints, gray, color, show_size)


# ---------------------------------------------------------------------------
# MSER
# ---------------------------------------------------------------------------
class MserRegions(Operator):
    """MSER 最大稳定极值区域。"""

    spec = OperatorSpec(
        id="keypoint.mser",
        name="MSER 区域",
        category=("特征与分析", "关键点"),
        description=(
            "用 MSER（Maximally Stable Extremal Regions）检测灰度稳定的斑块区域，"
            "适合文字与色块检测。输出区域标注图、区域质心点、区域轮廓与数量。"
        ),
        keywords=("mser", "稳定区域", "区域", "关键点", "文字"),
        inputs=(_INPUT, _MASK),
        outputs=(
            PortSpec("image", "区域标注图", accepts=("BGR",)),
            PortSpec("points", "区域质心", data_type=DataType.EDGES),
            PortSpec("regions", "区域轮廓", data_type=DataType.CONTOURS),
            PortSpec("count", "区域数", data_type=DataType.NUMBER),
        ),
        params=_keypoint_params(
            ParamSpec(
                "delta",
                "灰度步长",
                "int",
                default=5,
                min=1,
                max=50,
                step=1,
            ),
            ParamSpec(
                "min_area",
                "最小面积",
                "int",
                default=60,
                min=1,
                max=100000,
                step=1,
            ),
            ParamSpec(
                "max_area",
                "最大面积",
                "int",
                default=14400,
                min=1,
                max=1000000,
                step=1,
            ),
            ParamSpec(
                "max_variation",
                "最大变化率",
                "float",
                default=0.25,
                min=0.01,
                max=1.0,
                step=0.01,
            ),
            ParamSpec(
                "min_diversity",
                "最小差异度",
                "float",
                default=0.2,
                min=0.0,
                max=1.0,
                step=0.01,
            ),
            ParamSpec(
                "max_regions",
                "最大区域数",
                "int",
                default=200,
                min=1,
                max=2000,
                step=1,
            ),
        ),
    )

    def process(
        self,
        image=None,
        mask=None,
        delta: int = 5,
        min_area: int = 60,
        max_area: int = 14400,
        max_variation: float = 0.25,
        min_diversity: float = 0.2,
        max_regions: int = 200,
        color: str = "#00ff66",
        show_size: bool = True,
        max_points: int = 0,
        **_: Any,
    ):
        gray = _gray(image)
        mask_array = _mask_of(mask, gray)
        try:
            detector = cv2.MSER_create(
                int(delta),
                int(min_area),
                int(max_area),
                float(max_variation),
                float(min_diversity),
            )
            regions, _boxes = detector.detectRegions(gray)
        except cv2.error as exc:
            raise OperatorError(
                "MSER 区域检测失败：当前参数组合不受当前 OpenCV 支持，请调整参数后重试。"
            ) from exc

        keypoints = []
        region_polygons = []
        for region in list(regions)[: int(max_regions)]:
            points = np.asarray(region).reshape(-1, 2)
            if points.size == 0:
                continue
            centroid_x = float(points[:, 0].mean())
            centroid_y = float(points[:, 1].mean())
            keypoints.append(
                cv2.KeyPoint(
                    centroid_x,
                    centroid_y, max(1.0, float(np.sqrt(len(points))))
                )
            )
            region_polygons.append(points.astype(np.int32))

        mask_filtered = _filter_by_mask(keypoints, mask_array)
        keep = {id(keypoint) for keypoint in mask_filtered}
        paired = [
            (keypoint, polygon)
            for keypoint, polygon in zip(keypoints, region_polygons)
            if id(keypoint) in keep
        ]
        if int(max_points) > 0:
            paired = paired[: int(max_points)]
        keypoints = [keypoint for keypoint, _polygon in paired]
        region_polygons = [polygon for _keypoint, polygon in paired]

        height, width = gray.shape[:2]
        return {
            "image": _draw(image, keypoints, color, show_size),
            "points": _points_output(keypoints, width, height),
            "regions": ContoursData(
                contours=tuple(region_polygons),
                image_size=(width, height),
                label="MSER 区域",
            ),
            "count": NumberValue(value=len(keypoints), label="区域数", unit="个"),
        }


# ---------------------------------------------------------------------------
# Blob
# ---------------------------------------------------------------------------
class BlobKeypoints(Operator):
    """SimpleBlobDetector 斑点检测。"""

    spec = OperatorSpec(
        id="keypoint.blob",
        name="斑点检测",
        category=("特征与分析", "关键点"),
        description=(
            "用 SimpleBlobDetector 检测斑点：先按多阈值二值化，再按面积、圆度、"
            "惯性比和凸度过滤。输出斑点标注图、斑点集合与数量。"
        ),
        keywords=("blob", "斑点", "检测", "simpleblob", "关键点"),
        inputs=(_INPUT, _MASK),
        outputs=_OUTPUTS,
        params=_keypoint_params(
            ParamSpec(
                "min_threshold",
                "最小阈值",
                "int",
                default=10,
                min=0,
                max=255,
                step=1,
            ),
            ParamSpec(
                "max_threshold",
                "最大阈值",
                "int",
                default=200,
                min=0,
                max=255,
                step=1,
            ),
            ParamSpec(
                "threshold_step",
                "阈值步长",
                "float",
                default=10.0,
                min=1.0,
                max=50.0,
                step=1.0,
            ),
            ParamSpec("filter_by_area", "按面积过滤", "bool", default=True),
            ParamSpec(
                "min_area",
                "最小面积",
                "float",
                default=25.0,
                min=0.0,
                max=1000000.0,
                step=1.0,
                visible_when="filter_by_area",
                enabled_when="filter_by_area",
            ),
            ParamSpec(
                "max_area",
                "最大面积",
                "float",
                default=5000.0,
                min=0.0,
                max=1000000.0,
                step=1.0,
                visible_when="filter_by_area",
                enabled_when="filter_by_area",
            ),
            ParamSpec("filter_by_circularity", "按圆度过滤", "bool", default=False),
            ParamSpec(
                "min_circularity",
                "最小圆度",
                "float",
                default=0.1,
                min=0.0,
                max=1.0,
                step=0.01,
                visible_when="filter_by_circularity",
                enabled_when="filter_by_circularity",
            ),
            ParamSpec("filter_by_inertia", "按惯性比过滤", "bool", default=False),
            ParamSpec(
                "min_inertia",
                "最小惯性比",
                "float",
                default=0.1,
                min=0.0,
                max=1.0,
                step=0.01,
                visible_when="filter_by_inertia",
                enabled_when="filter_by_inertia",
            ),
            ParamSpec("filter_by_convexity", "按凸度过滤", "bool", default=False),
            ParamSpec(
                "min_convexity",
                "最小凸度",
                "float",
                default=0.5,
                min=0.0,
                max=1.0,
                step=0.01,
                visible_when="filter_by_convexity",
                enabled_when="filter_by_convexity",
            ),
            ParamSpec(
                "min_distance",
                "斑点最小间距",
                "float",
                default=10.0,
                min=0.0,
                max=1000.0,
                step=1.0,
                unit="px",
            ),
        ),
    )

    def validate(self, inputs, params) -> None:
        if int(params["max_threshold"]) < int(params["min_threshold"]):
            raise OperatorError("斑点的最大阈值不能小于最小阈值。")
        if float(params["max_area"]) < float(params["min_area"]):
            raise OperatorError("斑点的最大面积不能小于最小面积。")

    def process(
        self,
        image=None,
        mask=None,
        min_threshold: int = 10,
        max_threshold: int = 200,
        threshold_step: float = 10.0,
        filter_by_area: bool = True,
        min_area: float = 25.0,
        max_area: float = 5000.0,
        filter_by_circularity: bool = False,
        min_circularity: float = 0.1,
        filter_by_inertia: bool = False,
        min_inertia: float = 0.1,
        filter_by_convexity: bool = False,
        min_convexity: float = 0.5,
        min_distance: float = 10.0,
        color: str = "#00ff66",
        show_size: bool = True,
        max_points: int = 0,
        **_: Any,
    ):
        if int(max_threshold) < int(min_threshold):
            raise OperatorError("斑点的最大阈值不能小于最小阈值。")
        if float(max_area) < float(min_area):
            raise OperatorError("斑点的最大面积不能小于最小面积。")
        gray = _gray(image)
        mask_array = _mask_of(mask, gray)

        detector_params = cv2.SimpleBlobDetector_Params()
        detector_params.minThreshold = float(min_threshold)
        detector_params.maxThreshold = float(max_threshold)
        detector_params.thresholdStep = float(threshold_step)
        detector_params.filterByArea = bool(filter_by_area)
        detector_params.minArea = float(min_area)
        detector_params.maxArea = float(max_area)
        detector_params.filterByCircularity = bool(filter_by_circularity)
        detector_params.minCircularity = float(min_circularity)
        detector_params.filterByInertia = bool(filter_by_inertia)
        detector_params.minInertiaRatio = float(min_inertia)
        detector_params.filterByConvexity = bool(filter_by_convexity)
        detector_params.minConvexity = float(min_convexity)
        detector_params.minDistBetweenBlobs = float(min_distance)
        detector_params.filterByColor = False

        keypoints = _run_detector(
            lambda: cv2.SimpleBlobDetector_create(detector_params),
            gray,
            mask_array,
            "斑点",
        )
        keypoints = _limit(_filter_by_mask(keypoints, mask_array), max_points)
        return _finish(image, keypoints, gray, color, show_size)


# ---------------------------------------------------------------------------
# 关键点绘制
# ---------------------------------------------------------------------------
class DrawKeypoints(Operator):
    """用统一样式绘制指定检测器的关键点。"""

    spec = OperatorSpec(
        id="keypoint.draw",
        name="关键点绘制",
        category=("特征与分析", "关键点"),
        description=(
            "用选定的检测器（FAST / ORB / SIFT / AKAZE / BRISK / 优质角点 / Harris）"
            "检测关键点，并用统一样式绘制到彩色图上；可放大关键点尺度便于观察。"
            "输出标注图、关键点集合与数量。"
        ),
        keywords=("draw", "绘制", "关键点", "可视化", "keypoint"),
        inputs=(_INPUT, _MASK),
        outputs=_OUTPUTS,
        params=_keypoint_params(
            ParamSpec(
                "method",
                "检测器",
                "enum",
                choices=(
                    ("fast", "FAST"),
                    ("orb", "ORB"),
                    ("sift", "SIFT"),
                    ("akaze", "AKAZE"),
                    ("brisk", "BRISK"),
                    ("good_features", "优质角点"),
                    ("harris", "Harris 角点"),
                ),
                default="orb",
            ),
            ParamSpec(
                "max_features",
                "最大特征数",
                "int",
                default=500,
                min=0,
                max=5000,
                step=1,
                tooltip="0 表示不限制。",
            ),
            ParamSpec(
                "point_scale",
                "关键点放大系数",
                "float",
                default=1.0,
                min=0.1,
                max=10.0,
                step=0.1,
                tooltip="放大绘制关键点的尺度，便于观察小关键点。",
            ),
        ),
    )

    def _detect(
        self,
        method: str,
        gray: np.ndarray,
        max_features: int,
        mask_array: np.ndarray | None,
    ):
        if method == "fast":
            return _run_detector(
                lambda: cv2.FastFeatureDetector_create(
                    10, True, cv2.FAST_FEATURE_DETECTOR_TYPE_9_16
                ),
                gray,
                mask_array,
                "FAST 角点",
            )
        if method == "orb":
            return _run_detector(
                lambda: cv2.ORB_create(nfeatures=int(max_features)),
                gray,
                mask_array,
                "ORB 关键点",
            )
        if method == "sift":
            return _run_detector(
                lambda: cv2.SIFT_create(nfeatures=int(max_features)),
                gray,
                mask_array,
                "SIFT 关键点",
            )
        if method == "akaze":
            return _run_detector(
                lambda: cv2.AKAZE_create(), gray, mask_array, "AKAZE 关键点"
            )
        if method == "brisk":
            return _run_detector(
                lambda: cv2.BRISK_create(), gray, mask_array, "BRISK 关键点"
            )
        if method == "good_features":
            try:
                corners = cv2.goodFeaturesToTrack(
                    gray,
                    maxCorners=int(max_features),
                    qualityLevel=0.01,
                    minDistance=10,
                    mask=mask_array,
                )
            except cv2.error as exc:
                raise OperatorError(
                    "绘制关键点失败：优质角点检测参数不受当前 OpenCV 支持。"
                ) from exc
            if corners is None:
                return []
            return [
                cv2.KeyPoint(float(corner[0][0]), float(corner[0][1]), 7.0)
                for corner in corners
            ]
        if method == "harris":
            return _harris_keypoints(gray, 5, 3, 0.04, 0.01)
        raise OperatorError(f"不支持的关键点检测器：{method}")

    def process(
        self,
        image=None,
        mask=None,
        method: str = "orb",
        max_features: int = 500,
        point_scale: float = 1.0,
        color: str = "#00ff66",
        show_size: bool = True,
        max_points: int = 0,
        **_: Any,
    ):
        gray = _gray(image)
        mask_array = _mask_of(mask, gray)
        keypoints = self._detect(method, gray, max_features, mask_array)
        keypoints = _limit(_filter_by_mask(keypoints, mask_array), max_points)
        height, width = gray.shape[:2]
        return {
            "image": _draw(image, keypoints, color, show_size, scale=float(point_scale)),
            "points": _points_output(keypoints, width, height),
            "count": NumberValue(value=len(keypoints), label="关键点数", unit="个"),
        }


OPERATORS = (
    CornerHarris,
    CornerMinEigen,
    GoodFeatures,
    FastKeypoints,
    SiftKeypoints,
    OrbKeypoints,
    AkazeKeypoints,
    BriskKeypoints,
    KazeKeypoints,
    MserRegions,
    BlobKeypoints,
    DrawKeypoints,
)
