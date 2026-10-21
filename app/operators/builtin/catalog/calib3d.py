"""OpenCV 目录算子：立体与标定。

包含棋盘格角点、镜头畸变校正、双目视差、深度图、全景拼接与单应变换 6 个算子。

约定
----
* 双图算子的 ``image2`` 均可选，未连接时用第一幅图水平平移派生，保证默认参数可运行；
* 立体匹配的视差范围 / 块大小会按图像尺寸自动收缩，小图也能运行；
* 主输出为图像（视差、深度以热力图呈现），额外输出数值统计。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.core.data_types import DataType, NumberValue
from app.operators.builtin.catalog import _cv
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec

#: 第二幅图未连接时的默认水平平移像素数。
DEFAULT_SHIFT = 3


# ---------------------------------------------------------------------------
# 共享工具
# ---------------------------------------------------------------------------
def _derived(base: np.ndarray, shift: int = DEFAULT_SHIFT) -> np.ndarray:
    return np.roll(base, max(int(shift), 1), axis=1).copy()


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


def _stereo_range(width: int, num_disparities: int) -> int:
    """把视差范围裁剪为「小于图像宽度」的 16 的倍数。"""
    available = ((int(width) - 1) // 16) * 16
    if available < 16:
        raise OperatorError("图像宽度太小，无法进行立体匹配（至少需要 17 像素宽）。")
    return max(16, min(int(num_disparities) // 16 * 16, available))


def _stereo_block(block_size: int, width: int, height: int) -> int:
    """把匹配块大小裁剪为不超过图像尺寸的奇数。"""
    limit = max(3, min(int(width), int(height)))
    block = max(3, min(int(block_size), limit))
    if block % 2 == 0:
        block -= 1
    return max(block, 3)


def _heat_map(values: np.ndarray, mask: np.ndarray | None = None, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """把一维数值图按有效区域分位数归一化后渲染为 BGR 热力图。"""
    work = np.asarray(values, dtype=np.float32)
    valid = np.isfinite(work)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    upper = float(np.percentile(work[valid], 98)) if valid.any() else 1.0
    upper = max(upper, 1e-6)
    scaled = np.clip(np.where(valid, work, 0.0) / upper * 255.0, 0.0, 255.0)
    gray = np.rint(scaled).astype(np.uint8)
    return cv2.applyColorMap(gray, colormap)


def _orb_match(source: np.ndarray, target: np.ndarray, max_features: int):
    """用 ORB + Lowe 比值筛选 + RANSAC 估计 ``source → target`` 的单应矩阵。"""
    orb = cv2.ORB_create(nfeatures=max(int(max_features), 100))
    gray_source = cv2.cvtColor(source, cv2.COLOR_BGR2GRAY)
    gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
    key_source, desc_source = orb.detectAndCompute(gray_source, None)
    key_target, desc_target = orb.detectAndCompute(gray_target, None)
    if (
        desc_source is None
        or desc_target is None
        or len(key_source) < 8
        or len(key_target) < 8
    ):
        return None, 0
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    pairs = matcher.knnMatch(desc_source, desc_target, k=2)
    good = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance]
    if len(good) < 8:
        return None, len(good)
    points_source = np.float32([key_source[item.queryIdx].pt for item in good]).reshape(-1, 1, 2)
    points_target = np.float32([key_target[item.trainIdx].pt for item in good]).reshape(-1, 1, 2)
    homography, inliers = cv2.findHomography(points_source, points_target, cv2.RANSAC, 4.0)
    if homography is None:
        return None, len(good)
    count = int(inliers.sum()) if inliers is not None else len(good)
    if count < 8:
        return None, count
    return homography.astype(np.float32), count


def _trapezoid_homography(width: int, height: int, inset: float) -> np.ndarray:
    """生成默认的梯形单应矩阵：把上边缘向中间收拢。"""
    ratio = float(min(max(inset, 0.0), 0.45))
    source = np.float32(
        [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
    )
    target = np.float32(
        [
            [width * ratio, 0],
            [width * (1.0 - ratio), 0],
            [width - 1, height - 1],
            [0, height - 1],
        ]
    )
    return cv2.getPerspectiveTransform(source, target)


# ---------------------------------------------------------------------------
# 1. 棋盘格角点
# ---------------------------------------------------------------------------
class CalibChessboard(Operator):
    spec = OperatorSpec(
        id="calib.chessboard",
        name="棋盘格角点",
        category=("立体与标定", "标定"),
        description=(
            "在灰度图上检测棋盘格内角点并绘制出来，可选亚像素精化。"
            "内角点数量即棋盘格每行 / 每列的「格子数减一」。未检测到时输出原图副本与角点数 0。"
            "输出标注图与角点数量，默认参数即可运行。"
        ),
        keywords=("棋盘格", "角点", "chessboard", "标定", "calibration", "corner"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(
            PortSpec("image", "角点标注图", accepts=("BGR",)),
            PortSpec("count", "角点数量", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec("columns", "内角点列数", "int", default=9, min=2, max=50, step=1, group="棋盘格"),
            ParamSpec("rows", "内角点行数", "int", default=6, min=2, max=50, step=1, group="棋盘格"),
            ParamSpec(
                "refine",
                "亚像素精化",
                "bool",
                default=True,
                group="棋盘格",
                tooltip="找到角点后使用 cornerSubPix 提升精度。",
            ),
            ParamSpec("color", "标注颜色", "color", default="#00d26a", group="显示"),
        ),
    )

    def process(
        self,
        image=None,
        columns: int = 9,
        rows: int = 6,
        refine: bool = True,
        color: str = "#00d26a",
        **_: Any,
    ):
        gray = _cv.source_gray8(image, "输入图像")
        canvas = _cv.source_bgr8(image, "输入图像")
        pattern = (int(columns), int(rows))
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, pattern, flags=flags)
        count = 0
        if found and corners is not None:
            if bool(refine):
                criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.1)
                corners = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria)
            cv2.drawChessboardCorners(canvas, pattern, corners, found)
            count = int(len(corners))
        else:
            cv2.putText(
                canvas,
                "No chessboard corners",
                (8, max(min(canvas.shape[0] // 2, 24), 12)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                _cv.draw_color(canvas, color),
                1,
                _cv.draw_type(canvas),
            )
        return {
            "image": canvas,
            "count": NumberValue(value=float(count), label="角点数量", unit="个"),
        }


# ---------------------------------------------------------------------------
# 2. 畸变校正
# ---------------------------------------------------------------------------
class CalibUndistort(Operator):
    spec = OperatorSpec(
        id="calib.undistort",
        name="镜头畸变校正",
        category=("立体与标定", "标定"),
        description=(
            "按径向畸变系数 k1 / k2 与等效焦距构造相机内参，生成校正映射并重采样图像。"
            "等效焦距为 0 时自动取图像长边，默认 k1=-0.2 可以看到桶形校正效果。"
            "支持 L / BGR / BGRA 与 uint8 / uint16 / float32，输出与输入同类型。"
        ),
        keywords=("畸变", "畸变校正", "undistort", "k1", "镜头", "标定"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(
            PortSpec("image", "校正图"),
            PortSpec("focal", "等效焦距", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec("k1", "径向畸变 k1", "float", default=-0.2, min=-2.0, max=2.0, step=0.01, group="畸变"),
            ParamSpec("k2", "径向畸变 k2", "float", default=0.0, min=-2.0, max=2.0, step=0.01, group="畸变"),
            ParamSpec(
                "focal",
                "等效焦距",
                "float",
                default=0.0,
                min=0.0,
                max=20000.0,
                step=10.0,
                unit="px",
                group="相机",
                tooltip="0 表示自动取图像长边。",
            ),
        ),
    )

    def process(self, image=None, k1: float = -0.2, k2: float = 0.0, focal: float = 0.0, **_: Any):
        base = _cv.source(image, "输入图像")
        height, width = base.shape[:2]
        focal_length = float(focal) if float(focal) > 0.0 else float(max(width, height))
        camera = np.array(
            [
                [focal_length, 0.0, (width - 1) / 2.0],
                [0.0, focal_length, (height - 1) / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        distortion = np.array([float(k1), float(k2), 0.0, 0.0, 0.0], dtype=np.float64)
        map_x, map_y = cv2.initUndistortRectifyMap(
            camera,
            distortion,
            None,
            camera,
            (width, height),
            cv2.CV_32FC1,
        )
        corrected = cv2.remap(
            base,
            map_x,
            map_y,
            cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        return {
            "image": corrected,
            "focal": NumberValue(value=focal_length, label="等效焦距", unit="px"),
        }


# ---------------------------------------------------------------------------
# 3. 双目视差
# ---------------------------------------------------------------------------
class CalibStereoDisparity(Operator):
    spec = OperatorSpec(
        id="calib.stereo_disparity",
        name="立体视差",
        category=("立体与标定", "立体视觉"),
        description=(
            "用 StereoBM 计算左右视图的视差图并渲染为热力图。视差范围与匹配块大小会按图像尺寸"
            "自动收缩，小图也能运行；未连接右视图时自动平移左视图派生。输出视差热力图与平均视差（像素）。"
        ),
        keywords=("视差", "立体匹配", "StereoBM", "disparity", "双目", "深度"),
        inputs=(
            PortSpec("image", "左视图"),
            PortSpec("image2", "右视图（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "视差热力图", accepts=("BGR",)),
            PortSpec("mean_disparity", "平均视差", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "num_disparities",
                "视差范围",
                "int",
                default=64,
                min=16,
                max=512,
                step=16,
                unit="px",
                tooltip="必须是 16 的倍数，自动收缩到小于图像宽度。",
            ),
            ParamSpec(
                "block_size",
                "匹配块大小",
                "int",
                default=15,
                min=3,
                max=51,
                step=2,
                odd=True,
                tooltip="必须为奇数；图像较小时会自动收缩。",
            ),
        ),
    )

    def process(self, image=None, image2=None, num_disparities: int = 64, block_size: int = 15, **_: Any):
        left = _cv.source_gray8(image, "左视图")
        right = _second_gray8(image2, left)
        height, width = left.shape[:2]
        disparities = _stereo_range(width, int(num_disparities))
        block = _stereo_block(int(block_size), width, height)
        matcher = cv2.StereoBM_create(numDisparities=disparities, blockSize=block)
        disparity = matcher.compute(left, right).astype(np.float32) / 16.0
        valid = disparity > 0.0
        mean_disparity = float(disparity[valid].mean()) if valid.any() else 0.0
        return {
            "image": _heat_map(disparity, valid),
            "mean_disparity": NumberValue(value=mean_disparity, label="平均视差", unit="px"),
        }


# ---------------------------------------------------------------------------
# 4. 深度图
# ---------------------------------------------------------------------------
class CalibStereoDepth(Operator):
    spec = OperatorSpec(
        id="calib.stereo_depth",
        name="立体深度图",
        category=("立体与标定", "立体视觉"),
        description=(
            "用 StereoSGBM 计算视差，再按 Z = f × baseline / disparity 换算深度并渲染为热力图。"
            "焦距为 0 时自动取图像宽度，未连接右视图时自动平移左视图派生。"
            "输出深度热力图与平均深度（米）。"
        ),
        keywords=("深度", "深度图", "StereoSGBM", "depth", "双目", "视差"),
        inputs=(
            PortSpec("image", "左视图"),
            PortSpec("image2", "右视图（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "深度热力图", accepts=("BGR",)),
            PortSpec("mean_depth", "平均深度", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec("num_disparities", "视差范围", "int", default=64, min=16, max=512, step=16, unit="px"),
            ParamSpec("block_size", "匹配块大小", "int", default=15, min=3, max=51, step=2, odd=True),
            ParamSpec(
                "baseline",
                "基线距离",
                "float",
                default=0.1,
                min=0.001,
                max=10.0,
                step=0.01,
                unit="m",
                group="相机",
            ),
            ParamSpec(
                "focal",
                "焦距",
                "float",
                default=0.0,
                min=0.0,
                max=20000.0,
                step=10.0,
                unit="px",
                group="相机",
                tooltip="0 表示自动取图像宽度。",
            ),
            ParamSpec(
                "max_depth",
                "最大深度",
                "float",
                default=100.0,
                min=0.1,
                max=10000.0,
                step=1.0,
                unit="m",
                group="相机",
                tooltip="大于该值的深度会被截断。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        image2=None,
        num_disparities: int = 64,
        block_size: int = 15,
        baseline: float = 0.1,
        focal: float = 0.0,
        max_depth: float = 100.0,
        **_: Any,
    ):
        left = _cv.source_gray8(image, "左视图")
        right = _second_gray8(image2, left)
        height, width = left.shape[:2]
        disparities = _stereo_range(width, int(num_disparities))
        block = _stereo_block(int(block_size), width, height)
        matcher = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=disparities,
            blockSize=block,
        )
        disparity = matcher.compute(left, right).astype(np.float32) / 16.0
        valid = disparity > 0.0
        focal_length = float(focal) if float(focal) > 0.0 else float(width)
        depth = np.zeros_like(disparity)
        if valid.any():
            depth[valid] = focal_length * float(baseline) / disparity[valid]
            depth = np.clip(depth, 0.0, float(max_depth))
        mean_depth = float(depth[valid].mean()) if valid.any() else 0.0
        return {
            "image": _heat_map(depth, valid, cv2.COLORMAP_TURBO),
            "mean_depth": NumberValue(value=mean_depth, label="平均深度", unit="m"),
        }


# ---------------------------------------------------------------------------
# 5. 全景拼接
# ---------------------------------------------------------------------------
class CalibPanorama(Operator):
    spec = OperatorSpec(
        id="calib.panorama",
        name="全景拼接",
        category=("立体与标定", "拼接与变换"),
        description=(
            "用 ORB 特征匹配 + RANSAC 单应矩阵把第二幅图对齐到第一幅图的画布上拼接成全景。"
            "特征不足或匹配失败时退回左右并排拼接，保证默认参数一定有输出。"
            "输出拼接图与内点匹配数量。"
        ),
        keywords=("全景", "拼接", "panorama", "stitch", "ORB", "单应"),
        inputs=(
            PortSpec("image", "左图"),
            PortSpec("image2", "右图（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "拼接图", accepts=("BGR",)),
            PortSpec("inliers", "内点数量", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "mode",
                "拼接方式",
                "enum",
                choices=(("auto", "自动（特征优先）"), ("side_by_side", "并排拼接")),
                default="auto",
            ),
            ParamSpec("max_features", "最大特征点数", "int", default=800, min=100, max=5000, step=100),
        ),
    )

    def process(self, image=None, image2=None, mode: str = "auto", max_features: int = 800, **_: Any):
        left = _cv.source_bgr8(image, "左图")
        right = _second_bgr8(image2, left)
        height, width = left.shape[:2]
        if mode == "auto":
            homography, inliers = _orb_match(right, left, int(max_features))
            if homography is not None:
                warped = cv2.warpPerspective(
                    right,
                    homography,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                )
                coverage = cv2.warpPerspective(
                    np.full(right.shape[:2], 255, dtype=np.uint8),
                    homography,
                    (width, height),
                )
                canvas = left.copy()
                canvas[coverage > 0] = warped[coverage > 0]
                return {
                    "image": canvas,
                    "inliers": NumberValue(value=float(inliers), label="内点数量", unit="个"),
                }
        canvas = np.hstack([left, right])
        return {
            "image": np.ascontiguousarray(canvas),
            "inliers": NumberValue(value=0.0, label="内点数量", unit="个"),
        }


# ---------------------------------------------------------------------------
# 6. 单应变换
# ---------------------------------------------------------------------------
class CalibHomography(Operator):
    spec = OperatorSpec(
        id="calib.homography",
        name="单应变换",
        category=("立体与标定", "拼接与变换"),
        description=(
            "对图像施加透视（单应）变换。未连接第二幅图时按「上边缘收拢比例」生成梯形透视；"
            "连接第二幅图时用 ORB + RANSAC 估计第二幅到第一幅的单应矩阵并把第二幅对齐叠加。"
            "输出变换结果图与匹配内点数量。"
        ),
        keywords=("单应", "透视变换", "homography", "透视", "warp"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec("image2", "参考图像（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "变换图"),
            PortSpec("inliers", "内点数量", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "inset",
                "上边缘收拢比例",
                "float",
                default=0.12,
                min=0.0,
                max=0.45,
                step=0.01,
                tooltip="未连接第二幅图时使用；0 表示不变换。",
            ),
            ParamSpec(
                "border",
                "边界填充",
                "enum",
                choices=(
                    ("replicate", "复制边缘"),
                    ("reflect", "镜像"),
                    ("constant", "黑色"),
                ),
                default="replicate",
            ),
            ParamSpec("max_features", "最大特征点数", "int", default=800, min=100, max=5000, step=100),
        ),
    )

    def process(
        self,
        image=None,
        image2=None,
        inset: float = 0.12,
        border: str = "replicate",
        max_features: int = 800,
        **_: Any,
    ):
        if image2 is not None:
            left = _cv.source_bgr8(image, "输入图像")
            right = _cv.source_bgr8(image2, "参考图像")
            _cv.same_size(left, right, "参考图像")
            height, width = left.shape[:2]
            homography, inliers = _orb_match(right, left, int(max_features))
            if homography is not None:
                warped = cv2.warpPerspective(
                    right,
                    homography,
                    (width, height),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REPLICATE,
                )
                coverage = cv2.warpPerspective(
                    np.full(right.shape[:2], 255, dtype=np.uint8),
                    homography,
                    (width, height),
                )
                canvas = left.copy()
                canvas[coverage > 0] = warped[coverage > 0]
                return {
                    "image": canvas,
                    "inliers": NumberValue(value=float(inliers), label="内点数量", unit="个"),
                }

        base = _cv.source(image, "输入图像")
        height, width = base.shape[:2]
        homography = _trapezoid_homography(width, height, float(inset))
        transformed = cv2.warpPerspective(
            base,
            homography,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=_cv.match_border(border),
        )
        return {
            "image": transformed,
            "inliers": NumberValue(value=0.0, label="内点数量", unit="个"),
        }


OPERATORS = (
    CalibChessboard,
    CalibUndistort,
    CalibStereoDisparity,
    CalibStereoDepth,
    CalibPanorama,
    CalibHomography,
)
