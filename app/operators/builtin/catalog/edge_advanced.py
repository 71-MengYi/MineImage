"""OpenCV 目录算子：高级边缘检测（Scharr / Laplacian / 自动 Canny / 霍夫 / LSD）。

输入统一通过 :mod:`app.operators.builtin.catalog._cv` 转为 8 位灰度再交给
OpenCV 的 8 位接口；结果图统一为 BGR，线段类结果额外输出 ``DataType.EDGES``。
"""

from __future__ import annotations

from typing import Any

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

_APERTURES = ((3, "3×3"), (5, "5×5"), (7, "7×7"))


def _gray8(image) -> np.ndarray:
    return _cv.source_gray8(image, "输入图像")


def _preprocess(image, blur: int) -> np.ndarray:
    gray = _gray8(image)
    if int(blur) > 1:
        gray = cv2.GaussianBlur(gray, (int(blur), int(blur)), 0)
    return gray


def _normalize_uint8(values: np.ndarray) -> np.ndarray:
    """把浮点梯度图归一化到 0-255 的 uint8 预览图。"""
    peak = float(np.nanmax(values)) if values.size else 0.0
    if not np.isfinite(peak) or peak <= 0:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = values.astype(np.float32) / peak * 255.0
    return np.clip(np.rint(scaled), 0, 255).astype(np.uint8)


def _canny_params(low: int = 100, high: int = 200) -> tuple[ParamSpec, ...]:
    return (
        ParamSpec(
            "low",
            "低阈值",
            "int",
            default=low,
            min=0,
            max=255,
            step=1,
            tooltip="低于低阈值的梯度会被丢弃。",
        ),
        ParamSpec(
            "high",
            "高阈值",
            "int",
            default=high,
            min=0,
            max=255,
            step=1,
            tooltip="高于高阈值的梯度直接视为边缘，必须不小于低阈值。",
        ),
        ParamSpec(
            "blur",
            "预模糊核",
            "int",
            default=5,
            min=1,
            max=31,
            step=2,
            odd=True,
            tooltip="Canny 前的高斯模糊核，必须为奇数；1 表示不模糊。",
        ),
        ParamSpec(
            "aperture",
            "Sobel 孔径",
            "enum",
            choices=_APERTURES,
            default=3,
        ),
        ParamSpec("l2", "L2 梯度", "bool", default=False),
    )


def _check_low_high(low: int, high: int) -> None:
    if int(high) < int(low):
        raise OperatorError("Canny 的高阈值必须大于或等于低阈值。")


def _edges_output(
    segments, width: int, height: int, label: str = "线段"
) -> EdgesData:
    return EdgesData(segments=tuple(segments), image_size=(width, height), label=label)


# ---------------------------------------------------------------------------
# Scharr 梯度
# ---------------------------------------------------------------------------
class ScharrEdge(Operator):
    """Scharr 梯度（比 Sobel 更精确的 3×3 方向梯度）。"""

    spec = OperatorSpec(
        id="edge.scharr",
        name="Scharr 梯度",
        category=("边缘检测", "高级"),
        description=(
            "用 Scharr 算子计算 x / y 方向梯度，输出归一化到 0-255 的梯度幅值图"
            "以及最大梯度强度。"
        ),
        keywords=("scharr", "梯度", "边缘", "gradient", "edge"),
        inputs=(_INPUT,),
        outputs=(
            PortSpec("image", "梯度幅值图", accepts=("L",)),
            PortSpec("strength", "最大梯度", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "direction",
                "梯度方向",
                "enum",
                choices=(("x", "水平方向"), ("y", "垂直方向"), ("both", "两方向合成")),
                default="both",
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
                tooltip="计算前的高斯模糊核，必须为奇数；1 表示不模糊。",
            ),
        ),
    )

    def process(self, image=None, direction: str = "both", blur: int = 1, **_: Any):
        gray = _preprocess(image, int(blur))
        work = gray.astype(np.float32)
        horizontal = vertical = None
        if direction in ("x", "both"):
            horizontal = cv2.Scharr(work, cv2.CV_32F, 1, 0)
        if direction in ("y", "both"):
            vertical = cv2.Scharr(work, cv2.CV_32F, 0, 1)
        if horizontal is not None and vertical is not None:
            magnitude = cv2.magnitude(horizontal, vertical)
        elif horizontal is not None:
            magnitude = np.abs(horizontal)
        else:
            magnitude = np.abs(vertical)
        peak = float(magnitude.max()) if magnitude.size else 0.0
        return {
            "image": _normalize_uint8(magnitude),
            "strength": NumberValue(value=peak, label="最大梯度"),
        }


# ---------------------------------------------------------------------------
# Laplacian
# ---------------------------------------------------------------------------
class LaplacianEdge(Operator):
    """Laplacian 二阶边缘。"""

    spec = OperatorSpec(
        id="edge.laplacian",
        name="Laplacian 边缘",
        category=("边缘检测", "高级"),
        description=(
            "用 Laplacian 二阶微分检测边缘与细节，输出取绝对值并归一化到 0-255 的"
            "边缘图，以及 Laplacian 方差（越大说明细节越丰富）。"
        ),
        keywords=("laplacian", "拉普拉斯", "二阶", "边缘", "edge", "清晰度"),
        inputs=(_INPUT,),
        outputs=(
            PortSpec("image", "边缘图", accepts=("L",)),
            PortSpec("variance", "Laplacian 方差", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "ksize",
                "核大小",
                "enum",
                choices=((1, "1×1"), (3, "3×3"), (5, "5×5"), (7, "7×7")),
                default=3,
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
                tooltip="计算前的高斯模糊核，必须为奇数；1 表示不模糊。",
            ),
            ParamSpec(
                "normalized",
                "归一化显示",
                "bool",
                default=True,
                tooltip="关闭时直接裁剪到 0-255，开启时按最大值拉伸。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        ksize: int = 3,
        blur: int = 3,
        normalized: bool = True,
        **_: Any,
    ):
        gray = _preprocess(image, int(blur))
        work = gray.astype(np.float32)
        laplacian = cv2.Laplacian(work, cv2.CV_32F, ksize=int(ksize))
        magnitude = np.abs(laplacian)
        variance = float(laplacian.var()) if laplacian.size else 0.0
        if normalized:
            canvas = _normalize_uint8(magnitude)
        else:
            canvas = np.clip(np.rint(magnitude), 0, 255).astype(np.uint8)
        return {
            "image": canvas,
            "variance": NumberValue(value=variance, label="Laplacian 方差"),
        }


# ---------------------------------------------------------------------------
# 自动阈值 Canny
# ---------------------------------------------------------------------------
class CannyAuto(Operator):
    """按中位数自动确定双阈值的 Canny。"""

    spec = OperatorSpec(
        id="edge.canny_auto",
        name="Canny 自动阈值",
        category=("边缘检测", "高级"),
        description=(
            "用图像灰度中位数自动确定 Canny 的高低阈值（下界 = (1-σ)×中位数，"
            "上界 = (1+σ)×中位数），避免手动调参。输出边缘图与自动阈值。"
        ),
        keywords=("canny", "自动阈值", "auto", "边缘", "edge", "median"),
        inputs=(_INPUT,),
        outputs=(
            PortSpec("image", "边缘图", accepts=("L",)),
            PortSpec("low", "自动低阈值", data_type=DataType.NUMBER),
            PortSpec("high", "自动高阈值", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "sigma",
                "阈值系数 σ",
                "float",
                default=0.33,
                min=0.01,
                max=2.0,
                step=0.01,
                tooltip="阈值围绕灰度中位数的浮动比例，越大保留的边缘越多。",
            ),
            ParamSpec(
                "blur",
                "预模糊核",
                "int",
                default=5,
                min=1,
                max=31,
                step=2,
                odd=True,
                tooltip="Canny 前的高斯模糊核，必须为奇数；1 表示不模糊。",
            ),
            ParamSpec("aperture", "Sobel 孔径", "enum", choices=_APERTURES, default=3),
            ParamSpec("l2", "L2 梯度", "bool", default=False),
        ),
    )

    def process(
        self,
        image=None,
        sigma: float = 0.33,
        blur: int = 5,
        aperture: int = 3,
        l2: bool = False,
        **_: Any,
    ):
        gray = _preprocess(image, int(blur))
        median = float(np.median(gray))
        low = int(max(0.0, (1.0 - float(sigma)) * median))
        high = int(min(255.0, (1.0 + float(sigma)) * median))
        if high < low:
            low, high = high, low
        edges = cv2.Canny(
            gray,
            low,
            high,
            apertureSize=int(aperture),
            L2gradient=bool(l2),
        )
        return {
            "image": edges,
            "low": NumberValue(value=float(low), label="自动低阈值"),
            "high": NumberValue(value=float(high), label="自动高阈值"),
        }


# ---------------------------------------------------------------------------
# 标准霍夫直线
# ---------------------------------------------------------------------------
class HoughLines(Operator):
    """标准霍夫变换提取直线。"""

    spec = OperatorSpec(
        id="edge.hough_lines",
        name="霍夫直线",
        category=("边缘检测", "高级"),
        description=(
            "先做 Canny 边缘检测，再用标准霍夫变换（HoughLines）提取无限长直线，"
            "并把直线延长绘制到彩色图上。输出标注图、直线集合与数量。"
        ),
        keywords=("hough", "霍夫", "直线", "lines", "边缘"),
        inputs=(_INPUT,),
        outputs=(
            PortSpec("image", "直线标注图", accepts=("BGR",)),
            PortSpec("lines", "直线", data_type=DataType.EDGES),
            PortSpec("count", "直线数", data_type=DataType.NUMBER),
        ),
        params=_canny_params()
        + (
            ParamSpec(
                "threshold",
                "累加器阈值",
                "int",
                default=80,
                min=1,
                max=500,
                step=1,
                tooltip="累加器超过该值的直线才会被检出，越大越严格。",
            ),
            ParamSpec(
                "rho",
                "距离分辨率",
                "float",
                default=1.0,
                min=0.5,
                max=10.0,
                step=0.5,
                unit="px",
            ),
            ParamSpec(
                "theta",
                "角度分辨率",
                "float",
                default=1.0,
                min=0.1,
                max=10.0,
                step=0.1,
                unit="°",
            ),
            ParamSpec(
                "max_lines",
                "最大直线数",
                "int",
                default=100,
                min=1,
                max=1000,
                step=1,
            ),
            ParamSpec("color", "绘制颜色", "color", default="#ff3b30"),
            ParamSpec("thickness", "线宽", "int", default=2, min=1, max=10, step=1),
        ),
    )

    def process(
        self,
        image=None,
        low: int = 100,
        high: int = 200,
        blur: int = 5,
        aperture: int = 3,
        l2: bool = False,
        threshold: int = 80,
        rho: float = 1.0,
        theta: float = 1.0,
        max_lines: int = 100,
        color: str = "#ff3b30",
        thickness: int = 2,
        **_: Any,
    ):
        _check_low_high(low, high)
        gray = _preprocess(image, int(blur))
        edges = cv2.Canny(
            gray,
            int(low),
            int(high),
            apertureSize=int(aperture),
            L2gradient=bool(l2),
        )
        detected = cv2.HoughLines(
            edges,
            float(rho),
            float(np.deg2rad(float(theta))),
            int(threshold),
        )
        width, height = gray.shape[1], gray.shape[0]
        segments: list[np.ndarray] = []
        canvas = _cv.source_bgr8(image, "输入图像")
        draw_value = _cv.draw_color(canvas, color)
        if detected is not None:
            for index, line in enumerate(detected):
                if index >= int(max_lines):
                    break
                rho_value, theta_value = float(line[0][0]), float(line[0][1])
                cos_t, sin_t = np.cos(theta_value), np.sin(theta_value)
                x0, y0 = cos_t * rho_value, sin_t * rho_value
                x1 = int(round(x0 + 2000 * (-sin_t)))
                y1 = int(round(y0 + 2000 * cos_t))
                x2 = int(round(x0 - 2000 * (-sin_t)))
                y2 = int(round(y0 - 2000 * cos_t))
                segments.append(
                    np.array([[x1, y1], [x2, y2]], dtype=np.int32).reshape(2, 2)
                )
                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    draw_value,
                    int(thickness),
                    lineType=_cv.draw_type(canvas),
                )
        return {
            "image": canvas,
            "lines": _edges_output(segments, width, height, label="霍夫直线"),
            "count": NumberValue(value=len(segments), label="直线数", unit="条"),
        }


# ---------------------------------------------------------------------------
# 概率霍夫直线
# ---------------------------------------------------------------------------
class HoughLinesP(Operator):
    """概率霍夫变换提取线段。"""

    spec = OperatorSpec(
        id="edge.hough_lines_p",
        name="概率霍夫线段",
        category=("边缘检测", "高级"),
        description=(
            "先做 Canny 边缘检测，再用概率霍夫变换（HoughLinesP）提取有端点的线段，"
            "并绘制到彩色图上。输出标注图、线段集合与数量。"
        ),
        keywords=("houghlinesp", "概率霍夫", "线段", "lines", "边缘"),
        inputs=(_INPUT,),
        outputs=(
            PortSpec("image", "线段标注图", accepts=("BGR",)),
            PortSpec("lines", "线段", data_type=DataType.EDGES),
            PortSpec("count", "线段数", data_type=DataType.NUMBER),
        ),
        params=_canny_params()
        + (
            ParamSpec(
                "threshold",
                "累加器阈值",
                "int",
                default=60,
                min=1,
                max=500,
                step=1,
            ),
            ParamSpec(
                "min_length",
                "最小线段长度",
                "int",
                default=20,
                min=1,
                max=1000,
                step=1,
                unit="px",
            ),
            ParamSpec(
                "max_gap",
                "最大间隙",
                "int",
                default=3,
                min=0,
                max=100,
                step=1,
                unit="px",
            ),
            ParamSpec(
                "max_lines",
                "最大线段数",
                "int",
                default=200,
                min=1,
                max=1000,
                step=1,
            ),
            ParamSpec("color", "绘制颜色", "color", default="#00e5ff"),
            ParamSpec("thickness", "线宽", "int", default=2, min=1, max=10, step=1),
        ),
    )

    def process(
        self,
        image=None,
        low: int = 100,
        high: int = 200,
        blur: int = 5,
        aperture: int = 3,
        l2: bool = False,
        threshold: int = 60,
        min_length: int = 20,
        max_gap: int = 3,
        max_lines: int = 200,
        color: str = "#00e5ff",
        thickness: int = 2,
        **_: Any,
    ):
        _check_low_high(low, high)
        gray = _preprocess(image, int(blur))
        edges = cv2.Canny(
            gray,
            int(low),
            int(high),
            apertureSize=int(aperture),
            L2gradient=bool(l2),
        )
        detected = cv2.HoughLinesP(
            edges,
            1,
            np.pi / 180.0,
            threshold=int(threshold),
            minLineLength=int(min_length),
            maxLineGap=int(max_gap),
        )
        width, height = gray.shape[1], gray.shape[0]
        segments: list[np.ndarray] = []
        canvas = _cv.source_bgr8(image, "输入图像")
        draw_value = _cv.draw_color(canvas, color)
        if detected is not None:
            for index, line in enumerate(detected):
                if index >= int(max_lines):
                    break
                x1, y1, x2, y2 = (int(value) for value in line[0])
                segments.append(
                    np.array([[x1, y1], [x2, y2]], dtype=np.int32).reshape(2, 2)
                )
                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    draw_value,
                    int(thickness),
                    lineType=_cv.draw_type(canvas),
                )
        return {
            "image": canvas,
            "lines": _edges_output(segments, width, height, label="概率霍夫线段"),
            "count": NumberValue(value=len(segments), label="线段数", unit="条"),
        }


# ---------------------------------------------------------------------------
# 霍夫圆
# ---------------------------------------------------------------------------
class HoughCircles(Operator):
    """霍夫变换检测圆。"""

    spec = OperatorSpec(
        id="edge.hough_circles",
        name="霍夫圆检测",
        category=("边缘检测", "高级"),
        description=(
            "用霍夫梯度法（HoughCircles）检测圆形目标，并在彩色图上绘制圆。"
            "输出标注图、圆轮廓点集与数量。"
        ),
        keywords=("hough", "霍夫", "圆", "circles", "检测"),
        inputs=(_INPUT,),
        outputs=(
            PortSpec("image", "圆标注图", accepts=("BGR",)),
            PortSpec("circles", "圆轮廓", data_type=DataType.CONTOURS),
            PortSpec("count", "圆数量", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "dp",
                "分辨率比例",
                "float",
                default=1.5,
                min=1.0,
                max=5.0,
                step=0.1,
                tooltip="累加器分辨率与图像分辨率之比，越大越省时。",
            ),
            ParamSpec(
                "min_distance",
                "圆心最小距离",
                "float",
                default=20.0,
                min=1.0,
                max=2000.0,
                step=1.0,
                unit="px",
            ),
            ParamSpec(
                "param1",
                "Canny 高阈值",
                "int",
                default=100,
                min=1,
                max=255,
                step=1,
            ),
            ParamSpec(
                "param2",
                "圆心累加阈值",
                "int",
                default=30,
                min=1,
                max=300,
                step=1,
                tooltip="越小检出的圆越多，误检也越多。",
            ),
            ParamSpec("min_radius", "最小半径", "int", default=0, min=0, max=2000, step=1),
            ParamSpec("max_radius", "最大半径", "int", default=0, min=0, max=2000, step=1),
            ParamSpec(
                "blur",
                "预模糊核",
                "int",
                default=5,
                min=1,
                max=31,
                step=2,
                odd=True,
            ),
            ParamSpec("color", "绘制颜色", "color", default="#ffd400"),
            ParamSpec("thickness", "线宽", "int", default=2, min=1, max=10, step=1),
        ),
    )

    def validate(self, inputs, params) -> None:
        if int(params["max_radius"]) and int(params["max_radius"]) < int(params["min_radius"]):
            raise OperatorError("最大半径不能小于最小半径（0 表示自动）。")

    def process(
        self,
        image=None,
        dp: float = 1.5,
        min_distance: float = 20.0,
        param1: int = 100,
        param2: int = 30,
        min_radius: int = 0,
        max_radius: int = 0,
        blur: int = 5,
        color: str = "#ffd400",
        thickness: int = 2,
        **_: Any,
    ):
        if int(max_radius) and int(max_radius) < int(min_radius):
            raise OperatorError("最大半径不能小于最小半径（0 表示自动）。")
        gray = _preprocess(image, int(blur))
        detected = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=float(dp),
            minDist=float(min_distance),
            param1=float(param1),
            param2=float(param2),
            minRadius=int(min_radius),
            maxRadius=int(max_radius),
        )
        height, width = gray.shape[:2]
        canvas = _cv.source_bgr8(image, "输入图像")
        draw_value = _cv.draw_color(canvas, color)
        circles: list[np.ndarray] = []
        if detected is not None:
            for circle in detected[0]:
                center_x, center_y, radius = (float(value) for value in circle)
                if radius < 1:
                    continue
                cv2.circle(
                    canvas,
                    (int(round(center_x)), int(round(center_y))),
                    int(round(radius)),
                    draw_value,
                    int(thickness),
                    lineType=_cv.draw_type(canvas),
                )
                polygon = cv2.ellipse2Poly(
                    (int(round(center_x)), int(round(center_y))),
                    (max(1, int(round(radius))), max(1, int(round(radius)))),
                    0,
                    0,
                    360,
                    10,
                )
                circles.append(polygon.reshape(-1, 2))
        return {
            "image": canvas,
            "circles": ContoursData(
                contours=tuple(circles), image_size=(width, height), label="圆轮廓"
            ),
            "count": NumberValue(value=len(circles), label="圆数量", unit="个"),
        }


# ---------------------------------------------------------------------------
# LSD 线段检测
# ---------------------------------------------------------------------------
class LineSegmentDetector(Operator):
    """LSD 线段检测。"""

    spec = OperatorSpec(
        id="edge.lsd",
        name="LSD 线段检测",
        category=("边缘检测", "高级"),
        description=(
            "用 LSD（Line Segment Detector）在灰度图上直接检测直线段，不需要手动设阈值，"
            "适合检测建筑、网格等直线结构。输出标注图、线段集合与数量。"
        ),
        keywords=("lsd", "线段检测", "线段", "直线", "line segment"),
        inputs=(_INPUT,),
        outputs=(
            PortSpec("image", "线段标注图", accepts=("BGR",)),
            PortSpec("lines", "线段", data_type=DataType.EDGES),
            PortSpec("count", "线段数", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "refine",
                "精化方式",
                "enum",
                choices=(("none", "不精化"), ("standard", "标准精化"), ("advanced", "高级精化")),
                default="standard",
            ),
            ParamSpec(
                "scale",
                "缩放系数",
                "float",
                default=0.8,
                min=0.1,
                max=2.0,
                step=0.05,
                tooltip="检测前对图像做缩放的系数，越大对细节越敏感。",
            ),
            ParamSpec(
                "sigma_scale",
                "高斯 σ 比例",
                "float",
                default=0.6,
                min=0.1,
                max=2.0,
                step=0.05,
            ),
            ParamSpec(
                "max_lines",
                "最大线段数",
                "int",
                default=300,
                min=1,
                max=2000,
                step=1,
            ),
            ParamSpec("color", "绘制颜色", "color", default="#00ff9d"),
            ParamSpec("thickness", "线宽", "int", default=2, min=1, max=10, step=1),
        ),
    )

    def process(
        self,
        image=None,
        refine: str = "standard",
        scale: float = 0.8,
        sigma_scale: float = 0.6,
        max_lines: int = 300,
        color: str = "#00ff9d",
        thickness: int = 2,
        **_: Any,
    ):
        gray = _gray8(image)
        refine_flags = {
            "none": cv2.LSD_REFINE_NONE,
            "standard": cv2.LSD_REFINE_STD,
            "advanced": cv2.LSD_REFINE_ADV,
        }
        try:
            detector = cv2.createLineSegmentDetector(
                refine_flags[refine],
                scale=float(scale),
                sigma_scale=float(sigma_scale),
            )
        except (AttributeError, cv2.error) as exc:
            raise OperatorError(
                "当前 OpenCV 构建不支持 LSD 线段检测，请升级 OpenCV 或改用其他边缘算子。"
            ) from exc
        result = detector.detect(gray)
        detected = result[0] if isinstance(result, tuple) else result
        height, width = gray.shape[:2]
        canvas = _cv.source_bgr8(image, "输入图像")
        draw_value = _cv.draw_color(canvas, color)
        segments: list[np.ndarray] = []
        if detected is not None:
            for index, line in enumerate(detected):
                if index >= int(max_lines):
                    break
                x1, y1, x2, y2 = (int(round(float(value))) for value in np.ravel(line))
                segments.append(
                    np.array([[x1, y1], [x2, y2]], dtype=np.int32).reshape(2, 2)
                )
                cv2.line(
                    canvas,
                    (x1, y1),
                    (x2, y2),
                    draw_value,
                    int(thickness),
                    lineType=_cv.draw_type(canvas),
                )
        return {
            "image": canvas,
            "lines": _edges_output(segments, width, height, label="LSD 线段"),
            "count": NumberValue(value=len(segments), label="线段数", unit="条"),
        }


# ---------------------------------------------------------------------------
# 梯度幅值 / 方向
# ---------------------------------------------------------------------------
class GradientMagnitude(Operator):
    """Sobel 梯度幅值与方向。"""

    spec = OperatorSpec(
        id="edge.gradient_mag",
        name="梯度幅值与方向",
        category=("边缘检测", "高级"),
        description=(
            "用 Sobel 算子计算 x / y 方向梯度，输出归一化的梯度幅值图、"
            "方向角编码图（0-255 对应 0-360°）以及最大梯度幅值。"
        ),
        keywords=("gradient", "梯度", "幅值", "方向", "sobel", "edge"),
        inputs=(_INPUT,),
        outputs=(
            PortSpec("magnitude", "梯度幅值图", accepts=("L",)),
            PortSpec("angle", "梯度方向图", accepts=("L",)),
            PortSpec("max_magnitude", "最大幅值", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "ksize",
                "Sobel 核大小",
                "enum",
                choices=((1, "1×1"), (3, "3×3"), (5, "5×5"), (7, "7×7")),
                default=3,
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
                tooltip="计算前的高斯模糊核，必须为奇数；1 表示不模糊。",
            ),
            ParamSpec(
                "magnitude_mode",
                "幅值计算",
                "enum",
                choices=(("l2", "L2 范数"), ("l1", "L1 绝对值之和")),
                default="l2",
            ),
        ),
    )

    def process(
        self,
        image=None,
        ksize: int = 3,
        blur: int = 3,
        magnitude_mode: str = "l2",
        **_: Any,
    ):
        gray = _preprocess(image, int(blur))
        work = gray.astype(np.float32)
        horizontal = cv2.Sobel(work, cv2.CV_32F, 1, 0, ksize=int(ksize))
        vertical = cv2.Sobel(work, cv2.CV_32F, 0, 1, ksize=int(ksize))
        if magnitude_mode == "l1":
            magnitude = np.abs(horizontal) + np.abs(vertical)
        else:
            magnitude = cv2.magnitude(horizontal, vertical)
        angle = cv2.phase(horizontal, vertical, angleInDegrees=True)
        peak = float(magnitude.max()) if magnitude.size else 0.0
        angle_canvas = np.clip(np.rint(angle / 360.0 * 255.0), 0, 255).astype(np.uint8)
        return {
            "magnitude": _normalize_uint8(magnitude),
            "angle": angle_canvas,
            "max_magnitude": NumberValue(value=peak, label="最大梯度幅值"),
        }


OPERATORS = (
    ScharrEdge,
    LaplacianEdge,
    CannyAuto,
    HoughLines,
    HoughLinesP,
    HoughCircles,
    LineSegmentDetector,
    GradientMagnitude,
)
