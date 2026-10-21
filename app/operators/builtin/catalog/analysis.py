"""OpenCV 目录算子：特征与分析（度量与变换组）。

包含积分图、最值、均值标准差、矩与 Hu 矩、FFT / IFFT、相位相关、模板匹配、
PSNR、SSIM 与形状匹配共 12 个算子。

约定
----
* 主输出始终是图像（可视化结果或文本面板），数值统计以 ``DataType.NUMBER`` 额外输出；
* 双图算子的 ``image2`` / ``template`` 均可选，未连接时使用平移派生图或中心裁剪，
  保证默认参数可运行；
* 所有算子都先把输入转换为 8 位或原生 dtype 的安全形式，支持 L / BGR / BGRA 与
  uint8 / uint16 / float32，返回新数组、不修改输入。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.core.data_types import DataType, NumberValue
from app.operators.builtin._utils import full_scale
from app.operators.builtin.catalog import _cv
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec

#: 第二幅图未连接时的默认水平平移像素数。
DEFAULT_SHIFT = 3


# ---------------------------------------------------------------------------
# 共享工具
# ---------------------------------------------------------------------------
def _derived(base: np.ndarray, shift: int = DEFAULT_SHIFT) -> np.ndarray:
    return np.roll(base, max(int(shift), 1), axis=1).copy()


def _second_gray(second, base: np.ndarray, shift: int = DEFAULT_SHIFT) -> np.ndarray:
    """第二幅图的灰度版本（保持 dtype）；未连接时由第一幅图平移派生。"""
    if second is None:
        return _derived(base, shift)
    other = _cv.source_gray(second, "第二幅图像")
    _cv.same_size(base, other, "第二幅图像")
    return other


def _second_gray8(second, base: np.ndarray, shift: int = DEFAULT_SHIFT) -> np.ndarray:
    """第二幅图的 8 位灰度版本；未连接时由第一幅图平移派生。"""
    if second is None:
        return _derived(base, shift)
    other = _cv.source_gray8(second, "第二幅图像")
    _cv.same_size(base, other, "第二幅图像")
    return other


def _colorize(values: np.ndarray, colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    """把任意数值图按 min-max 归一化后渲染为 BGR 热力图。"""
    work = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low = float(work.min()) if work.size else 0.0
    high = float(work.max()) if work.size else 0.0
    if high - low > 1e-12:
        work = (work - low) / (high - low) * 255.0
    else:
        work = np.zeros_like(work)
    gray = np.clip(np.rint(work), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(gray, colormap)


def _restore_gray(values: np.ndarray) -> np.ndarray:
    """把逆变换结果恢复为 8 位灰度：0–255 内保持原尺度，超出时才拉伸。"""
    work = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low = float(work.min()) if work.size else 0.0
    high = float(work.max()) if work.size else 0.0
    if not (low >= -0.5 and high <= 255.5):
        if high - low > 1e-12:
            work = (work - low) / (high - low) * 255.0
        else:
            work = np.zeros_like(work)
    return np.clip(np.rint(work), 0, 255).astype(np.uint8)


def _encode_spectrum(spectrum: np.ndarray) -> np.ndarray:
    """把 ``(H, W, 2)`` 复频谱编码为 float32 BGR：B=实部，G=虚部，R=0。"""
    height, width = spectrum.shape[:2]
    encoded = np.zeros((height, width, 3), dtype=np.float32)
    encoded[:, :, 0] = spectrum[:, :, 0]
    encoded[:, :, 1] = spectrum[:, :, 1]
    return encoded


def _decode_spectrum(spectrum) -> np.ndarray:
    """把 FFT 输出的 float32 BGR 复频谱解码为 ``(H, W, 2)``。"""
    array = _cv.source_bgr(spectrum, "复频谱")
    if array.shape[2] != 3:
        raise OperatorError("复频谱必须是三通道图像（B=实部，G=虚部）。")
    if array.dtype != np.float32:
        array = array.astype(np.float32)
    return np.dstack([array[:, :, 0], array[:, :, 1]]).astype(np.float32)


def _lowpass_mask(height: int, width: int, radius: float) -> np.ndarray:
    """生成中心化频谱的低通掩码，``radius`` 为相对短边的半径比例。"""
    mask = np.zeros((height, width, 2), dtype=np.float32)
    pixel_radius = max(1, int(round(min(height, width) * float(radius) / 2.0)))
    cv2.circle(mask, (width // 2, height // 2), pixel_radius, (1, 1), -1)
    return mask


def _gaussian_kernel(ksize: int, sigma: float) -> np.ndarray:
    size = max(int(ksize), 1)
    kernel = cv2.getGaussianKernel(size, float(sigma) if float(sigma) > 0 else 0.0)
    return kernel @ kernel.T


def _largest_contour(gray8: np.ndarray):
    """Otsu 二值化后取面积最大的外轮廓；没有轮廓时返回 ``None``。"""
    _value, binary = cv2.threshold(gray8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _hierarchy = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    return max(contours, key=cv2.contourArea)


# ---------------------------------------------------------------------------
# 1. 积分图
# ---------------------------------------------------------------------------
class AnalysisIntegral(Operator):
    spec = OperatorSpec(
        id="analysis.integral",
        name="积分图",
        category=("特征与分析", "积分"),
        description=(
            "计算灰度图的积分图（累加和），并用对数拉伸后渲染成热力图，"
            "便于观察亮度在空间上的累积分布。输出积分图可视化与总和。"
        ),
        keywords=("积分图", "integral", "累加和", "summed area", "分析"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(
            PortSpec("image", "积分图热力", accepts=("BGR",)),
            PortSpec("total", "积分图总和", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "log_scale",
                "对数拉伸",
                "bool",
                default=True,
                tooltip="积分值很大时建议开启，避免热力图只有少量亮块。",
            ),
        ),
    )

    def process(self, image=None, log_scale: bool = True, **_: Any):
        gray = _cv.source_gray(image, "输入图像")
        integral = cv2.integral(gray.astype(np.float64))
        total = float(integral[-1, -1]) if integral.size else 0.0
        body = integral[1:, 1:]
        if bool(log_scale):
            body = np.log1p(np.maximum(body, 0.0))
        return {
            "image": _colorize(body, cv2.COLORMAP_INFERNO),
            "total": NumberValue(value=total, label="积分图总和"),
        }


# ---------------------------------------------------------------------------
# 2. 最值定位
# ---------------------------------------------------------------------------
class AnalysisMinMax(Operator):
    spec = OperatorSpec(
        id="analysis.min_max",
        name="最值定位",
        category=("特征与分析", "度量"),
        description=(
            "统计灰度图的最小值、最大值及其坐标，并在画面上用圆形标记标出这两个位置。"
            "最小值与最大值按输入原始数值统计，不做 0-255 归一化。输出标注图与两个数值。"
        ),
        keywords=("最小值", "最大值", "最值", "min", "max", "定位"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(
            PortSpec("image", "最值标注图", accepts=("BGR",)),
            PortSpec("min", "最小值", data_type=DataType.NUMBER),
            PortSpec("max", "最大值", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec("radius", "标记半径", "int", default=8, min=2, max=60, step=1, unit="px", group="显示"),
            ParamSpec("color", "标记颜色", "color", default="#ffb020", group="显示"),
        ),
    )

    def process(self, image=None, radius: int = 8, color: str = "#ffb020", **_: Any):
        gray = _cv.source_gray(image, "输入图像")
        min_value, max_value, min_location, max_location = cv2.minMaxLoc(gray)
        canvas = _cv.source_bgr8(image, "输入图像")
        draw_color = _cv.draw_color(canvas, color)
        line_type = _cv.draw_type(canvas)
        cv2.circle(canvas, min_location, int(radius), draw_color, 2, line_type)
        cv2.circle(canvas, max_location, int(radius), draw_color, 2, line_type)
        return {
            "image": canvas,
            "min": NumberValue(value=float(min_value), label="最小值"),
            "max": NumberValue(value=float(max_value), label="最大值"),
        }


# ---------------------------------------------------------------------------
# 3. 均值与标准差
# ---------------------------------------------------------------------------
class AnalysisMeanStd(Operator):
    spec = OperatorSpec(
        id="analysis.mean_std",
        name="均值与标准差",
        category=("特征与分析", "度量"),
        description=(
            "统计灰度图的均值与标准差，并把每个像素与均值的偏差渲染成热力图，"
            "便于观察整体亮度与离散程度。输出偏差热力图、均值与标准差。"
        ),
        keywords=("均值", "标准差", "mean", "std", "统计", "偏差"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(
            PortSpec("image", "偏差热力图", accepts=("BGR",)),
            PortSpec("mean", "均值", data_type=DataType.NUMBER),
            PortSpec("std", "标准差", data_type=DataType.NUMBER),
        ),
    )

    def process(self, image=None, **_: Any):
        gray = _cv.source_gray(image, "输入图像")
        values = gray.astype(np.float64)
        mean = float(values.mean()) if values.size else 0.0
        std = float(values.std()) if values.size else 0.0
        deviation = np.abs(values - mean)
        return {
            "image": _colorize(deviation, cv2.COLORMAP_VIRIDIS),
            "mean": NumberValue(value=mean, label="均值"),
            "std": NumberValue(value=std, label="标准差"),
        }


# ---------------------------------------------------------------------------
# 4. 图像矩
# ---------------------------------------------------------------------------
class AnalysisMoments(Operator):
    spec = OperatorSpec(
        id="analysis.moments",
        name="图像矩与质心",
        category=("特征与分析", "矩"),
        description=(
            "计算灰度图的零阶矩、一阶矩与质心，并在画面上用十字标记标出质心位置。"
            "零阶矩为像素值之和，质心为亮度加权中心。输出标注图与三个数值。"
        ),
        keywords=("矩", "moments", "质心", "零阶矩", "一阶矩"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(
            PortSpec("image", "质心标注图", accepts=("BGR",)),
            PortSpec("m00", "零阶矩", data_type=DataType.NUMBER),
            PortSpec("cx", "质心 X", data_type=DataType.NUMBER),
            PortSpec("cy", "质心 Y", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec("radius", "十字半径", "int", default=10, min=2, max=80, step=1, unit="px", group="显示"),
            ParamSpec("color", "标记颜色", "color", default="#00d26a", group="显示"),
        ),
    )

    def process(self, image=None, radius: int = 10, color: str = "#00d26a", **_: Any):
        gray = _cv.source_gray(image, "输入图像")
        height, width = gray.shape[:2]
        moments = cv2.moments(gray)
        m00 = float(moments.get("m00", 0.0))
        if abs(m00) > 1e-12:
            centroid_x = float(moments["m10"]) / m00
            centroid_y = float(moments["m01"]) / m00
        else:
            centroid_x = (width - 1) / 2.0
            centroid_y = (height - 1) / 2.0
        canvas = _cv.source_bgr8(image, "输入图像")
        draw_color = _cv.draw_color(canvas, color)
        line_type = _cv.draw_type(canvas)
        center = (int(round(centroid_x)), int(round(centroid_y)))
        cv2.line(canvas, (center[0] - int(radius), center[1]), (center[0] + int(radius), center[1]), draw_color, 1, line_type)
        cv2.line(canvas, (center[0], center[1] - int(radius)), (center[0], center[1] + int(radius)), draw_color, 1, line_type)
        cv2.circle(canvas, center, 3, draw_color, -1, line_type)
        return {
            "image": canvas,
            "m00": NumberValue(value=m00, label="零阶矩"),
            "cx": NumberValue(value=centroid_x, label="质心 X", unit="px"),
            "cy": NumberValue(value=centroid_y, label="质心 Y", unit="px"),
        }


# ---------------------------------------------------------------------------
# 5. Hu 不变矩
# ---------------------------------------------------------------------------
class AnalysisHuMoments(Operator):
    spec = OperatorSpec(
        id="analysis.hu_moments",
        name="Hu 不变矩",
        category=("特征与分析", "矩"),
        description=(
            "计算 7 个 Hu 不变矩（对平移、旋转、缩放不敏感），并渲染成文本面板。"
            "Hu 矩数值跨度很大，默认取 -sign(h)·log10(|h|) 的对数形式，便于比较。"
            "输出文本面板与第一不变矩。"
        ),
        keywords=("Hu", "不变矩", "hu moments", "形状特征", "矩"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(
            PortSpec("image", "Hu 矩面板", accepts=("BGR",)),
            PortSpec("hu1", "第一不变矩", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "log_scale",
                "对数形式",
                "bool",
                default=True,
                tooltip="开启后输出 -sign(h)·log10(|h|)，数值更易读。",
            ),
        ),
    )

    def process(self, image=None, log_scale: bool = True, **_: Any):
        gray = _cv.source_gray(image, "输入图像")
        moments = cv2.moments(gray)
        hu = cv2.HuMoments(moments).ravel()
        if bool(log_scale):
            values = np.where(
                np.abs(hu) > 0.0,
                -np.sign(hu) * np.log10(np.maximum(np.abs(hu), 1e-300)),
                0.0,
            )
            title = "Hu moments (log)"
        else:
            values = hu
            title = "Hu moments"
        lines = [title]
        lines.extend(f"Hu{index + 1}: {float(value):.6g}" for index, value in enumerate(values))
        height = max(40 + 22 * len(lines), 120)
        return {
            "image": _cv.text_panel(360, height, lines),
            "hu1": NumberValue(value=float(values[0]), label="第一不变矩"),
        }


# ---------------------------------------------------------------------------
# 6. FFT 频谱
# ---------------------------------------------------------------------------
class AnalysisFft(Operator):
    spec = OperatorSpec(
        id="analysis.fft",
        name="傅里叶变换（FFT）",
        category=("特征与分析", "频域"),
        description=(
            "把灰度图转换为 float32，做二维离散傅里叶变换，输出频谱热力图、频谱总能量，"
            "以及供“逆傅里叶变换（IFFT）”使用的复频谱。复频谱用 float32 BGR 承载："
            "B 通道为实部、G 通道为虚部、R 通道为 0，未中心化保存。"
        ),
        keywords=("FFT", "fft", "傅里叶", "频谱", "dft", "频域", "复频谱"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(
            PortSpec("image", "频谱热力图", accepts=("BGR",)),
            PortSpec("spectrum", "复频谱（B=实部 G=虚部）", accepts=("BGR",)),
            PortSpec("energy", "频谱总能量", data_type=DataType.NUMBER),
        ),
    )

    def process(self, image=None, **_: Any):
        gray = _cv.source_gray8(image, "输入图像").astype(np.float32)
        spectrum = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)
        shifted = np.fft.fftshift(spectrum, axes=(0, 1))
        magnitude = cv2.magnitude(shifted[:, :, 0], shifted[:, :, 1])
        energy = float(magnitude.sum()) if magnitude.size else 0.0
        return {
            "image": _colorize(np.log1p(magnitude), cv2.COLORMAP_VIRIDIS),
            "spectrum": _encode_spectrum(spectrum),
            "energy": NumberValue(value=energy, label="频谱总能量"),
        }


# ---------------------------------------------------------------------------
# 7. IFFT 逆变换 / 低通重建
# ---------------------------------------------------------------------------
class AnalysisIfft(Operator):
    spec = OperatorSpec(
        id="analysis.ifft",
        name="逆傅里叶变换（IFFT）",
        category=("特征与分析", "频域"),
        description=(
            "优先使用 spectrum 端口输入的复频谱（来自“傅里叶变换（FFT）”的复频谱输出，"
            "B=实部、G=虚部）做逆变换，得到与 FFT 严格配对的 IFFT；未连接 spectrum 时"
            "对 image 先做正向 FFT 再逆变换。勾选低通后仅保留中心半径内的频率，"
            "半径越小图像越模糊。输出重建图与保留频率比例。"
        ),
        keywords=("IFFT", "ifft", "逆傅里叶", "逆变换", "低通", "频域滤波", "idft"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec(
                "spectrum",
                "复频谱（可选）",
                accepts=("BGR",),
                optional=True,
                description="连接 FFT 的“复频谱”输出即可做真正的逆变换。",
            ),
        ),
        outputs=(
            PortSpec("image", "重建图", accepts=("L",)),
            PortSpec("ratio", "保留频率比例", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "lowpass",
                "低通滤波",
                "bool",
                default=False,
                tooltip="开启后只保留中心半径内的频率；关闭时做完整逆变换。",
            ),
            ParamSpec(
                "radius",
                "保留半径比例",
                "float",
                default=0.25,
                min=0.02,
                max=1.0,
                step=0.01,
                visible_when="lowpass",
                enabled_when="lowpass",
                tooltip="相对于图像短边的半径比例，1.0 表示保留全部频率。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        spectrum=None,
        lowpass: bool = False,
        radius: float = 0.25,
        **_: Any,
    ):
        gray = _cv.source_gray8(image, "输入图像").astype(np.float32)
        height, width = gray.shape[:2]
        if spectrum is not None:
            complex_spectrum = _decode_spectrum(spectrum)
            if complex_spectrum.shape[:2] != (height, width):
                raise OperatorError(
                    f"复频谱尺寸 {complex_spectrum.shape[1]}×{complex_spectrum.shape[0]} "
                    f"与输入图像 {width}×{height} 不一致。"
                )
        else:
            complex_spectrum = cv2.dft(gray, flags=cv2.DFT_COMPLEX_OUTPUT)

        shifted = np.fft.fftshift(complex_spectrum, axes=(0, 1))
        if lowpass:
            mask = _lowpass_mask(height, width, radius)
            shifted = shifted * mask
            ratio = float(mask[:, :, 0].mean()) * 100.0 if mask.size else 0.0
        else:
            ratio = 100.0
        restored = cv2.idft(
            np.fft.ifftshift(shifted, axes=(0, 1)),
            flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT,
        )
        return {
            "image": _restore_gray(restored),
            "ratio": NumberValue(value=ratio, label="保留频率比例", unit="%"),
        }


# ---------------------------------------------------------------------------
# 8. 相位相关
# ---------------------------------------------------------------------------
class AnalysisPhaseCorrelate(Operator):
    spec = OperatorSpec(
        id="analysis.phase_correlate",
        name="相位相关位移",
        category=("特征与分析", "频域"),
        description=(
            "用相位相关（phaseCorrelate）估计两幅图之间的整像素平移，"
            "并同时渲染归一化互相关峰值响应图。未连接第二幅图时自动平移第一幅图派生。"
            "输出响应热力图、位移 X / Y 与响应强度。"
        ),
        keywords=("相位相关", "phase correlation", "位移", "平移估计", "频域"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec("image2", "参考图像（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "相关响应图", accepts=("BGR",)),
            PortSpec("dx", "位移 X", data_type=DataType.NUMBER),
            PortSpec("dy", "位移 Y", data_type=DataType.NUMBER),
            PortSpec("response", "响应强度", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "use_window",
                "汉宁窗",
                "bool",
                default=False,
                tooltip="开启后抑制边界效应，但可能降低整像素位移精度。",
            ),
        ),
    )

    def process(self, image=None, image2=None, use_window: bool = False, **_: Any):
        first = _cv.source_gray8(image, "输入图像").astype(np.float32) / 255.0
        second = _second_gray8(image2, _cv.source_gray8(image, "输入图像")).astype(np.float32) / 255.0
        height, width = first.shape[:2]
        window = cv2.createHanningWindow((width, height), cv2.CV_32F) if bool(use_window) else None
        try:
            (dx, dy), response = cv2.phaseCorrelate(first, second, window)
        except cv2.error as exc:
            raise OperatorError("相位相关计算失败：请检查两幅图尺寸是否一致。") from exc
        spectrum_a = np.fft.fft2(first)
        spectrum_b = np.fft.fft2(second)
        cross = spectrum_a * np.conj(spectrum_b)
        cross = cross / np.maximum(np.abs(cross), 1e-9)
        correlation = np.abs(np.fft.ifft2(cross))
        correlation = np.fft.fftshift(correlation)
        canvas = _colorize(correlation, cv2.COLORMAP_HOT)
        peak = np.unravel_index(int(np.argmax(correlation)), correlation.shape)
        cv2.circle(canvas, (int(peak[1]), int(peak[0])), 5, (255, 255, 255), 1, cv2.LINE_AA)
        return {
            "image": canvas,
            "dx": NumberValue(value=float(np.nan_to_num(dx, nan=0.0)), label="位移 X", unit="px"),
            "dy": NumberValue(value=float(np.nan_to_num(dy, nan=0.0)), label="位移 Y", unit="px"),
            "response": NumberValue(
                value=float(np.nan_to_num(response, nan=0.0, posinf=0.0, neginf=0.0)),
                label="响应强度",
            ),
        }


# ---------------------------------------------------------------------------
# 9. 模板匹配
# ---------------------------------------------------------------------------
class AnalysisTemplateMatch(Operator):
    spec = OperatorSpec(
        id="analysis.template_match",
        name="模板匹配",
        category=("特征与分析", "匹配"),
        description=(
            "在输入图像中搜索模板最相似的位置并画出匹配框。模板端口可选，"
            "未连接时自动从图像中心裁剪 1/4 区域作为模板，因此默认参数即可运行。"
            "输出标注图、匹配度与匹配位置 X / Y。"
        ),
        keywords=("模板匹配", "template match", "匹配", "定位", "相关系数"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec("template", "模板图像（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "匹配标注图", accepts=("BGR",)),
            PortSpec("score", "匹配度", data_type=DataType.NUMBER),
            PortSpec("x", "匹配位置 X", data_type=DataType.NUMBER),
            PortSpec("y", "匹配位置 Y", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "method",
                "匹配方法",
                "enum",
                choices=(("ccoeff", "归一化相关（越大越好）"), ("sqdiff", "归一化平方差（越小越好）")),
                default="ccoeff",
            ),
            ParamSpec("color", "匹配框颜色", "color", default="#00d26a", group="显示"),
        ),
    )

    def process(self, image=None, template=None, method: str = "ccoeff", color: str = "#00d26a", **_: Any):
        gray = _cv.source_gray8(image, "输入图像")
        height, width = gray.shape[:2]
        if template is None:
            x, y, rect_w, rect_h = _cv.center_rect(width, height, 0.25)
            patch = gray[y : y + rect_h, x : x + rect_w].copy()
        else:
            patch = _cv.source_gray8(template, "模板图像")
            if patch.shape[0] > height or patch.shape[1] > width:
                raise OperatorError(
                    f"模板图像（{patch.shape[1]}×{patch.shape[0]}）不能大于输入图像（{width}×{height}）。"
                )
        if patch.size == 0:
            raise OperatorError("模板图像为空，无法进行模板匹配。")
        if method == "sqdiff":
            method_flag = cv2.TM_SQDIFF_NORMED
        else:
            method_flag = cv2.TM_CCOEFF_NORMED
        similarity = cv2.matchTemplate(gray, patch, method_flag)
        similarity = np.nan_to_num(similarity, nan=0.0, posinf=1.0, neginf=-1.0)
        min_value, max_value, min_location, max_location = cv2.minMaxLoc(similarity)
        if method == "sqdiff":
            score = 1.0 - float(min_value)
            location = min_location
        else:
            score = float(max_value)
            location = max_location
        canvas = _cv.source_bgr8(image, "输入图像")
        rect_h, rect_w = patch.shape[:2]
        cv2.rectangle(
            canvas,
            (int(location[0]), int(location[1])),
            (int(location[0]) + rect_w, int(location[1]) + rect_h),
            _cv.draw_color(canvas, color),
            2,
            _cv.draw_type(canvas),
        )
        return {
            "image": canvas,
            "score": NumberValue(value=score, label="匹配度"),
            "x": NumberValue(value=float(location[0]), label="匹配位置 X", unit="px"),
            "y": NumberValue(value=float(location[1]), label="匹配位置 Y", unit="px"),
        }


# ---------------------------------------------------------------------------
# 10. PSNR
# ---------------------------------------------------------------------------
class AnalysisPsnr(Operator):
    spec = OperatorSpec(
        id="analysis.psnr",
        name="峰值信噪比（PSNR）",
        category=("特征与分析", "度量"),
        description=(
            "计算两幅图的均方误差与峰值信噪比（PSNR），并把差异渲染成热力图。"
            "两幅图完全相同时 PSNR 为无穷大。峰值按输入 dtype 取满量程："
            "uint8 为 255、uint16 为 65535、float 按 0-1 或 0-255 范围判断。"
            "未连接第二幅图时自动平移第一幅图派生。"
        ),
        keywords=("PSNR", "峰值信噪比", "图像质量", "均方误差", "MSE"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec("image2", "参考图像（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "差异热力图", accepts=("BGR",)),
            PortSpec("psnr", "PSNR", data_type=DataType.NUMBER),
            PortSpec("mse", "均方误差", data_type=DataType.NUMBER),
        ),
    )

    def process(self, image=None, image2=None, **_: Any):
        first = _cv.source_gray(image, "输入图像")
        second = _second_gray(image2, first)
        first_values = first.astype(np.float64)
        second_values = second.astype(np.float64)
        difference = np.abs(first_values - second_values)
        mse = float(np.mean((first_values - second_values) ** 2)) if difference.size else 0.0
        peak = float(full_scale(first))
        if mse <= 1e-12:
            psnr = float("inf")
        else:
            psnr = float(10.0 * np.log10((peak * peak) / mse))
        return {
            "image": _colorize(difference, cv2.COLORMAP_INFERNO),
            "psnr": NumberValue(value=psnr, label="PSNR", unit="dB"),
            "mse": NumberValue(value=mse, label="均方误差"),
        }


# ---------------------------------------------------------------------------
# 11. SSIM
# ---------------------------------------------------------------------------
class AnalysisSsim(Operator):
    spec = OperatorSpec(
        id="analysis.ssim",
        name="结构相似度（SSIM）",
        category=("特征与分析", "度量"),
        description=(
            "用高斯加权滑动窗口计算两幅灰度图的结构相似度（SSIM）图与平均值，"
            "SSIM 越接近 1 表示结构越相似，完全相同的图像为 1。"
            "未连接第二幅图时自动平移第一幅图派生。输出 SSIM 热力图与平均值。"
        ),
        keywords=("SSIM", "结构相似度", "图像质量", "相似度", "structural similarity"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec("image2", "参考图像（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "SSIM 热力图", accepts=("BGR",)),
            PortSpec("ssim", "SSIM 平均值", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "window",
                "窗口大小",
                "int",
                default=11,
                min=3,
                max=51,
                step=2,
                odd=True,
                group="窗口",
            ),
            ParamSpec(
                "sigma",
                "高斯标准差",
                "float",
                default=1.5,
                min=0.1,
                max=20.0,
                step=0.1,
                group="窗口",
            ),
        ),
    )

    def process(self, image=None, image2=None, window: int = 11, sigma: float = 1.5, **_: Any):
        first = _cv.source_gray8(image, "输入图像").astype(np.float64)
        second = _second_gray8(image2, _cv.source_gray8(image, "输入图像")).astype(np.float64)
        kernel = _gaussian_kernel(int(window), float(sigma))
        mu_first = cv2.filter2D(first, cv2.CV_64F, kernel, borderType=cv2.BORDER_REFLECT)
        mu_second = cv2.filter2D(second, cv2.CV_64F, kernel, borderType=cv2.BORDER_REFLECT)
        mu_first_sq = mu_first * mu_first
        mu_second_sq = mu_second * mu_second
        mu_product = mu_first * mu_second
        sigma_first_sq = np.maximum(
            cv2.filter2D(first * first, cv2.CV_64F, kernel, borderType=cv2.BORDER_REFLECT) - mu_first_sq,
            0.0,
        )
        sigma_second_sq = np.maximum(
            cv2.filter2D(second * second, cv2.CV_64F, kernel, borderType=cv2.BORDER_REFLECT) - mu_second_sq,
            0.0,
        )
        sigma_product = (
            cv2.filter2D(first * second, cv2.CV_64F, kernel, borderType=cv2.BORDER_REFLECT) - mu_product
        )
        c1 = (0.01 * 255.0) ** 2
        c2 = (0.03 * 255.0) ** 2
        numerator = (2.0 * mu_product + c1) * (2.0 * sigma_product + c2)
        denominator = (mu_first_sq + mu_second_sq + c1) * (sigma_first_sq + sigma_second_sq + c2)
        map_values = numerator / np.maximum(denominator, 1e-12)
        score = float(map_values.mean()) if map_values.size else 0.0
        display = np.clip((map_values + 1.0) * 127.5, 0.0, 255.0)
        return {
            "image": cv2.applyColorMap(np.rint(display).astype(np.uint8), cv2.COLORMAP_VIRIDIS),
            "ssim": NumberValue(value=score, label="SSIM 平均值"),
        }


# ---------------------------------------------------------------------------
# 12. 形状匹配
# ---------------------------------------------------------------------------
class AnalysisMatchShapes(Operator):
    spec = OperatorSpec(
        id="analysis.match_shapes",
        name="形状匹配",
        category=("特征与分析", "匹配"),
        description=(
            "对两幅图分别做 Otsu 二值化并取面积最大的外轮廓，用 cv2.matchShapes 比较形状相似度，"
            "同时把两个轮廓左右并排画出来。匹配度越小越相似，0 表示完全一致；-1 表示未找到轮廓。"
            "未连接第二幅图时自动平移第一幅图派生。"
        ),
        keywords=("形状匹配", "matchShapes", "轮廓", "形状相似度", "Hu"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec("image2", "参考图像（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "轮廓对比图", accepts=("BGR",)),
            PortSpec("score", "形状匹配度", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "method",
                "匹配方法",
                "enum",
                choices=(("I1", "I1（Hu 矩差值）"), ("I2", "I2"), ("I3", "I3")),
                default="I1",
            ),
            ParamSpec("color", "轮廓颜色", "color", default="#00d26a", group="显示"),
        ),
    )

    def process(self, image=None, image2=None, method: str = "I1", color: str = "#00d26a", **_: Any):
        first = _cv.source_gray8(image, "输入图像")
        second = _second_gray8(image2, first)
        first_contour = _largest_contour(first)
        second_contour = _largest_contour(second)
        left = _cv.source_bgr8(image, "输入图像")
        right = _cv.source_bgr8(second, "参考图像")
        if left.shape[:2] != right.shape[:2]:
            right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_NEAREST)
        left_panel = left.copy()
        right_panel = right.copy()
        draw_color = _cv.draw_color(left_panel, color)
        line_type = _cv.draw_type(left_panel)
        if first_contour is not None:
            cv2.drawContours(left_panel, [first_contour], -1, draw_color, 2, line_type)
        if second_contour is not None:
            cv2.drawContours(right_panel, [second_contour], -1, draw_color, 2, line_type)
        canvas = np.ascontiguousarray(np.hstack([left_panel, right_panel]))
        if first_contour is None or second_contour is None:
            return {
                "image": canvas,
                "score": NumberValue(value=-1.0, label="形状匹配度"),
            }
        method_flag = {
            "I1": cv2.CONTOURS_MATCH_I1,
            "I2": cv2.CONTOURS_MATCH_I2,
            "I3": cv2.CONTOURS_MATCH_I3,
        }.get(str(method), cv2.CONTOURS_MATCH_I1)
        score = float(cv2.matchShapes(first_contour, second_contour, method_flag, 0.0))
        return {
            "image": canvas,
            "score": NumberValue(value=score, label="形状匹配度"),
        }


OPERATORS = (
    AnalysisIntegral,
    AnalysisMinMax,
    AnalysisMeanStd,
    AnalysisMoments,
    AnalysisHuMoments,
    AnalysisFft,
    AnalysisIfft,
    AnalysisPhaseCorrelate,
    AnalysisTemplateMatch,
    AnalysisPsnr,
    AnalysisSsim,
    AnalysisMatchShapes,
)
