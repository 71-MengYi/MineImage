"""高级滤波算子：双边、方框、可分离卷积、锐化、拉普拉斯与金字塔。

本模块按 OpenCV 的滤波族组织，一个文件导出多个算子（loader 支持
``OPERATORS``）。所有算子都通过 :mod:`app.operators.builtin.catalog._cv`
读取输入与恢复 dtype，保证 L/BGR/BGRA 与 uint8/uint16/float32 均可运行。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.operators.builtin.catalog import _cv
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec

# 0..255 参数在浮点接口里使用的“8 位语义”满量程。
_U8_MAX = 255.0


def _is_unit_scale(array: np.ndarray) -> bool:
    """判断浮点图像是否采用 ``0..1`` 归一化范围。"""
    if array.dtype.kind != "f" or array.size == 0:
        return False
    data = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return float(data.min()) >= 0.0 and float(data.max()) <= 1.0


def _float_work(array: np.ndarray) -> np.ndarray:
    """转成 float32 的「0..255 语义」工作数组，便于统一使用 0..255 参数。"""
    work = np.asarray(array, dtype=np.float32)
    if array.dtype == np.uint16:
        work = work / 257.0
    elif array.dtype.kind == "f" and _is_unit_scale(array):
        work = work * _U8_MAX
    return np.ascontiguousarray(work)


def _restore_work(work: np.ndarray, template: np.ndarray) -> np.ndarray:
    """把「0..255 语义」结果写回 ``template`` 的 dtype，并裁剪到有效范围。"""
    values = np.asarray(work, dtype=np.float32)
    if template.dtype == np.uint16:
        values = values * 257.0
    elif template.dtype.kind == "f":
        if _is_unit_scale(template):
            values = values / _U8_MAX
        span = _cv.full_scale(template)
        values = np.clip(values, 0.0, span)
    return _cv.restore(values, template)


def _split_alpha(array: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """BGRA 拆成 BGR + alpha；其它颜色空间返回 ``(array, None)``。"""
    if array.ndim == 3 and array.shape[2] == 4:
        return np.ascontiguousarray(array[:, :, :3]), array[:, :, 3]
    return array, None


def _merge_alpha(color: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if alpha is None:
        return color
    if alpha.dtype != color.dtype:
        alpha = alpha.astype(color.dtype, copy=False)
    return np.dstack([color, alpha])


def _apply_no_alpha(array: np.ndarray, func) -> np.ndarray:
    """4 通道不支持的接口先处理 BGR，再把 alpha 原样接回。"""
    color, alpha = _split_alpha(array)
    return _merge_alpha(func(color), alpha)


class BilateralFilter(Operator):
    """双边滤波：保边去噪。"""

    spec = OperatorSpec(
        id="filter.bilateral",
        name="双边滤波",
        category=("滤波与平滑",),
        description=(
            "同时考虑空间距离与颜色差异的双边滤波，平滑噪声的同时保留边缘。"
            "uint16 / float32 与 BGRA 会在内部做归一化与 alpha 分离，输出保持输入 dtype 与通道数。"
        ),
        keywords=("双边", "bilateral", "保边", "去噪", "滤波", "edge preserving"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec(
                "d",
                "邻域直径",
                "int",
                default=9,
                min=3,
                max=51,
                step=2,
                odd=True,
                unit="px",
                tooltip="参与计算的邻域直径，越大越慢、平滑范围越广。",
            ),
            ParamSpec(
                "sigma_color",
                "颜色标准差 σ",
                "float",
                default=75.0,
                min=1.0,
                max=255.0,
                step=1.0,
                tooltip="颜色差异容忍度，越大越接近普通高斯模糊。",
            ),
            ParamSpec(
                "sigma_space",
                "空间标准差 σ",
                "float",
                default=75.0,
                min=1.0,
                max=255.0,
                step=1.0,
                tooltip="空间距离权重，越大邻域影响越远。",
            ),
            ParamSpec(
                "border",
                "边界模式",
                "enum",
                choices=(
                    ("reflect101", "镜像（不含边缘）"),
                    ("reflect", "镜像"),
                    ("replicate", "复制边缘"),
                    ("constant", "常量填充（黑）"),
                ),
                default="reflect101",
            ),
        ),
    )

    def process(
        self,
        image=None,
        d: int = 9,
        sigma_color: float = 75.0,
        sigma_space: float = 75.0,
        border: str = "reflect101",
        **_: Any,
    ):
        array = _cv.source(image)
        flag = _cv.match_border(border)

        def run(source: np.ndarray) -> np.ndarray:
            work = _float_work(source)
            result = cv2.bilateralFilter(
                work, int(d), float(sigma_color), float(sigma_space), borderType=flag
            )
            return _restore_work(result, source)

        return _apply_no_alpha(array, run)


class BoxFilter(Operator):
    """方框滤波（均值或邻域求和）。"""

    spec = OperatorSpec(
        id="filter.box",
        name="方框滤波",
        category=("滤波与平滑",),
        description=(
            "用矩形邻域做方框滤波。归一化时为均值模糊；关闭归一化则输出邻域和，"
            "适合观察局部能量。uint8/uint16/float32 与 BGRA 均原生支持。"
        ),
        keywords=("方框", "box", "均值", "mean", "求和", "滤波"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec(
                "ksize",
                "核大小",
                "int",
                default=5,
                min=1,
                max=99,
                step=2,
                odd=True,
                unit="px",
                tooltip="矩形邻域的边长，必须为奇数。",
            ),
            ParamSpec(
                "normalize",
                "归一化（均值）",
                "bool",
                default=True,
                tooltip="关闭后输出邻域像素之和，可能被 dtype 上限截断。",
            ),
            ParamSpec(
                "border",
                "边界模式",
                "enum",
                choices=(
                    ("reflect101", "镜像（不含边缘）"),
                    ("reflect", "镜像"),
                    ("replicate", "复制边缘"),
                    ("constant", "常量填充（黑）"),
                ),
                default="reflect101",
            ),
        ),
    )

    def process(
        self,
        image=None,
        ksize: int = 5,
        normalize: bool = True,
        border: str = "reflect101",
        **_: Any,
    ):
        array = _cv.source(image)
        return cv2.boxFilter(
            array,
            -1,
            (int(ksize), int(ksize)),
            normalize=bool(normalize),
            borderType=_cv.match_border(border),
        )


class SquareBoxFilter(Operator):
    """平方方框滤波（输出均方根）。"""

    spec = OperatorSpec(
        id="filter.sqr_box",
        name="平方方框滤波",
        category=("滤波与平滑",),
        description=(
            "先对邻域像素平方求均值，再开方得到均方根强度，适合观察局部能量分布。"
            "结果会裁剪回输入 dtype；float32 的 0..1 归一化图像同样适用。"
        ),
        keywords=("平方", "sqrbox", "均方根", "rms", "能量", "滤波"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec(
                "ksize",
                "核大小",
                "int",
                default=5,
                min=1,
                max=99,
                step=2,
                odd=True,
                unit="px",
                tooltip="矩形邻域的边长，必须为奇数。",
            ),
            ParamSpec(
                "border",
                "边界模式",
                "enum",
                choices=(
                    ("reflect101", "镜像（不含边缘）"),
                    ("replicate", "复制边缘"),
                    ("constant", "常量填充（黑）"),
                ),
                default="reflect101",
            ),
        ),
    )

    def process(self, image=None, ksize: int = 5, border: str = "reflect101", **_: Any):
        array = _cv.source(image)
        squared = cv2.sqrBoxFilter(
            array.astype(np.float32),
            cv2.CV_32F,
            (int(ksize), int(ksize)),
            normalize=True,
            borderType=_cv.match_border(border),
        )
        return _cv.restore(np.sqrt(np.maximum(squared, 0.0)), array)


class Filter2D(Operator):
    """自定义卷积（预设核）。"""

    spec = OperatorSpec(
        id="filter.filter2d",
        name="自定义卷积",
        category=("滤波与平滑",),
        description=(
            "用内置预设卷积核做二维卷积：锐化、边缘、浮雕或邻域平均。"
            "强度会整体缩放卷积核，默认 1.0 即经典核；支持全部颜色空间与 dtype。"
        ),
        keywords=("卷积", "filter2d", "kernel", "锐化", "边缘", "浮雕", "convolution"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec(
                "preset",
                "预设卷积核",
                "enum",
                choices=(
                    ("sharpen", "锐化"),
                    ("edge", "边缘强化"),
                    ("emboss", "浮雕"),
                    ("blur", "邻域平均"),
                ),
                default="sharpen",
            ),
            ParamSpec(
                "strength",
                "强度",
                "float",
                default=1.0,
                min=0.0,
                max=5.0,
                step=0.1,
                unit="×",
                tooltip="整体缩放卷积核；邻域平均核大于 1 会变亮。",
            ),
            ParamSpec(
                "border",
                "边界模式",
                "enum",
                choices=(
                    ("reflect101", "镜像（不含边缘）"),
                    ("reflect", "镜像"),
                    ("replicate", "复制边缘"),
                    ("constant", "常量填充（黑）"),
                ),
                default="reflect101",
            ),
        ),
    )

    _KERNELS = {
        "sharpen": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
        "edge": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
        "emboss": np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32),
        "blur": np.full((3, 3), 1.0 / 9.0, dtype=np.float32),
    }

    def process(
        self,
        image=None,
        preset: str = "sharpen",
        strength: float = 1.0,
        border: str = "reflect101",
        **_: Any,
    ):
        kernel = self._KERNELS.get(str(preset))
        if kernel is None:
            raise OperatorError(f"不支持的卷积核预设：{preset}")
        array = _cv.source(image)
        work = _float_work(array)
        result = cv2.filter2D(
            work,
            -1,
            (kernel * float(strength)).astype(np.float32),
            borderType=_cv.match_border(border),
        )
        return _restore_work(result, array)


class SepFilter(Operator):
    """可分离卷积（高斯 1D 核）。"""

    spec = OperatorSpec(
        id="filter.sep_filter",
        name="可分离卷积",
        category=("滤波与平滑",),
        description=(
            "用两个一维高斯核分别沿 X/Y 方向做可分离卷积，速度优于同等二维核。"
            "sigma 为 0 时按核大小自动计算；输出与输入同尺寸、同 dtype。"
        ),
        keywords=("可分离", "sepfilter", "高斯", "gaussian", "卷积", "滤波"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec(
                "ksize",
                "核大小",
                "int",
                default=5,
                min=1,
                max=99,
                step=2,
                odd=True,
                unit="px",
            ),
            ParamSpec(
                "sigma",
                "标准差 σ",
                "float",
                default=1.2,
                min=0.0,
                max=50.0,
                step=0.1,
                tooltip="0 表示根据核大小自动计算。",
            ),
            ParamSpec(
                "border",
                "边界模式",
                "enum",
                choices=(
                    ("reflect101", "镜像（不含边缘）"),
                    ("reflect", "镜像"),
                    ("replicate", "复制边缘"),
                    ("constant", "常量填充（黑）"),
                ),
                default="reflect101",
            ),
        ),
    )

    def process(self, image=None, ksize: int = 5, sigma: float = 1.2, border: str = "reflect101", **_: Any):
        array = _cv.source(image)
        work = _float_work(array)
        kernel = cv2.getGaussianKernel(int(ksize), float(sigma)).astype(np.float32)
        result = cv2.sepFilter2D(
            work,
            -1,
            kernel,
            kernel,
            borderType=_cv.match_border(border),
        )
        return _restore_work(result, array)


class Sharpen(Operator):
    """锐化（中心加权的拉普拉斯锐化核）。"""

    spec = OperatorSpec(
        id="filter.sharpen",
        name="锐化",
        category=("滤波与平滑",),
        description=(
            "用中心加权卷积核增强边缘对比度。强度越大中心权重越高；"
            "对 uint8/uint16/float32 与 L/BGR/BGRA 均可用。"
        ),
        keywords=("锐化", "sharpen", "清晰", "边缘", "unsharp"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec(
                "amount",
                "锐化强度",
                "float",
                default=1.0,
                min=0.0,
                max=4.0,
                step=0.1,
                unit="×",
                tooltip="0 表示不锐化（恒等），1 为经典锐化核。",
            ),
            ParamSpec(
                "border",
                "边界模式",
                "enum",
                choices=(
                    ("reflect101", "镜像（不含边缘）"),
                    ("reflect", "镜像"),
                    ("replicate", "复制边缘"),
                    ("constant", "常量填充（黑）"),
                ),
                default="reflect101",
            ),
        ),
    )

    def process(self, image=None, amount: float = 1.0, border: str = "reflect101", **_: Any):
        array = _cv.source(image)
        work = _float_work(array)
        kernel = np.array(
            [[0.0, -1.0, 0.0], [-1.0, 4.0 + float(amount), -1.0], [0.0, -1.0, 0.0]],
            dtype=np.float32,
        )
        result = cv2.filter2D(work, -1, kernel, borderType=_cv.match_border(border))
        return _restore_work(result, array)


class UnsharpMask(Operator):
    """非锐化掩模（Unsharp Mask）。"""

    spec = OperatorSpec(
        id="filter.unsharp",
        name="非锐化掩模",
        category=("滤波与平滑",),
        description=(
            "用高斯模糊构造低频层，再把原图与低频层的差值按强度加回，得到可调的非锐化掩模锐化。"
            "阈值用于跳过平坦区域，避免放大噪点。"
        ),
        keywords=("非锐化", "unsharp", "锐化", "usm", "细节", "高斯"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec(
                "radius",
                "模糊半径",
                "int",
                default=5,
                min=1,
                max=99,
                step=2,
                odd=True,
                unit="px",
            ),
            ParamSpec(
                "amount",
                "增强强度",
                "float",
                default=1.0,
                min=0.0,
                max=5.0,
                step=0.1,
                unit="×",
            ),
            ParamSpec(
                "threshold",
                "平坦区阈值",
                "int",
                default=0,
                min=0,
                max=255,
                step=1,
                tooltip="差值小于该阈值的像素不增强，用于抑制噪点。",
            ),
            ParamSpec(
                "border",
                "边界模式",
                "enum",
                choices=(
                    ("reflect101", "镜像（不含边缘）"),
                    ("replicate", "复制边缘"),
                    ("constant", "常量填充（黑）"),
                ),
                default="reflect101",
            ),
        ),
    )

    def process(
        self,
        image=None,
        radius: int = 5,
        amount: float = 1.0,
        threshold: int = 0,
        border: str = "reflect101",
        **_: Any,
    ):
        array = _cv.source(image)
        work = _float_work(array)
        blurred = cv2.GaussianBlur(
            work,
            (int(radius), int(radius)),
            0.0,
            borderType=_cv.match_border(border),
        )
        detail = work - blurred
        if int(threshold) > 0:
            mask = np.abs(detail) >= float(threshold)
            detail = np.where(mask, detail, 0.0)
        result = work + float(amount) * detail
        return _restore_work(result, array)


class LaplacianFilter(Operator):
    """拉普拉斯滤波（取绝对值后的边缘响应）。"""

    spec = OperatorSpec(
        id="filter.laplacian",
        name="拉普拉斯滤波",
        category=("滤波与平滑",),
        description=(
            "用拉普拉斯算子提取二阶边缘响应，并取绝对值方便显示。"
            "核大小 1/3/5/7 可选；对全部颜色空间与 dtype 可用。"
        ),
        keywords=("拉普拉斯", "laplacian", "二阶", "边缘", "锐化"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec(
                "ksize",
                "核大小",
                "enum",
                choices=((1, "1×1"), (3, "3×3"), (5, "5×5"), (7, "7×7")),
                default=3,
                tooltip="拉普拉斯算子的孔径大小。",
            ),
            ParamSpec(
                "scale",
                "响应缩放",
                "float",
                default=1.0,
                min=0.0,
                max=10.0,
                step=0.1,
                unit="×",
            ),
            ParamSpec(
                "border",
                "边界模式",
                "enum",
                choices=(
                    ("reflect101", "镜像（不含边缘）"),
                    ("replicate", "复制边缘"),
                    ("constant", "常量填充（黑）"),
                ),
                default="reflect101",
            ),
        ),
    )

    def process(self, image=None, ksize=3, scale: float = 1.0, border: str = "reflect101", **_: Any):
        array = _cv.source(image)
        work = _float_work(array)
        response = cv2.Laplacian(
            work,
            cv2.CV_32F,
            ksize=int(ksize),
            scale=float(scale),
            borderType=_cv.match_border(border),
        )
        return _restore_work(np.abs(response), array)


class PyrDown(Operator):
    """金字塔下采样。"""

    spec = OperatorSpec(
        id="filter.pyrdown",
        name="金字塔下采样",
        category=("滤波与平滑",),
        description=(
            "先用高斯核平滑再去掉偶数行/列，得到宽高各约一半的图像。"
            "输出宽高为 ceil(宽/2) × ceil(高/2)，保持输入 dtype 与通道数。"
        ),
        keywords=("金字塔", "pyrdown", "下采样", "降采样", "多尺度"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
    )

    def process(self, image=None, **_: Any):
        return cv2.pyrDown(_cv.source(image))


class PyrUp(Operator):
    """金字塔上采样。"""

    spec = OperatorSpec(
        id="filter.pyrup",
        name="金字塔上采样",
        category=("滤波与平滑",),
        description=(
            "在偶数行/列之间插入零并做高斯平滑，得到宽高约两倍的图像。"
            "输出为输入的 2 倍尺寸，保持输入 dtype 与通道数。"
        ),
        keywords=("金字塔", "pyrup", "上采样", "升采样", "多尺度"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
    )

    def process(self, image=None, **_: Any):
        return cv2.pyrUp(_cv.source(image))


class PyrMeanShift(Operator):
    """均值漂移滤波（彩色）。"""

    spec = OperatorSpec(
        id="filter.pyr_mean_shift",
        name="均值漂移滤波",
        category=("滤波与平滑",),
        description=(
            "金字塔均值漂移滤波，把颜色相近的邻域收敛到同一颜色，产生绘画般的平滑效果。"
            "该 OpenCV 接口只支持 uint8 三通道，内部会统一转换为 8 位 BGR（灰度输入输出仍为灰度）。"
        ),
        keywords=("均值漂移", "mean shift", "pyrMeanShift", "彩色平滑", "风格化"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec(
                "sp",
                "空间半径",
                "float",
                default=10.0,
                min=1.0,
                max=100.0,
                step=1.0,
                unit="px",
                tooltip="空间窗口半径，越大平滑范围越广。",
            ),
            ParamSpec(
                "sr",
                "颜色半径",
                "float",
                default=20.0,
                min=1.0,
                max=255.0,
                step=1.0,
                tooltip="颜色窗口半径，越大颜色合并越强。",
            ),
            ParamSpec(
                "max_level",
                "金字塔层数",
                "int",
                default=1,
                min=0,
                max=5,
                step=1,
            ),
        ),
    )

    def process(self, image=None, sp: float = 10.0, sr: float = 20.0, max_level: int = 1, **_: Any):
        u8 = _cv.source_u8(image)
        if u8.ndim == 2:
            bgr = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
            result = cv2.pyrMeanShiftFiltering(bgr, float(sp), float(sr), maxLevel=int(max_level))
            return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
        if u8.ndim == 3 and u8.shape[2] == 4:
            result = cv2.pyrMeanShiftFiltering(
                np.ascontiguousarray(u8[:, :, :3]), float(sp), float(sr), maxLevel=int(max_level)
            )
            return np.dstack([result, u8[:, :, 3]])
        return cv2.pyrMeanShiftFiltering(u8, float(sp), float(sr), maxLevel=int(max_level))


OPERATORS = (
    BilateralFilter,
    BoxFilter,
    SquareBoxFilter,
    Filter2D,
    SepFilter,
    Sharpen,
    UnsharpMask,
    LaplacianFilter,
    PyrDown,
    PyrUp,
    PyrMeanShift,
)
