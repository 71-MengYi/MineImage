"""图像增强（去噪与风格化）算子。

包含非局部均值 / 彩色去噪 / TV-L1 去噪、保边滤波、细节增强、铅笔素描与
风格化。只支持 8 位的 OpenCV 接口会在内部先转 uint8 并保证输出仍有像素。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.operators.builtin.catalog import _cv
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec

_BORDERS = (
    ("reflect101", "镜像（不含边缘）"),
    ("reflect", "镜像"),
    ("replicate", "复制边缘"),
    ("constant", "常量填充（黑）"),
)

_U8_MAX = 255.0


# ---------------------------------------------------------------------------
# 通用 dtype / 通道辅助
# ---------------------------------------------------------------------------
def _is_unit_scale(array: np.ndarray) -> bool:
    if array.dtype.kind != "f" or array.size == 0:
        return False
    data = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    return float(data.min()) >= 0.0 and float(data.max()) <= 1.0


def _float_work(array: np.ndarray) -> np.ndarray:
    work = np.asarray(array, dtype=np.float32)
    if array.dtype == np.uint16:
        work = work / 257.0
    elif array.dtype.kind == "f" and _is_unit_scale(array):
        work = work * _U8_MAX
    return np.ascontiguousarray(work)


def _restore_work(work: np.ndarray, template: np.ndarray) -> np.ndarray:
    values = np.asarray(work, dtype=np.float32)
    if template.dtype == np.uint16:
        values = values * 257.0
    elif template.dtype.kind == "f":
        if _is_unit_scale(template):
            values = values / _U8_MAX
        values = np.clip(values, 0.0, _cv.full_scale(template))
    return _cv.restore(values, template)


def _split_alpha(array: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    if array.ndim == 3 and array.shape[2] == 4:
        return np.ascontiguousarray(array[:, :, :3]), array[:, :, 3]
    return array, None


def _merge_alpha(color: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    if alpha is None:
        return color
    if alpha.dtype != color.dtype:
        alpha = alpha.astype(color.dtype, copy=False)
    return np.dstack([color, alpha])


def _ensure_bgr(u8: np.ndarray) -> np.ndarray:
    if u8.ndim == 2:
        return cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    if u8.shape[2] == 4:
        return np.ascontiguousarray(u8[:, :, :3])
    return u8


def _restore_space(result: np.ndarray, u8: np.ndarray) -> np.ndarray:
    if u8.ndim == 2:
        return cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
    if u8.ndim == 3 and u8.shape[2] == 4:
        return np.dstack([result, u8[:, :, 3]])
    return result


def _denoise_channels(u8: np.ndarray, func) -> np.ndarray:
    """逐通道去噪；BGRA 保持 alpha 不变。"""
    if u8.ndim == 2:
        return func(u8)
    if u8.shape[2] == 4:
        channels = [func(np.ascontiguousarray(u8[:, :, index])) for index in range(3)]
        return np.dstack(channels + [u8[:, :, 3]])
    channels = [func(np.ascontiguousarray(u8[:, :, index])) for index in range(u8.shape[2])]
    return np.dstack(channels)


class Denoise(Operator):
    """单图去噪（非局部均值 / 中值 / 高斯）。"""

    spec = OperatorSpec(
        id="enhance.denoise",
        name="图像去噪",
        category=("图像增强",),
        description=(
            "按所选方法去除图像噪点：非局部均值适合高斯噪声，中值滤波适合椒盐噪声，"
            "高斯模糊适合快速柔化。非局部均值只支持 8 位，内部会先做 dtype 转换。"
        ),
        keywords=("去噪", "denoise", "降噪", "非局部均值", "nlmeans", "中值", "median", "高斯"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec(
                "method",
                "去噪方法",
                "enum",
                choices=(
                    ("nlmeans", "非局部均值"),
                    ("median", "中值滤波"),
                    ("gaussian", "高斯模糊"),
                ),
                default="nlmeans",
            ),
            ParamSpec(
                "h",
                "滤波强度 h",
                "float",
                default=10.0,
                min=0.1,
                max=100.0,
                step=0.5,
                group="非局部均值",
                visible_when="method == 'nlmeans'",
                tooltip="控制非局部均值的去噪强度，越大越平滑。",
            ),
            ParamSpec(
                "ksize",
                "核大小",
                "int",
                default=5,
                min=1,
                max=31,
                step=2,
                odd=True,
                unit="px",
                group="中值 / 高斯",
                visible_when="method != 'nlmeans'",
                tooltip="中值/高斯滤波的核大小；uint16/float32 的中值核最大为 5。",
            ),
        ),
    )

    def process(self, image=None, method: str = "nlmeans", h: float = 10.0, ksize: int = 5, **_: Any):
        array = _cv.source(image)
        if method == "nlmeans":
            u8 = _cv.source_u8(array)
            return _denoise_channels(
                u8,
                lambda channel: cv2.fastNlMeansDenoising(channel, None, float(h), 7, 21),
            )
        if method == "median":
            if array.dtype != np.uint8 and int(ksize) > 5:
                raise OperatorError(
                    "uint16/float32 图像的中值滤波核大小不能大于 5，请改用 3 或 5，"
                    "或先把图像转换为 8 位。"
                )
            return cv2.medianBlur(array, int(ksize))
        if method == "gaussian":
            return cv2.GaussianBlur(array, (int(ksize), int(ksize)), 0.0)
        raise OperatorError(f"不支持的去噪方法：{method}")


class DenoiseColor(Operator):
    """彩色非局部均值去噪。"""

    spec = OperatorSpec(
        id="enhance.denoise_color",
        name="彩色去噪",
        category=("图像增强",),
        description=(
            "使用彩色非局部均值（fastNlMeansDenoisingColored）同时利用亮度与色度信息去噪，"
            "比逐通道去噪更少产生色斑。只支持 8 位，内部统一转换为 uint8。"
        ),
        keywords=("彩色去噪", "denoise color", "nlmeans", "非局部均值", "降噪"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec("h", "亮度强度", "float", default=10.0, min=1.0, max=100.0, step=0.5),
            ParamSpec("h_color", "色度强度", "float", default=10.0, min=1.0, max=100.0, step=0.5),
            ParamSpec("template_window", "模板窗口", "int", default=7, min=3, max=21, step=2, odd=True),
            ParamSpec("search_window", "搜索窗口", "int", default=21, min=5, max=51, step=2, odd=True),
        ),
    )

    def process(
        self,
        image=None,
        h: float = 10.0,
        h_color: float = 10.0,
        template_window: int = 7,
        search_window: int = 21,
        **_: Any,
    ):
        u8 = _cv.source_u8(image)
        template = int(template_window)
        search = int(search_window)
        if u8.ndim == 2:
            return cv2.fastNlMeansDenoising(u8, None, float(h), template, search)
        if u8.shape[2] == 4:
            color = cv2.fastNlMeansDenoisingColored(
                np.ascontiguousarray(u8[:, :, :3]),
                None,
                float(h),
                float(h_color),
                template,
                search,
            )
            return np.dstack([color, u8[:, :, 3]])
        return cv2.fastNlMeansDenoisingColored(
            u8, None, float(h), float(h_color), template, search
        )


class DenoiseTVL1(Operator):
    """TV-L1 全变分去噪。"""

    spec = OperatorSpec(
        id="enhance.denoise_tvl1",
        name="全变分去噪",
        category=("图像增强",),
        description=(
            "用 TV-L1 全变分模型在保边的前提下抑制噪声，适合去除小幅随机噪声。"
            "OpenCV 实现只接受 8 位输入，内部会统一转 uint8 并输出 8 位图像。"
        ),
        keywords=("全变分", "tvl1", "denoise", "去噪", "保边", "变分"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec(
                "weight",
                "平滑强度 λ",
                "float",
                default=1.0,
                min=0.01,
                max=10.0,
                step=0.05,
                tooltip="越大越平滑，过大会丢失细节。",
            ),
            ParamSpec("niters", "迭代次数", "int", default=30, min=1, max=100, step=1),
        ),
    )

    def process(self, image=None, weight: float = 1.0, niters: int = 30, **_: Any):
        u8 = _cv.source_u8(image)
        iterations = int(niters)
        if u8.ndim == 2:
            result = np.empty_like(u8)
            cv2.denoise_TVL1([u8], result, float(weight), iterations)
            return result
        channels: list[np.ndarray] = []
        for index in range(u8.shape[2]):
            channel = np.ascontiguousarray(u8[:, :, index])
            result = np.empty_like(channel)
            cv2.denoise_TVL1([channel], result, float(weight), iterations)
            channels.append(result)
        return np.stack(channels, axis=2)


class EdgePreserving(Operator):
    """保边滤波（edgePreservingFilter）。"""

    spec = OperatorSpec(
        id="enhance.edge_preserving",
        name="保边滤波",
        category=("图像增强",),
        description=(
            "在平滑颜色的同时保持边缘锐利，常用于人像/绘画风格预处理。"
            "OpenCV 接口只支持 8 位三通道，内部统一转 8 位 BGR；灰度输入输出仍为灰度。"
        ),
        keywords=("保边", "edge preserving", "边缘保持", "平滑", "滤波"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec(
                "mode",
                "滤波模式",
                "enum",
                choices=(("recursive", "递归滤波"), ("normconv", "归一化卷积")),
                default="recursive",
            ),
            ParamSpec("sigma_s", "空间标准差", "float", default=60.0, min=0.0, max=200.0, step=1.0),
            ParamSpec("sigma_r", "颜色标准差", "float", default=0.4, min=0.0, max=1.0, step=0.01),
        ),
    )

    def process(
        self,
        image=None,
        mode: str = "recursive",
        sigma_s: float = 60.0,
        sigma_r: float = 0.4,
        **_: Any,
    ):
        if mode == "recursive":
            flag = cv2.RECURS_FILTER
        elif mode == "normconv":
            flag = cv2.NORMCONV_FILTER
        else:
            raise OperatorError(f"不支持的保边滤波模式：{mode}")
        u8 = _cv.source_u8(image)
        result = cv2.edgePreservingFilter(
            _ensure_bgr(u8), flags=flag, sigma_s=float(sigma_s), sigma_r=float(sigma_r)
        )
        return _restore_space(result, u8)


class DetailEnhance(Operator):
    """细节增强（可选细节参考图）。"""

    spec = OperatorSpec(
        id="enhance.detail",
        name="细节增强",
        category=("图像增强",),
        description=(
            "用高斯低频层构造高频细节并加回原图，提升纹理清晰度。"
            "可连接“细节参考图”从另一幅图像提取细节；未连接时使用输入自身的高频层。"
            "阈值用于跳过平坦区域，避免放大噪点。"
        ),
        keywords=("细节", "detail", "细节增强", "高频", "锐化", "纹理"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec(
                "guide",
                "细节参考图（可选）",
                optional=True,
                description="从中提取高频细节的图像；留空时使用输入图像自身。",
            ),
        ),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec("radius", "模糊半径", "int", default=3, min=1, max=99, step=2, odd=True, unit="px"),
            ParamSpec("amount", "细节强度", "float", default=1.0, min=0.0, max=5.0, step=0.1, unit="×"),
            ParamSpec(
                "threshold",
                "平坦区阈值",
                "int",
                default=0,
                min=0,
                max=255,
                step=1,
                tooltip="细节差值小于该阈值的像素不增强。",
            ),
            ParamSpec("border", "边界模式", "enum", choices=_BORDERS, default="reflect101"),
        ),
    )

    def process(
        self,
        image=None,
        guide=None,
        radius: int = 3,
        amount: float = 1.0,
        threshold: int = 0,
        border: str = "reflect101",
        **_: Any,
    ):
        array = _cv.source(image)
        color, alpha = _split_alpha(array)
        if guide is None:
            guide_array = array
        else:
            guide_array = _cv.source(guide, "细节参考图")
            _cv.same_size(array, guide_array, "细节参考图")

        work = _float_work(color)
        guide_work = _float_work(_cv.source_gray(guide_array))
        blurred = cv2.GaussianBlur(
            guide_work,
            (int(radius), int(radius)),
            0.0,
            borderType=_cv.match_border(border),
        )
        detail = guide_work - blurred
        if int(threshold) > 0:
            detail = np.where(np.abs(detail) >= float(threshold), detail, 0.0)
        if work.ndim == 3:
            detail = detail[:, :, None]
        result = _restore_work(work + float(amount) * detail, color)
        return _merge_alpha(result, alpha)


class PencilSketch(Operator):
    """铅笔素描。"""

    spec = OperatorSpec(
        id="enhance.pencil_sketch",
        name="铅笔素描",
        category=("图像增强",),
        description=(
            "把图像转换为铅笔素描风格，输出单通道灰度线稿。"
            "OpenCV 接口只支持 8 位三通道，内部统一转 8 位 BGR。"
        ),
        keywords=("素描", "pencil", "sketch", "铅笔画", "风格化"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像", accepts=("L",)),),
        params=(
            ParamSpec("sigma_s", "空间标准差", "float", default=60.0, min=0.0, max=200.0, step=1.0),
            ParamSpec("sigma_r", "颜色标准差", "float", default=0.07, min=0.0, max=1.0, step=0.01),
            ParamSpec("shade_factor", "明暗系数", "float", default=0.02, min=0.0, max=0.1, step=0.005),
        ),
    )

    def process(
        self,
        image=None,
        sigma_s: float = 60.0,
        sigma_r: float = 0.07,
        shade_factor: float = 0.02,
        **_: Any,
    ):
        u8 = _cv.source_u8(image)
        gray, _color = cv2.pencilSketch(
            _ensure_bgr(u8),
            sigma_s=float(sigma_s),
            sigma_r=float(sigma_r),
            shade_factor=float(shade_factor),
        )
        return gray


class Stylization(Operator):
    """风格化（油画/水彩感）。"""

    spec = OperatorSpec(
        id="enhance.stylization",
        name="风格化",
        category=("图像增强",),
        description=(
            "用保边滤波与边缘处理生成类似油画/水彩的平滑风格化效果。"
            "OpenCV 接口只支持 8 位三通道；灰度输入会先转 BGR 再转回灰度，BGRA 保留 alpha。"
        ),
        keywords=("风格化", "stylization", "油画", "水彩", "艺术效果"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec("sigma_s", "空间标准差", "float", default=60.0, min=0.0, max=200.0, step=1.0),
            ParamSpec("sigma_r", "颜色标准差", "float", default=0.45, min=0.0, max=1.0, step=0.01),
        ),
    )

    def process(self, image=None, sigma_s: float = 60.0, sigma_r: float = 0.45, **_: Any):
        u8 = _cv.source_u8(image)
        result = cv2.stylization(
            _ensure_bgr(u8), sigma_s=float(sigma_s), sigma_r=float(sigma_r)
        )
        return _restore_space(result, u8)


OPERATORS = (
    Denoise,
    DenoiseColor,
    DenoiseTVL1,
    EdgePreserving,
    DetailEnhance,
    PencilSketch,
    Stylization,
)
