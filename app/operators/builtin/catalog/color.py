"""OpenCV 目录算子：通道与色彩。

本模块把颜色空间转换、通道交换、通道混合、伪彩色、白平衡 / 色温、alpha 合成与
查找曲线导出为多个算子。约定与 ``arithmetic.py`` 一致：

* 统一用 :mod:`app.operators.builtin.catalog._cv` 读取输入、校验尺寸并恢复 dtype；
* 支持 L / BGR / BGRA 与 uint8 / uint16 / float32；
* 需要 8 位输入的 OpenCV 接口（颜色空间转换、CLAHE 等）在内部转 uint8，
  再按满量程恢复到输入 dtype；
* 返回新数组，不修改输入。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.operators.builtin.catalog import _cv
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec

_U8 = 255.0

#: 伪彩色可选项。
_COLORMAPS = {
    "jet": ("Jet 彩虹", cv2.COLORMAP_JET),
    "turbo": ("Turbo 高对比", cv2.COLORMAP_TURBO),
    "rainbow": ("Rainbow 彩虹", cv2.COLORMAP_RAINBOW),
    "viridis": ("Viridis 感知均匀", cv2.COLORMAP_VIRIDIS),
    "inferno": ("Inferno 火焰", cv2.COLORMAP_INFERNO),
    "magma": ("Magma 熔岩", cv2.COLORMAP_MAGMA),
    "plasma": ("Plasma 等离子", cv2.COLORMAP_PLASMA),
    "hot": ("Hot 热力", cv2.COLORMAP_HOT),
    "bone": ("Bone 骨白", cv2.COLORMAP_BONE),
    "cool": ("Cool 冷色", cv2.COLORMAP_COOL),
    "spring": ("Spring 春色", cv2.COLORMAP_SPRING),
    "autumn": ("Autumn 秋色", cv2.COLORMAP_AUTUMN),
    "winter": ("Winter 冬色", cv2.COLORMAP_WINTER),
    "ocean": ("Ocean 海洋", cv2.COLORMAP_OCEAN),
}

#: 颜色空间转换目标。
_CVT_TARGETS = (
    ("gray", "灰度（L）"),
    ("bgr", "BGR 彩色"),
    ("rgb", "RGB 顺序"),
    ("hsv", "HSV"),
    ("lab", "Lab"),
    ("ycrcb", "YCrCb"),
    ("hls", "HLS"),
)


# ---------------------------------------------------------------------------
# 共享辅助
# ---------------------------------------------------------------------------
def _unit_like(reference: np.ndarray, other: np.ndarray) -> np.ndarray:
    """把 ``other`` 映射到 0..1 并对齐到 ``reference`` 的通道数。"""
    unit = _cv.unit_array(other)
    ref_channels = _cv.channels(reference)
    other_channels = _cv.channels(unit)
    if ref_channels == other_channels:
        return unit
    if ref_channels == 1:
        if other_channels >= 3:
            return 0.299 * unit[..., 2] + 0.587 * unit[..., 1] + 0.114 * unit[..., 0]
        return unit
    if other_channels == 1:
        return np.repeat(unit[:, :, None], ref_channels, axis=2)
    if ref_channels == 3:
        return unit[:, :, :3]
    alpha = np.ones((unit.shape[0], unit.shape[1], 1), dtype=np.float32)
    return np.concatenate([unit[:, :, :3], alpha], axis=2)


def _finish_u8(work: np.ndarray, template: np.ndarray) -> np.ndarray:
    """把 uint8 工作结果恢复为 ``template`` 的通道数 / dtype（BGRA 保留原 alpha）。"""
    work = np.asarray(work)
    if template.ndim == 2:
        if work.ndim == 3:
            code = cv2.COLOR_BGR2GRAY if work.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
            work = cv2.cvtColor(work, code)
    elif template.shape[2] == 4:
        if work.ndim == 2:
            work = np.repeat(work[:, :, None], 3, axis=2)
        if work.shape[2] == 3:
            alpha = _cv.to_uint8(template[:, :, 3])
            work = np.concatenate([work, alpha[:, :, None]], axis=2)
    else:  # 三通道模板
        if work.ndim == 2:
            work = np.repeat(work[:, :, None], 3, axis=2)
        elif work.shape[2] == 4:
            work = work[:, :, :3]
    if template.dtype == np.uint8:
        return np.ascontiguousarray(work)
    return _cv.from_unit(work.astype(np.float32) / _U8, template)


def _curve_table(
    preset: str,
    strength: float,
    points,
) -> np.ndarray:
    """生成 256 级查找表；自定义控制点优先于预设曲线。"""
    x = np.linspace(0.0, 1.0, 256, dtype=np.float32)
    strength = float(np.clip(float(strength), 0.0, 1.0))
    if preset == "identity":
        y = x
    elif preset == "invert":
        y = 1.0 - x
    elif preset == "brighten":
        y = np.power(x, 1.0 / (1.0 + 1.8 * strength))
    elif preset == "darken":
        y = np.power(x, 1.0 + 1.8 * strength)
    else:  # s_curve
        smooth = x * x * (3.0 - 2.0 * x)
        y = (1.0 - strength) * x + strength * smooth
    if points and len(points) >= 2:
        ordered = sorted((float(px), float(py)) for px, py in points)
        xs = np.clip(np.array([item[0] for item in ordered], dtype=np.float64), 0.0, 255.0)
        ys = np.clip(np.array([item[1] for item in ordered], dtype=np.float64), 0.0, 255.0) / _U8
        order = np.argsort(xs)
        y = np.interp(np.arange(256, dtype=np.float64), xs[order], ys[order]).astype(np.float32)
    return np.clip(np.rint(y * _U8), 0, 255).astype(np.uint8)


def _apply_table(array: np.ndarray, table: np.ndarray) -> np.ndarray:
    """把 256 级查找表应用到整幅图像，保持 dtype 与通道数。"""
    alpha = array[:, :, 3:4] if (array.ndim == 3 and array.shape[2] == 4) else None
    core = array[:, :, :3] if alpha is not None else array
    if core.dtype == np.uint8:
        mapped = cv2.LUT(core, table)
    else:
        scale = _cv.full_scale(core)
        domain = np.linspace(0.0, scale, 256, dtype=np.float32)
        values = table.astype(np.float32) / _U8 * scale
        work = np.nan_to_num(
            core.astype(np.float32, copy=False), nan=0.0, posinf=scale, neginf=0.0
        )
        mapped = np.interp(work.reshape(-1), domain, values).reshape(core.shape)
        if core.dtype.kind == "f":
            mapped = mapped.astype(np.float32)
        else:
            mapped = _cv.restore(mapped, core)
    if alpha is not None:
        return np.concatenate([mapped, alpha.copy()], axis=2)
    return mapped


# ---------------------------------------------------------------------------
# 颜色空间与通道
# ---------------------------------------------------------------------------
class ColorConvert(Operator):
    """颜色空间转换。"""

    spec = OperatorSpec(
        id="color.cvt",
        name="颜色空间转换",
        category=("通道与色彩",),
        description=(
            "把图像转换到指定颜色空间。灰度为单通道输出；BGR / RGB 保持输入 dtype；"
            "HSV / HLS / Lab 只支持 8 位接口，内部先转 uint8 再输出 8 位三通道图。"
        ),
        keywords=("颜色空间", "cvt", "color", "hsv", "lab", "ycrcb", "灰度"),
        inputs=(PortSpec("image", "输入图像", description="要转换颜色空间的图像，支持 L / BGR / BGRA。"),),
        outputs=(PortSpec("image", "转换结果"),),
        params=(
            ParamSpec(
                "target",
                "目标颜色空间",
                "enum",
                choices=_CVT_TARGETS,
                default="gray",
                tooltip="输出通道顺序与 OpenCV 一致（HLS/HSV 的 H 范围为 0..180）。",
            ),
        ),
    )

    def process(self, image=None, target: str = "gray", **_):
        array = _cv.source(image)
        if target == "gray":
            return _cv.source_gray(array)
        if target == "bgr":
            return _cv.source_bgr(array)
        if target == "rgb":
            bgr = _cv.source_bgr(array)
            return np.ascontiguousarray(bgr[:, :, ::-1])
        if target == "ycrcb":
            # YCrCb 的 cvtColor 支持全部原生 dtype，直接保留 dtype。
            return cv2.cvtColor(_cv.source_bgr(array), cv2.COLOR_BGR2YCrCb)
        work = _cv.source_bgr8(array)
        codes = {
            "hsv": cv2.COLOR_BGR2HSV,
            "hls": cv2.COLOR_BGR2HLS,
            "lab": cv2.COLOR_BGR2Lab,
        }
        if target not in codes:
            raise OperatorError(f"不支持的目标颜色空间：{target}")
        return cv2.cvtColor(work, codes[target])


class ChannelSwap(Operator):
    """通道交换。"""

    spec = OperatorSpec(
        id="color.swap",
        name="通道交换",
        category=("通道与色彩",),
        description=(
            "按指定顺序重排 B / G / R 三个通道，例如把 BGR 变成 RGB。"
            "BGRA 输入的透明通道保持不变；灰度图原样返回。"
        ),
        keywords=("通道交换", "swap", "rgb", "bgr", "重排"),
        inputs=(PortSpec("image", "输入图像", description="要重排通道的图像，支持 L / BGR / BGRA。"),),
        outputs=(PortSpec("image", "交换结果"),),
        params=(
            ParamSpec(
                "order",
                "输出通道顺序",
                "enum",
                choices=(
                    ("bgr", "BGR（原序）"),
                    ("brg", "BRG"),
                    ("gbr", "GBR"),
                    ("grb", "GRB"),
                    ("rbg", "RBG"),
                    ("rgb", "RGB"),
                ),
                default="rgb",
                tooltip="按该顺序输出三个颜色通道。",
            ),
        ),
    )

    _PERMUTATIONS = {
        "bgr": (0, 1, 2),
        "brg": (0, 2, 1),
        "gbr": (1, 0, 2),
        "grb": (1, 2, 0),
        "rbg": (2, 0, 1),
        "rgb": (2, 1, 0),
    }

    def process(self, image=None, order: str = "rgb", **_):
        array = _cv.source(image)
        if array.ndim == 2:
            return array.copy()
        permutation = self._PERMUTATIONS.get(order)
        if permutation is None:
            raise OperatorError(f"不支持的通道顺序：{order}")
        core = array[:, :, :3][:, :, permutation]
        if array.shape[2] == 4:
            return np.ascontiguousarray(
                np.concatenate([core, array[:, :, 3:4]], axis=2)
            )
        return np.ascontiguousarray(core)


class HsvAdjust(Operator):
    """HSV 调整。"""

    spec = OperatorSpec(
        id="color.hsv_adjust",
        name="HSV 调整",
        category=("通道与色彩",),
        description=(
            "在 HSV 空间分别调整色相、饱和度和明度，再转回 BGR。"
            "色相按 0..180 环形偏移；饱和度和明度用倍数缩放。结果恢复到输入 dtype / 通道数。"
        ),
        keywords=("hsv", "色相", "饱和度", "明度", "hue", "saturation", "value"),
        inputs=(PortSpec("image", "输入图像", description="要调整颜色的图像，支持 L / BGR / BGRA。"),),
        outputs=(PortSpec("image", "调整结果"),),
        params=(
            ParamSpec(
                "hue_shift",
                "色相偏移",
                "int",
                default=0,
                min=-90,
                max=90,
                step=1,
                unit="度",
                tooltip="在 0..180 的 OpenCV 色相环上环形偏移，正负方向均可。",
            ),
            ParamSpec(
                "saturation",
                "饱和度倍数",
                "float",
                default=1.0,
                min=0.0,
                max=3.0,
                step=0.05,
                unit="×",
                tooltip="0 表示完全去色，1.0 表示不变。",
            ),
            ParamSpec(
                "brightness",
                "明度倍数",
                "float",
                default=1.0,
                min=0.0,
                max=3.0,
                step=0.05,
                unit="×",
                tooltip="1.0 表示不变，小于 1 变暗。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        hue_shift: int = 0,
        saturation: float = 1.0,
        brightness: float = 1.0,
        **_,
    ):
        array = _cv.source(image)
        if int(hue_shift) == 0 and float(saturation) == 1.0 and float(brightness) == 1.0:
            # 中性参数保持逐位不变，避免 HSV 往返的舍入误差。
            return array.copy()
        work = _cv.source_bgr8(array)
        hsv = cv2.cvtColor(work, cv2.COLOR_BGR2HSV)
        if int(hue_shift) % 180:
            hue = hsv[:, :, 0].astype(np.int16) + int(hue_shift)
            hsv[:, :, 0] = np.mod(hue, 180).astype(np.uint8)
        if float(saturation) != 1.0:
            sat = hsv[:, :, 1].astype(np.float32) * float(saturation)
            hsv[:, :, 1] = np.clip(np.rint(sat), 0, 255).astype(np.uint8)
        if float(brightness) != 1.0:
            val = hsv[:, :, 2].astype(np.float32) * float(brightness)
            hsv[:, :, 2] = np.clip(np.rint(val), 0, 255).astype(np.uint8)
        adjusted = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        return _finish_u8(adjusted, array)


class ChannelMix(Operator):
    """通道混合（3×3 矩阵）。"""

    spec = OperatorSpec(
        id="color.mix",
        name="通道混合",
        category=("通道与色彩",),
        description=(
            "用 3×3 权重矩阵把输入的 B / G / R 重新组合成输出的 B / G / R，"
            "可实现去色、通道互换、色彩分离等效果。默认矩阵为恒等变换；BGRA 保留 alpha。"
        ),
        keywords=("通道混合", "mix", "channel", "矩阵", "色彩分离"),
        inputs=(PortSpec("image", "输入图像", description="要混合通道的图像，支持 L / BGR / BGRA。"),),
        outputs=(PortSpec("image", "混合结果"),),
        params=(
            ParamSpec("b_from_b", "蓝 ← 蓝", "float", default=1.0, min=-1.0, max=2.0, step=0.05, group="蓝色输出"),
            ParamSpec("b_from_g", "蓝 ← 绿", "float", default=0.0, min=-1.0, max=2.0, step=0.05, group="蓝色输出"),
            ParamSpec("b_from_r", "蓝 ← 红", "float", default=0.0, min=-1.0, max=2.0, step=0.05, group="蓝色输出"),
            ParamSpec("g_from_b", "绿 ← 蓝", "float", default=0.0, min=-1.0, max=2.0, step=0.05, group="绿色输出"),
            ParamSpec("g_from_g", "绿 ← 绿", "float", default=1.0, min=-1.0, max=2.0, step=0.05, group="绿色输出"),
            ParamSpec("g_from_r", "绿 ← 红", "float", default=0.0, min=-1.0, max=2.0, step=0.05, group="绿色输出"),
            ParamSpec("r_from_b", "红 ← 蓝", "float", default=0.0, min=-1.0, max=2.0, step=0.05, group="红色输出"),
            ParamSpec("r_from_g", "红 ← 绿", "float", default=0.0, min=-1.0, max=2.0, step=0.05, group="红色输出"),
            ParamSpec("r_from_r", "红 ← 红", "float", default=1.0, min=-1.0, max=2.0, step=0.05, group="红色输出"),
        ),
    )

    def process(
        self,
        image=None,
        b_from_b: float = 1.0,
        b_from_g: float = 0.0,
        b_from_r: float = 0.0,
        g_from_b: float = 0.0,
        g_from_g: float = 1.0,
        g_from_r: float = 0.0,
        r_from_b: float = 0.0,
        r_from_g: float = 0.0,
        r_from_r: float = 1.0,
        **_,
    ):
        array = _cv.source(image)
        if array.ndim == 2:
            # 灰度图没有可混合的颜色通道，原样返回副本。
            return array.copy()
        unit = _cv.unit_array(array)
        blue = unit[:, :, 0]
        green = unit[:, :, 1]
        red = unit[:, :, 2]
        out_blue = b_from_b * blue + b_from_g * green + b_from_r * red
        out_green = g_from_b * blue + g_from_g * green + g_from_r * red
        out_red = r_from_b * blue + r_from_g * green + r_from_r * red
        result = np.stack([out_blue, out_green, out_red], axis=2)
        if unit.shape[2] == 4:
            result = np.concatenate([result, unit[:, :, 3:4]], axis=2)
        return _cv.from_unit(np.clip(result, 0.0, 1.0), array)


class Palette(Operator):
    """伪彩色映射。"""

    spec = OperatorSpec(
        id="color.palette",
        name="伪彩色",
        category=("通道与色彩",),
        description=(
            "把灰度（或彩色图像的亮度）按指定色图映射为伪彩色，输出 BGR 三通道图像。"
            "适合热力图、深度图与特征响应可视化。"
        ),
        keywords=("伪彩色", "palette", "colormap", "热力图", "jet"),
        inputs=(PortSpec("image", "输入图像", description="要映射为伪彩色的图像，支持 L / BGR / BGRA。"),),
        outputs=(PortSpec("image", "伪彩色图像", accepts=("BGR",)),),
        params=(
            ParamSpec(
                "colormap",
                "色图",
                "enum",
                choices=tuple((key, label) for key, (label, _flag) in _COLORMAPS.items()),
                default="jet",
                tooltip="OpenCV 内置色图，jet 适合热力图，viridis 更感知均匀。",
            ),
        ),
    )

    def process(self, image=None, colormap: str = "jet", **_):
        entry = _COLORMAPS.get(colormap)
        if entry is None:
            raise OperatorError(f"不支持的色图：{colormap}")
        gray = _cv.source_gray8(image)
        return cv2.applyColorMap(gray, entry[1])


class WhiteBalance(Operator):
    """灰度世界白平衡。"""

    spec = OperatorSpec(
        id="color.white_balance",
        name="白平衡",
        category=("通道与色彩",),
        description=(
            "按灰度世界假设估计每个通道的增益，把偏色图像的通道均值拉平。"
            "“强度”控制校正力度；灰度图原样返回。结果保持输入 dtype / 通道数。"
        ),
        keywords=("白平衡", "white_balance", "偏色", "gray_world", "色偏"),
        inputs=(PortSpec("image", "输入图像", description="需要校正偏色的图像，支持 L / BGR / BGRA。"),),
        outputs=(PortSpec("image", "白平衡结果"),),
        params=(
            ParamSpec(
                "strength",
                "校正强度",
                "float",
                default=1.0,
                min=0.0,
                max=1.0,
                step=0.05,
                tooltip="0 表示不校正，1 表示完全把通道均值拉平。",
            ),
        ),
    )

    def process(self, image=None, strength: float = 1.0, **_):
        array = _cv.source(image)
        if array.ndim == 2:
            return array.copy()
        if float(strength) <= 0.0:
            return array.copy()
        work = _cv.source_bgr8(array)
        means = work.reshape(-1, 3).mean(axis=0).astype(np.float64)
        average = float(means.mean())
        if average <= 1e-6:
            return _finish_u8(work, array)
        gains = average / np.maximum(means, 1e-6)
        gains = 1.0 + (gains - 1.0) * float(strength)
        adjusted = np.clip(
            work.astype(np.float32) * gains.astype(np.float32).reshape(1, 1, 3), 0.0, _U8
        )
        return _finish_u8(np.rint(adjusted).astype(np.uint8), array)


class Temperature(Operator):
    """色温调整。"""

    spec = OperatorSpec(
        id="color.temperature",
        name="色温",
        category=("通道与色彩",),
        description=(
            "模拟色温变化：正值偏暖（增强红、削弱蓝），负值偏冷。"
            "调整在 0..1 单位域完成，保持输入 dtype 与通道数；灰度图原样返回。"
        ),
        keywords=("色温", "temperature", "暖色", "冷色", "白平衡"),
        inputs=(PortSpec("image", "输入图像", description="要调整色温的图像，支持 L / BGR / BGRA。"),),
        outputs=(PortSpec("image", "色温结果"),),
        params=(
            ParamSpec(
                "temperature",
                "色温偏移",
                "int",
                default=20,
                min=-100,
                max=100,
                step=1,
                tooltip="正值偏暖，负值偏冷，0 表示不变。",
            ),
        ),
    )

    def process(self, image=None, temperature: int = 20, **_):
        array = _cv.source(image)
        if array.ndim == 2:
            return array.copy()
        if int(temperature) == 0:
            return array.copy()
        unit = _cv.unit_array(array)
        amount = float(temperature) / 100.0
        gains = np.array([1.0 - 0.3 * amount, 1.0, 1.0 + 0.3 * amount], dtype=np.float32)
        result = unit.copy()
        result[:, :, :3] *= gains.reshape(1, 1, 3)
        return _cv.from_unit(np.clip(result, 0.0, 1.0), array)


# ---------------------------------------------------------------------------
# Alpha 与曲线
# ---------------------------------------------------------------------------
class AlphaComposite(Operator):
    """Alpha 合成。"""

    spec = OperatorSpec(
        id="color.alpha_composite",
        name="Alpha 合成",
        category=("通道与色彩",),
        description=(
            "把第二幅图像按 alpha 叠加到第一幅图上：第二幅为 BGRA 时使用其透明通道，"
            "否则使用“不透明度”。未连接第二幅图时改为叠加纯色。BGRA 背景保留原 alpha。"
        ),
        keywords=("alpha", "合成", "composite", "叠加", "透明度", "overlay"),
        inputs=(
            PortSpec("image", "输入图像", description="被叠加的背景图，支持 L / BGR / BGRA。"),
            PortSpec(
                "image2",
                "前景图像（可选）",
                optional=True,
                description="叠加的前景图；未连接时使用参数“叠加颜色”。",
            ),
        ),
        outputs=(PortSpec("image", "合成结果"),),
        params=(
            ParamSpec(
                "opacity",
                "不透明度",
                "float",
                default=0.5,
                min=0.0,
                max=1.0,
                step=0.05,
                tooltip="前景没有 alpha 通道（或未连接第二幅图）时使用的不透明度。",
            ),
            ParamSpec(
                "color",
                "叠加颜色",
                "color",
                default="#ffffff",
                tooltip="第二幅图像未连接时叠加的纯色。",
            ),
        ),
    )

    def process(self, image=None, image2=None, opacity: float = 0.5, color: str = "#ffffff", **_):
        array = _cv.source(image)
        background = _cv.unit_array(array)
        alpha: Any = float(opacity)
        if image2 is not None:
            raw = _cv.source(image2, "第二幅图像")
            _cv.same_size(array, raw, "第二幅图像")
            has_alpha = raw.ndim == 3 and raw.shape[2] == 4
            foreground = _unit_like(array, raw[:, :, :3] if has_alpha else raw)
            if has_alpha:
                alpha = _cv.unit_array(raw)[:, :, 3]
                if background.ndim == 3 and background.shape[2] >= 3:
                    alpha = alpha[:, :, None]
        else:
            red, green, blue = _cv.hex_rgb(color)
            luma = (0.299 * red + 0.587 * green + 0.114 * blue) / _U8
            if background.ndim == 2:
                foreground = np.full_like(background, luma)
            else:
                channels = background.shape[2]
                values = [blue / _U8, green / _U8, red / _U8]
                if channels == 4:
                    values.append(1.0)
                foreground = np.empty_like(background)
                foreground[...] = np.array(values, dtype=np.float32)
        result = background * (1.0 - alpha) + foreground * alpha
        if background.ndim == 3 and background.shape[2] == 4:
            result = result.copy()
            result[:, :, 3] = background[:, :, 3]
        return _cv.from_unit(np.clip(result, 0.0, 1.0), array)


class ExtractAlpha(Operator):
    """提取透明通道。"""

    spec = OperatorSpec(
        id="color.extract_alpha",
        name="提取 Alpha",
        category=("通道与色彩",),
        description=(
            "输出 BGRA 图像的透明通道；对没有 alpha 的灰度 / BGR 图像输出全不透明（满量程）。"
            "输出为单通道灰度，dtype 与输入一致。"
        ),
        keywords=("alpha", "透明通道", "extract", "不透明度"),
        inputs=(PortSpec("image", "输入图像", description="要提取 alpha 的图像，支持 L / BGR / BGRA。"),),
        outputs=(PortSpec("image", "Alpha 通道", accepts=("L",)),),
    )

    def process(self, image=None, **_):
        array = _cv.source(image)
        if array.ndim == 3 and array.shape[2] == 4:
            return array[:, :, 3].copy()
        if array.dtype.kind in "ui":
            value = np.iinfo(array.dtype).max
        elif array.dtype.kind == "b":
            value = True
        else:
            value = 1.0
        return np.full(array.shape[:2], value, dtype=array.dtype)


class SetAlpha(Operator):
    """设置透明通道。"""

    spec = OperatorSpec(
        id="color.set_alpha",
        name="设置 Alpha",
        category=("通道与色彩",),
        description=(
            "把图像转换为 BGRA 并把透明通道统一设置为指定值；灰度 / BGR 输入会自动补 alpha。"
            "dtype 与输入保持一致。"
        ),
        keywords=("alpha", "透明通道", "set", "不透明度", "bgra"),
        inputs=(PortSpec("image", "输入图像", description="要写入 alpha 的图像，支持 L / BGR / BGRA。"),),
        outputs=(PortSpec("image", "BGRA 图像", accepts=("BGRA",)),),
        params=(
            ParamSpec(
                "alpha",
                "Alpha 值",
                "int",
                default=255,
                min=0,
                max=255,
                step=1,
                tooltip="0 表示完全透明，255 表示完全不透明（按 0..255 满量程换算）。",
            ),
        ),
    )

    def process(self, image=None, alpha: int = 255, **_):
        array = _cv.source(image)
        if array.dtype.kind in "ui":
            info = np.iinfo(array.dtype)
            alpha_value = int(np.clip(round(float(alpha) / _U8 * info.max), info.min, info.max))
        else:
            alpha_value = float(alpha) / _U8
        if array.ndim == 2:
            core = np.repeat(array[:, :, None], 3, axis=2)
        else:
            core = array[:, :, :3]
        layer = np.full(array.shape[:2] + (1,), alpha_value, dtype=array.dtype)
        return np.ascontiguousarray(np.concatenate([core, layer], axis=2))


class LutCurve(Operator):
    """曲线映射。"""

    spec = OperatorSpec(
        id="color.lut_curve",
        name="曲线映射",
        category=("通道与色彩",),
        description=(
            "用预设曲线或自定义控制点生成 256 级查找表，再映射整幅图像。"
            "uint8 走 OpenCV LUT，其它 dtype 按满量程线性插值；BGRA 的 alpha 保持不变。"
        ),
        keywords=("曲线", "lut_curve", "查找表", "tone", "s_curve", "对比度"),
        inputs=(PortSpec("image", "输入图像", description="要应用查找表曲线的图像，支持 L / BGR / BGRA。"),),
        outputs=(PortSpec("image", "映射结果"),),
        params=(
            ParamSpec(
                "preset",
                "预设曲线",
                "enum",
                choices=(
                    ("identity", "恒等"),
                    ("s_curve", "S 曲线（增强对比度）"),
                    ("brighten", "提亮中间调"),
                    ("darken", "压暗中间调"),
                    ("invert", "反相"),
                ),
                default="s_curve",
                tooltip="提供至少 2 个自定义控制点时，以控制点为准。",
            ),
            ParamSpec(
                "strength",
                "曲线强度",
                "float",
                default=0.5,
                min=0.0,
                max=1.0,
                step=0.05,
                visible_when="preset != 'identity' and preset != 'invert'",
                enabled_when="preset != 'identity' and preset != 'invert'",
                tooltip="预设曲线的作用力度，0 接近恒等。",
            ),
            ParamSpec(
                "points",
                "自定义控制点",
                "points",
                default=[],
                tooltip="在预览中拾取 (x, y) 控制点（0..255），至少 2 个时按控制点插值生成曲线。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        preset: str = "s_curve",
        strength: float = 0.5,
        points=None,
        **_,
    ):
        array = _cv.source(image)
        table = _curve_table(preset, strength, points or [])
        return _apply_table(array, table)


OPERATORS = (
    ColorConvert,
    ChannelSwap,
    HsvAdjust,
    ChannelMix,
    Palette,
    WhiteBalance,
    Temperature,
    AlphaComposite,
    ExtractAlpha,
    SetAlpha,
    LutCurve,
)
