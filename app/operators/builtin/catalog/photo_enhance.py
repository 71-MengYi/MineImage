"""OpenCV 图像增强目录算子（photo 组）。

本模块按 OpenCV ``photo`` 模块组织 10 个算子，覆盖修复、无缝克隆、颜色调整、
光照校正、纹理平滑、曝光融合、色调映射、Retinex、局部对比度与去雾。

约定
----
* 输入端口统一命名为 ``image``；需要附加图像时使用 ``optional=True`` 端口，
  未连接时回退到由第一幅图派生的结果，保证默认参数直接可运行；
* 不修改输入数组，始终返回新数组；
* 支持 L / BGR / BGRA 与 uint8 / uint16 / float32；只接受 8 位图像的 OpenCV
  接口（inpaint / seamlessClone / mergeMertens 等）内部先转 uint8 再还原量程；
* 所有错误使用 :class:`app.operators.spec.OperatorError` 的中文提示。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.operators.builtin.catalog import _cv
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec

#: 图像端口支持的颜色空间。
SPACES = ("L", "BGR", "BGRA")


# ---------------------------------------------------------------------------
# 本地辅助：uint8 工作图 <-> 原始 dtype / 通道数
# ---------------------------------------------------------------------------
def _split_alpha(array: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """拆出 BGRA 的 alpha；其它颜色空间返回 ``(array, None)``。"""
    if array.ndim == 3 and array.shape[2] == 4:
        return array[:, :, :3], array[:, :, 3]
    return array, None


def _scale_u8(values: np.ndarray, template: np.ndarray) -> np.ndarray:
    """把 uint8 结果映射回 ``template`` 的 dtype 与数值量程。"""
    if template.dtype == np.uint8:
        return np.ascontiguousarray(values)
    unit = np.asarray(values, dtype=np.float32) / 255.0
    return np.ascontiguousarray(_cv.from_unit(unit, template))


def _finish(image, work_u8: np.ndarray) -> np.ndarray:
    """把 uint8 工作结果（1 或 3 通道）写回输入的 dtype 与通道数。"""
    template = _cv.source(image)
    channels = _cv.channels(template)
    if channels == 1:
        gray = work_u8 if work_u8.ndim == 2 else cv2.cvtColor(work_u8, cv2.COLOR_BGR2GRAY)
        return _scale_u8(gray, template)
    bgr = work_u8 if work_u8.ndim == 3 else cv2.cvtColor(work_u8, cv2.COLOR_GRAY2BGR)
    if channels == 3:
        return _scale_u8(bgr, template)
    color = _scale_u8(np.ascontiguousarray(bgr), template[:, :, :3])
    return np.dstack([color, template[:, :, 3]])


def _finish_unit(values: np.ndarray, template: np.ndarray, alpha: np.ndarray | None) -> np.ndarray:
    """把 0..1 的浮点结果写回 ``template`` 的 dtype；BGRA 保留 alpha。"""
    color_template = template[:, :, :3] if alpha is not None else template
    color = _cv.from_unit(
        np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0), color_template
    )
    return color if alpha is None else np.dstack([color, alpha])


# ---------------------------------------------------------------------------
# 修复与重建
# ---------------------------------------------------------------------------
class Inpaint(Operator):
    """按蒙版修复图像中缺失或不需要的区域。"""

    spec = OperatorSpec(
        id="enhance.inpaint",
        name="图像修复",
        category=("图像增强", "修复与重建"),
        description=(
            "用 Telea 或 Navier-Stokes 算法按蒙版修复图像：蒙版中的白色像素会被周围内容"
            "填补。未连接蒙版时自动用 Otsu 生成，适合直接试跑。"
        ),
        keywords=("修复", "inpaint", "去瑕疵", "蒙版", "补全", "修补"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec(
                "mask",
                "蒙版（可选）",
                accepts=("L",),
                optional=True,
                description="白色区域参与修复；未连接时按 Otsu 自动生成。",
            ),
        ),
        params=(
            ParamSpec(
                "radius",
                "修复半径",
                "int",
                default=3,
                min=1,
                max=21,
                step=1,
                unit="px",
                tooltip="每个待修复像素参考的邻域半径。",
            ),
            ParamSpec(
                "method",
                "修复算法",
                "enum",
                choices=(("telea", "Telea（快速）"), ("ns", "Navier-Stokes（平滑）")),
                default="telea",
            ),
        ),
    )

    def process(self, image=None, mask=None, radius=3, method="telea", **_: Any):
        array = _cv.source(image)
        mask8 = np.ascontiguousarray(_cv.safe_roi_for_mask(mask, array))
        flag = cv2.INPAINT_TELEA if method == "telea" else cv2.INPAINT_NS
        if _cv.channels(array) == 1:
            work = np.ascontiguousarray(_cv.source_u8(array))
        else:
            work = np.ascontiguousarray(_cv.source_bgr8(array))
        try:
            result = cv2.inpaint(work, mask8, int(radius), flag)
        except cv2.error as exc:
            raise OperatorError("图像修复失败，请确认图像与蒙版均为有效图像。") from exc
        return _finish(array, result)


class SeamlessClone(Operator):
    """把克隆源区域无缝融合到当前图像。"""

    spec = OperatorSpec(
        id="enhance.seamless_clone",
        name="无缝克隆",
        category=("图像增强", "修复与重建"),
        description=(
            "把克隆源图像在矩形区域内的内容无缝融合到当前图像，自动匹配周围色调与边缘。"
            "未连接克隆源时使用当前图像平移后的副本，方便直接预览效果。"
        ),
        keywords=("无缝克隆", "seamless", "clone", "融合", "泊松"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec(
                "source",
                "克隆源（可选）",
                optional=True,
                description="未连接时使用当前图像平移后的副本。",
            ),
        ),
        params=(
            ParamSpec(
                "roi",
                "融合区域",
                "roi",
                default=None,
                placeholder="[x, y, w, h]",
                tooltip="在预览中框选融合区域；缺省使用图像中心 50% 的区域。",
            ),
            ParamSpec(
                "mode",
                "融合方式",
                "enum",
                choices=(("normal", "普通克隆"), ("mixed", "混合克隆")),
                default="normal",
            ),
        ),
    )

    def process(self, image=None, source=None, roi=None, mode="normal", **_: Any):
        array = _cv.source(image)
        dst = np.ascontiguousarray(_cv.source_bgr8(array))
        height, width = dst.shape[:2]
        x, y, rect_w, rect_h = _cv.resolve_rect(roi, width, height, fraction=0.5)
        fallback = np.roll(
            dst, (max(1, rect_h // 4), max(1, rect_w // 4)), axis=(0, 1)
        )
        clone_source = _cv.optional_second(source, fallback, "克隆源图像")
        src = np.ascontiguousarray(_cv.source_bgr8(clone_source, "克隆源图像"))
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.rectangle(mask, (x, y), (x + rect_w - 1, y + rect_h - 1), 255, -1)
        center = (x + rect_w // 2, y + rect_h // 2)
        flag = cv2.NORMAL_CLONE if mode == "normal" else cv2.MIXED_CLONE
        try:
            result = cv2.seamlessClone(src, dst, mask, center, flag)
        except cv2.error as exc:
            raise OperatorError(
                "无缝克隆失败，请确认融合区域位于图像内且克隆源与图像尺寸一致。"
            ) from exc
        return _finish(array, result)


# ---------------------------------------------------------------------------
# 颜色与光照
# ---------------------------------------------------------------------------
class ColorChange(Operator):
    """在 HSV 空间调整色相、饱和度并整体染色。"""

    spec = OperatorSpec(
        id="enhance.color_change",
        name="颜色调整",
        category=("图像增强", "颜色与光照"),
        description=(
            "在 HSV 空间调整色相与饱和度，并把画面整体向指定颜色染色，用于风格化或突出主体。"
            "染色强度为 0 时只做色相 / 饱和度调整。"
        ),
        keywords=("颜色", "color", "染色", "色相", "饱和度", "hsv"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec("color", "染色颜色", "color", default="#ff8a00"),
            ParamSpec(
                "strength",
                "染色强度",
                "float",
                default=0.35,
                min=0.0,
                max=1.0,
                step=0.05,
            ),
            ParamSpec(
                "hue_shift",
                "色相偏移",
                "int",
                default=0,
                min=-180,
                max=180,
                step=1,
                unit="°",
            ),
            ParamSpec(
                "saturation",
                "饱和度增益",
                "float",
                default=1.15,
                min=0.0,
                max=3.0,
                step=0.05,
                unit="×",
            ),
        ),
    )

    def process(
        self,
        image=None,
        color="#ff8a00",
        strength=0.35,
        hue_shift=0,
        saturation=1.15,
        **_: Any,
    ):
        bgr = _cv.source_bgr8(image)
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 0] = np.mod(hsv[:, :, 0] + float(hue_shift), 180.0)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * float(saturation), 0.0, 255.0)
        adjusted = cv2.cvtColor(
            np.clip(np.rint(hsv), 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR
        )
        if float(strength) > 0.0:
            tint = np.empty_like(adjusted)
            tint[:] = _cv.draw_color(adjusted, color)
            adjusted = cv2.addWeighted(
                adjusted, 1.0 - float(strength), tint, float(strength), 0.0
            )
        return _finish(image, adjusted)


class Illumination(Operator):
    """按低频背景做光照均匀化（平场校正）。"""

    spec = OperatorSpec(
        id="enhance.illumination",
        name="光照校正",
        category=("图像增强", "颜色与光照"),
        description=(
            "用大尺度高斯模糊估计低频光照背景，再用原图除以背景并拉回整体亮度，"
            "用于校正不均匀照明与暗角。强度为 0 时输出原图。"
        ),
        keywords=("光照", "illumination", "平场", "暗角", "背景", "校正"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec(
                "radius",
                "背景尺度",
                "int",
                default=41,
                min=3,
                max=201,
                step=2,
                odd=True,
                unit="px",
                tooltip="越大越接近全局亮度校正，越小越强调局部。",
            ),
            ParamSpec(
                "strength",
                "校正强度",
                "float",
                default=1.0,
                min=0.0,
                max=1.0,
                step=0.05,
            ),
        ),
    )

    def process(self, image=None, radius=41, strength=1.0, **_: Any):
        array = _cv.source(image)
        color, alpha = _split_alpha(array)
        unit = np.clip(_cv.unit_array(color).astype(np.float32), 0.0, 1.0)
        ksize = max(3, int(radius) | 1)
        background = cv2.GaussianBlur(unit, (ksize, ksize), 0.0)
        mean = float(unit.mean()) if unit.size else 0.5
        mean = mean if mean > 1e-6 else 0.5
        corrected = unit / np.maximum(background, 1e-4) * mean
        corrected = np.clip(corrected, 0.0, 1.0)
        result = unit * (1.0 - float(strength)) + corrected * float(strength)
        return _finish_unit(result, array, alpha)


# ---------------------------------------------------------------------------
# 平滑与细节
# ---------------------------------------------------------------------------
class TextureFlatten(Operator):
    """用双边滤波压平纹理并保留边缘。"""

    spec = OperatorSpec(
        id="enhance.texture_flatten",
        name="纹理平滑",
        category=("图像增强", "平滑与细节"),
        description=(
            "使用双边滤波压平皮肤、织物等细碎纹理，同时保留明显边缘。可多次迭代，"
            "次数越多越平滑。"
        ),
        keywords=("纹理", "texture", "平滑", "双边滤波", "磨皮", "bilateral"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec(
                "sigma_color",
                "颜色相似度",
                "float",
                default=45.0,
                min=5.0,
                max=150.0,
                step=5.0,
                tooltip="越大越容易把相近颜色一起平滑。",
            ),
            ParamSpec(
                "sigma_space",
                "空间范围",
                "float",
                default=45.0,
                min=5.0,
                max=150.0,
                step=5.0,
            ),
            ParamSpec(
                "diameter",
                "邻域直径",
                "int",
                default=0,
                min=0,
                max=31,
                step=2,
                unit="px",
                tooltip="0 表示由空间范围自动推算。",
            ),
            ParamSpec(
                "iterations",
                "迭代次数",
                "int",
                default=1,
                min=1,
                max=5,
                step=1,
            ),
        ),
    )

    def process(
        self,
        image=None,
        sigma_color=45.0,
        sigma_space=45.0,
        diameter=0,
        iterations=1,
        **_: Any,
    ):
        template = _cv.source(image)
        result = np.ascontiguousarray(_cv.source_bgr8(template))
        radius = 0 if int(diameter) <= 0 else int(diameter)
        for _ in range(int(iterations)):
            result = cv2.bilateralFilter(
                result, radius, float(sigma_color), float(sigma_space)
            )
        return _finish(template, result)


class Retinex(Operator):
    """基于 Retinex 理论提升细节与动态范围。"""

    spec = OperatorSpec(
        id="enhance.retinex",
        name="Retinex 增强",
        category=("图像增强", "平滑与细节"),
        description=(
            "用 Retinex 理论分离照射分量与反射分量，提升暗部细节和整体动态范围。"
            "单尺度适合细节增强，多尺度更均衡。"
        ),
        keywords=("retinex", "细节", "动态范围", "均匀照明", "增强"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec(
                "mode",
                "尺度模式",
                "enum",
                choices=(("single", "单尺度"), ("multi", "多尺度")),
                default="multi",
            ),
            ParamSpec(
                "sigma",
                "单尺度半径",
                "int",
                default=80,
                min=5,
                max=200,
                step=5,
                unit="px",
                tooltip="仅在单尺度模式下生效。",
                visible_when="mode == 'single'",
                enabled_when="mode == 'single'",
            ),
            ParamSpec(
                "gain",
                "增强增益",
                "float",
                default=1.0,
                min=0.5,
                max=3.0,
                step=0.05,
                unit="×",
            ),
        ),
    )

    def process(self, image=None, mode="multi", sigma=80, gain=1.0, **_: Any):
        array = _cv.source(image)
        color, alpha = _split_alpha(array)
        unit = np.clip(_cv.unit_array(color).astype(np.float32), 1e-4, 1.0)
        log_image = np.log(unit + 1e-4)
        if mode == "single":
            blur = cv2.GaussianBlur(unit, (0, 0), float(sigma))
            retinex = log_image - np.log(blur + 1e-4)
        else:
            retinex = np.zeros_like(unit)
            total = 0.0
            for scale in (15.0, 80.0, 250.0):
                blur = cv2.GaussianBlur(unit, (0, 0), scale)
                retinex += log_image - np.log(blur + 1e-4)
                total += 1.0
            retinex /= total
        if retinex.ndim == 2:
            low, high = np.percentile(retinex, (1.0, 99.0))
            normalized = (retinex - low) / max(float(high - low), 1e-6)
        else:
            low = np.percentile(retinex, 1.0, axis=(0, 1))
            high = np.percentile(retinex, 99.0, axis=(0, 1))
            normalized = (retinex - low) / np.maximum(high - low, 1e-6)
        result = np.clip(normalized * float(gain), 0.0, 1.0)
        return _finish_unit(result, array, alpha)


class LocalContrast(Operator):
    """用 CLAHE 做局部对比度增强。"""

    spec = OperatorSpec(
        id="enhance.local_contrast",
        name="局部对比度",
        category=("图像增强", "平滑与细节"),
        description=(
            "用限制对比度的自适应直方图均衡（CLAHE）增强局部对比度。彩色图像在 LAB "
            "亮度通道上处理，避免色偏。"
        ),
        keywords=("对比度", "contrast", "clahe", "直方图均衡", "局部"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec(
                "clip_limit",
                "对比度限制",
                "float",
                default=2.0,
                min=0.1,
                max=10.0,
                step=0.1,
                tooltip="越大对比度越强，过大会放大噪声。",
            ),
            ParamSpec(
                "grid",
                "网格大小",
                "int",
                default=8,
                min=1,
                max=32,
                step=1,
                tooltip="把图像分成 grid×grid 个局部窗口。",
            ),
        ),
    )

    def process(self, image=None, clip_limit=2.0, grid=8, **_: Any):
        array = _cv.source(image)
        clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit), tileGridSize=(int(grid), int(grid))
        )
        if _cv.channels(array) == 1:
            work = np.ascontiguousarray(_cv.source_u8(array))
            result = clahe.apply(work)
        else:
            work = np.ascontiguousarray(_cv.source_bgr8(array))
            lab = cv2.cvtColor(work, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return _finish(array, result)


# ---------------------------------------------------------------------------
# 动态范围与去雾
# ---------------------------------------------------------------------------
class HdrMerge(Operator):
    """对多曝光图像做曝光融合。"""

    spec = OperatorSpec(
        id="enhance.hdr_merge",
        name="曝光融合",
        category=("图像增强", "动态范围"),
        description=(
            "用 Mertens 曝光融合合并两幅不同曝光的图像，兼顾亮部与暗部细节。未连接"
            "第二幅图像时使用第一幅的提亮副本，便于直接试跑。"
        ),
        keywords=("hdr", "曝光", "融合", "merge", "mertens", "动态范围"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec(
                "image2",
                "第二曝光图像（可选）",
                optional=True,
                description="未连接时使用第一幅图像的提亮副本。",
            ),
        ),
        params=(
            ParamSpec(
                "contrast",
                "对比度权重",
                "float",
                default=1.0,
                min=0.0,
                max=2.0,
                step=0.05,
            ),
            ParamSpec(
                "saturation",
                "饱和度权重",
                "float",
                default=1.0,
                min=0.0,
                max=2.0,
                step=0.05,
            ),
            ParamSpec(
                "exposure",
                "曝光权重",
                "float",
                default=1.0,
                min=0.0,
                max=2.0,
                step=0.05,
            ),
        ),
    )

    def process(
        self,
        image=None,
        image2=None,
        contrast=1.0,
        saturation=1.0,
        exposure=1.0,
        **_: Any,
    ):
        array = _cv.source(image)
        first = np.ascontiguousarray(_cv.source_bgr8(array))
        fallback = np.clip(
            first.astype(np.float32) * 1.25 + 16.0, 0.0, 255.0
        ).astype(np.uint8)
        second_input = _cv.optional_second(image2, fallback, "第二曝光图像")
        second = np.ascontiguousarray(_cv.source_bgr8(second_input, "第二曝光图像"))
        merge = cv2.createMergeMertens(
            float(contrast), float(saturation), float(exposure)
        )
        try:
            merged = merge.process([first, second])
        except cv2.error as exc:
            raise OperatorError(
                "曝光融合失败，请确认两幅图像尺寸一致且都是有效图像。"
            ) from exc
        merged = np.nan_to_num(merged, nan=0.0, posinf=1.0, neginf=0.0)
        work = _cv.to_uint8(np.clip(merged, 0.0, 1.0))
        return _finish(array, work)


class Tonemap(Operator):
    """Reinhard 色调映射。"""

    spec = OperatorSpec(
        id="enhance.tonemap",
        name="色调映射",
        category=("图像增强", "动态范围"),
        description=(
            "用 Reinhard 算子压缩高动态范围图像的亮度，保留亮部层次并减少过曝。"
            "适合对归一化后的浮点图像或普通 8 位图像做风格化处理。"
        ),
        keywords=("色调映射", "tonemap", "reinhard", "hdr", "压缩"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec(
                "gamma",
                "Gamma",
                "float",
                default=1.8,
                min=0.1,
                max=5.0,
                step=0.1,
            ),
            ParamSpec(
                "intensity",
                "亮度强度",
                "float",
                default=0.0,
                min=0.0,
                max=1.0,
                step=0.05,
            ),
            ParamSpec(
                "light_adapt",
                "光照自适应",
                "float",
                default=0.8,
                min=0.0,
                max=1.0,
                step=0.05,
            ),
            ParamSpec(
                "color_adapt",
                "颜色自适应",
                "float",
                default=0.0,
                min=0.0,
                max=1.0,
                step=0.05,
            ),
        ),
    )

    def process(
        self,
        image=None,
        gamma=1.8,
        intensity=0.0,
        light_adapt=0.8,
        color_adapt=0.0,
        **_: Any,
    ):
        array = _cv.source(image)
        color, alpha = _split_alpha(array)
        unit = np.clip(_cv.unit_array(color).astype(np.float32), 0.0, 1.0)
        bgr = unit if unit.ndim == 3 else cv2.cvtColor(unit, cv2.COLOR_GRAY2BGR)
        tonemap = cv2.createTonemapReinhard(
            gamma=float(gamma),
            intensity=float(intensity),
            light_adapt=float(light_adapt),
            color_adapt=float(color_adapt),
        )
        try:
            result = tonemap.process(np.ascontiguousarray(bgr))
        except cv2.error as exc:
            raise OperatorError("色调映射失败，请检查图像内容与参数设置。") from exc
        result = np.nan_to_num(result, nan=0.0, posinf=1.0, neginf=0.0)
        result = np.clip(result, 0.0, 1.0)
        if _cv.channels(array) == 1:
            result = cv2.cvtColor(result.astype(np.float32), cv2.COLOR_BGR2GRAY)
        return _finish_unit(result, array, alpha)


class Dehaze(Operator):
    """基于暗通道先验的快速去雾。"""

    spec = OperatorSpec(
        id="enhance.dehaze",
        name="去雾",
        category=("图像增强", "修复与重建"),
        description=(
            "基于暗通道先验估计大气光与透射率，恢复被雾、霾或烟尘降低的对比度。"
            "强度为 0 时输出原图，可逐步提高避免过度去雾。"
        ),
        keywords=("去雾", "dehaze", "暗通道", "雾霾", "清晰化"),
        inputs=(PortSpec("image", "输入图像"),),
        params=(
            ParamSpec(
                "radius",
                "暗通道半径",
                "int",
                default=15,
                min=3,
                max=51,
                step=2,
                odd=True,
                unit="px",
            ),
            ParamSpec(
                "omega",
                "去雾强度",
                "float",
                default=0.95,
                min=0.5,
                max=1.0,
                step=0.01,
            ),
            ParamSpec(
                "t0",
                "透射率下限",
                "float",
                default=0.1,
                min=0.05,
                max=0.5,
                step=0.01,
            ),
            ParamSpec(
                "strength",
                "输出混合",
                "float",
                default=1.0,
                min=0.0,
                max=1.0,
                step=0.05,
            ),
        ),
    )

    def process(
        self,
        image=None,
        radius=15,
        omega=0.95,
        t0=0.1,
        strength=1.0,
        **_: Any,
    ):
        array = _cv.source(image)
        bgr = np.ascontiguousarray(_cv.source_bgr8(array))
        image_unit = bgr.astype(np.float32) / 255.0
        dark = np.min(image_unit, axis=2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (max(3, int(radius) | 1), max(3, int(radius) | 1))
        )
        dark = cv2.erode(dark, kernel)
        flat = dark.reshape(-1)
        count = max(1, int(flat.size * 0.001))
        index = np.argpartition(flat, -count)[-count:]
        atmosphere = image_unit.reshape(-1, 3)[index].max(axis=0)
        atmosphere = np.maximum(atmosphere, 0.1)
        transmission = 1.0 - float(omega) * dark / float(np.max(atmosphere))
        transmission = np.maximum(transmission, float(t0))
        recovered = (image_unit - atmosphere) / transmission[:, :, None] + atmosphere
        recovered = np.clip(recovered, 0.0, 1.0)
        result = image_unit * (1.0 - float(strength)) + recovered * float(strength)
        return _finish(array, _cv.to_uint8(np.clip(result, 0.0, 1.0)))


#: 模块导出的算子类；loader 支持一个模块注册多个算子。
OPERATORS = (
    Inpaint,
    SeamlessClone,
    ColorChange,
    Illumination,
    TextureFlatten,
    HdrMerge,
    Tonemap,
    Retinex,
    LocalContrast,
    Dehaze,
)
