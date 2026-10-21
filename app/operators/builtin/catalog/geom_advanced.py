"""高级几何变换算子：转置、仿射/透视、重映射、极坐标、填充、错切、旋转与裁剪。

所有算子都返回新数组，不修改输入；对 L/BGR/BGRA 与 uint8/uint16/float32
保持 OpenCV 原生支持（个别接口内部会先做 dtype 转换并说明）。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.operators.builtin.catalog import _cv
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec

_BORDERS = (
    ("constant", "常量填充（黑）"),
    ("replicate", "复制边缘"),
    ("reflect", "镜像"),
    ("reflect101", "镜像（不含边缘）"),
    ("wrap", "平铺环绕"),
)

_INTERPOLATIONS = (
    ("nearest", "最近邻"),
    ("linear", "双线性"),
    ("cubic", "双三次"),
    ("area", "区域"),
    ("lanczos", "Lanczos"),
)


class GeomTranspose(Operator):
    """图像转置（交换 X/Y）。"""

    spec = OperatorSpec(
        id="geom.transpose",
        name="转置",
        category=("几何变换",),
        description=(
            "沿主对角线转置图像，等价于 x/y 坐标互换，宽高会互换。"
            "对全部颜色空间与 dtype 可用。"
        ),
        keywords=("转置", "transpose", "坐标互换", "几何"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
    )

    def process(self, image=None, **_: Any):
        return np.ascontiguousarray(cv2.transpose(_cv.source(image)))


class GeomWarpAffine(Operator):
    """仿射变换（旋转 + 缩放 + 平移）。"""

    spec = OperatorSpec(
        id="geom.warp_affine",
        name="仿射变换",
        category=("几何变换",),
        description=(
            "围绕图像中心做旋转与等比缩放，再按像素做水平/垂直平移。"
            "默认参数为恒等变换；输出尺寸与输入相同，超出部分按边界模式填充。"
        ),
        keywords=("仿射", "affine", "warp", "旋转", "缩放", "平移"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec("angle", "旋转角度", "float", default=0.0, min=-360.0, max=360.0, step=1.0, unit="°"),
            ParamSpec("scale", "缩放比例", "float", default=1.0, min=0.05, max=10.0, step=0.05, unit="×"),
            ParamSpec("translate_x", "水平平移", "int", default=0, min=-1000, max=1000, step=1, unit="px"),
            ParamSpec("translate_y", "垂直平移", "int", default=0, min=-1000, max=1000, step=1, unit="px"),
            ParamSpec("interpolation", "插值方式", "enum", choices=_INTERPOLATIONS, default="linear"),
            ParamSpec("border", "边界填充", "enum", choices=_BORDERS, default="constant"),
        ),
    )

    def process(
        self,
        image=None,
        angle: float = 0.0,
        scale: float = 1.0,
        translate_x: int = 0,
        translate_y: int = 0,
        interpolation: str = "linear",
        border: str = "constant",
        **_: Any,
    ):
        array = _cv.source(image)
        height, width = array.shape[:2]
        center = ((width - 1) / 2.0, (height - 1) / 2.0)
        matrix = cv2.getRotationMatrix2D(center, float(angle), float(scale))
        matrix[0, 2] += float(translate_x)
        matrix[1, 2] += float(translate_y)
        return cv2.warpAffine(
            array,
            matrix,
            (width, height),
            flags=_cv.match_interpolation(interpolation),
            borderMode=_cv.match_border(border),
            borderValue=0,
        )


class GeomWarpPerspective(Operator):
    """透视变换（四点校正）。"""

    spec = OperatorSpec(
        id="geom.warp_perspective",
        name="透视变换",
        category=("几何变换",),
        description=(
            "把 source 参数中的四个源点映射到整幅输出画面的四角，适合文档/标定板透视校正。"
            "未选点时默认取图像内缩 1/8 的四边形，直接预览校正效果。"
        ),
        keywords=("透视", "perspective", "warpPerspective", "四点", "校正"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec(
                "points",
                "源四边形",
                "points",
                default=[],
                interactive="points",
                placeholder="依次点选左上、右上、右下、左下四个点",
                tooltip="按左上、右上、右下、左下顺序点选 4 个点；留空时使用默认内缩四边形。",
            ),
            ParamSpec("interpolation", "插值方式", "enum", choices=_INTERPOLATIONS, default="linear"),
            ParamSpec("border", "边界填充", "enum", choices=_BORDERS, default="constant"),
        ),
    )

    def process(self, image=None, points=None, interpolation: str = "linear", border: str = "constant", **_: Any):
        array = _cv.source(image)
        height, width = array.shape[:2]
        if width < 3 or height < 3:
            raise OperatorError("图像太小，无法进行透视变换（宽高需至少 3 像素）。")

        if points:
            resolved = _cv.resolve_points(points, width, height)
            if len(resolved) != 4:
                raise OperatorError(f"透视变换需要 4 个源点，当前为 {len(resolved)} 个。")
            source = np.asarray(resolved, dtype=np.float32)
        else:
            margin_x = max(1, width // 8)
            margin_y = max(1, height // 8)
            source = np.asarray(
                [
                    (margin_x, margin_y),
                    (width - 1 - margin_x, margin_y),
                    (width - 1 - margin_x, height - 1 - margin_y),
                    (margin_x, height - 1 - margin_y),
                ],
                dtype=np.float32,
            )

        target = np.asarray(
            [(0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)],
            dtype=np.float32,
        )
        matrix = cv2.getPerspectiveTransform(source, target)
        return cv2.warpPerspective(
            array,
            matrix,
            (width, height),
            flags=_cv.match_interpolation(interpolation),
            borderMode=_cv.match_border(border),
            borderValue=0,
        )


class GeomRemap(Operator):
    """非线性重映射（波浪 / 涟漪 / 漩涡 / 鱼眼）。"""

    spec = OperatorSpec(
        id="geom.remap",
        name="非线性重映射",
        category=("几何变换",),
        description=(
            "用逐像素坐标映射制造波浪、涟漪、漩涡或鱼眼畸变。"
            "振幅控制形变强度，频率控制波纹疏密；输出尺寸与输入相同。"
        ),
        keywords=("remap", "重映射", "波浪", "wave", "涟漪", "ripple", "漩涡", "swirl", "鱼眼", "fisheye"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec(
                "mode",
                "畸变类型",
                "enum",
                choices=(
                    ("wave", "波浪"),
                    ("ripple", "涟漪"),
                    ("swirl", "漩涡"),
                    ("fisheye", "鱼眼"),
                ),
                default="wave",
            ),
            ParamSpec("amplitude", "振幅", "float", default=10.0, min=0.0, max=100.0, step=1.0, unit="px"),
            ParamSpec(
                "frequency",
                "频率",
                "float",
                default=0.08,
                min=0.001,
                max=1.0,
                step=0.005,
                tooltip="每像素的波纹周期数，越大越密。",
            ),
            ParamSpec("interpolation", "插值方式", "enum", choices=_INTERPOLATIONS, default="linear"),
            ParamSpec("border", "边界填充", "enum", choices=_BORDERS, default="constant"),
        ),
    )

    def process(
        self,
        image=None,
        mode: str = "wave",
        amplitude: float = 10.0,
        frequency: float = 0.08,
        interpolation: str = "linear",
        border: str = "constant",
        **_: Any,
    ):
        array = _cv.source(image)
        height, width = array.shape[:2]
        yy, xx = np.indices((height, width), dtype=np.float32)
        center_x = (width - 1) / 2.0
        center_y = (height - 1) / 2.0
        amp = float(amplitude)
        freq = float(frequency)

        if mode == "wave":
            map_x = xx + amp * np.sin(2.0 * np.pi * freq * yy)
            map_y = yy.copy()
        elif mode == "ripple":
            radius = np.hypot(xx - center_x, yy - center_y)
            offset = amp * np.sin(2.0 * np.pi * freq * radius)
            safe = np.maximum(radius, 1e-3)
            map_x = xx + offset * (xx - center_x) / safe
            map_y = yy + offset * (yy - center_y) / safe
        elif mode == "swirl":
            delta_x = xx - center_x
            delta_y = yy - center_y
            radius = np.hypot(delta_x, delta_y)
            scale = max(min(width, height) / 2.0, 1.0)
            theta = np.deg2rad(amp) * np.exp(-radius / scale)
            cos_t = np.cos(theta)
            sin_t = np.sin(theta)
            map_x = center_x + delta_x * cos_t - delta_y * sin_t
            map_y = center_y + delta_x * sin_t + delta_y * cos_t
        elif mode == "fisheye":
            delta_x = xx - center_x
            delta_y = yy - center_y
            radius = np.hypot(delta_x, delta_y)
            scale = max(min(width, height) / 2.0, 1.0)
            factor = 1.0 + (amp / 100.0) * (radius / scale) ** 2
            map_x = center_x + delta_x * factor
            map_y = center_y + delta_y * factor
        else:  # pragma: no cover - 枚举已限制取值
            raise OperatorError(f"不支持的畸变类型：{mode}")

        return cv2.remap(
            array,
            map_x.astype(np.float32),
            map_y.astype(np.float32),
            _cv.match_interpolation(interpolation),
            borderMode=_cv.match_border(border),
            borderValue=0,
        )


class GeomPolar(Operator):
    """极坐标变换 / 逆变换。"""

    spec = OperatorSpec(
        id="geom.polar",
        name="极坐标变换",
        category=("几何变换",),
        description=(
            "把笛卡尔图像映射到极坐标（角度/半径），或把极坐标图还原为直角坐标。"
            "线性模式输出 半径×360 的带状图；对数模式压缩远距离半径。"
        ),
        keywords=("极坐标", "polar", "warpPolar", "展开", "对数极坐标"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec(
                "mode",
                "半径映射",
                "enum",
                choices=(("linear", "线性"), ("log", "对数")),
                default="linear",
            ),
            ParamSpec("inverse", "极坐标转回直角坐标", "bool", default=False),
            ParamSpec(
                "radius_scale",
                "最大半径比例",
                "float",
                default=1.0,
                min=0.1,
                max=2.0,
                step=0.05,
                unit="×",
                tooltip="相对 min(宽, 高)/2 的最大采样半径。",
            ),
            ParamSpec("interpolation", "插值方式", "enum", choices=_INTERPOLATIONS, default="linear"),
        ),
    )

    def process(
        self,
        image=None,
        mode: str = "linear",
        inverse: bool = False,
        radius_scale: float = 1.0,
        interpolation: str = "linear",
        **_: Any,
    ):
        array = _cv.source(image)
        height, width = array.shape[:2]
        radius = max(float(min(width, height)) / 2.0 * float(radius_scale), 1.0)
        center = ((width - 1) / 2.0, (height - 1) / 2.0)

        if inverse:
            size = (width, height)
        else:
            size = (max(1, int(round(radius))), 360)

        flags = _cv.match_interpolation(interpolation)
        flags |= cv2.WARP_POLAR_LOG if mode == "log" else cv2.WARP_POLAR_LINEAR
        if inverse:
            flags |= cv2.WARP_INVERSE_MAP
        return cv2.warpPolar(array, size, center, radius, flags)


class GeomPad(Operator):
    """图像边界填充。"""

    spec = OperatorSpec(
        id="geom.pad",
        name="边界填充",
        category=("几何变换",),
        description=(
            "在上下左右按像素扩充画布，可选择常量（黑）、复制边缘或镜像等填充方式。"
            "常用于给滤波/形态学预留边界，避免边缘被裁掉。"
        ),
        keywords=("填充", "pad", "padding", "边界", "copyMakeBorder", "扩充"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec("top", "上边距", "int", default=16, min=0, max=500, step=1, unit="px"),
            ParamSpec("bottom", "下边距", "int", default=16, min=0, max=500, step=1, unit="px"),
            ParamSpec("left", "左边距", "int", default=16, min=0, max=500, step=1, unit="px"),
            ParamSpec("right", "右边距", "int", default=16, min=0, max=500, step=1, unit="px"),
            ParamSpec("border", "填充方式", "enum", choices=_BORDERS, default="replicate"),
        ),
    )

    def process(
        self,
        image=None,
        top: int = 16,
        bottom: int = 16,
        left: int = 16,
        right: int = 16,
        border: str = "replicate",
        **_: Any,
    ):
        array = _cv.source(image)
        if int(top) == 0 and int(bottom) == 0 and int(left) == 0 and int(right) == 0:
            return array.copy()
        return cv2.copyMakeBorder(
            array,
            int(top),
            int(bottom),
            int(left),
            int(right),
            _cv.match_border(border),
            value=0,
        )


class GeomShear(Operator):
    """错切（倾斜）变换。"""

    spec = OperatorSpec(
        id="geom.shear",
        name="错切变换",
        category=("几何变换",),
        description=(
            "沿水平或垂直方向做错切，产生斜切/倾斜效果。"
            "X 方向错切按行偏移，Y 方向错切按列偏移，正负号控制倾斜方向。"
        ),
        keywords=("错切", "shear", "倾斜", "skew", "仿射"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec("shear_x", "X 方向错切", "float", default=0.3, min=-2.0, max=2.0, step=0.05),
            ParamSpec("shear_y", "Y 方向错切", "float", default=0.0, min=-2.0, max=2.0, step=0.05),
            ParamSpec("interpolation", "插值方式", "enum", choices=_INTERPOLATIONS, default="linear"),
            ParamSpec("border", "边界填充", "enum", choices=_BORDERS, default="replicate"),
        ),
    )

    def process(
        self,
        image=None,
        shear_x: float = 0.3,
        shear_y: float = 0.0,
        interpolation: str = "linear",
        border: str = "replicate",
        **_: Any,
    ):
        array = _cv.source(image)
        height, width = array.shape[:2]
        matrix = np.asarray(
            [
                [1.0, float(shear_x), -float(shear_x) * height / 2.0],
                [float(shear_y), 1.0, -float(shear_y) * width / 2.0],
            ],
            dtype=np.float32,
        )
        return cv2.warpAffine(
            array,
            matrix,
            (width, height),
            flags=_cv.match_interpolation(interpolation),
            borderMode=_cv.match_border(border),
            borderValue=0,
        )


class GeomRotate90(Operator):
    """正交旋转（0/90/180/270 度）。"""

    spec = OperatorSpec(
        id="geom.rotate90",
        name="正交旋转",
        category=("几何变换",),
        description=(
            "按 90 度的整数倍旋转图像，不产生插值误差。"
            "90/270 度时宽高互换；对全部颜色空间与 dtype 可用。"
        ),
        keywords=("旋转", "rotate90", "90度", "正交", "翻转"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec(
                "turns",
                "旋转方向",
                "enum",
                choices=((0, "0°（不变）"), (1, "90° 顺时针"), (2, "180°"), (3, "90° 逆时针")),
                default=1,
            ),
        ),
    )

    def process(self, image=None, turns=1, **_: Any):
        array = _cv.source(image)
        turns = int(turns) % 4
        if turns == 0:
            return array.copy()
        if turns == 1:
            return cv2.rotate(array, cv2.ROTATE_90_CLOCKWISE)
        if turns == 2:
            return cv2.rotate(array, cv2.ROTATE_180)
        return cv2.rotate(array, cv2.ROTATE_90_COUNTERCLOCKWISE)


class GeomCrop(Operator):
    """裁剪指定矩形区域。"""

    spec = OperatorSpec(
        id="geom.crop",
        name="区域裁剪",
        category=("几何变换",),
        description=(
            "按 ROI 矩形 [x, y, w, h] 裁剪图像并自动夹取到图像范围内。"
            "未框选时默认裁出居中的一块（比例由“默认比例”控制）。"
        ),
        keywords=("裁剪", "crop", "roi", "区域", "cut"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec(
                "roi",
                "裁剪区域",
                "roi",
                default=None,
                placeholder="[x, y, w, h]",
                tooltip="在预览中框选裁剪区域；留空时使用居中区域。",
            ),
            ParamSpec(
                "fraction",
                "默认比例",
                "float",
                default=0.6,
                min=0.05,
                max=1.0,
                step=0.05,
                tooltip="未框选 ROI 时，居中裁剪区域占原图宽高的比例。",
                visible_when="roi is None",
            ),
        ),
    )

    def process(self, image=None, roi=None, fraction: float = 0.6, **_: Any):
        array = _cv.source(image)
        height, width = array.shape[:2]
        x, y, rect_w, rect_h = _cv.resolve_rect(roi, width, height, fraction=float(fraction))
        return array[y : y + rect_h, x : x + rect_w].copy()


class GeomTile(Operator):
    """平铺复制图像。"""

    spec = OperatorSpec(
        id="geom.tile",
        name="图像平铺",
        category=("几何变换",),
        description=(
            "按行列数把输入图像平铺成网格大图，常用于制作纹理或对比阵列。"
            "输出尺寸为 宽×列数 与 高×行数。"
        ),
        keywords=("平铺", "tile", "网格", "重复", "阵列"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(PortSpec("image", "输出图像"),),
        params=(
            ParamSpec("cols", "列数", "int", default=2, min=1, max=8, step=1),
            ParamSpec("rows", "行数", "int", default=2, min=1, max=8, step=1),
        ),
    )

    def process(self, image=None, cols: int = 2, rows: int = 2, **_: Any):
        array = _cv.source(image)
        reps = (int(rows), int(cols)) + ((1,) if array.ndim == 3 else ())
        return np.tile(array, reps)


OPERATORS = (
    GeomTranspose,
    GeomWarpAffine,
    GeomWarpPerspective,
    GeomRemap,
    GeomPolar,
    GeomPad,
    GeomShear,
    GeomRotate90,
    GeomCrop,
    GeomTile,
)
