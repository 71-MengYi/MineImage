"""OpenCV 目录算子：直方图分析（特征与分析）。

包含直方图绘制、均衡化、CLAHE、直方图比较、反投影、直方图匹配与二维直方图。
约定与其它目录算子一致：

* 输入统一经 :mod:`app.operators.builtin.catalog._cv` 读取；
* 第一个输出始终是图像（预览依赖），``hist_compare`` 额外输出 ``DataType.NUMBER``；
* 直方图类 OpenCV 接口只支持 8 位，内部用 ``_cv.source_gray8`` / ``source_bgr8``
  转换，结果再按输入满量程恢复到原 dtype / 通道数；
* 返回新数组，不修改输入。
"""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np

from app.core.data_types import DataType, NumberValue
from app.operators.builtin.catalog import _cv
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec

_U8 = 255.0

#: 直方图比较方式。
_HIST_METHODS = {
    "correlation": ("相关系数（越大越相似）", cv2.HISTCMP_CORREL),
    "chi_square": ("卡方（越小越相似）", cv2.HISTCMP_CHISQR),
    "intersection": ("直方图交集（越大越相似）", cv2.HISTCMP_INTERSECT),
    "bhattacharyya": ("巴氏距离（越小越相似）", cv2.HISTCMP_BHATTACHARYYA),
}

#: 二维直方图的通道组合。
_HIST2D_PAIRS = {
    "hs": ("H-S（色相 / 饱和度）", None),
    "gb": ("G-B（绿 / 蓝）", (1, 0)),
    "gr": ("G-R（绿 / 红）", (1, 2)),
    "br": ("B-R（蓝 / 红）", (0, 2)),
}


# ---------------------------------------------------------------------------
# 共享辅助
# ---------------------------------------------------------------------------
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
    else:
        if work.ndim == 2:
            work = np.repeat(work[:, :, None], 3, axis=2)
        elif work.shape[2] == 4:
            work = work[:, :, :3]
    if template.dtype == np.uint8:
        return np.ascontiguousarray(work)
    return _cv.from_unit(work.astype(np.float32) / _U8, template)


def _plot_curves(
    curves: Sequence[tuple[np.ndarray, tuple[int, int, int]]],
    width: int = 480,
    height: int = 300,
) -> np.ndarray:
    """把一条或多条直方图曲线画成 BGR 预览图（各自按峰值归一化）。"""
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    for hist, color in curves:
        values = np.nan_to_num(np.asarray(hist, dtype=np.float64).ravel())
        peak = float(values.max()) if values.size else 0.0
        if peak <= 0.0:
            continue
        xs = np.linspace(8.0, width - 8.0, values.size, dtype=np.float64)
        ys = (height - 24.0) - (values / peak) * (height - 48.0)
        points = np.stack([xs, ys], axis=1).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [points], False, color, 2, cv2.LINE_AA)
    cv2.line(canvas, (8, height - 20), (width - 8, height - 20), (80, 80, 80), 1)
    return canvas


def _match_table(source_gray8: np.ndarray, target_cdf: np.ndarray) -> np.ndarray:
    """按源 / 目标累计分布构建 256 级直方图匹配查找表。"""
    hist = cv2.calcHist([source_gray8], [0], None, [256], [0, 256]).ravel()
    total = float(hist.sum())
    if total <= 0.0:
        raise OperatorError("输入图像的直方图为空，无法做直方图匹配。")
    cdf = np.cumsum(hist, dtype=np.float64)
    cdf /= total
    target = np.asarray(target_cdf, dtype=np.float64)
    if target.size != 256 or target[-1] <= 0.0:
        raise OperatorError("参考图像的直方图为空，无法做直方图匹配。")
    mapped = np.interp(cdf, target, np.arange(256, dtype=np.float64))
    return np.clip(np.rint(mapped), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 直方图绘制与均衡化
# ---------------------------------------------------------------------------
class Histogram(Operator):
    """直方图绘制。"""

    spec = OperatorSpec(
        id="analysis.histogram",
        name="直方图",
        category=("特征与分析", "直方图"),
        description=(
            "统计灰度或指定颜色通道的像素分布，并绘制成 480×300 的直方图预览。"
            "柱数越多曲线越细；输出为 BGR 预览图（绘制类结果固定使用 uint8）。"
        ),
        keywords=("直方图", "histogram", "统计", "分布", "灰度"),
        inputs=(PortSpec("image", "输入图像", description="要统计直方图的图像，支持 L / BGR / BGRA。"),),
        outputs=(PortSpec("image", "直方图", accepts=("BGR",)),),
        params=(
            ParamSpec(
                "bins",
                "柱数",
                "int",
                default=64,
                min=8,
                max=256,
                step=1,
                group="统计",
                tooltip="直方图的区间数量，越大曲线越细。",
            ),
            ParamSpec(
                "channel",
                "统计通道",
                "enum",
                choices=(
                    ("gray", "灰度 / 亮度"),
                    ("blue", "蓝色通道 (B)"),
                    ("green", "绿色通道 (G)"),
                    ("red", "红色通道 (R)"),
                ),
                default="gray",
                group="统计",
                tooltip="灰度图始终按亮度统计；彩色图可选择单个颜色通道。",
            ),
        ),
    )

    def process(self, image=None, bins: int = 64, channel: str = "gray", **_):
        if channel == "gray":
            data = _cv.source_gray8(image)
        else:
            index = {"blue": 0, "green": 1, "red": 2}.get(channel)
            if index is None:
                raise OperatorError(f"不支持的统计通道：{channel}")
            data = _cv.source_bgr8(image)[:, :, index]
        hist = cv2.calcHist([data], [0], None, [int(bins)], [0, 256])
        return _plot_curves([(hist, (90, 220, 90))])


class Equalize(Operator):
    """直方图均衡化。"""

    spec = OperatorSpec(
        id="analysis.equalize",
        name="直方图均衡",
        category=("特征与分析", "直方图"),
        description=(
            "用累计分布做直方图均衡，拉伸低对比度图像。灰度图直接均衡；彩色图默认均衡亮度，"
            "也可选择逐通道均衡。结果保持输入 dtype / 通道数。"
        ),
        keywords=("直方图均衡", "equalize", "equalizeHist", "对比度", "增强"),
        inputs=(PortSpec("image", "输入图像", description="需要均衡化的图像，支持 L / BGR / BGRA。"),),
        outputs=(PortSpec("image", "均衡结果"),),
        params=(
            ParamSpec(
                "mode",
                "均衡方式",
                "enum",
                choices=(("luma", "按亮度均衡（保持色彩）"), ("channels", "逐通道均衡")),
                default="luma",
                tooltip="彩色图选择“按亮度均衡”可避免色偏；灰度图两种方式等价。",
            ),
        ),
    )

    def process(self, image=None, mode: str = "luma", **_):
        source = _cv.source(image)
        gray8 = _cv.source_gray8(image)
        equalized = cv2.equalizeHist(gray8)
        if source.ndim == 2:
            return _finish_u8(equalized, source)
        bgr8 = _cv.source_bgr8(image)
        if mode == "channels":
            blue, green, red = cv2.split(bgr8)
            merged = cv2.merge(
                [cv2.equalizeHist(blue), cv2.equalizeHist(green), cv2.equalizeHist(red)]
            )
            return _finish_u8(merged, source)
        ycrcb = cv2.cvtColor(bgr8, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = equalized
        return _finish_u8(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), source)


class Clahe(Operator):
    """限制对比度的自适应直方图均衡。"""

    spec = OperatorSpec(
        id="analysis.clahe",
        name="CLAHE 均衡",
        category=("特征与分析", "直方图"),
        description=(
            "用限制对比度的自适应直方图均衡（CLAHE）增强局部对比度。"
            "彩色图默认只处理亮度通道；结果保持输入 dtype / 通道数。"
        ),
        keywords=("clahe", "自适应均衡", "局部对比度", "限制对比度"),
        inputs=(PortSpec("image", "输入图像", description="需要局部增强的图像，支持 L / BGR / BGRA。"),),
        outputs=(PortSpec("image", "增强结果"),),
        params=(
            ParamSpec(
                "clip_limit",
                "对比度限制",
                "float",
                default=2.0,
                min=0.5,
                max=40.0,
                step=0.5,
                group="CLAHE",
                tooltip="值越大对比度越强，也越容易放大噪声。",
            ),
            ParamSpec(
                "tiles",
                "网格数",
                "int",
                default=8,
                min=2,
                max=32,
                step=1,
                group="CLAHE",
                tooltip="把图像分成 tiles×tiles 个小块，块越大局部性越弱。",
            ),
        ),
    )

    def process(self, image=None, clip_limit: float = 2.0, tiles: int = 8, **_):
        source = _cv.source(image)
        gray8 = _cv.source_gray8(image)
        clahe = cv2.createCLAHE(
            clipLimit=float(clip_limit), tileGridSize=(int(tiles), int(tiles))
        )
        enhanced = clahe.apply(gray8)
        if source.ndim == 2:
            return _finish_u8(enhanced, source)
        bgr8 = _cv.source_bgr8(image)
        ycrcb = cv2.cvtColor(bgr8, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = enhanced
        return _finish_u8(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), source)


# ---------------------------------------------------------------------------
# 直方图比较 / 反投影 / 匹配
# ---------------------------------------------------------------------------
class HistCompare(Operator):
    """直方图比较。"""

    spec = OperatorSpec(
        id="analysis.hist_compare",
        name="直方图比较",
        category=("特征与分析", "直方图"),
        description=(
            "比较两幅灰度直方图的相似程度，输出“对比图”和“相似度”两个结果。"
            "第二幅图未连接时与第一幅图比较（相似度恒为 1）。两个输入尺寸可以不同。"
        ),
        keywords=("直方图比较", "hist_compare", "相似度", "compareHist", "匹配"),
        inputs=(
            PortSpec("image", "输入图像", description="第一幅图像，按亮度统计直方图。"),
            PortSpec(
                "image2",
                "第二幅图像（可选）",
                optional=True,
                description="参与比较的图像；未连接时使用第一幅图。",
            ),
        ),
        outputs=(
            PortSpec("image", "对比图", accepts=("BGR",)),
            PortSpec("similarity", "相似度", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "method",
                "比较方式",
                "enum",
                choices=tuple((key, label) for key, (label, _flag) in _HIST_METHODS.items()),
                default="correlation",
                group="比较",
                tooltip="不同方式输出的相似度都已归一化到 0..1，越大越相似。",
            ),
            ParamSpec(
                "bins",
                "柱数",
                "int",
                default=64,
                min=8,
                max=256,
                step=1,
                group="比较",
                tooltip="统计直方图时使用的区间数量，两幅图保持一致。",
            ),
        ),
    )

    def process(self, image=None, image2=None, method: str = "correlation", bins: int = 64, **_):
        entry = _HIST_METHODS.get(method)
        if entry is None:
            raise OperatorError(f"不支持的直方图比较方式：{method}")
        gray1 = _cv.source_gray8(image)
        gray2 = _cv.source_gray8(image2, "第二幅图像") if image2 is not None else gray1
        hist1 = cv2.calcHist([gray1], [0], None, [int(bins)], [0, 256])
        hist2 = cv2.calcHist([gray2], [0], None, [int(bins)], [0, 256])
        norm1 = cv2.normalize(hist1.copy(), None, 0.0, 1.0, cv2.NORM_MINMAX)
        norm2 = cv2.normalize(hist2.copy(), None, 0.0, 1.0, cv2.NORM_MINMAX)
        raw = float(cv2.compareHist(norm1, norm2, entry[1]))
        if method == "correlation":
            similarity = raw
        elif method == "chi_square":
            similarity = 1.0 / (1.0 + max(raw, 0.0))
        elif method == "bhattacharyya":
            similarity = 1.0 - raw
        else:  # intersection
            similarity = raw
        similarity = float(np.clip(similarity, 0.0, 1.0))
        plot = _plot_curves(
            [
                (hist1, (90, 220, 90)),
                (hist2, (60, 140, 255)),
            ]
        )
        return {
            "image": plot,
            "similarity": NumberValue(similarity, label="相似度"),
        }


class BackProject(Operator):
    """直方图反投影。"""

    spec = OperatorSpec(
        id="analysis.backproject",
        name="直方图反投影",
        category=("特征与分析", "直方图"),
        description=(
            "用矩形区域内的颜色直方图做反投影，高亮整幅图中与区域颜色相近的像素。"
            "彩色图使用 H-S 二维直方图，灰度图使用一维直方图；输出单通道 uint8 响应图。"
        ),
        keywords=("反投影", "backproject", "calcBackProject", "目标定位", "颜色"),
        inputs=(PortSpec("image", "输入图像", description="要计算反投影的图像，支持 L / BGR / BGRA。"),),
        outputs=(PortSpec("image", "反投影响应", accepts=("L",)),),
        params=(
            ParamSpec(
                "roi",
                "取样区域",
                "roi",
                default=None,
                tooltip="用于统计直方图的矩形区域；未框选时使用图像中心一半区域。",
            ),
            ParamSpec(
                "bins",
                "柱数",
                "int",
                default=32,
                min=4,
                max=128,
                step=1,
                tooltip="每个通道的直方图区间数量。",
            ),
        ),
    )

    def process(self, image=None, roi=None, bins: int = 32, **_):
        source = _cv.source(image)
        if source.ndim == 2:
            gray8 = _cv.source_gray8(image)
            x, y, width, height = _cv.resolve_rect(
                roi, gray8.shape[1], gray8.shape[0], fraction=0.5, label="取样区域"
            )
            patch = gray8[y : y + height, x : x + width]
            hist = cv2.calcHist([patch], [0], None, [int(bins)], [0, 256])
            cv2.normalize(hist, hist, 0.0, 255.0, cv2.NORM_MINMAX)
            response = cv2.calcBackProject([gray8], [0], hist, [0, 256], 1)
        else:
            bgr8 = _cv.source_bgr8(image)
            hsv = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV)
            x, y, width, height = _cv.resolve_rect(
                roi, bgr8.shape[1], bgr8.shape[0], fraction=0.5, label="取样区域"
            )
            patch = hsv[y : y + height, x : x + width]
            hist = cv2.calcHist(
                [patch], [0, 1], None, [int(bins), int(bins)], [0, 180, 0, 256]
            )
            cv2.normalize(hist, hist, 0.0, 255.0, cv2.NORM_MINMAX)
            response = cv2.calcBackProject(
                [hsv], [0, 1], hist, [0, 180, 0, 256], 1
            )
        return np.ascontiguousarray(response)


class MatchHist(Operator):
    """直方图匹配（规定化）。"""

    spec = OperatorSpec(
        id="analysis.match_hist",
        name="直方图匹配",
        category=("特征与分析", "直方图"),
        description=(
            "把图像的直方图变换到参考图像的分布；参考图未连接时使用均匀分布（等价于均衡化）。"
            "彩色图只匹配亮度通道以保持色彩，结果保持输入 dtype / 通道数。"
        ),
        keywords=("直方图匹配", "match_hist", "规定化", "直方图规定化", "迁移"),
        inputs=(
            PortSpec("image", "输入图像", description="待匹配的图像。"),
            PortSpec(
                "image2",
                "参考图像（可选）",
                optional=True,
                description="提供目标直方图的图像；未连接时使用均匀分布。",
            ),
        ),
        outputs=(PortSpec("image", "匹配结果"),),
    )

    def process(self, image=None, image2=None, **_):
        source = _cv.source(image)
        gray8 = _cv.source_gray8(image)
        if image2 is None:
            target_cdf = np.linspace(0.0, 1.0, 256, dtype=np.float64)
        else:
            reference = _cv.source_gray8(image2, "参考图像")
            hist = cv2.calcHist([reference], [0], None, [256], [0, 256]).ravel()
            total = float(hist.sum())
            if total <= 0.0:
                raise OperatorError("参考图像的直方图为空，无法做直方图匹配。")
            target_cdf = np.cumsum(hist, dtype=np.float64) / total
        table = _match_table(gray8, target_cdf)
        if source.ndim == 2:
            return _finish_u8(table[gray8], source)
        bgr8 = _cv.source_bgr8(image)
        ycrcb = cv2.cvtColor(bgr8, cv2.COLOR_BGR2YCrCb)
        ycrcb[:, :, 0] = table[ycrcb[:, :, 0]]
        return _finish_u8(cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR), source)


class Hist2D(Operator):
    """二维直方图。"""

    spec = OperatorSpec(
        id="analysis.hist_2d",
        name="二维直方图",
        category=("特征与分析", "直方图"),
        description=(
            "统计两个通道的联合分布并渲染成彩色热力图，输出 BGR 预览图。"
            "H-S 组合适合分析颜色分布，通道两两组合适合观察通道相关性。"
        ),
        keywords=("二维直方图", "hist_2d", "2d histogram", "联合分布", "热力图"),
        inputs=(PortSpec("image", "输入图像", description="要统计二维直方图的图像，支持 L / BGR / BGRA。"),),
        outputs=(PortSpec("image", "二维直方图", accepts=("BGR",)),),
        params=(
            ParamSpec(
                "pair",
                "通道组合",
                "enum",
                choices=tuple((key, label) for key, (label, _pair) in _HIST2D_PAIRS.items()),
                default="hs",
                group="统计",
                tooltip="H-S 需要彩色图；灰度输入会先按灰度复制成三通道。",
            ),
            ParamSpec(
                "bins",
                "柱数",
                "int",
                default=64,
                min=4,
                max=128,
                step=1,
                group="统计",
                tooltip="每个通道的区间数量，网格为 bins×bins。",
            ),
        ),
    )

    def process(self, image=None, pair: str = "hs", bins: int = 64, **_):
        entry = _HIST2D_PAIRS.get(pair)
        if entry is None:
            raise OperatorError(f"不支持的通道组合：{pair}")
        bgr8 = _cv.source_bgr8(image)
        if entry[1] is None:
            work = cv2.cvtColor(bgr8, cv2.COLOR_BGR2HSV)
            first, second = work[:, :, 0], work[:, :, 1]
            ranges = [0, 180, 0, 256]
        else:
            first_index, second_index = entry[1]
            first, second = bgr8[:, :, first_index], bgr8[:, :, second_index]
            ranges = [0, 256, 0, 256]
        hist = cv2.calcHist(
            [first, second], [0, 1], None, [int(bins), int(bins)], ranges
        )
        cv2.normalize(hist, hist, 0.0, 255.0, cv2.NORM_MINMAX)
        heat = np.clip(np.rint(hist), 0, 255).astype(np.uint8)
        colored = cv2.applyColorMap(heat, cv2.COLORMAP_JET)
        return cv2.resize(colored, (320, 320), interpolation=cv2.INTER_NEAREST)


OPERATORS = (
    Histogram,
    Equalize,
    Clahe,
    HistCompare,
    BackProject,
    MatchHist,
    Hist2D,
)
