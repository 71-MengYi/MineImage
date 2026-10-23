"""端口数据类型与结构化数据容器。

节点之间的连线不仅传递图像，也可以传递数值、边缘线段、轮廓和连通域标记图。
:class:`DataType` 是端口类型的唯一枚举来源，:class:`~app.operators.spec.PortSpec`
通过它约束输入 / 输出；引擎在写回结果时把算子返回值规范化为本模块的容器。

约定
----
* 图像：:class:`app.core.types.ImageData`；
* 数值：:class:`NumberValue`，可携带名称与单位；
* 边缘特征：:class:`EdgesData`，一组由 ``(x, y)`` 点组成的折线 / 线段；
* 轮廓特征：:class:`ContoursData`，一组闭合或开放轮廓点；
* 连通域：:class:`ComponentsData`，标签图 + 每个连通域的面积 / 质心。

所有容器都只读使用（``frozen`` 或约定不修改），并实现 ``summary()`` 供界面展示。
结构化数据可以通过 :func:`render_preview` 渲染成图像，供算子显式生成可视化输出；
主预览与节点缩略图不再自动渲染结构化数据，数组详情在参数面板中以文本展示。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar, Iterable, Mapping, Sequence

import cv2
import numpy as np

from app.core.types import ImageData, ImageList, first_image, image_values  # noqa: F401 - 兼容旧导入


class DataType(str, Enum):
    """端口数据类型；值稳定，可用于工程文件 / 算子声明。"""

    IMAGE = "image"
    NUMBER = "number"
    EDGES = "edges"
    CONTOURS = "contours"
    COMPONENTS = "components"

    @property
    def label(self) -> str:
        return _TYPE_LABELS[self]

    @property
    def color(self) -> str:
        """端口圆点使用的主色（十六进制）。"""

        return _TYPE_COLORS[self]


_TYPE_LABELS = {
    DataType.IMAGE: "图像",
    DataType.NUMBER: "数值",
    DataType.EDGES: "边缘特征",
    DataType.CONTOURS: "轮廓特征",
    DataType.COMPONENTS: "连通域",
}

_TYPE_COLORS = {
    DataType.IMAGE: "#3b82f6",
    DataType.NUMBER: "#d97706",
    DataType.EDGES: "#7c3aed",
    DataType.CONTOURS: "#0d9488",
    DataType.COMPONENTS: "#e11d48",
}

_TYPE_ALIASES = {
    "image": DataType.IMAGE,
    "img": DataType.IMAGE,
    "图像": DataType.IMAGE,
    "图片": DataType.IMAGE,
    "number": DataType.NUMBER,
    "numeric": DataType.NUMBER,
    "scalar": DataType.NUMBER,
    "数值": DataType.NUMBER,
    "数字": DataType.NUMBER,
    "标量": DataType.NUMBER,
    "edge": DataType.EDGES,
    "edges": DataType.EDGES,
    "线段": DataType.EDGES,
    "边缘": DataType.EDGES,
    "边缘特征": DataType.EDGES,
    "contour": DataType.CONTOURS,
    "contours": DataType.CONTOURS,
    "轮廓": DataType.CONTOURS,
    "轮廓特征": DataType.CONTOURS,
    "component": DataType.COMPONENTS,
    "components": DataType.COMPONENTS,
    "connected_component": DataType.COMPONENTS,
    "connected_components": DataType.COMPONENTS,
    "连通域": DataType.COMPONENTS,
    "连通分量": DataType.COMPONENTS,
}


def coerce_data_type(value: DataType | str) -> DataType:
    """把字符串 / 枚举规范化为 :class:`DataType`，失败时抛出中文 ``ValueError``。"""

    if isinstance(value, DataType):
        return value
    if isinstance(value, str):
        key = value.strip().lower()
        if key in _TYPE_ALIASES:
            return _TYPE_ALIASES[key]
    supported = "、".join(f"{item.label}（{item.value}）" for item in DataType)
    raise ValueError(f"不支持的端口数据类型：{value!r}；可选值：{supported}")


# ---------------------------------------------------------------------------
# 数值
# ---------------------------------------------------------------------------
def format_number(value: float) -> str:
    """数值的紧凑显示：整数不带小数点，浮点数保留 6 位有效数字。"""

    number = float(value)
    if number.is_integer() and abs(number) < 1e15:
        return str(int(number))
    return f"{number:.6g}"


@dataclass(frozen=True)
class NumberValue:
    """单个数值结果。"""

    value: float
    label: str = "数值"
    unit: str = ""

    data_type: ClassVar[DataType] = DataType.NUMBER

    def __post_init__(self) -> None:
        if isinstance(self.value, bool):
            raise ValueError(f"{self.label} 必须是数值")
        object.__setattr__(self, "value", float(self.value))
        object.__setattr__(self, "label", str(self.label or "数值"))
        object.__setattr__(self, "unit", str(self.unit or ""))

    def summary(self) -> str:
        suffix = f" {self.unit}" if self.unit else ""
        return f"{self.label}：{format_number(self.value)}{suffix}"


# ---------------------------------------------------------------------------
# 点集辅助
# ---------------------------------------------------------------------------
def _is_point(item: Any) -> bool:
    if isinstance(item, np.ndarray):
        return item.ndim == 1 and item.size == 2
    if isinstance(item, (list, tuple)):
        return len(item) == 2 and all(
            isinstance(part, (int, float, np.integer, np.floating)) and not isinstance(part, bool)
            for part in item
        )
    return False


def _as_points(value: Any, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim == 3 and array.shape[1] == 1 and array.shape[2] == 2:
        array = array[:, 0, :]
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"{label}必须是 (N, 2) 的点数组，收到形状 {array.shape}")
    if array.size and not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{label}的坐标必须是数字")
    return np.rint(array.astype(np.float32)).astype(np.int32)


def _normalize_polylines(value: Any, label: str) -> tuple[np.ndarray, ...]:
    """把多种常见写法统一成元组，每个元素是一条 ``(N, 2)`` 折线。"""

    if value is None:
        return ()
    if isinstance(value, np.ndarray):
        if value.ndim == 2 and value.shape[1] == 2:
            return (_as_points(value, label),)
        if value.ndim == 3 and value.shape[2] == 2:
            return tuple(_as_points(item, label) for item in value)
        raise ValueError(f"{label}必须是 (N, 2) 或 (K, N, 2) 的点数组，收到形状 {value.shape}")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{label}必须是点集或点集列表，收到 {type(value).__name__}")
    if not value:
        return ()
    if _is_point(value[0]):
        return (_as_points(value, label),)
    return tuple(_as_points(item, label) for item in value)


def _normalize_size(image_size: tuple[int, int] | Sequence[int] | None) -> tuple[int, int] | None:
    if image_size is None:
        return None
    try:
        width = int(image_size[0])
        height = int(image_size[1])
    except (TypeError, ValueError, IndexError) as exc:
        raise ValueError("image_size 必须是 (宽度, 高度)") from exc
    if width <= 0 or height <= 0:
        raise ValueError("image_size 的宽高必须大于 0")
    return (width, height)


def _canvas_for(
    image_size: tuple[int, int] | None,
    polylines: Iterable[np.ndarray],
) -> np.ndarray:
    """按给定尺寸或点集包围盒创建黑色 BGR 预览画布。"""

    if image_size is not None:
        width, height = image_size
    else:
        points = [item.reshape(-1, 2) for item in polylines if item.size]
        if points:
            stacked = np.vstack(points)
            width = max(1, int(stacked[:, 0].max()) + 5)
            height = max(1, int(stacked[:, 1].max()) + 5)
        else:
            width = height = 64
        width = min(max(width, 64), 4096)
        height = min(max(height, 64), 4096)
    return np.zeros((height, width, 3), dtype=np.uint8)


# ---------------------------------------------------------------------------
# 边缘线段
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EdgesData:
    """边缘特征：一组折线 / 线段，坐标为像素坐标 ``(x, y)``。"""

    segments: tuple[np.ndarray, ...] = ()
    image_size: tuple[int, int] | None = None
    label: str = "边缘线段"

    data_type: ClassVar[DataType] = DataType.EDGES

    def __post_init__(self) -> None:
        object.__setattr__(self, "segments", _normalize_polylines(self.segments, "边缘线段"))
        object.__setattr__(self, "image_size", _normalize_size(self.image_size))
        object.__setattr__(self, "label", str(self.label or "边缘线段"))

    @property
    def count(self) -> int:
        return len(self.segments)

    def summary(self) -> str:
        text = f"{self.label}：{self.count} 条"
        if self.image_size is not None:
            text += f" · 画布 {self.image_size[0]}×{self.image_size[1]}"
        return text

    def render_preview(self) -> ImageData:
        canvas = _canvas_for(self.image_size, self.segments)
        for segment in self.segments:
            if segment.shape[0] >= 2:
                cv2.polylines(canvas, [segment.reshape(-1, 1, 2)], False, (255, 255, 255), 2)
        return ImageData(canvas, self.label, "BGR")


# ---------------------------------------------------------------------------
# 轮廓
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ContoursData:
    """轮廓特征：一组轮廓点，``hierarchy`` 与 OpenCV ``findContours`` 一致。"""

    contours: tuple[np.ndarray, ...] = ()
    image_size: tuple[int, int] | None = None
    hierarchy: np.ndarray | None = None
    label: str = "轮廓"

    data_type: ClassVar[DataType] = DataType.CONTOURS

    def __post_init__(self) -> None:
        object.__setattr__(self, "contours", _normalize_polylines(self.contours, "轮廓"))
        object.__setattr__(self, "image_size", _normalize_size(self.image_size))
        if self.hierarchy is not None:
            object.__setattr__(self, "hierarchy", np.asarray(self.hierarchy, dtype=np.int32))
        object.__setattr__(self, "label", str(self.label or "轮廓"))

    @property
    def count(self) -> int:
        return len(self.contours)

    def summary(self) -> str:
        text = f"{self.label}：{self.count} 个"
        if self.image_size is not None:
            text += f" · 画布 {self.image_size[0]}×{self.image_size[1]}"
        return text

    def render_preview(self) -> ImageData:
        canvas = _canvas_for(self.image_size, self.contours)
        for contour in self.contours:
            if contour.shape[0] >= 2:
                cv2.polylines(canvas, [contour.reshape(-1, 1, 2)], True, (255, 255, 255), 2)
            elif contour.shape[0] == 1:
                cv2.circle(canvas, tuple(int(v) for v in contour[0]), 2, (255, 255, 255), -1)
        return ImageData(canvas, self.label, "BGR")


# ---------------------------------------------------------------------------
# 连通域
# ---------------------------------------------------------------------------
_COMPONENT_PALETTE = np.array(
    [
        (226, 107, 66),  # BGR
        (74, 144, 226),
        (92, 184, 92),
        (230, 168, 42),
        (144, 92, 210),
        (60, 190, 190),
        (200, 90, 150),
        (110, 170, 60),
        (70, 120, 220),
        (180, 180, 70),
    ],
    dtype=np.uint8,
)


@dataclass(frozen=True)
class ComponentsData:
    """连通域：``labels`` 为 ``(H, W)`` int32 标签图（0 为背景）。"""

    labels: np.ndarray
    areas: tuple[int, ...] = ()
    centroids: tuple[tuple[float, float], ...] = ()
    image_size: tuple[int, int] | None = None
    label: str = "连通域"

    data_type: ClassVar[DataType] = DataType.COMPONENTS

    def __post_init__(self) -> None:
        labels = np.asarray(self.labels)
        if labels.ndim != 2:
            raise ValueError(f"连通域标签图必须是二维数组，收到形状 {labels.shape}")
        labels = labels.astype(np.int32, copy=False)
        if labels.size and int(labels.min()) < 0:
            raise ValueError("连通域标签图不能包含负值（0 为背景）")
        object.__setattr__(self, "labels", labels)
        height, width = labels.shape
        size = self.image_size if self.image_size is not None else (width, height)
        object.__setattr__(self, "image_size", _normalize_size(size))
        object.__setattr__(self, "label", str(self.label or "连通域"))

        if not self.areas or not self.centroids:
            areas, centroids = _component_statistics(labels)
            if not self.areas:
                object.__setattr__(self, "areas", areas)
            if not self.centroids:
                object.__setattr__(self, "centroids", centroids)
        object.__setattr__(self, "areas", tuple(int(value) for value in self.areas))
        object.__setattr__(
            self,
            "centroids",
            tuple((float(point[0]), float(point[1])) for point in self.centroids),
        )

    @classmethod
    def from_labels(
        cls,
        labels: np.ndarray,
        *,
        image_size: tuple[int, int] | None = None,
        label: str = "连通域",
    ) -> "ComponentsData":
        size = image_size
        if size is None:
            labels_array = np.asarray(labels)
            if labels_array.ndim == 2:
                size = (int(labels_array.shape[1]), int(labels_array.shape[0]))
        return cls(labels=np.asarray(labels), image_size=size, label=label)

    @property
    def count(self) -> int:
        return len(self.areas)

    @property
    def total_area(self) -> int:
        return int(sum(self.areas))

    @property
    def largest_area(self) -> int:
        return int(max(self.areas)) if self.areas else 0

    def summary(self) -> str:
        return (
            f"{self.label}：{self.count} 个 · 总面积 {self.total_area} 像素"
            f" · 最大 {self.largest_area} 像素"
        )

    def render_preview(self) -> ImageData:
        labels = self.labels
        max_label = int(labels.max()) if labels.size else 0
        colors = np.zeros((max_label + 1, 3), dtype=np.uint8)
        for index in range(1, max_label + 1):
            colors[index] = _COMPONENT_PALETTE[(index - 1) % len(_COMPONENT_PALETTE)]
        return ImageData(colors[labels], self.label, "BGR")


def _component_statistics(labels: np.ndarray) -> tuple[tuple[int, ...], tuple[tuple[float, float], ...]]:
    if labels.size == 0:
        return (), ()
    max_label = int(labels.max())
    if max_label <= 0:
        return (), ()
    flat = labels.reshape(-1)
    counts = np.bincount(flat, minlength=max_label + 1)
    rows, cols = np.nonzero(labels)
    ids = labels[rows, cols]
    xs = np.bincount(ids, weights=cols.astype(np.float64), minlength=max_label + 1)
    ys = np.bincount(ids, weights=rows.astype(np.float64), minlength=max_label + 1)
    areas: list[int] = []
    centroids: list[tuple[float, float]] = []
    for index in range(1, max_label + 1):
        count = int(counts[index])
        if count <= 0:
            continue
        areas.append(count)
        centroids.append((float(xs[index] / count), float(ys[index] / count)))
    return tuple(areas), tuple(centroids)


# ---------------------------------------------------------------------------
# 通用工具
# ---------------------------------------------------------------------------
def data_type_of(value: Any) -> DataType | None:
    """返回数据值对应的 :class:`DataType`；未知类型返回 None。

    批量结果以同构的 list / tuple 表示（例如图像矩阵逐张计算后的数值端口），
    这里在元素类型一致时沿用元素的数据类型，使下游端口仍能识别并继续逐张执行。
    """

    if isinstance(value, ImageData):
        return DataType.IMAGE
    if isinstance(value, ImageList):
        return DataType.IMAGE
    if isinstance(value, NumberValue):
        return DataType.NUMBER
    if isinstance(value, EdgesData):
        return DataType.EDGES
    if isinstance(value, ContoursData):
        return DataType.CONTOURS
    if isinstance(value, ComponentsData):
        return DataType.COMPONENTS
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        kinds = {data_type_of(item) for item in value}
        if len(kinds) == 1:
            return kinds.pop()
    return None


def is_data_value(value: Any) -> bool:
    return data_type_of(value) is not None


def summarize_value(value: Any) -> str:
    """生成适合状态栏 / 预览提示的一行摘要。"""

    if isinstance(value, ImageData):
        return value.summary()
    if isinstance(value, ImageList):
        return value.summary()
    if isinstance(value, (list, tuple)):
        kind = data_type_of(value)
        if kind is not None:
            return f"{kind.label} ×{len(value)}"
        return f"{len(value)} 项"
    summary = getattr(value, "summary", None)
    if callable(summary):
        return str(summary())
    return str(value)


def render_preview(value: Any) -> ImageData | None:
    """把结构化数据渲染成图像，供算子显式生成可视化输出。

    引擎的主预览与节点缩略图不再自动调用本函数；数值等不可渲染的数据返回 None。
    图像矩阵取首张图作为兼容预览。
    """

    if isinstance(value, ImageData):
        return value
    if isinstance(value, ImageList):
        return first_image(value)
    renderer = getattr(value, "render_preview", None)
    if callable(renderer):
        return renderer()
    return None


def coerce_number(value: Any, label: str = "数值") -> NumberValue:
    if isinstance(value, NumberValue):
        return value
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise ValueError(f"{label}必须是数值，收到 {type(value).__name__}")
    return NumberValue(value=float(value), label=label)


def coerce_edges(value: Any, label: str = "边缘线段") -> EdgesData:
    if isinstance(value, EdgesData):
        return value
    return EdgesData(segments=_normalize_polylines(value, label), label=label)


def coerce_contours(value: Any, label: str = "轮廓") -> ContoursData:
    if isinstance(value, ContoursData):
        return value
    return ContoursData(contours=_normalize_polylines(value, label), label=label)


def coerce_components(value: Any, label: str = "连通域") -> ComponentsData:
    if isinstance(value, ComponentsData):
        return value
    if isinstance(value, np.ndarray):
        return ComponentsData.from_labels(value, label=label)
    raise ValueError(f"{label}必须是 ComponentsData 或二维标签图，收到 {type(value).__name__}")


def summarize_outputs(outputs: Mapping[str, Any], labels: Mapping[str, str] | None = None) -> list[str]:
    """把多个输出整理成 ``端口：类型 · 摘要`` 的行列表。"""

    rows: list[str] = []
    for key, value in outputs.items():
        name = (labels or {}).get(key, key)
        data_type = data_type_of(value)
        type_text = f"（{data_type.label}）" if data_type is not None else ""
        rows.append(f"{name}{type_text}：{summarize_value(value)}")
    return rows
