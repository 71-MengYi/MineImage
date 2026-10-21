"""输出节点：汇总显示全部已知数据类型，并把每路输入原样透传。

节点位于「其他 → 可视化」，面向流程的终点与中途检查：

* 五个可选输入分别接收图像、数值、边缘特征、轮廓特征与连通域，覆盖
  app.core.data_types.DataType 的全部已知类型；
* 节点预览是一张汇总显示面板：图像缩略图 + 每个结构化数据的缩略图 / 数值卡片，
  因此只连接数值或特征数据时也有可看的预览；
* 与 image / number / edges / contours / components 同名的输出端口把输入对象
  **原样**返回（不做类型转换、不复制数组），可直接把本节点插在任意链路中当检查点；
* display 端口额外输出汇总显示面板，供需要把面板继续接线的场景使用。

显示面板只使用 OpenCV 内置字体，文本统一为 ASCII，避免缺少中文字体时出现乱码；
中文的端口名、节点名与摘要仍会显示在界面其它位置。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.core.data_types import DataType, format_number, render_preview
from app.core.types import ImageData, to_display_rgb
from app.operators.spec import Operator, OperatorSpec, PortSpec

#: 透传端口顺序，也是显示面板中数据块的排列顺序。
_PASS_KEYS = ("image", "number", "edges", "contours", "components")

#: 每种数据类型在显示面板中的英文标题。
_TITLES = {
    "image": "IMAGE",
    "number": "NUMBER",
    "edges": "EDGES",
    "contours": "CONTOURS",
    "components": "COMPONENTS",
}


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    """把 `#RRGGBB` 转换为 OpenCV 使用的 BGR 三元组。"""

    text = str(color).lstrip("#")
    red = int(text[0:2], 16)
    green = int(text[2:4], 16)
    blue = int(text[4:6], 16)
    return (blue, green, red)


#: 每种数据类型对应的端口圆点颜色（BGR），直接复用 DataType.color 保持与画布一致。
_TYPE_COLORS = {data_type: _hex_to_bgr(data_type.color) for data_type in DataType}

_TYPE_BY_KEY = {
    "image": DataType.IMAGE,
    "number": DataType.NUMBER,
    "edges": DataType.EDGES,
    "contours": DataType.CONTOURS,
    "components": DataType.COMPONENTS,
}

# ---------------------------------------------------------------------------
# 面板布局常量
# ---------------------------------------------------------------------------
_PANEL_WIDTH = 720
_PADDING = 20
_GAP = 12
_HEADER_HEIGHT = 58
_LABEL_HEIGHT = 22
_TILE_WIDTH = 160
_TILE_HEIGHT = 150
_TILE_INNER_TOP = 26
_TILE_INNER_HEIGHT = 92
_IMAGE_MAX_HEIGHT = 300
_CONTENT_TOP_GAP = 14
_BOTTOM_PAD = 18

# 配色（BGR）。
_BG = (26, 20, 15)
_HEADER_BG = (33, 25, 19)
_CARD_BG = (52, 38, 29)
_CARD_BORDER = (78, 62, 49)
_TEXT = (238, 234, 230)
_MUTED = (168, 153, 141)
_ACCENT = (235, 99, 37)

_FONT = cv2.FONT_HERSHEY_SIMPLEX

#: 每种输入的端口标签与说明。
_PORT_TEXT = {
    "image": ("图像", "待显示 / 原样透传的图像。"),
    "number": ("数值", "待显示 / 原样透传的数值。"),
    "edges": ("边缘特征", "待显示 / 原样透传的边缘线段。"),
    "contours": ("轮廓特征", "待显示 / 原样透传的轮廓。"),
    "components": ("连通域", "待显示 / 原样透传的连通域标签图。"),
}


def _input_ports() -> tuple[PortSpec, ...]:
    return tuple(
        PortSpec(
            key,
            label,
            optional=True,
            data_type=_TYPE_BY_KEY[key],
            description=description,
        )
        for key, (label, description) in _PORT_TEXT.items()
    )


def _output_ports() -> tuple[PortSpec, ...]:
    return tuple(
        PortSpec(key, label, data_type=_TYPE_BY_KEY[key])
        for key, (label, _description) in _PORT_TEXT.items()
    ) + (PortSpec("display", "显示面板", accepts=("BGR",)),)


def _clean(text: Any, fallback: str = "") -> str:
    """只保留可被 OpenCV 字体绘制的 ASCII 文本。"""

    cleaned = "".join(char for char in str(text) if 32 <= ord(char) <= 126)
    cleaned = " ".join(cleaned.split())
    return cleaned or fallback


def _text_width(text: Any, scale: float, thickness: int = 1) -> int:
    cleaned = _clean(text)
    if not cleaned:
        return 0
    return int(cv2.getTextSize(cleaned, _FONT, float(scale), thickness)[0][0])


def _text_height(scale: float, thickness: int = 1) -> int:
    return int(cv2.getTextSize("Ag", _FONT, float(scale), thickness)[0][1])


def _elide(text: Any, max_width: int, scale: float, thickness: int = 1) -> str:
    """按像素宽度截断 ASCII 文本，超出时追加 ...。"""

    cleaned = _clean(text)
    if not cleaned or _text_width(cleaned, scale, thickness) <= max_width:
        return cleaned
    suffix = "..."
    while cleaned and _text_width(cleaned + suffix, scale, thickness) > max_width:
        cleaned = cleaned[:-1]
    return (cleaned + suffix) if cleaned else ""


def _put_text(
    canvas: np.ndarray,
    text: Any,
    x: float,
    y: float,
    *,
    scale: float = 0.5,
    color: tuple[int, int, int] = _TEXT,
    thickness: int = 1,
) -> None:
    cleaned = _clean(text)
    if not cleaned:
        return
    cv2.putText(
        canvas,
        cleaned,
        (int(round(x)), int(round(y))),
        _FONT,
        float(scale),
        color,
        thickness,
        cv2.LINE_AA,
    )


def _to_bgr8(value: Any) -> np.ndarray | None:
    """把图像或结构化数据转换为显示面板使用的 uint8 BGR 缩略图。"""

    if isinstance(value, ImageData):
        image = value
    else:
        image = render_preview(value)
        if image is None:
            return None
    rgb = to_display_rgb(image)
    if rgb.size == 0:
        return None
    return np.ascontiguousarray(rgb[:, :, ::-1])


def _fit(
    array: np.ndarray,
    max_width: int,
    max_height: int,
    max_scale: float = 4.0,
    upscale_interpolation: int = cv2.INTER_LINEAR,
) -> np.ndarray:
    """等比缩放到给定范围内；不放大超过 max_scale，避免小图糊成色块。"""

    height, width = array.shape[:2]
    if width <= 0 or height <= 0:
        return array
    scale = min(float(max_width) / width, float(max_height) / height, float(max_scale))
    if scale <= 0:
        return array
    if abs(scale - 1.0) < 1e-3:
        return array
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else upscale_interpolation
    return cv2.resize(array, size, interpolation=interpolation)


def _title_color(key: str) -> tuple[int, int, int]:
    return _TYPE_COLORS[_TYPE_BY_KEY[key]]


def _draw_header(canvas: np.ndarray, connected: int) -> None:
    width = canvas.shape[1]
    cv2.rectangle(canvas, (0, 0), (width - 1, _HEADER_HEIGHT - 1), _HEADER_BG, -1)
    cv2.rectangle(canvas, (0, 0), (width - 1, 2), _ACCENT, -1)
    cv2.rectangle(
        canvas,
        (_PADDING, 17),
        (_PADDING + 3, _HEADER_HEIGHT - 16),
        _ACCENT,
        -1,
    )
    _put_text(canvas, "OUTPUT VIEW", _PADDING + 12, 30, scale=0.58, color=_TEXT)
    _put_text(
        canvas,
        f"{connected}/5 INPUTS | PASS-THROUGH ENABLED",
        _PADDING + 12,
        47,
        scale=0.36,
        color=_MUTED,
    )
    status = f"{connected} CONNECTED"
    _put_text(
        canvas,
        status,
        width - _PADDING - _text_width(status, 0.4),
        33,
        scale=0.4,
        color=_ACCENT,
    )


def _number_subtitle(value: Any) -> str:
    label = _clean(getattr(value, "label", ""))
    unit = _clean(getattr(value, "unit", ""))
    if label:
        return f"{label} {unit}".strip()
    return f"VALUE {unit}".strip()


def _tile_subtitle(key: str, value: Any) -> str:
    image_size = getattr(value, "image_size", None)
    size_text = f" | {image_size[0]}x{image_size[1]}" if image_size else ""
    if key == "number":
        return _number_subtitle(value)
    if key == "edges":
        return f"{value.count} SEGMENTS{size_text}"
    if key == "contours":
        return f"{value.count} CONTOURS{size_text}"
    if key == "components":
        return f"{value.count} COMPONENTS | {value.total_area} PX"
    return _TITLES[key]


def _draw_number_tile(canvas: np.ndarray, x: int, y: int, value: Any) -> None:
    text = _clean(format_number(getattr(value, "value", value)), "0")
    inner_width = _TILE_WIDTH - 20
    scale = 0.95
    while scale > 0.4 and _text_width(text, scale) > inner_width:
        scale = round(scale - 0.05, 2)
    text_width = _text_width(text, scale)
    text_height = _text_height(scale)
    center_y = y + _TILE_INNER_TOP + _TILE_INNER_HEIGHT / 2.0
    _put_text(
        canvas,
        text,
        x + (_TILE_WIDTH - text_width) / 2.0,
        center_y + text_height / 2.0,
        scale=scale,
        color=_TEXT,
    )


def _draw_tile(canvas: np.ndarray, x: int, y: int, key: str, value: Any) -> None:
    accent = _title_color(key)
    cv2.rectangle(canvas, (x, y), (x + _TILE_WIDTH - 1, y + _TILE_HEIGHT - 1), _CARD_BG, -1)
    cv2.rectangle(canvas, (x, y), (x + _TILE_WIDTH - 1, y + _TILE_HEIGHT - 1), _CARD_BORDER, 1)
    cv2.rectangle(canvas, (x, y), (x + _TILE_WIDTH - 1, y + 2), accent, -1)
    _put_text(canvas, _TITLES[key], x + 10, y + 18, scale=0.4, color=accent)

    if key == "number":
        _draw_number_tile(canvas, x, y, value)
    else:
        preview = _to_bgr8(value)
        if preview is not None:
            # 连通域标签图通常分辨率很小，允许放大更多并用最近邻保持色块边界。
            component = key == "components"
            preview = _fit(
                preview,
                _TILE_WIDTH - 16,
                _TILE_INNER_HEIGHT,
                max_scale=8.0 if component else 3.0,
                upscale_interpolation=cv2.INTER_NEAREST if component else cv2.INTER_LINEAR,
            )
            preview_h, preview_w = preview.shape[:2]
            origin_x = x + (_TILE_WIDTH - preview_w) // 2
            origin_y = y + _TILE_INNER_TOP + (_TILE_INNER_HEIGHT - preview_h) // 2
            canvas[
                origin_y : origin_y + preview_h,
                origin_x : origin_x + preview_w,
            ] = preview

    subtitle = _elide(_tile_subtitle(key, value), _TILE_WIDTH - 20, 0.33)
    _put_text(canvas, subtitle, x + 10, y + _TILE_HEIGHT - 9, scale=0.33, color=_MUTED)


def _draw_image_block(
    canvas: np.ndarray,
    top: int,
    image: ImageData,
    preview: np.ndarray,
) -> int:
    """绘制图像块，返回到下一块内容的纵向位置。"""

    width = canvas.shape[1]
    _put_text(
        canvas,
        f"IMAGE | {image.width}x{image.height} | {image.color_space} | {image.dtype_name}",
        _PADDING,
        top + 15,
        scale=0.42,
        color=_title_color("image"),
    )
    area_height = max(120, preview.shape[0])
    area_top = top + _LABEL_HEIGHT
    origin_x = (width - preview.shape[1]) // 2
    origin_y = area_top + (area_height - preview.shape[0]) // 2
    canvas[
        origin_y : origin_y + preview.shape[0],
        origin_x : origin_x + preview.shape[1],
    ] = preview
    cv2.rectangle(
        canvas,
        (origin_x - 1, origin_y - 1),
        (origin_x + preview.shape[1], origin_y + preview.shape[0]),
        _CARD_BORDER,
        1,
    )
    return area_top + area_height


def _build_display(values: dict[str, Any]) -> ImageData:
    """把已连接的输入汇总成一张显示面板；没有任何输入时给出提示面板。"""

    connected = [key for key in _PASS_KEYS if values.get(key) is not None]
    image: ImageData | None = values.get("image")
    tile_keys = [key for key in connected if key != "image"]

    if not connected:
        width, height = 640, 210
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
        canvas[:] = _BG
        _draw_header(canvas, 0)
        title = "NO INPUT CONNECTED"
        _put_text(
            canvas,
            title,
            (width - _text_width(title, 0.62)) / 2.0,
            120,
            scale=0.62,
            color=_TEXT,
        )
        hint = "CONNECT IMAGE / NUMBER / EDGES / CONTOURS / COMPONENTS"
        _put_text(
            canvas,
            hint,
            (width - _text_width(hint, 0.4)) / 2.0,
            150,
            scale=0.4,
            color=_MUTED,
        )
        note = "EVERY INPUT IS OPTIONAL AND PASSES THROUGH UNCHANGED"
        _put_text(
            canvas,
            note,
            (width - _text_width(note, 0.34)) / 2.0,
            174,
            scale=0.34,
            color=_MUTED,
        )
        return ImageData(canvas, "输出预览", "BGR")

    width = _PANEL_WIDTH if image is not None else max(
        320,
        2 * _PADDING + len(tile_keys) * _TILE_WIDTH + max(0, len(tile_keys) - 1) * _GAP,
    )

    image_preview = _to_bgr8(image) if image is not None else None
    image_fit: np.ndarray | None = None
    if image_preview is not None:
        image_fit = _fit(
            image_preview,
            width - 2 * _PADDING,
            _IMAGE_MAX_HEIGHT,
            max_scale=4.0,
        )

    image_block_height = 0
    if image is not None:
        image_block_height = _LABEL_HEIGHT + max(120, image_fit.shape[0] if image_fit is not None else 0)

    content_height = image_block_height
    if tile_keys:
        content_height += (_GAP if image_block_height else 0) + _TILE_HEIGHT
    height = _HEADER_HEIGHT + _CONTENT_TOP_GAP + content_height + _BOTTOM_PAD

    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:] = _BG
    _draw_header(canvas, len(connected))

    y = _HEADER_HEIGHT + _CONTENT_TOP_GAP
    if image is not None:
        if image_fit is not None:
            y = _draw_image_block(canvas, y, image, image_fit)
        else:
            _put_text(
                canvas,
                "IMAGE | EMPTY",
                _PADDING,
                y + 15,
                scale=0.42,
                color=_title_color("image"),
            )
            y += _LABEL_HEIGHT + 120
        if tile_keys:
            y += _GAP

    if tile_keys:
        row_width = len(tile_keys) * _TILE_WIDTH + (len(tile_keys) - 1) * _GAP
        x = (width - row_width) // 2
        for key in tile_keys:
            _draw_tile(canvas, x, y, key, values[key])
            x += _TILE_WIDTH + _GAP

    return ImageData(canvas, "输出预览", "BGR")


class OutputDisplay(Operator):
    """输出预览：汇总显示全部已知类型输入，并逐路原样透传。"""

    spec = OperatorSpec(
        id="output.display",
        name="输出预览",
        category=("其他", "可视化"),
        description=(
            "接收图像、数值、边缘特征、轮廓特征与连通域等全部已知类型输入，"
            "把所有已连接的数据汇总成一张显示面板供预览；同时从同名输出端口把"
            "每一路输入原样透传，可作为流程终点或链路中途的检查点。"
        ),
        keywords=(
            "输出",
            "显示",
            "预览",
            "可视化",
            "透传",
            "检查点",
            "output",
            "display",
            "viewer",
            "sink",
            "debug",
        ),
        notes=(
            "五个输入端口均为可选。面板按 IMAGE、NUMBER、EDGES、CONTOURS、"
            "COMPONENTS 的顺序展示已连接的数据；display 端口输出汇总面板，"
            "其余输出端口与输入一一对应，数据不做任何转换。"
        ),
        inputs=_input_ports(),
        outputs=_output_ports(),
    )

    def process(
        self,
        image: ImageData | None = None,
        number: Any = None,
        edges: Any = None,
        contours: Any = None,
        components: Any = None,
        **_: Any,
    ) -> dict[str, Any]:
        values = {
            "image": image,
            "number": number,
            "edges": edges,
            "contours": contours,
            "components": components,
        }
        # display 放在字典首位，NodeResult.preview_image 会优先把它作为节点预览；
        # spec.outputs 中 image 仍排在第一位，画布自动插入连线时优先使用透传端口。
        result: dict[str, Any] = {"display": _build_display(values)}
        for key in _PASS_KEYS:
            value = values[key]
            if value is not None:
                result[key] = value
        return result


OPERATOR = OutputDisplay
