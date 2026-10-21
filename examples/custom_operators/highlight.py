"""示例自定义算子：交互标注（矩形 / 线段 / 点集），输出图像 + 蒙版。

这个文件演示了算子开发指南中的大部分能力：

* 两个输入端口，其中 ``mask`` 为可选输入；
* 两个输出端口（BGR 标注图 + 单通道蒙版），``process`` 返回字典；
* ``int / float / bool / enum / color / points / roi / line`` 之外的常用类型；
* ``visible_when`` 的两种写法：裸变量名 ``mode`` 与 ``params['mode']``；
* 交互参数在预览中拾取后自动回填。

使用方式：把本文件复制到 ``operators/``（或任意在“设置 → 算子目录”里
添加的目录），按 F5 刷新算子库即可在“其他 / 示例”分类下看到它。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec


def _to_bgr_u8(array: np.ndarray) -> np.ndarray:
    """把任意支持 dtype 的图像转换为 BGR uint8，仅用于预览标注。"""

    if array.dtype != np.uint8:
        work = array.astype(np.float32)
        if array.dtype == np.uint16:
            work = work / 257.0
        else:
            low = float(np.nanmin(work))
            high = float(np.nanmax(work))
            if low >= 0.0 and high <= 1.0:
                work = work * 255.0
            elif high > low:
                work = (work - low) / (high - low) * 255.0
        array = np.clip(np.rint(work), 0, 255).astype(np.uint8)
    if array.ndim == 2:
        return cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
    if array.ndim == 3 and array.shape[2] == 4:
        return array[:, :, :3].copy()
    if array.ndim == 3 and array.shape[2] == 3:
        return array.copy()
    raise OperatorError(f"不支持的图像形状：{array.shape}")


def _to_mask_u8(array: np.ndarray) -> np.ndarray:
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise OperatorError("蒙版输入必须是单通道灰度图。")
    if array.dtype == np.uint8:
        return array
    return np.clip(np.rint(array.astype(np.float32)), 0, 255).astype(np.uint8)


def _hex_to_bgr(color: str) -> tuple[int, int, int]:
    text = str(color or "").lstrip("#")
    if len(text) == 8:  # #RRGGBBAA，标注忽略 alpha
        text = text[:6]
    if len(text) != 6:
        raise OperatorError(f"颜色必须是 #RRGGBB 或 #RRGGBBAA，当前为 {color!r}。")
    try:
        red, green, blue = int(text[0:2], 16), int(text[2:4], 16), int(text[4:6], 16)
    except ValueError as exc:
        raise OperatorError(f"颜色值无效：{color!r}。") from exc
    return blue, green, red


class Highlight(Operator):
    spec = OperatorSpec(
        id="example.highlight",
        name="示例：交互标注",
        category=("其他", "示例"),
        description=(
            "在图像上绘制矩形、线段或点集，并同时输出标注图与二值蒙版。"
            "交互参数可在右侧预览中拾取；输出统一为 uint8，便于预览。"
        ),
        version="1.0.0",
        author="MineImage examples",
        keywords=("示例", "标注", "roi", "line", "points", "mask"),
        inputs=(
            PortSpec("image", "图像", accepts=("L", "BGR", "BGRA")),
            PortSpec("mask", "蒙版（可选）", accepts=("L",), optional=True),
        ),
        outputs=(
            PortSpec("image", "标注图", accepts=("BGR",)),
            PortSpec("mask", "高亮蒙版", accepts=("L",)),
        ),
        params=(
            ParamSpec(
                "mode",
                "标注方式",
                "enum",
                choices=(("rect", "矩形"), ("line", "线段"), ("points", "点集")),
                default="rect",
                group="形状",
                tooltip="选择后下方只显示对应的交互参数。",
            ),
            ParamSpec(
                "roi",
                "矩形区域",
                "roi",
                default=(80, 60, 320, 200),
                group="形状",
                visible_when="params['mode'] == 'rect'",
                enabled_when="params['mode'] == 'rect'",
                tooltip="点击“拾取”后在预览中拖拽框选。",
            ),
            ParamSpec(
                "line",
                "线段",
                "line",
                default=(60, 60, 400, 260),
                group="形状",
                visible_when="mode == 'line'",
                enabled_when="mode == 'line'",
                tooltip="裸变量名与 params['mode'] 两种写法等价。",
            ),
            ParamSpec(
                "points",
                "参考点",
                "points",
                default=((120, 120), (240, 120), (240, 240)),
                count=3,
                group="形状",
                visible_when="params['mode'] == 'points'",
                tooltip="需要恰好 3 个点；在预览中依次点击，Enter 或右键完成。",
            ),
            ParamSpec(
                "fill",
                "填充矩形",
                "bool",
                default=False,
                group="形状",
                visible_when="mode == 'rect'",
            ),
            ParamSpec(
                "color",
                "颜色",
                "color",
                default="#ff3b30",
                group="外观",
            ),
            ParamSpec(
                "thickness",
                "线宽",
                "int",
                default=2,
                min=1,
                max=31,
                step=1,
                unit="px",
                group="外观",
            ),
        ),
    )

    def process(
        self,
        image=None,
        mask=None,
        mode: str = "rect",
        roi=None,
        line=None,
        points=None,
        fill: bool = False,
        color: str = "#ff3b30",
        thickness: int = 2,
        **_: Any,
    ):
        if image is None:
            raise OperatorError("缺少图像输入。")
        source = image.array
        preview = _to_bgr_u8(source)
        height, width = preview.shape[:2]
        shape_mask = np.zeros((height, width), np.uint8)
        overlay = np.zeros_like(preview)
        bgr = _hex_to_bgr(color)
        stroke = max(1, int(thickness))

        if mode == "rect":
            if not roi or len(roi) != 4:
                raise OperatorError("请先在预览中框选矩形区域。")
            x, y, w, h = (int(round(float(value))) for value in roi)
            if w <= 0 or h <= 0:
                raise OperatorError("矩形区域的宽高必须大于 0。")
            p1, p2 = (x, y), (x + w, y + h)
            cv2.rectangle(shape_mask, p1, p2, 255, -1 if fill else stroke)
            cv2.rectangle(overlay, p1, p2, bgr, -1 if fill else stroke)
        elif mode == "line":
            if not line or len(line) != 4:
                raise OperatorError("请先在预览中绘制一条线段。")
            x1, y1, x2, y2 = (int(round(float(value))) for value in line)
            if (x1, y1) == (x2, y2):
                raise OperatorError("线段的两个端点不能重合。")
            cv2.line(shape_mask, (x1, y1), (x2, y2), 255, stroke)
            cv2.line(overlay, (x1, y1), (x2, y2), bgr, stroke)
        elif mode == "points":
            if not points:
                raise OperatorError("请先在预览中拾取 3 个参考点。")
            radius = max(3, stroke * 2)
            for point in points:
                px, py = int(round(float(point[0]))), int(round(float(point[1])))
                cv2.circle(shape_mask, (px, py), radius, 255, -1)
                cv2.circle(overlay, (px, py), radius, bgr, -1)
        else:  # pragma: no cover - 枚举已由框架校验
            raise OperatorError(f"不支持的标注方式：{mode}")

        if mask is not None:
            mask_array = _to_mask_u8(mask.array)
            if mask_array.shape != shape_mask.shape:
                raise OperatorError(
                    f"蒙版尺寸 {mask_array.shape[1]}×{mask_array.shape[0]} 与图像 "
                    f"{width}×{height} 不一致。"
                )
            shape_mask = cv2.bitwise_and(shape_mask, mask_array)

        blended = cv2.addWeighted(preview, 0.35, overlay, 0.65, 0)
        annotated = np.where(shape_mask[:, :, None] > 0, blended, preview)
        return {"image": annotated, "mask": shape_mask}


OPERATOR = Highlight
