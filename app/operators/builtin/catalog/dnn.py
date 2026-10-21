"""OpenCV 目录算子：深度学习（DNN）。

包含 ``dnn.blob``（DNN 预处理可视化）与 ``dnn.infer``（DNN 推理可视化）两个算子。

约定
----
* ``dnn.blob`` 默认参数即可运行，不需要模型文件；
* ``dnn.infer`` 的模型路径默认为空字符串，未选择模型时抛出中文 ``OperatorError``，
  这属于预期行为（没有模型无法推理），测试中会跳过它；
* 推理结果统一渲染成文本面板 / 特征条图，主输出保持图像。
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from app.core.data_types import DataType, NumberValue
from app.operators.builtin.catalog import _cv
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec


def _to_display(values: np.ndarray) -> np.ndarray:
    """把任意浮点特征图按 min-max 归一化到 0..255 的 uint8。"""
    work = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    low = float(work.min()) if work.size else 0.0
    high = float(work.max()) if work.size else 0.0
    if high - low > 1e-12:
        work = (work - low) / (high - low) * 255.0
    else:
        work = np.zeros_like(work)
    return np.clip(np.rint(work), 0, 255).astype(np.uint8)


def _blob_from(
    image,
    width: int,
    height: int,
    mean_b: float,
    mean_g: float,
    mean_r: float,
    scale: float,
    swap_rb: bool,
) -> np.ndarray:
    """按统一参数生成 DNN 输入 blob（形状 ``1×3×H×W``）。"""
    bgr = _cv.source_bgr8(image, "输入图像")
    return cv2.dnn.blobFromImage(
        bgr,
        scalefactor=float(scale),
        size=(int(width), int(height)),
        mean=(float(mean_b), float(mean_g), float(mean_r)),
        swapRB=bool(swap_rb),
        crop=False,
    )


def _feature_panel(values: np.ndarray, top_k: int) -> np.ndarray:
    """把一维特征向量渲染成黑底柱状图，便于预览。"""
    flat = np.asarray(values, dtype=np.float32).ravel()
    width, height = 480, 320
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    if flat.size == 0:
        return _cv.text_panel(width, height, ["Empty output"])
    top = flat[np.argsort(np.abs(flat))[::-1][: max(int(top_k), 1)]]
    maximum = float(np.max(np.abs(top))) if top.size else 1.0
    maximum = max(maximum, 1e-12)
    bar_width = max(4, width // (len(top) * 2 + 1))
    baseline = height - 24
    for index, value in enumerate(top):
        bar_height = int(abs(float(value)) / maximum * (height - 60))
        left = 16 + index * (bar_width + 8)
        right = min(left + bar_width, width - 4)
        cv2.rectangle(
            canvas,
            (left, baseline - bar_height),
            (right, baseline),
            (0, 200, 255) if value >= 0 else (60, 60, 255),
            -1,
        )
    cv2.putText(
        canvas,
        f"top{len(top)} of {flat.size}",
        (12, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return canvas


# ---------------------------------------------------------------------------
# 1. DNN 预处理可视化
# ---------------------------------------------------------------------------
class DnnBlob(Operator):
    spec = OperatorSpec(
        id="dnn.blob",
        name="DNN 预处理（Blob）",
        category=("深度学习", "DNN"),
        description=(
            "执行与 cv2.dnn.blobFromImage 一致的预处理：缩放到指定尺寸、减均值、乘缩放系数，"
            "并把归一化后的三个通道可视化出来，用于检查 DNN 输入是否符合预期。"
            "不需要模型文件，默认参数即可运行。输出可视化图与 Blob 均值。"
        ),
        keywords=("DNN", "blob", "预处理", "归一化", "preprocess", "深度学习"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(
            PortSpec("image", "预处理可视化", accepts=("BGR",)),
            PortSpec("mean", "Blob 均值", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec("width", "输入宽度", "int", default=128, min=8, max=1024, step=8, unit="px", group="尺寸"),
            ParamSpec("height", "输入高度", "int", default=128, min=8, max=1024, step=8, unit="px", group="尺寸"),
            ParamSpec("mean_b", "蓝色均值", "float", default=104.0, min=0.0, max=255.0, step=1.0, group="归一化"),
            ParamSpec("mean_g", "绿色均值", "float", default=117.0, min=0.0, max=255.0, step=1.0, group="归一化"),
            ParamSpec("mean_r", "红色均值", "float", default=124.0, min=0.0, max=255.0, step=1.0, group="归一化"),
            ParamSpec(
                "scale",
                "缩放系数",
                "float",
                default=1.0 / 255.0,
                min=1e-6,
                max=10.0,
                step=0.001,
                group="归一化",
            ),
            ParamSpec("swap_rb", "交换红蓝通道", "bool", default=True, group="归一化"),
        ),
    )

    def process(
        self,
        image=None,
        width: int = 128,
        height: int = 128,
        mean_b: float = 104.0,
        mean_g: float = 117.0,
        mean_r: float = 124.0,
        scale: float = 1.0 / 255.0,
        swap_rb: bool = True,
        **_: Any,
    ):
        blob = _blob_from(image, width, height, mean_b, mean_g, mean_r, scale, swap_rb)
        planes = blob[0]
        canvas = np.zeros((int(height), int(width), 3), dtype=np.uint8)
        for index in range(min(3, planes.shape[0])):
            canvas[:, :, index] = _to_display(planes[index])
        return {
            "image": canvas,
            "mean": NumberValue(value=float(blob.mean()), label="Blob 均值"),
        }


# ---------------------------------------------------------------------------
# 2. DNN 推理
# ---------------------------------------------------------------------------
class DnnInfer(Operator):
    spec = OperatorSpec(
        id="dnn.infer",
        name="DNN 推理",
        category=("深度学习", "DNN"),
        description=(
            "加载 ONNX / Caffe / TensorFlow 等模型文件，按统一参数生成 Blob 并前向推理，"
            "把输出向量中最强的若干响应画成柱状图。模型路径为空时不会执行推理，"
            "而是提示先选择模型文件；请先通过参数面板选择模型后再运行。"
        ),
        keywords=("DNN", "推理", "infer", "ONNX", "模型", "深度学习"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(
            PortSpec("image", "推理响应图", accepts=("BGR",)),
            PortSpec("count", "输出元素数", data_type=DataType.NUMBER),
            PortSpec("max_response", "最大响应", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "path",
                "模型文件",
                "path",
                default="",
                placeholder="请选择 ONNX / Caffe 模型",
                group="模型",
                tooltip="必填；留空时算子会提示先选择模型文件。",
            ),
            ParamSpec("width", "输入宽度", "int", default=224, min=8, max=2048, step=8, unit="px", group="尺寸"),
            ParamSpec("height", "输入高度", "int", default=224, min=8, max=2048, step=8, unit="px", group="尺寸"),
            ParamSpec("mean_b", "蓝色均值", "float", default=104.0, min=0.0, max=255.0, step=1.0, group="归一化"),
            ParamSpec("mean_g", "绿色均值", "float", default=117.0, min=0.0, max=255.0, step=1.0, group="归一化"),
            ParamSpec("mean_r", "红色均值", "float", default=124.0, min=0.0, max=255.0, step=1.0, group="归一化"),
            ParamSpec(
                "scale",
                "缩放系数",
                "float",
                default=1.0 / 255.0,
                min=1e-6,
                max=10.0,
                step=0.001,
                group="归一化",
            ),
            ParamSpec("swap_rb", "交换红蓝通道", "bool", default=True, group="归一化"),
            ParamSpec("top_k", "显示前 K 项", "int", default=5, min=1, max=32, step=1, group="显示"),
        ),
    )

    def process(
        self,
        image=None,
        path: str = "",
        width: int = 224,
        height: int = 224,
        mean_b: float = 104.0,
        mean_g: float = 117.0,
        mean_r: float = 124.0,
        scale: float = 1.0 / 255.0,
        swap_rb: bool = True,
        top_k: int = 5,
        **_: Any,
    ):
        model_path = str(path or "").strip()
        if not model_path:
            raise OperatorError("请先选择模型文件（ONNX 等）后再运行 DNN 推理。")
        if not os.path.isfile(model_path):
            raise OperatorError(f"模型文件不存在：{model_path}")
        try:
            network = cv2.dnn.readNet(model_path)
        except cv2.error as exc:
            raise OperatorError(f"模型加载失败，请确认文件格式是否受支持：{model_path}") from exc
        blob = _blob_from(image, width, height, mean_b, mean_g, mean_r, scale, swap_rb)
        network.setInput(blob)
        try:
            output = network.forward()
        except cv2.error as exc:
            raise OperatorError("模型推理失败：输入尺寸或网络结构与模型不匹配。") from exc
        flat = np.asarray(output, dtype=np.float32).ravel()
        return {
            "image": _feature_panel(flat, int(top_k)),
            "count": NumberValue(value=float(flat.size), label="输出元素数", unit="个"),
            "max_response": NumberValue(
                value=float(np.max(flat)) if flat.size else 0.0,
                label="最大响应",
            ),
        }


OPERATORS = (
    DnnBlob,
    DnnInfer,
)
