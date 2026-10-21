"""OpenCV 目录算子：深度学习 / 检测。

包含 Haar 人脸检测、二维码识别、条码识别与 HOG 行人检测 4 个算子。

约定
----
* 只有在需要 8 位图像时内部统一走 ``_cv.source_gray8`` / ``_cv.source_bgr8``；
* 主输出是画了检测框的 BGR 图像，额外输出检测数量（``DataType.NUMBER``）；
* 默认参数即可运行：检测不到目标时输出原图副本与数量 0，不抛异常；
* 模型加载失败、参数非法等情况统一抛中文 ``OperatorError``。
"""

from __future__ import annotations

import os
from typing import Any

import cv2
import numpy as np

from app.core.data_types import DataType, NumberValue
from app.operators.builtin.catalog import _cv
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec

#: HOG 默认 SVM 行人检测器要求的最小输入尺寸（宽, 高）。
HOG_MIN_WIDTH = 64
HOG_MIN_HEIGHT = 128


def _draw_rects(canvas: np.ndarray, rects: Any, color: str, thickness: int = 2) -> np.ndarray:
    """在 BGR 画布上批量画矩形；``rects`` 为 ``(x, y, w, h)`` 序列。"""
    draw_color = _cv.draw_color(canvas, color)
    line_type = _cv.draw_type(canvas)
    for rect in rects:
        x, y, width, height = (int(round(float(value))) for value in rect)
        cv2.rectangle(canvas, (x, y), (x + width, y + height), draw_color, int(thickness), line_type)
    return canvas


def _draw_label(canvas: np.ndarray, text: str, origin: tuple[int, int], color: str) -> None:
    """在 BGR 画布上写英文短标签（OpenCV 内置字体不支持中文）。"""
    draw_color = _cv.draw_color(canvas, color)
    cv2.putText(
        canvas,
        str(text),
        (int(origin[0]), int(origin[1])),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        draw_color,
        1,
        _cv.draw_type(canvas),
    )


# ---------------------------------------------------------------------------
# 1. Haar 人脸检测
# ---------------------------------------------------------------------------
class DetectFace(Operator):
    spec = OperatorSpec(
        id="detect.face",
        name="人脸检测（Haar）",
        category=("深度学习", "目标检测"),
        description=(
            "用 OpenCV 自带的 Haar 级联分类器检测正面人脸，并在画面上画出人脸框。"
            "模型路径留空时使用 cv2.data.haarcascades 中的默认正面人脸模型；"
            "输出标注图和检测到的人脸数量。默认参数即可运行。"
        ),
        keywords=("人脸", "人脸检测", "Haar", "face", "cascade", "检测"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(
            PortSpec("image", "人脸标注图", accepts=("BGR",)),
            PortSpec("count", "人脸数量", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "path",
                "模型路径",
                "path",
                default="",
                placeholder="留空使用 OpenCV 默认模型",
                group="模型",
                tooltip="自定义 Haar 级联 XML 路径；留空使用 haarcascade_frontalface_default.xml。",
            ),
            ParamSpec(
                "scale_factor",
                "缩放步长",
                "float",
                default=1.1,
                min=1.01,
                max=3.0,
                step=0.01,
                group="检测",
            ),
            ParamSpec("min_neighbors", "最小邻居数", "int", default=3, min=0, max=50, step=1, group="检测"),
            ParamSpec(
                "min_size",
                "最小人脸尺寸",
                "int",
                default=20,
                min=4,
                max=2000,
                step=1,
                unit="px",
                group="检测",
            ),
            ParamSpec("color", "标注颜色", "color", default="#00d26a", group="显示"),
        ),
    )

    def _cascade_path(self, path: str) -> str:
        custom = str(path or "").strip()
        if custom:
            if not os.path.isfile(custom):
                raise OperatorError(f"人脸检测模型文件不存在：{custom}")
            return custom
        return os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")

    def process(
        self,
        image=None,
        path: str = "",
        scale_factor: float = 1.1,
        min_neighbors: int = 3,
        min_size: int = 20,
        color: str = "#00d26a",
        **_: Any,
    ):
        gray = _cv.source_gray8(image, "输入图像")
        canvas = _cv.source_bgr8(image, "输入图像")
        cascade = cv2.CascadeClassifier(self._cascade_path(path))
        if cascade.empty():
            raise OperatorError("无法加载人脸检测模型，请检查模型路径是否正确。")
        faces = cascade.detectMultiScale(
            gray,
            scaleFactor=float(scale_factor),
            minNeighbors=int(min_neighbors),
            minSize=(int(min_size), int(min_size)),
        )
        if faces is not None and len(faces) > 0:
            _draw_rects(canvas, faces, color)
        return {
            "image": canvas,
            "count": NumberValue(value=0.0 if faces is None else float(len(faces)), label="人脸数量", unit="个"),
        }


# ---------------------------------------------------------------------------
# 2. 二维码识别
# ---------------------------------------------------------------------------
class DetectQrCode(Operator):
    spec = OperatorSpec(
        id="detect.qr",
        name="二维码识别",
        category=("深度学习", "目标检测"),
        description=(
            "用 OpenCV 的 QRCodeDetector 定位并解码二维码，在画面上画出二维码四边形轮廓。"
            "输出标注图与识别到的二维码数量；未识别到二维码时输出原图副本与数量 0。"
        ),
        keywords=("二维码", "QR", "qrcode", "解码", "detect"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(
            PortSpec("image", "二维码标注图", accepts=("BGR",)),
            PortSpec("count", "二维码数量", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec("min_size", "最小尺寸", "int", default=0, min=0, max=2000, step=1, unit="px",
                      tooltip="大于 0 时忽略短边小于该值的检测结果。"),
            ParamSpec("color", "标注颜色", "color", default="#00d26a", group="显示"),
        ),
    )

    def process(self, image=None, min_size: int = 0, color: str = "#00d26a", **_: Any):
        gray = _cv.source_gray8(image, "输入图像")
        canvas = _cv.source_bgr8(image, "输入图像")
        detector = cv2.QRCodeDetector()
        try:
            decoded, points, _straight = detector.detectAndDecode(gray)
        except cv2.error as exc:
            raise OperatorError("二维码识别失败：输入图像格式不受支持。") from exc
        count = 0
        if points is not None and len(points) > 0:
            polygon = np.asarray(points[0], dtype=np.float32).reshape(-1, 2)
            if len(polygon) >= 4:
                width = float(np.linalg.norm(polygon[0] - polygon[1]))
                height = float(np.linalg.norm(polygon[1] - polygon[2]))
                if int(min_size) <= 0 or min(width, height) >= float(min_size):
                    count = 1
                    cv2.polylines(
                        canvas,
                        [polygon.astype(np.int32).reshape(-1, 1, 2)],
                        True,
                        _cv.draw_color(canvas, color),
                        2,
                        _cv.draw_type(canvas),
                    )
                    _draw_label(canvas, decoded or "QR", (int(polygon[0][0]), max(int(polygon[0][1]) - 6, 12)), color)
        return {
            "image": canvas,
            "count": NumberValue(value=float(count), label="二维码数量", unit="个"),
        }


# ---------------------------------------------------------------------------
# 3. 条码识别
# ---------------------------------------------------------------------------
class DetectBarcode(Operator):
    spec = OperatorSpec(
        id="detect.barcode",
        name="条码识别",
        category=("深度学习", "目标检测"),
        description=(
            "用 OpenCV 的 BarcodeDetector 定位并解码一维条码，画出条码四边形轮廓。"
            "输出标注图与条码数量；未识别时输出原图副本与数量 0。需要 8 位彩色图像，"
            "其它 dtype 会自动转换为 8 位。"
        ),
        keywords=("条码", "条形码", "barcode", "EAN", "解码"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(
            PortSpec("image", "条码标注图", accepts=("BGR",)),
            PortSpec("count", "条码数量", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec("min_size", "最小尺寸", "int", default=0, min=0, max=2000, step=1, unit="px",
                      tooltip="大于 0 时忽略短边小于该值的检测结果。"),
            ParamSpec("color", "标注颜色", "color", default="#00d26a", group="显示"),
        ),
    )

    def process(self, image=None, min_size: int = 0, color: str = "#00d26a", **_: Any):
        bgr = _cv.source_bgr8(image, "输入图像")
        canvas = bgr.copy()
        try:
            decoded, points, _straight = cv2.barcode.BarcodeDetector().detectAndDecode(bgr)
        except cv2.error as exc:
            raise OperatorError("条码识别失败：输入图像格式不受支持。") from exc
        count = 0
        if points is not None and len(points) > 0:
            polygon = np.asarray(points[0], dtype=np.float32).reshape(-1, 2)
            if len(polygon) >= 4:
                width = float(np.linalg.norm(polygon[0] - polygon[1]))
                height = float(np.linalg.norm(polygon[1] - polygon[2]))
                if int(min_size) <= 0 or min(width, height) >= float(min_size):
                    count = 1
                    cv2.polylines(
                        canvas,
                        [polygon.astype(np.int32).reshape(-1, 1, 2)],
                        True,
                        _cv.draw_color(canvas, color),
                        2,
                        _cv.draw_type(canvas),
                    )
                    _draw_label(canvas, decoded or "barcode", (int(polygon[0][0]), max(int(polygon[0][1]) - 6, 12)), color)
        return {
            "image": canvas,
            "count": NumberValue(value=float(count), label="条码数量", unit="个"),
        }


# ---------------------------------------------------------------------------
# 4. HOG 行人检测
# ---------------------------------------------------------------------------
class DetectHogPeople(Operator):
    spec = OperatorSpec(
        id="detect.hog_people",
        name="行人检测（HOG）",
        category=("深度学习", "目标检测"),
        description=(
            "用 HOG + SVM 默认行人检测器扫描图像，输出带行人框的标注图与人数。"
            "检测器要求图像至少 64×128 像素，小图会先放大再检测，检测框会映射回原尺寸，"
            "因此默认参数即可运行。"
        ),
        keywords=("行人", "行人检测", "HOG", "people", "SVM", "检测"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(
            PortSpec("image", "行人标注图", accepts=("BGR",)),
            PortSpec("count", "行人数量", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "scale",
                "扫描缩放步长",
                "float",
                default=1.05,
                min=1.01,
                max=2.0,
                step=0.01,
                tooltip="多尺度扫描的缩放系数，越小越慢但越细致。",
            ),
            ParamSpec("win_stride", "滑动步长", "int", default=8, min=2, max=64, step=2, unit="px"),
            ParamSpec("color", "标注颜色", "color", default="#00d26a", group="显示"),
        ),
    )

    def process(self, image=None, scale: float = 1.05, win_stride: int = 8, color: str = "#00d26a", **_: Any):
        bgr = _cv.source_bgr8(image, "输入图像")
        height, width = bgr.shape[:2]
        canvas = bgr.copy()
        target_width = max(width, HOG_MIN_WIDTH)
        target_height = max(height, HOG_MIN_HEIGHT)
        if (target_width, target_height) == (width, height):
            work = bgr
        else:
            work = cv2.resize(bgr, (target_width, target_height), interpolation=cv2.INTER_LINEAR)
        ratio_x = width / float(target_width)
        ratio_y = height / float(target_height)
        hog = cv2.HOGDescriptor()
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        try:
            rects, _weights = hog.detectMultiScale(
                work,
                winStride=(int(win_stride), int(win_stride)),
                padding=(8, 8),
                scale=float(scale),
            )
        except cv2.error as exc:
            raise OperatorError("行人检测失败：图像尺寸或参数不受支持。") from exc
        boxes: list[tuple[int, int, int, int]] = []
        for rect in rects:
            x, y, rect_w, rect_h = (float(value) for value in rect)
            boxes.append(
                (
                    int(round(x * ratio_x)),
                    int(round(y * ratio_y)),
                    max(1, int(round(rect_w * ratio_x))),
                    max(1, int(round(rect_h * ratio_y))),
                )
            )
        if boxes:
            _draw_rects(canvas, boxes, color)
        return {
            "image": canvas,
            "count": NumberValue(value=float(len(boxes)), label="行人数量", unit="个"),
        }


OPERATORS = (
    DetectFace,
    DetectQrCode,
    DetectBarcode,
    DetectHogPeople,
)
