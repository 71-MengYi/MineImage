"""OpenCV 目录算子：深度学习 / 检测。

包含 YuNet / Haar 人脸检测、二维码识别、条码识别与 HOG 行人检测 4 个算子。

注意：OpenCV 5.0 已移除 `cv2.CascadeClassifier` 与 `cv2.HOGDescriptor`，
因此人脸检测默认改用 `cv2.FaceDetectorYN` + OpenCV Zoo 的 YuNet ONNX 模型；
模型缺失时可由参数面板下载或由算子在运行前自动下载。

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

from app.core import models
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
# 1. 人脸检测（YuNet / Haar）
# ---------------------------------------------------------------------------
#: OpenCV Zoo YuNet 人脸检测模型在 app.core.models 中的注册 key。
YUNET_MODEL_KEY = "face_detection_yunet"


def _yunet_available() -> bool:
    return callable(getattr(cv2, "FaceDetectorYN", None))


def _haar_available() -> bool:
    return callable(getattr(cv2, "CascadeClassifier", None))


def face_detection_backend() -> str:
    """返回当前 OpenCV 可用的人脸检测后端：'yunet' / 'haar' / 空字符串。"""

    if _yunet_available():
        return "yunet"
    if _haar_available():
        return "haar"
    return ""


def face_detection_available() -> bool:
    """当前环境能否直接用于人脸检测（YuNet 模型已就绪，或 OpenCV 4 带 Haar）。"""

    if _yunet_available() and models.is_model_ready(YUNET_MODEL_KEY):
        return True
    return _haar_available()


def _face_rows(faces: Any, columns: int) -> np.ndarray:
    """把检测器返回的二维结果整理为 N×columns 的 float32 数组。"""

    if faces is None:
        return np.empty((0, columns), dtype=np.float32)
    array = np.asarray(faces, dtype=np.float32)
    if array.size == 0:
        return np.empty((0, columns), dtype=np.float32)
    if array.ndim == 1:
        array = array.reshape(1, -1)
    if array.ndim != 2 or array.shape[1] < columns:
        raise OperatorError(f"人脸检测结果格式异常：{array.shape}")
    return array[:, :columns]


def _draw_landmarks(canvas: np.ndarray, landmarks: np.ndarray, color: str) -> None:
    """在画布上绘制 YuNet 的 5 个五官关键点。"""

    if landmarks.size == 0:
        return
    draw_color = _cv.draw_color(canvas, color)
    line_type = _cv.draw_type(canvas)
    for row in landmarks:
        for index in range(0, min(10, row.size), 2):
            x = int(round(float(row[index])))
            y = int(round(float(row[index + 1])))
            if x > 0 and y > 0:
                cv2.circle(canvas, (x, y), 2, draw_color, -1, line_type)


class DetectFace(Operator):
    spec = OperatorSpec(
        id="detect.face",
        name="人脸检测（YuNet / Haar）",
        category=("深度学习", "目标检测"),
        description=(
            "用 OpenCV Zoo 的 YuNet ONNX 模型检测正面人脸，并在画面上画出人脸框与五官关键点。"
            "OpenCV 5.0 已移除 Haar 级联，因此不再依赖 cv2.CascadeClassifier；"
            "模型缺失时可从参数面板一键下载，或在运行前自动下载（默认开启）。"
            "OpenCV 4 环境仍可自动回退到 Haar 级联。输出标注图与检测到的人脸数量。"
        ),
        keywords=("人脸", "人脸检测", "YuNet", "ONNX", "Haar", "face", "cascade", "检测", "模型下载"),
        inputs=(PortSpec("image", "输入图像"),),
        outputs=(
            PortSpec("image", "人脸标注图", accepts=("BGR",)),
            PortSpec("count", "人脸数量", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "detector",
                "检测器",
                "enum",
                default="auto",
                choices=(
                    ("auto", "自动（优先 YuNet）"),
                    ("yunet", "YuNet（ONNX）"),
                    ("haar", "Haar 级联（OpenCV 4）"),
                ),
                group="模型",
                tooltip="自动模式优先使用 YuNet；OpenCV 4 缺少模型时可回退到 Haar。",
            ),
            ParamSpec(
                "path",
                "模型文件",
                "path",
                default="",
                placeholder="留空使用模型目录中的 YuNet 模型",
                group="模型",
                tooltip=(
                    "自定义 YuNet ONNX（或 Haar XML）路径；留空时自动使用模型目录中的 "
                    "face_detection_yunet_2023mar.onnx。"
                ),
                widget="model_file",
                widget_options={"model": YUNET_MODEL_KEY},
            ),
            ParamSpec(
                "auto_download",
                "自动下载模型",
                "bool",
                default=True,
                group="模型",
                tooltip="模型缺失时自动从 OpenCV Zoo 下载 YuNet ONNX（约 230 KB）。",
            ),
            ParamSpec(
                "score_threshold",
                "置信度阈值",
                "float",
                default=0.9,
                min=0.0,
                max=1.0,
                step=0.01,
                group="YuNet",
                visible_when="detector != 'haar'",
                tooltip="YuNet 人脸置信度阈值，越高越严格。",
            ),
            ParamSpec(
                "nms_threshold",
                "NMS 阈值",
                "float",
                default=0.3,
                min=0.0,
                max=1.0,
                step=0.01,
                group="YuNet",
                visible_when="detector != 'haar'",
                tooltip="YuNet 非极大值抑制的 IoU 阈值。",
            ),
            ParamSpec(
                "top_k",
                "候选框上限",
                "int",
                default=5000,
                min=1,
                max=100000,
                step=100,
                group="YuNet",
                visible_when="detector != 'haar'",
                tooltip="NMS 之前保留的候选框数量。",
            ),
            ParamSpec(
                "draw_landmarks",
                "绘制五官关键点",
                "bool",
                default=True,
                group="显示",
                tooltip="仅 YuNet 输出五官关键点；Haar 不支持。",
            ),
            ParamSpec(
                "scale_factor",
                "缩放步长（Haar）",
                "float",
                default=1.1,
                min=1.01,
                max=3.0,
                step=0.01,
                group="Haar",
                visible_when="detector != 'yunet'",
            ),
            ParamSpec(
                "min_neighbors",
                "最小邻居数（Haar）",
                "int",
                default=3,
                min=0,
                max=50,
                step=1,
                group="Haar",
                visible_when="detector != 'yunet'",
            ),
            ParamSpec(
                "min_size",
                "最小人脸尺寸（Haar）",
                "int",
                default=20,
                min=4,
                max=2000,
                step=1,
                unit="px",
                group="Haar",
                visible_when="detector != 'yunet'",
            ),
            ParamSpec("color", "标注颜色", "color", default="#00d26a", group="显示"),
        ),
    )

    # ------------------------------------------------------------------ 资源
    @staticmethod
    def _backend_hint() -> str:
        return (
            "当前 OpenCV 版本没有可用的 Haar 级联（OpenCV 5 已移除 "
            "cv2.CascadeClassifier），人脸检测需要 YuNet ONNX 模型："
            f"{models.download_url(YUNET_MODEL_KEY)}"
            "。可在参数面板点击「下载模型」，或手动下载后选择该文件。"
        )

    def _select_backend(self, detector: str, path: str) -> str:
        mode = str(detector or "auto").strip().lower()
        suffix = os.path.splitext(str(path or "").strip())[1].lower()
        if mode == "haar":
            if not _haar_available():
                raise OperatorError(
                    "当前 OpenCV 版本不支持 Haar 级联（cv2.CascadeClassifier 不存在），"
                    "请把「检测器」改为 YuNet 或自动。" + self._backend_hint()
                )
            return "haar"
        if mode == "yunet":
            if not _yunet_available():
                raise OperatorError(
                    "当前 OpenCV 版本不支持 YuNet（cv2.FaceDetectorYN 不存在），"
                    "请升级 opencv-python。" + self._backend_hint()
                )
            return "yunet"
        # 自动模式：按模型后缀 / 当前 OpenCV 能力选择。
        if suffix == ".onnx":
            if not _yunet_available():
                raise OperatorError("所选模型是 ONNX，但当前 OpenCV 不支持 YuNet。" + self._backend_hint())
            return "yunet"
        if suffix == ".xml":
            if not _haar_available():
                raise OperatorError("所选模型是 Haar XML，但当前 OpenCV 已移除 Haar 级联。" + self._backend_hint())
            return "haar"
        if _yunet_available():
            return "yunet"
        if _haar_available():
            return "haar"
        raise OperatorError(self._backend_hint())

    def _cascade_path(self, path: str) -> str:
        custom = str(path or "").strip()
        if custom:
            if custom.lower().endswith(".onnx"):
                raise OperatorError("Haar 检测器需要 .xml 级联文件；请把「检测器」设为 YuNet 或自动。")
            if not os.path.isfile(custom):
                raise OperatorError(f"人脸检测模型文件不存在：{custom}")
            return custom
        data = getattr(cv2, "data", None)
        folder = str(getattr(data, "haarcascades", "") or "") if data is not None else ""
        default = os.path.join(folder, "haarcascade_frontalface_default.xml") if folder else ""
        if not default or not os.path.isfile(default):
            raise OperatorError(self._backend_hint())
        return default

    def _resolve_yunet_model(self, path: str, auto_download: bool) -> str:
        custom = str(path or "").strip()
        if custom:
            if custom.lower().endswith(".xml"):
                raise OperatorError("YuNet 检测器需要 .onnx 模型；请把「检测器」设为 Haar 或自动。")
            if not os.path.isfile(custom):
                raise OperatorError(f"人脸检测模型文件不存在：{custom}")
            return custom
        auto = bool(auto_download) and models.auto_download_enabled()
        try:
            return str(models.ensure_model(YUNET_MODEL_KEY, auto=auto))
        except models.ModelError as exc:
            raise OperatorError(str(exc)) from exc

    # ------------------------------------------------------------------ 检测
    @staticmethod
    def _detect_yunet(
        bgr: np.ndarray,
        model_path: str,
        score_threshold: float,
        nms_threshold: float,
        top_k: int,
    ) -> tuple[np.ndarray, list[float], np.ndarray]:
        height, width = bgr.shape[:2]
        try:
            detector = cv2.FaceDetectorYN.create(
                model_path,
                "",
                (int(width), int(height)),
                float(score_threshold),
                float(nms_threshold),
                int(top_k),
            )
            detector.setInputSize((int(width), int(height)))
            _retval, faces = detector.detect(bgr)
        except (AttributeError, cv2.error) as exc:
            raise OperatorError(
                f"YuNet 模型加载/推理失败：{model_path}。"
                "请确认文件是 OpenCV Zoo 的 YuNet ONNX 模型，并检查 OpenCV 版本。"
            ) from exc
        rows = _face_rows(faces, 15)
        if rows.shape[0] == 0:
            return np.empty((0, 4), dtype=np.float32), [], np.empty((0, 10), dtype=np.float32)
        boxes = rows[:, :4].copy()
        scores = [float(value) for value in rows[:, 14]]
        landmarks = rows[:, 4:14].copy()
        return boxes, scores, landmarks

    def _detect_haar(
        self,
        bgr: np.ndarray,
        path: str,
        scale_factor: float,
        min_neighbors: int,
        min_size: int,
    ) -> np.ndarray:
        if not _haar_available():
            raise OperatorError(self._backend_hint())
        cascade_path = self._cascade_path(path)
        try:
            cascade = cv2.CascadeClassifier(cascade_path)
        except (AttributeError, cv2.error) as exc:
            raise OperatorError(f"Haar 级联模型加载失败：{cascade_path}") from exc
        if cascade.empty():
            raise OperatorError(f"无法加载 Haar 人脸模型：{cascade_path}。请检查 XML 文件是否完整。")
        gray = _cv.source_gray8(bgr, "输入图像")
        try:
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=float(scale_factor),
                minNeighbors=int(min_neighbors),
                minSize=(int(min_size), int(min_size)),
            )
        except cv2.error as exc:
            raise OperatorError("Haar 人脸检测失败：图像尺寸或参数不受支持。") from exc
        return _face_rows(faces, 4)

    # ------------------------------------------------------------------ 执行
    def process(
        self,
        image=None,
        detector: str = "auto",
        path: str = "",
        auto_download: bool = True,
        score_threshold: float = 0.9,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
        draw_landmarks: bool = True,
        scale_factor: float = 1.1,
        min_neighbors: int = 3,
        min_size: int = 20,
        color: str = "#00d26a",
        **_: Any,
    ):
        bgr = _cv.source_bgr8(image, "输入图像")
        canvas = bgr.copy()
        backend = self._select_backend(detector, path)
        if backend == "yunet":
            try:
                model_path = self._resolve_yunet_model(path, auto_download)
            except OperatorError:
                # 自动模式 + OpenCV 4：YuNet 模型不可用时回退 Haar，保证默认参数可运行。
                if str(detector or "auto").strip().lower() == "auto" and _haar_available():
                    backend = "haar"
                else:
                    raise
        if backend == "yunet":
            boxes, scores, landmarks = self._detect_yunet(
                bgr,
                model_path,
                score_threshold,
                nms_threshold,
                top_k,
            )
        else:
            boxes = self._detect_haar(bgr, path, scale_factor, min_neighbors, min_size)
            scores = []
            landmarks = np.empty((0, 10), dtype=np.float32)
        if len(boxes) > 0:
            _draw_rects(canvas, boxes, color)
            for box, score in zip(boxes, scores):
                _draw_label(
                    canvas,
                    f"{float(score):.2f}",
                    (int(box[0]), max(int(box[1]) - 6, 10)),
                    color,
                )
            if draw_landmarks and landmarks.size > 0:
                _draw_landmarks(canvas, landmarks, color)
        return {
            "image": canvas,
            "count": NumberValue(value=float(len(boxes)), label="人脸数量", unit="个"),
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
            "检测器要求图像至少 64×128 像素，小图会先放大再检测，检测框会映射回原尺寸。"
            "注意：OpenCV 5.0 已移除 cv2.HOGDescriptor，本算子仅在 OpenCV 4.x 可用；"
            "在 OpenCV 5 中运行会给出明确提示。"
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
        if not hasattr(cv2, "HOGDescriptor") or not hasattr(cv2, "HOGDescriptor_getDefaultPeopleDetector"):
            raise OperatorError(
                "当前 OpenCV 版本已移除 HOGDescriptor（OpenCV 5.0 移除了传统 HOG 行人检测），"
                "无法运行 HOG 行人检测；请改用 OpenCV 4.x，或使用其它目标检测算子。"
            )
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
