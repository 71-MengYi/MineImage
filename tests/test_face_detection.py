"""人脸检测算子的 YuNet / Haar / 缺失后端分支测试。

用假的 cv2.FaceDetectorYN / cv2.CascadeClassifier 与临时模型文件验证分支逻辑，
不依赖真实 ONNX 模型，也不访问外网。
"""

from __future__ import annotations

import os
import shutil
import sys
import types
import unittest
import uuid
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MINEIMAGE_AUTO_DOWNLOAD_MODELS", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from app.core import models  # noqa: E402
from app.core.types import ImageData  # noqa: E402
from app.operators.builtin.catalog import detection  # noqa: E402
from app.operators.spec import OperatorError, normalize_params, output_to_dict  # noqa: E402

FACE_ROW = np.array(
    [[4.0, 5.0, 20.0, 20.0, 7.0, 7.0, 12.0, 7.0, 9.0, 13.0, 6.0, 19.0, 16.0, 19.0, 0.98]],
    dtype=np.float32,
)


class _FakeDetector:
    def __init__(self, faces: np.ndarray) -> None:
        self.faces = faces
        self.input_size: tuple[int, int] | None = None

    def setInputSize(self, size) -> None:  # noqa: N802 - OpenCV 命名
        self.input_size = tuple(int(value) for value in size)

    def detect(self, _image):
        return 1, self.faces.copy()


class _FakeFaceDetectorYN:
    faces = FACE_ROW
    last_model: str = ""

    @classmethod
    def create(cls, model, _config, input_size, _score, _nms, _top_k):
        cls.last_model = str(model)
        return _FakeDetector(cls.faces)


class _FakeCascade:
    last_path: str = ""

    def __init__(self, path: str) -> None:
        _FakeCascade.last_path = str(path)

    def empty(self) -> bool:
        return False

    def detectMultiScale(self, _gray, scaleFactor, minNeighbors, minSize):  # noqa: N802
        return np.array([[2.0, 3.0, 12.0, 12.0]], dtype=np.float32)


class FaceDetectionTests(unittest.TestCase):
    def setUp(self) -> None:
        base = ROOT / "tests" / ".tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.tmp = base / f"face-{uuid.uuid4().hex[:10]}"
        self.tmp.mkdir()
        self.image = ImageData(np.full((48, 64, 3), 30, dtype=np.uint8))
        self.onnx = self.tmp / "yunet.onnx"
        self.onnx.write_bytes(b"fake-onnx")
        self.xml = self.tmp / "haarcascade_frontalface_default.xml"
        self.xml.write_bytes(b"<opencv_storage/>")

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_face(self, **overrides):
        spec = detection.DetectFace.spec
        params = normalize_params(spec, overrides)
        result = output_to_dict(
            spec,
            detection.DetectFace().process(image=self.image, **params),
        )
        return spec, result

    # ------------------------------------------------------------------ 元数据
    def test_face_operator_uses_model_download_widget(self) -> None:
        spec = detection.DetectFace.spec
        model_param = spec.param("path")
        self.assertEqual(model_param.widget, "model_file")
        self.assertEqual(model_param.widget_options.get("model"), "face_detection_yunet")
        self.assertEqual(spec.param("detector").default_value(), "auto")
        self.assertIn("YuNet", spec.description)

    # ------------------------------------------------------------------ YuNet
    def test_yunet_path_detects_and_draws(self) -> None:
        with mock.patch.object(detection.cv2, "FaceDetectorYN", _FakeFaceDetectorYN, create=True):
            spec, result = self.run_face(
                detector="yunet",
                path=str(self.onnx),
                auto_download=False,
            )
        self.assertEqual(_FakeFaceDetectorYN.last_model, str(self.onnx))
        self.assertEqual(result["count"].value, 1.0)
        output = result["image"]
        self.assertIsInstance(output, ImageData)
        self.assertFalse(np.array_equal(output.array, self.image.array), "应画上人脸框/关键点")
        self.assertEqual(output.array.shape, self.image.array.shape)

    def test_auto_mode_prefers_yunet_when_model_ready(self) -> None:
        with mock.patch.dict(os.environ, {"MINEIMAGE_MODEL_DIR": str(self.tmp)}):
            # 假模型只需通过 models.is_model_ready() 的最小尺寸校验；
            # _FakeFaceDetectorYN 不会真的解析 ONNX 内容。
            cached = self.tmp / models.MODELS["face_detection_yunet"].filename
            cached.write_bytes(b"fake-onnx" + b"x" * 120_000)
            with mock.patch.object(detection.cv2, "FaceDetectorYN", _FakeFaceDetectorYN, create=True):
                _spec, result = self.run_face(auto_download=False)
        self.assertEqual(result["count"].value, 1.0)

    # ------------------------------------------------------------------ Haar 回退
    def test_auto_mode_falls_back_to_haar(self) -> None:
        with mock.patch.object(detection.cv2, "FaceDetectorYN", None):
            with mock.patch.object(detection.cv2, "CascadeClassifier", _FakeCascade, create=True):
                with mock.patch.object(detection.cv2, "data", types.SimpleNamespace(haarcascades=str(self.tmp))):
                    _spec, result = self.run_face(auto_download=False)
        self.assertEqual(result["count"].value, 1.0)
        self.assertEqual(_FakeCascade.last_path, str(self.xml))

    # ------------------------------------------------------------------ 错误提示
    def test_missing_all_backends_reports_download_url(self) -> None:
        with mock.patch.object(detection.cv2, "FaceDetectorYN", None):
            with mock.patch.object(detection.cv2, "CascadeClassifier", None, create=True):
                with self.assertRaises(OperatorError) as context:
                    self.run_face(auto_download=False)
        message = str(context.exception)
        self.assertIn(models.MODELS["face_detection_yunet"].url, message)

    def test_missing_model_without_download_reports_url(self) -> None:
        with mock.patch.object(detection.cv2, "FaceDetectorYN", _FakeFaceDetectorYN, create=True):
            with mock.patch.object(detection.cv2, "CascadeClassifier", None, create=True):
                with mock.patch.object(
                    detection.models,
                    "ensure_model",
                    side_effect=models.ModelNotFoundError(
                        models.missing_model_message("face_detection_yunet")
                    ),
                ):
                    with self.assertRaises(OperatorError) as context:
                        self.run_face(detector="yunet", path="", auto_download=False)
        self.assertIn(models.MODELS["face_detection_yunet"].url, str(context.exception))

    def test_available_helper_without_model_is_false_on_opencv5(self) -> None:
        if hasattr(cv2, "CascadeClassifier"):
            self.skipTest("OpenCV 4 带 Haar，天然可用")
        with mock.patch.dict(os.environ, {"MINEIMAGE_MODEL_DIR": str(self.tmp / "empty")}):
            self.assertFalse(detection.face_detection_available())
            self.assertEqual(detection.face_detection_backend(), "yunet")


if __name__ == "__main__":
    unittest.main(verbosity=2)
