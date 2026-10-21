"""DNN 推理算子的预置模型 / 下载控件单元测试。

用假网络与假模型文件验证分支，不加载真实 ONNX，也不访问外网。
"""

from __future__ import annotations

import os
import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MINEIMAGE_AUTO_DOWNLOAD_MODELS", "0")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from app.core import models  # noqa: E402
from app.core.types import ImageData  # noqa: E402
from app.operators.builtin.catalog import dnn  # noqa: E402
from app.operators.spec import OperatorError, normalize_params, output_to_dict  # noqa: E402

PRESET = "image_classification_mobilenetv2"


class _FakeNet:
    def __init__(self, output: np.ndarray) -> None:
        self.output = np.asarray(output, dtype=np.float32)
        self.input = None
        self.input_shape = None

    def setInput(self, blob) -> None:  # noqa: N802 - OpenCV 命名
        self.input = blob
        self.input_shape = tuple(blob.shape)

    def forward(self):
        return self.output


class DnnInferPresetTests(unittest.TestCase):
    def setUp(self) -> None:
        base = ROOT / "tests" / ".tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.tmp = base / f"dnn-infer-{uuid.uuid4().hex[:10]}"
        self.tmp.mkdir()
        self.image = ImageData(np.full((32, 32, 3), 64, dtype=np.uint8))
        self.model = self.tmp / "preset.onnx"
        self.model.write_bytes(b"fake-onnx")
        self.net = _FakeNet(np.array([[0.1, 0.9, 0.5, -0.2, 0.3]], dtype=np.float32))

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    def run_operator(self, **overrides):
        spec = dnn.DnnInfer.spec
        params = normalize_params(spec, overrides)
        outputs = output_to_dict(
            spec,
            dnn.DnnInfer().process(image=self.image, **params),
        )
        return spec, params, outputs

    # ------------------------------------------------------------------ 参数声明
    def test_spec_exposes_model_download_widget(self) -> None:
        spec = dnn.DnnInfer.spec
        model_param = spec.param("path")
        self.assertEqual(model_param.widget, "model_file")
        self.assertEqual(model_param.widget_options.get("model"), PRESET)
        self.assertTrue(spec.param("auto_download").default_value())
        self.assertIn("MobileNetV2", spec.description)
        self.assertIn(PRESET, models.MODELS)

    # ------------------------------------------------------------------ 预置模型
    def test_empty_path_uses_and_auto_downloads_preset(self) -> None:
        with mock.patch.dict(os.environ, {"MINEIMAGE_AUTO_DOWNLOAD_MODELS": "1"}):
            with mock.patch.object(dnn.models, "ensure_model", return_value=self.model) as ensure:
                with mock.patch.object(dnn.cv2.dnn, "readNet", return_value=self.net) as read_net:
                    _spec, _params, outputs = self.run_operator(
                        path="",
                        auto_download=True,
                        top_k=3,
                    )
        ensure.assert_called_once()
        self.assertEqual(ensure.call_args.args[0], PRESET)
        self.assertIs(ensure.call_args.kwargs.get("auto"), True)
        read_net.assert_called_once_with(str(self.model))
        self.assertEqual(self.net.input_shape, (1, 3, 224, 224))
        self.assertEqual(outputs["count"].value, 5.0)
        self.assertAlmostEqual(outputs["max_response"].value, 0.9)
        self.assertIsInstance(outputs["image"], ImageData)
        self.assertGreater(outputs["image"].array.size, 0)

    def test_env_switch_disables_auto_download(self) -> None:
        with mock.patch.dict(os.environ, {"MINEIMAGE_AUTO_DOWNLOAD_MODELS": "0"}):
            with mock.patch.object(dnn.models, "ensure_model", return_value=self.model) as ensure:
                with mock.patch.object(dnn.cv2.dnn, "readNet", return_value=self.net):
                    self.run_operator(path="", auto_download=True)
        self.assertIs(ensure.call_args.kwargs.get("auto"), False)

    def test_custom_path_skips_preset_download(self) -> None:
        with mock.patch.object(dnn.models, "ensure_model") as ensure:
            with mock.patch.object(dnn.cv2.dnn, "readNet", return_value=self.net):
                _spec, _params, outputs = self.run_operator(path=str(self.model))
        ensure.assert_not_called()
        self.assertEqual(outputs["count"].value, 5.0)

    def test_missing_preset_reports_download_url(self) -> None:
        missing = models.ModelNotFoundError(
            models.missing_model_message(PRESET)
        )
        with mock.patch.object(dnn.models, "ensure_model", side_effect=missing):
            with self.assertRaises(OperatorError) as context:
                self.run_operator(path="", auto_download=False)
        self.assertIn(models.MODELS[PRESET].url, str(context.exception))

    def test_missing_custom_path_reports_chinese_error(self) -> None:
        missing = self.tmp / "not-found.onnx"
        with self.assertRaises(OperatorError) as context:
            self.run_operator(path=str(missing))
        self.assertIn("模型文件不存在", str(context.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
