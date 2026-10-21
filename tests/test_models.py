"""OpenCV ONNX 模型下载基础设施的单元测试。

测试使用本机 127.0.0.1 的临时 HTTP 服务模拟 OpenCV Zoo，不访问外网；
重点验证：下载进度、原子落盘、失败清理、自动 / 手动两种模式与官方地址。
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
import unittest
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import models  # noqa: E402

#: 假模型内容（足够大，能通过默认的最小尺寸校验）。
FAKE_MODEL = b"ONNX-FAKE-MODEL\x00" + bytes(range(256)) * 16
HTML_PAGE = b"<!DOCTYPE html><html><head><title>Not Found</title></head><body>404</body></html>" + b" " * 2048


class _Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802 - http.server 接口
        routes = getattr(self.server, "routes", {})
        hits = getattr(self.server, "hits", {})
        hits[self.path] = hits.get(self.path, 0) + 1
        payload = routes.get(self.path)
        if payload is None:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *_args: object) -> None:
        # 测试中不打印访问日志。
        return


class ModelDownloadTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.server = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
        cls.server.routes = {}
        cls.server.hits = {}
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base_url = f"http://127.0.0.1:{cls.server.server_port}"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.server.shutdown()
        cls.server.server_close()
        cls.thread.join(timeout=5)

    def setUp(self) -> None:
        # 使用工作区内目录：本机沙箱可能拒绝写系统临时目录，也便于失败后清理。
        base = ROOT / "tests" / ".tmp"
        base.mkdir(parents=True, exist_ok=True)
        self.tmp = base / f"model-download-{uuid.uuid4().hex[:10]}"
        self.tmp.mkdir()
        self.server.routes = {}
        self.server.hits = {}

    def tearDown(self) -> None:
        shutil.rmtree(self.tmp, ignore_errors=True)

    # ------------------------------------------------------------------ 工具
    def make_spec(self, path: str = "/fake.onnx", **overrides) -> models.ModelSpec:
        data = overrides.pop("data", FAKE_MODEL)
        self.server.routes[path] = data
        values = {
            "key": "test_model",
            "name": "测试模型",
            "filename": "test_model.onnx",
            "url": self.base_url + path,
            "min_size": 1024,
            "size": len(data),
        }
        values.update(overrides)
        return models.ModelSpec(**values)

    def part_files(self) -> list[Path]:
        return list(self.tmp.glob("*.part"))

    # ------------------------------------------------------------------ 官方清单
    def test_builtin_yunet_points_to_opencv_zoo(self) -> None:
        spec = models.MODELS["face_detection_yunet"]
        self.assertEqual(spec.filename, "face_detection_yunet_2023mar.onnx")
        self.assertTrue(
            spec.url.startswith(
                "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
            ),
            spec.url,
        )
        self.assertIn("opencv_zoo", spec.homepage)
        self.assertEqual(spec.size, 232_589)
        self.assertEqual(models.model_url(spec), spec.url)
        self.assertTrue(
            any("media.githubusercontent.com" in url for url in spec.urls),
            f"应保留 GitHub LFS 备用地址：{spec.urls}",
        )

    def test_builtin_mobilenet_is_available_for_dnn_infer(self) -> None:
        spec = models.MODELS["image_classification_mobilenetv2"]
        self.assertEqual(spec.filename, "image_classification_mobilenetv2_2022apr.onnx")
        self.assertTrue(
            spec.url.startswith(
                "https://github.com/opencv/opencv_zoo/raw/main/"
                "models/image_classification_mobilenet/"
            ),
            spec.url,
        )
        self.assertIn("image_classification_mobilenet", spec.homepage)
        self.assertGreaterEqual(spec.size, 10_000_000)
        self.assertGreaterEqual(spec.min_size, 1_000_000)
        self.assertTrue(
            any("media.githubusercontent.com" in url for url in spec.urls),
            f"应保留 GitHub LFS 备用地址：{spec.urls}",
        )

    def test_unknown_model_raises(self) -> None:
        with self.assertRaises(models.ModelError):
            models.resolve_model("not-a-real-model")

    def test_model_dir_can_be_overridden_by_env(self) -> None:
        with mock.patch.dict(os.environ, {"MINEIMAGE_MODEL_DIR": str(self.tmp)}):
            self.assertEqual(models.model_dir(), self.tmp)
            self.assertEqual(models.model_path("face_detection_yunet").parent, self.tmp)

    # ------------------------------------------------------------------ 下载
    def test_download_writes_file_and_reports_progress(self) -> None:
        spec = self.make_spec()
        target = self.tmp / "downloaded.onnx"
        seen: list[tuple[int, int]] = []
        result = models.download_model(
            spec,
            target=target,
            progress=lambda done, total: seen.append((done, total)),
        )
        self.assertEqual(result, target)
        self.assertEqual(target.read_bytes(), FAKE_MODEL)
        self.assertGreaterEqual(seen[-1][0], len(FAKE_MODEL))
        self.assertEqual(seen[-1][1], len(FAKE_MODEL))
        self.assertEqual(self.part_files(), [])

    def test_download_skips_existing_file_unless_forced(self) -> None:
        spec = self.make_spec()
        target = self.tmp / "cached.onnx"
        target.write_bytes(FAKE_MODEL)
        models.download_model(spec, target=target)
        self.assertNotIn("/fake.onnx", self.server.hits)

        models.download_model(spec, target=target, force=True)
        self.assertGreaterEqual(self.server.hits.get("/fake.onnx", 0), 1)
        self.assertEqual(target.read_bytes(), FAKE_MODEL)

    def test_download_failure_leaves_no_partial_file(self) -> None:
        spec = models.ModelSpec(
            key="missing",
            name="不存在的模型",
            filename="missing.onnx",
            url=self.base_url + "/missing.onnx",
            min_size=1024,
        )
        target = self.tmp / "missing.onnx"
        with self.assertRaises(models.ModelDownloadError) as context:
            models.download_model(spec, target=target)
        self.assertFalse(target.exists())
        self.assertEqual(self.part_files(), [])
        self.assertIn(spec.url, str(context.exception))

    def test_download_rejects_html_page(self) -> None:
        spec = self.make_spec(path="/html.onnx", data=HTML_PAGE)
        target = self.tmp / "html.onnx"
        with self.assertRaises(models.ModelDownloadError):
            models.download_model(spec, target=target)
        self.assertFalse(target.exists())
        self.assertEqual(self.part_files(), [])

    def test_download_can_be_cancelled(self) -> None:
        big = FAKE_MODEL + b"x" * (2 * 1024 * 1024)
        spec = self.make_spec(path="/big.onnx", data=big, min_size=1024)
        target = self.tmp / "big.onnx"
        flags: list[bool] = []

        def progress(done: int, _total: int) -> None:
            flags.append(True)
            if done > 0:
                cancel[0] = True

        cancel = [False]
        with self.assertRaises(models.ModelDownloadError):
            models.download_model(
                spec,
                target=target,
                progress=progress,
                cancel=lambda: cancel[0],
            )
        self.assertTrue(flags, "应至少报告一次进度")
        self.assertFalse(target.exists())
        self.assertEqual(self.part_files(), [])

    # ------------------------------------------------------------------ ensure
    def test_ensure_auto_false_raises_with_manual_url(self) -> None:
        spec = self.make_spec()
        with mock.patch.dict(os.environ, {"MINEIMAGE_MODEL_DIR": str(self.tmp)}):
            with self.assertRaises(models.ModelNotFoundError) as context:
                models.ensure_model(spec, auto=False)
        message = str(context.exception)
        self.assertIn(spec.url, message)
        self.assertIn("下载地址", message)

    def test_ensure_auto_true_downloads_and_caches(self) -> None:
        spec = self.make_spec()
        with mock.patch.dict(os.environ, {"MINEIMAGE_MODEL_DIR": str(self.tmp)}):
            path = models.ensure_model(spec, auto=True)
            self.assertEqual(path, self.tmp / spec.filename)
            self.assertEqual(path.read_bytes(), FAKE_MODEL)
            self.assertTrue(models.is_model_ready(spec))
            # 第二次调用命中缓存，不再发起网络请求。
            hits_before = self.server.hits.get("/fake.onnx", 0)
            models.ensure_model(spec, auto=True)
            self.assertEqual(self.server.hits.get("/fake.onnx", 0), hits_before)

    def test_auto_download_env_switch(self) -> None:
        with mock.patch.dict(os.environ, {"MINEIMAGE_AUTO_DOWNLOAD_MODELS": "0"}):
            self.assertFalse(models.auto_download_enabled())
        with mock.patch.dict(os.environ, {"MINEIMAGE_AUTO_DOWNLOAD_MODELS": "1"}):
            self.assertTrue(models.auto_download_enabled())

    def test_status_text_and_size_format(self) -> None:
        spec = self.make_spec()
        with mock.patch.dict(os.environ, {"MINEIMAGE_MODEL_DIR": str(self.tmp)}):
            self.assertEqual(models.model_status_text(spec), "未下载")
            target = self.tmp / spec.filename
            target.write_bytes(FAKE_MODEL)
            self.assertIn("已就绪", models.model_status_text(spec))
        self.assertEqual(models.format_size(0), "0 B")
        self.assertTrue(models.format_size(2048).endswith("KB"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
