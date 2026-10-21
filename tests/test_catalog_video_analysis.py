"""OpenCV 目录算子 E（视频 / 检测 / 标定 / 分析）的单元测试。

覆盖文件：
``video_motion.py`` / ``detection.py`` / ``calib3d.py`` / ``dnn.py`` / ``analysis.py``。

测试只依赖 ``OperatorRegistry.build()`` 与公开的 spec 工具，不启动 GUI；
遍历 7 种样例数组（L/BGR/BGRA × uint8/uint16/float32）验证默认参数可执行、
输出非空且不修改输入；再对 PSNR / SSIM / FFT / 立体视差 / 模板匹配做正确性断言。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# 测试环境不自动下载模型；缺少 YuNet 模型时跳过 detect.face 的默认执行。
os.environ.setdefault("MINEIMAGE_AUTO_DOWNLOAD_MODELS", "0")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import models  # noqa: E402
from app.core.data_types import DataType, is_data_value  # noqa: E402
from app.core.types import ImageData  # noqa: E402
from app.operators.loader import OperatorRegistry  # noqa: E402
from app.operators.spec import (  # noqa: E402
    OperatorError,
    normalize_params,
    output_to_dict,
    validate_inputs,
)

#: 本组算子的完整 id 清单（含允许默认失败的 dnn.infer）。
GROUP_ALL_IDS = (
    "video.frame_diff",
    "video.optical_flow_farneback",
    "video.optical_flow_lk",
    "video.background_mog2",
    "video.background_knn",
    "video.mean_shift",
    "video.camshift",
    "video.ecc_align",
    "video.tracker",
    "video.motion_energy",
    "detect.face",
    "detect.qr",
    "detect.barcode",
    "detect.hog_people",
    "calib.chessboard",
    "calib.undistort",
    "calib.stereo_disparity",
    "calib.stereo_depth",
    "calib.panorama",
    "calib.homography",
    "dnn.blob",
    "dnn.infer",
    "analysis.integral",
    "analysis.min_max",
    "analysis.mean_std",
    "analysis.moments",
    "analysis.hu_moments",
    "analysis.fft",
    "analysis.ifft",
    "analysis.phase_correlate",
    "analysis.template_match",
    "analysis.psnr",
    "analysis.ssim",
    "analysis.match_shapes",
)

#: 默认参数必须成功的算子（dnn.infer 需要模型文件，单独测试）。
RUNNABLE_IDS = tuple(op_id for op_id in GROUP_ALL_IDS if op_id != "dnn.infer")

#: 本组允许出现的一级分类。
ALLOWED_TABS = {"视频与运动", "深度学习", "立体与标定", "特征与分析"}

#: 本组文件，用于过滤注册表加载错误。
GROUP_FILES = ("video_motion.py", "detection.py", "calib3d.py", "dnn.py", "analysis.py")

CJK_START, CJK_END = "\u4e00", "\u9fff"


def contains_cjk(text: str) -> bool:
    return any(CJK_START <= char <= CJK_END for char in str(text))


def face_detection_runnable() -> bool:
    """OpenCV 4 带 Haar，或 OpenCV 5 已缓存 YuNet 模型时可运行 detect.face。"""

    if hasattr(cv2, "CascadeClassifier"):
        return True
    return hasattr(cv2, "FaceDetectorYN") and models.is_model_ready("face_detection_yunet")


def hog_people_runnable() -> bool:
    """OpenCV 5.0 移除了 cv2.HOGDescriptor，HOG 行人检测仅在 4.x 可运行。"""

    return hasattr(cv2, "HOGDescriptor") and hasattr(cv2, "HOGDescriptor_getDefaultPeopleDetector")


# ---------------------------------------------------------------------------
# 样例数组（只在本文件内复用，不导入 tests 下的其他文件）
# ---------------------------------------------------------------------------
def bgr8(height: int = 24, width: int = 32) -> np.ndarray:
    rng = np.random.default_rng(11)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


def gray8(height: int = 24, width: int = 32) -> np.ndarray:
    return bgr8(height, width)[:, :, 1].copy()


def bgra8() -> np.ndarray:
    array = bgr8()
    return np.dstack([array, np.full(array.shape[:2], 200, dtype=np.uint8)])


def bgr16() -> np.ndarray:
    return (bgr8().astype(np.uint16) * 257).astype(np.uint16)


def gray16() -> np.ndarray:
    return (gray8().astype(np.uint16) * 257).astype(np.uint16)


def bgrf() -> np.ndarray:
    return (bgr8().astype(np.float32) / 255.0).astype(np.float32)


def grayf() -> np.ndarray:
    return (gray8().astype(np.float32) / 255.0).astype(np.float32)


def sample_arrays() -> dict[str, np.ndarray]:
    return {
        "gray8": gray8(),
        "bgr8": bgr8(),
        "bgra8": bgra8(),
        "gray16": gray16(),
        "bgr16": bgr16(),
        "grayf": grayf(),
        "bgrf": bgrf(),
    }


class CatalogVideoAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = OperatorRegistry.build()

    # ------------------------------------------------------------------ 工具
    def run_operator(self, op_id: str, inputs: dict | None = None, params: dict | None = None):
        entry = self.registry.entry(op_id)
        self.assertIsNotNone(entry, f"缺少算子 {op_id}")
        normalized = normalize_params(entry.spec, params or {})
        resolved = {}
        for port in entry.spec.inputs:
            raw = (inputs or {}).get(port.key)
            resolved[port.key] = raw if raw is None or is_data_value(raw) else ImageData(raw)
        validate_inputs(entry.spec, resolved)
        entry.operator.validate(resolved, normalized)
        kwargs = dict(resolved)
        kwargs.update(normalized)
        return output_to_dict(entry.spec, entry.operator.process(**kwargs))

    def first_image(self, op_id: str, outputs: dict) -> ImageData:
        spec = self.registry.spec(op_id)
        self.assertIsNotNone(spec)
        key = spec.outputs[0].key
        self.assertIn(key, outputs, f"{op_id} 缺少第一个输出 {key}")
        image = outputs[key]
        self.assertIsInstance(image, ImageData, f"{op_id} 第一个输出不是图像")
        return image

    # ------------------------------------------------------------------ 注册表
    def test_group_operators_registered(self) -> None:
        missing = [op_id for op_id in GROUP_ALL_IDS if self.registry.entry(op_id) is None]
        self.assertEqual(missing, [], f"缺少本组算子：{missing}")
        group_errors = [
            issue.display() for issue in self.registry.errors if issue.path.name in GROUP_FILES
        ]
        self.assertEqual(group_errors, [], f"本组算子文件加载报错：{group_errors}")

        for op_id in GROUP_ALL_IDS:
            entry = self.registry.entry(op_id)
            self.assertTrue(entry.builtin, f"{op_id} 应为内置算子")
            self.assertIn(entry.spec.category[0], ALLOWED_TABS, f"{op_id} 一级分类不合法")
            self.assertTrue(contains_cjk(entry.spec.name), f"{op_id} 名称应为中文")
            self.assertTrue(contains_cjk(entry.spec.description), f"{op_id} 描述应为中文")
            self.assertEqual(entry.spec.inputs[0].key, "image", f"{op_id} 第一个输入必须是 image")
            self.assertEqual(
                entry.spec.outputs[0].data_type,
                DataType.IMAGE,
                f"{op_id} 第一个输出必须是图像",
            )
            for param in entry.spec.params:
                self.assertTrue(param.label, f"{op_id}.{param.key} 缺少中文参数标签")
                if param.visible_when:
                    self.assertNotIn("params.", param.visible_when, f"{op_id}.{param.key} 用了属性访问")
                if param.enabled_when:
                    self.assertNotIn("params.", param.enabled_when, f"{op_id}.{param.key} 用了属性访问")

    def test_dnn_infer_default_requires_model(self) -> None:
        with self.assertRaises(OperatorError) as context:
            self.run_operator("dnn.infer", {"image": gray8()})
        message = str(context.exception)
        self.assertTrue(message, "dnn.infer 未给出提示")
        self.assertTrue(contains_cjk(message), f"dnn.infer 的提示应为中文：{message}")

    # ------------------------------------------------------------------ 默认参数矩阵
    @unittest.skipIf(face_detection_runnable(), "当前环境已有 Haar 或 YuNet 模型")
    def test_face_without_model_reports_download_url(self) -> None:
        with self.assertRaises(OperatorError) as context:
            self.run_operator("detect.face", {"image": gray8()})
        message = str(context.exception)
        self.assertIn("opencv_zoo", message)
        self.assertTrue(contains_cjk(message), message)

    def test_all_group_operators_run_and_do_not_mutate_inputs(self) -> None:
        arrays = sample_arrays()
        for op_id in RUNNABLE_IDS:
            if op_id == "detect.face" and not face_detection_runnable():
                continue
            if op_id == "detect.hog_people" and not hog_people_runnable():
                continue
            for name, array in arrays.items():
                before = array.copy()
                outputs = self.run_operator(op_id, {"image": array})
                self.assertTrue(outputs, f"{op_id}/{name} 没有输出")
                image = self.first_image(op_id, outputs)
                self.assertIn(image.array.ndim, (2, 3), f"{op_id}/{name} 输出不是图像")
                self.assertGreater(image.array.size, 0, f"{op_id}/{name} 输出为空图")
                self.assertIn(image.color_space, ("L", "BGR", "BGRA"), f"{op_id}/{name} 颜色空间异常")
                self.assertTrue(
                    np.array_equal(array, before),
                    f"{op_id}/{name} 修改了输入图像",
                )

    # ------------------------------------------------------------------ 正确性
    def test_psnr_identical_images_is_infinite(self) -> None:
        array = gray8(48, 64)
        same = self.run_operator("analysis.psnr", {"image": array, "image2": array.copy()})
        value = same["psnr"].value
        self.assertTrue(np.isinf(value) or value >= 100.0, f"相同图 PSNR 应极大，实际 {value}")

        shifted = np.roll(array, 4, axis=1)
        different = self.run_operator("analysis.psnr", {"image": array, "image2": shifted})
        self.assertTrue(np.isfinite(different["psnr"].value), "不同图 PSNR 应为有限值")
        self.assertGreater(different["mse"].value, 0.0, "不同图均方误差应大于 0")

    def test_ssim_identical_images_is_one(self) -> None:
        array = gray8(48, 64)
        same = self.run_operator("analysis.ssim", {"image": array, "image2": array.copy()})
        self.assertAlmostEqual(same["ssim"].value, 1.0, places=4)

        darker = (array.astype(np.float32) * 0.5).astype(np.uint8)
        different = self.run_operator("analysis.ssim", {"image": array, "image2": darker})
        self.assertLess(different["ssim"].value, 0.999, "亮度不同的图 SSIM 应小于 1")

    def test_fft_and_ifft_produce_spectrum_images(self) -> None:
        array = gray8(48, 64)
        spectrum = self.first_image("analysis.fft", self.run_operator("analysis.fft", {"image": array}))
        self.assertEqual(spectrum.array.shape[:2], array.shape)
        self.assertGreater(float(spectrum.array.max()), 0.0, "FFT 频谱图不应全黑")

        rebuilt = self.first_image(
            "analysis.ifft",
            self.run_operator("analysis.ifft", {"image": array}, {"radius": 1.0}),
        )
        self.assertEqual(rebuilt.array.ndim, 2, "IFFT 重建图应为单通道")
        self.assertGreater(rebuilt.array.size, 0)

    def test_stereo_disparity_and_depth_are_non_empty(self) -> None:
        for op_id, key in (("calib.stereo_disparity", "mean_disparity"), ("calib.stereo_depth", "mean_depth")):
            outputs = self.run_operator(op_id, {"image": gray8(48, 64)})
            image = self.first_image(op_id, outputs)
            self.assertEqual(image.array.shape, (48, 64, 3), f"{op_id} 热力图尺寸异常")
            self.assertGreaterEqual(outputs[key].value, 0.0, f"{op_id} 平均值不应为负")

        # 随机噪声缺少可匹配纹理，使用带周期纹理的图像验证视差确实被计算出来。
        height, width = 48, 64
        columns = np.arange(width, dtype=np.float32)[None, :]
        rows = np.arange(height, dtype=np.float32)[:, None]
        texture = (127.0 + 60.0 * np.sin(columns / 3.0) + 40.0 * np.cos(rows / 4.0)).astype(np.uint8)
        left = np.ascontiguousarray(texture)
        right = np.roll(left, 4, axis=1)
        disparity = self.run_operator(
            "calib.stereo_disparity",
            {"image": left, "image2": right},
        )
        self.assertGreater(disparity["mean_disparity"].value, 0.0, "存在平移时平均视差应大于 0")

    def test_template_match_finds_center_pattern(self) -> None:
        image = np.zeros((64, 64), dtype=np.uint8)
        cv2.rectangle(image, (24, 24), (44, 44), 255, -1)
        cv2.circle(image, (34, 34), 6, 90, -1)
        template = image[22:47, 22:47].copy()
        outputs = self.run_operator(
            "analysis.template_match",
            {"image": image, "template": template},
        )
        self.assertGreater(outputs["score"].value, 0.9, "模板匹配度应接近 1")
        self.assertLessEqual(abs(outputs["x"].value - 22.0), 2.0, "匹配位置 X 应接近模板左上角")
        self.assertLessEqual(abs(outputs["y"].value - 22.0), 2.0, "匹配位置 Y 应接近模板左上角")


if __name__ == "__main__":
    unittest.main()
