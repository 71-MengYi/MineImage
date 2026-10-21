"""OpenCV 5.0 目录算子的集成测试。

本测试不启动 GUI，只验证：

* 目录算子全部注册成功，没有加载错误 / id 冲突；
* 每个算子都有中文元数据、合法默认参数、第一个输出是图像；
* 每个算子用默认参数在 7 种代表性数组上都能运行，且不修改输入；
* 任务书要求的算子 id 全部存在；
* 新增的一级分类出现在 ``CATEGORY_ORDER`` 与注册表标签页中。
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
# 测试环境不自动下载模型；人脸检测在模型缺失时跳过默认执行。
os.environ.setdefault("MINEIMAGE_AUTO_DOWNLOAD_MODELS", "0")

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core import models  # noqa: E402
from app.core.constants import CATEGORY_ORDER  # noqa: E402
from app.core.data_types import (  # noqa: E402
    ComponentsData,
    ContoursData,
    DataType,
    EdgesData,
    NumberValue,
    data_type_of,
)
from app.core.types import ImageData  # noqa: E402
from app.operators.loader import OperatorRegistry  # noqa: E402
from app.operators.spec import (  # noqa: E402
    is_param_enabled,
    is_param_visible,
    normalize_params,
    output_to_dict,
    validate_inputs,
)

#: 新增的一级分类（OpenCV 5.0 目录）。
NEW_CATEGORIES = ("图像增强", "视频与运动", "立体与标定", "深度学习")
#: 需要外部模型 / 大文件，默认参数允许失败的算子。
ASSET_REQUIRED = {"dnn.infer"}


def face_detection_runnable() -> bool:
    """OpenCV 4 带 Haar，或 OpenCV 5 已缓存 YuNet 模型时可运行 detect.face。"""

    if hasattr(cv2, "CascadeClassifier"):
        return True
    return hasattr(cv2, "FaceDetectorYN") and models.is_model_ready("face_detection_yunet")


def hog_people_runnable() -> bool:
    """OpenCV 5.0 移除了 cv2.HOGDescriptor，HOG 行人检测仅在 4.x 可运行。"""

    return hasattr(cv2, "HOGDescriptor") and hasattr(cv2, "HOGDescriptor_getDefaultPeopleDetector")

#: 任务书要求交付的算子 id（缺失即失败）。
REQUIRED_IDS = {
    # A：滤波 / 形态学 / 几何 / 去噪
    "filter.bilateral", "filter.box", "filter.sqr_box", "filter.filter2d",
    "filter.sep_filter", "filter.sharpen", "filter.unsharp", "filter.laplacian",
    "filter.pyrdown", "filter.pyrup", "filter.pyr_mean_shift",
    "morph.gradient", "morph.tophat", "morph.blackhat", "morph.hitmiss",
    "morph.ex", "morph.kernel",
    "geom.transpose", "geom.warp_affine", "geom.warp_perspective", "geom.remap",
    "geom.polar", "geom.pad", "geom.shear", "geom.rotate90", "geom.crop", "geom.tile",
    "enhance.denoise", "enhance.denoise_color", "enhance.denoise_tvl1",
    "enhance.edge_preserving", "enhance.detail", "enhance.pencil_sketch",
    "enhance.stylization",
    # B：阈值 / 分割 / 边缘 / 关键点
    "threshold.otsu", "threshold.triangle", "threshold.trunc", "threshold.tozero",
    "threshold.range", "threshold.adaptive_advanced",
    "segment.watershed", "segment.grabcut", "segment.floodfill", "segment.distance",
    "segment.components", "segment.kmeans", "segment.contours", "segment.convex_hull",
    "segment.approx_poly", "segment.min_area_rect", "segment.fit_ellipse",
    "edge.scharr", "edge.laplacian", "edge.canny_auto", "edge.hough_lines",
    "edge.hough_lines_p", "edge.hough_circles", "edge.lsd", "edge.gradient_mag",
    "keypoint.corner_harris", "keypoint.corner_min_eigen", "keypoint.good_features",
    "keypoint.fast", "keypoint.sift", "keypoint.orb", "keypoint.akaze",
    "keypoint.brisk", "keypoint.kaze", "keypoint.mser", "keypoint.blob",
    "keypoint.draw",
    # C：算术 / 颜色 / 直方图
    "basic.add", "basic.subtract", "basic.multiply", "basic.divide", "basic.absdiff",
    "basic.add_weighted", "basic.bitwise_and", "basic.bitwise_or", "basic.bitwise_xor",
    "basic.bitwise_not", "basic.compare", "basic.normalize", "basic.convert_scale",
    "basic.gamma",
    "color.cvt", "color.swap", "color.hsv_adjust", "color.mix", "color.palette",
    "color.white_balance", "color.temperature", "color.alpha_composite",
    "color.extract_alpha", "color.set_alpha", "color.lut_curve",
    "analysis.histogram", "analysis.equalize", "analysis.clahe",
    "analysis.hist_compare", "analysis.backproject", "analysis.match_hist",
    "analysis.hist_2d",
    # D：图像增强 / 绘图 / 布局
    "enhance.inpaint", "enhance.seamless_clone", "enhance.color_change",
    "enhance.illumination", "enhance.texture_flatten", "enhance.hdr_merge",
    "enhance.tonemap", "enhance.retinex", "enhance.local_contrast", "enhance.dehaze",
    "draw.rectangle", "draw.circle", "draw.ellipse", "draw.polygon", "draw.text",
    "draw.arrow", "draw.marker", "draw.crosshair", "draw.grid", "draw.scale_bar",
    "draw.watermark", "draw.roi_highlight",
    "layout.montage", "layout.overlay", "layout.contact_sheet", "layout.border",
    "layout.split",
    # E：视频 / 检测 / 标定 / 分析
    "video.frame_diff", "video.optical_flow_farneback", "video.optical_flow_lk",
    "video.background_mog2", "video.background_knn", "video.mean_shift",
    "video.camshift", "video.ecc_align", "video.tracker", "video.motion_energy",
    "detect.face", "detect.qr", "detect.barcode", "detect.hog_people",
    "calib.chessboard", "calib.undistort", "calib.stereo_disparity",
    "calib.stereo_depth", "calib.panorama", "calib.homography",
    "dnn.blob", "dnn.infer",
    "analysis.integral", "analysis.min_max", "analysis.mean_std",
    "analysis.moments", "analysis.hu_moments", "analysis.fft", "analysis.ifft",
    "analysis.phase_correlate", "analysis.template_match", "analysis.psnr",
    "analysis.ssim", "analysis.match_shapes",
    # 通用能力与特征匹配
    "feature.match", "misc.cv2_call", "misc.cv2_info", "output.display",
}

CJK = ("\u4e00", "\u9fff")


def contains_cjk(text: str) -> bool:
    return any(CJK[0] <= char <= CJK[1] for char in str(text))


def make_samples() -> dict[str, np.ndarray]:
    """构造 32×32 的 7 种代表性输入，包含纹理、边缘与渐变，便于特征类算子工作。"""
    rng = np.random.default_rng(20251021)
    height = width = 32
    base = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    base[:, : width // 2, 0] = np.linspace(0, 255, width // 2, dtype=np.uint8)
    base[height // 4 : height // 2, :, 1] = 200
    base[height // 2 :, width // 2 :, 2] = 255
    bgr8 = base
    bgra8 = np.dstack([bgr8, np.full((height, width), 180, dtype=np.uint8)])
    gray8 = (bgr8.astype(np.float32).mean(axis=2)).astype(np.uint8)
    bgr16 = (bgr8.astype(np.uint16) * 257).astype(np.uint16)
    gray16 = (gray8.astype(np.uint16) * 257).astype(np.uint16)
    bgrf = bgr8.astype(np.float32) / 255.0
    grayf = gray8.astype(np.float32) / 255.0
    return {
        "bgr8": bgr8,
        "bgra8": bgra8,
        "gray8": gray8,
        "bgr16": bgr16,
        "gray16": gray16,
        "bgrf": bgrf,
        "grayf": grayf,
    }


def sample_for_port(port, samples: dict[str, np.ndarray]) -> np.ndarray:
    accepts = port.accepts or ("L", "BGR", "BGRA")
    for array in samples.values():
        if ImageData(array).color_space in accepts:
            return array
    return samples["bgr8"]


def value_for_required_port(port):
    """为（不推荐的）必填非图像端口构造一个合法样例值。"""
    size = (32, 32)
    if port.data_type is DataType.NUMBER:
        return NumberValue(0.5, port.label)
    if port.data_type is DataType.EDGES:
        return EdgesData((np.array([[0, 0], [10, 10]], dtype=np.int32),), image_size=size)
    if port.data_type is DataType.CONTOURS:
        return ContoursData((np.array([[0, 0], [10, 0], [10, 10]], dtype=np.int32),), image_size=size)
    if port.data_type is DataType.COMPONENTS:
        return ComponentsData.from_labels(np.zeros((32, 32), dtype=np.int32), image_size=size)
    if port.data_type is DataType.IMAGE:
        return ImageData(np.zeros((32, 32, 3), dtype=np.uint8))
    raise AssertionError(f"未处理的端口类型：{port.data_type}")


class OpenCVCatalogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = OperatorRegistry.build()
        cls.catalog_entries = [
            entry
            for entry in cls.registry.all_entries()
            if entry.path.parent.name == "catalog" and entry.path.name != "__init__.py"
        ]
        cls.samples = make_samples()

    # ------------------------------------------------------------------ 结构
    def test_registry_has_no_load_errors(self) -> None:
        errors = [(issue.path.name, issue.message) for issue in self.registry.errors]
        self.assertEqual(errors, [], f"算子加载错误：{errors}")

    def test_all_required_ids_are_registered(self) -> None:
        missing = sorted(REQUIRED_IDS - set(self.registry.entries))
        self.assertEqual(missing, [], f"缺少任务书要求的算子：{missing}")

    def test_catalog_has_enough_operators(self) -> None:
        self.assertGreaterEqual(len(self.catalog_entries), 130, "目录算子数量不足")

    def test_new_categories_are_registered(self) -> None:
        tabs = self.registry.tabs()
        for category in NEW_CATEGORIES:
            self.assertIn(category, CATEGORY_ORDER)
            self.assertIn(category, tabs, f"分类 {category} 下没有算子")
        self.assertEqual(
            list(tabs),
            [category for category in CATEGORY_ORDER if category in tabs],
        )

    def test_catalog_metadata(self) -> None:
        for entry in self.catalog_entries:
            with self.subTest(op=entry.id):
                spec = entry.spec
                self.assertTrue(entry.builtin, "目录算子应为内置算子")
                self.assertTrue(contains_cjk(spec.name), f"{entry.id} 名称必须是中文")
                self.assertTrue(contains_cjk(spec.description), f"{entry.id} 描述必须是中文")
                self.assertTrue(spec.category, f"{entry.id} 缺少分类")
                self.assertIn(spec.category[0], CATEGORY_ORDER)
                self.assertTrue(spec.outputs, f"{entry.id} 没有输出")
                self.assertTrue(
                    spec.outputs[0].any_type
                    or spec.outputs[0].data_type is DataType.IMAGE,
                    f"{entry.id} 第一个输出必须是图像或万能端口（预览依赖）",
                )
                defaults = normalize_params(spec, None)
                self.assertEqual(set(defaults), {param.key for param in spec.params})
                for param in spec.params:
                    self.assertTrue(param.label, f"{entry.id}.{param.key} 缺少标签")
                    for expression in (param.visible_when, param.enabled_when):
                        if expression:
                            self.assertNotIn("params.", expression)
                            self.assertIsInstance(is_param_visible(param, defaults), bool)
                            self.assertIsInstance(is_param_enabled(param, defaults), bool)

    # ------------------------------------------------------------------ 冒烟
    def test_all_catalog_operators_run_with_defaults(self) -> None:
        for entry in self.catalog_entries:
            if entry.id in ASSET_REQUIRED:
                continue
            if entry.id == "detect.face" and not face_detection_runnable():
                # 与 dnn.infer 同理：需要外部 YuNet 模型，测试环境不自动下载。
                continue
            if entry.id == "detect.hog_people" and not hog_people_runnable():
                # OpenCV 5.0 移除了 cv2.HOGDescriptor。
                continue
            spec = entry.spec
            sample_items = self.samples.items() if spec.inputs else [("gray8", self.samples["gray8"])]
            for name, array in sample_items:
                first = spec.inputs[0] if spec.inputs else None
                color_space = ImageData(array).color_space
                if first is not None and first.accepts and color_space not in first.accepts:
                    continue
                with self.subTest(op=entry.id, sample=name):
                    before = array.copy()
                    inputs = {}
                    for index, port in enumerate(spec.inputs):
                        if index == 0:
                            inputs[port.key] = ImageData(array)
                        elif port.optional:
                            inputs[port.key] = None
                        else:
                            inputs[port.key] = value_for_required_port(port)
                    params = normalize_params(spec, None)
                    validate_inputs(spec, inputs)
                    entry.operator.validate(inputs, params)
                    outputs = output_to_dict(
                        spec, entry.operator.process(**inputs, **params)
                    )
                    self.assertTrue(outputs, f"{entry.id}/{name} 没有输出")
                    np.testing.assert_array_equal(
                        array, before, err_msg=f"{entry.id}/{name} 修改了输入图像"
                    )
                    for port in spec.outputs:
                        if port.key not in outputs:
                            continue
                        value = outputs[port.key]
                        if port.any_type:
                            self.assertIsNotNone(
                                data_type_of(value),
                                f"{entry.id} 万能输出 {port.key} 必须是已知数据类型",
                            )
                        else:
                            self.assertEqual(
                                data_type_of(value),
                                port.data_type,
                                f"{entry.id} 输出 {port.key} 类型不匹配",
                            )
                        if isinstance(value, ImageData):
                            self.assertGreater(value.array.size, 0)
                            self.assertIn(value.array.ndim, (2, 3))
                            self.assertIn(value.color_space, ("L", "BGR", "BGRA"))

    # ------------------------------------------------------------------ 代表性行为
    def test_representative_behaviours(self) -> None:
        filters = {
            "bilateral": "filter.bilateral",
            "warp": "geom.warp_affine",
            "gradient": "morph.gradient",
            "denoise": "enhance.denoise",
        }
        for label, op_id in filters.items():
            with self.subTest(op=op_id):
                entry = self.registry.entry(op_id)
                self.assertIsNotNone(entry, f"{label} 算子缺失")
                params = normalize_params(entry.spec, None)
                inputs = {port.key: ImageData(self.samples["bgr8"]) for port in entry.spec.inputs if port.key == "image"}
                result = output_to_dict(entry.spec, entry.operator.process(**inputs, **params))
                first = result[entry.spec.outputs[0].key]
                self.assertGreater(first.array.size, 0)

    def test_feature_match_finds_correspondences(self) -> None:
        import cv2

        base = np.zeros((96, 96, 3), dtype=np.uint8)
        cv2.rectangle(base, (10, 10), (40, 40), (255, 255, 255), -1)
        cv2.circle(base, (70, 30), 14, (0, 255, 0), -1)
        cv2.line(base, (10, 70), (85, 85), (255, 0, 0), 3)
        cv2.putText(base, "ABC", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        shifted = np.roll(base, 6, axis=1)

        entry = self.registry.entry("feature.match")
        self.assertIsNotNone(entry)
        params = normalize_params(entry.spec, {"detector": "orb", "matcher": "bf", "ratio": 0.8})
        outputs = output_to_dict(
            entry.spec,
            entry.operator.process(image=ImageData(base), image2=ImageData(shifted), **params),
        )
        self.assertGreaterEqual(outputs["count"].value, 4.0)
        self.assertGreater(outputs["image"].array.size, 0)

    def test_fft_ifft_roundtrip_uses_spectrum_port(self) -> None:
        """FFT 的复频谱输出应能接入 IFFT 并精确重建原图。"""
        array = self.samples["gray8"]
        fft = self.registry.entry("analysis.fft")
        ifft = self.registry.entry("analysis.ifft")
        self.assertIsNotNone(fft)
        self.assertIsNotNone(ifft)
        self.assertIn(
            "spectrum",
            {port.key for port in fft.spec.outputs},
            "FFT 应输出复频谱端口",
        )
        self.assertIn(
            "spectrum",
            {port.key for port in ifft.spec.inputs},
            "IFFT 应接受复频谱输入",
        )

        fft_outputs = output_to_dict(
            fft.spec,
            fft.operator.process(
                image=ImageData(array),
                **normalize_params(fft.spec, None),
            ),
        )
        spectrum = fft_outputs["spectrum"]
        self.assertEqual(spectrum.array.dtype, np.float32)
        self.assertEqual(spectrum.color_space, "BGR")

        restored = output_to_dict(
            ifft.spec,
            ifft.operator.process(
                image=ImageData(array),
                spectrum=spectrum,
                **normalize_params(ifft.spec, None),
            ),
        )["image"].array
        self.assertEqual(restored.shape, array.shape)
        self.assertLessEqual(
            int(np.abs(restored.astype(np.int16) - array.astype(np.int16)).max()),
            1,
            "FFT → IFFT 往返误差应不超过 1 个灰度级",
        )

    def test_histogram_compare_returns_number(self) -> None:
        entry = self.registry.entry("analysis.hist_compare")
        self.assertIsNotNone(entry)
        params = normalize_params(entry.spec, None)
        inputs = {}
        for port in entry.spec.inputs:
            if port.key == "image":
                inputs[port.key] = ImageData(self.samples["bgr8"])
            elif port.optional:
                inputs[port.key] = None
            else:
                inputs[port.key] = value_for_required_port(port)
        result = output_to_dict(entry.spec, entry.operator.process(**inputs, **params))
        number_ports = [port for port in entry.spec.outputs if port.data_type is DataType.NUMBER]
        self.assertTrue(number_ports, "hist_compare 应输出数值端口")
        for port in number_ports:
            if port.key in result:
                self.assertIsInstance(result[port.key], NumberValue)


if __name__ == "__main__":
    unittest.main(verbosity=2)
