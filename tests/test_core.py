"""核心数据结构、工程文件与算子规范的单元测试。"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
import zipfile
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from app.core.imageio import load_image, save_image
from app.core.model import GraphError, GraphModel, NodeModel, WorkspaceModel
from app.core.project import ProjectFormatError, load_project, save_project
from app.core.types import (
    ImageData,
    PixelSample,
    image_intensity,
    intensity_grid,
    sample_pixel,
    to_display_rgb,
    to_scaled_rgb,
)
from app.operators.spec import (
    OperatorError,
    OperatorSpec,
    ParamSpec,
    ParamValidationError,
    PortSpec,
    coerce_param_value,
    normalize_params,
    output_to_dict,
    validate_inputs,
)

_TMP_ROOT = Path(__file__).resolve().parent.parent / ".test_tmp"
_TMP_ROOT.mkdir(exist_ok=True)


class TempDirMixin:
    """为测试提供 .test_tmp 下的唯一文件名。

    注意：不使用嵌套临时目录，部分受限运行环境对新目录 ACL 不友好。
    """

    def setUp(self) -> None:  # noqa: N802
        _TMP_ROOT.mkdir(exist_ok=True)
        self.tmpdir = _TMP_ROOT
        self._cleanup: list[Path] = []
        self.addCleanup(self._cleanup_files)

    def _cleanup_files(self) -> None:
        for path in self._cleanup:
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass

    def unique(self, name: str) -> Path:
        import uuid

        path = self.tmpdir / f"{uuid.uuid4().hex[:8]}_{name}"
        self._cleanup.append(path)
        return path


class TestImageTypes(unittest.TestCase):
    def test_display_normalization(self) -> None:
        gray16 = np.array([[0, 65535]], dtype=np.uint16)
        rgb = to_display_rgb(ImageData(gray16))
        self.assertEqual(rgb.shape, (1, 2, 3))
        self.assertEqual(rgb[0, 0, 0], 0)
        self.assertEqual(rgb[0, 1, 0], 255)

        flt = np.array([[[0.0, 0.0, 0.0], [1.0, 0.5, 0.25]]], dtype=np.float32)
        rgb = to_display_rgb(ImageData(flt))
        self.assertEqual(rgb[0, 0].tolist(), [0, 0, 0])
        self.assertEqual(rgb[0, 1].tolist(), [63, 127, 255])

    def test_bgr_order_and_pixel_sample(self) -> None:
        bgr = np.zeros((2, 3, 3), dtype=np.uint8)
        bgr[1, 2] = (10, 20, 30)  # B,G,R
        image = ImageData(bgr)
        self.assertEqual(image.color_space, "BGR")
        rgb = to_display_rgb(image)
        self.assertEqual(rgb[1, 2].tolist(), [30, 20, 10])
        sample = sample_pixel(image, 1, 2)
        self.assertIsInstance(sample, PixelSample)
        self.assertTrue(sample.in_bounds)
        self.assertEqual(sample.values, (30.0, 20.0, 10.0))
        self.assertIn("R=30", sample.value_text())

        outside = sample_pixel(image, 99, 99)
        self.assertFalse(outside.in_bounds)
        self.assertEqual(outside.value_text(), "--")

    def test_image_io_roundtrip_uint16_png(self) -> None:
        path = _TMP_ROOT / "图像_16bit.png"
        try:
            data = np.arange(4 * 5, dtype=np.uint16).reshape(4, 5) * 1000
            save_image(path, ImageData(data))
            loaded = load_image(path)
            self.assertEqual(loaded.array.dtype, np.uint16)
            np.testing.assert_array_equal(loaded.array, data)
        finally:
            path.unlink(missing_ok=True)


class TestGraphModel(unittest.TestCase):
    def _chain(self) -> tuple[GraphModel, NodeModel, NodeModel, NodeModel]:
        graph = GraphModel()
        a = graph.create_operator_node("op.a", "A")
        b = graph.create_operator_node("op.b", "B")
        c = graph.create_operator_node("op.c", "C")
        graph.add_connection(a.id, "image", b.id, "image")
        graph.add_connection(b.id, "image", c.id, "image")
        return graph, a, b, c

    def test_topological_and_traversal(self) -> None:
        graph, a, b, c = self._chain()
        order = graph.topological_order()
        self.assertLess(order.index(a.id), order.index(b.id))
        self.assertLess(order.index(b.id), order.index(c.id))
        self.assertEqual(graph.descendants(a.id), [b.id, c.id])
        self.assertEqual(graph.ancestors(c.id), [b.id, a.id])

    def test_cycle_rejected(self) -> None:
        graph, a, b, _c = self._chain()
        with self.assertRaises(GraphError):
            graph.add_connection(b.id, "image", a.id, "image")

    def test_duplicate_input_rejected_and_self_loop(self) -> None:
        graph, a, b, _c = self._chain()
        with self.assertRaises(GraphError):
            graph.add_connection(a.id, "image", b.id, "image")
        with self.assertRaises(GraphError):
            graph.add_connection(a.id, "image", a.id, "image")

    def test_topological_order_with_parallel_port_edges(self) -> None:
        """同一对节点之间的多条端口连线不应被误判为环。"""
        graph = GraphModel()
        split = graph.create_operator_node("channel.split", "拆分")
        merge = graph.create_operator_node("channel.merge", "合并")
        for port, dst_port in (("b", "b"), ("g", "g"), ("r", "r")):
            graph.add_connection(split.id, port, merge.id, dst_port)
        order = graph.topological_order()
        self.assertEqual(order, [split.id, merge.id])

    def test_remove_node_cleans_connections(self) -> None:
        graph, _a, b, _c = self._chain()
        graph.remove_node(b.id)
        self.assertEqual(len(graph.connections), 0)
        with self.assertRaises(GraphError):
            graph.node(b.id)

    def test_serialization(self) -> None:
        graph, a, b, _c = self._chain()
        data = graph.to_dict()
        restored = GraphModel.from_dict(data)
        self.assertEqual(set(restored.nodes), set(graph.nodes))
        self.assertEqual(len(restored.connections), 2)
        self.assertEqual(restored.node(a.id).title, "A")
        self.assertEqual(restored.node(b.id).params, {})


class TestWorkspaceProject(TempDirMixin, unittest.TestCase):
    def _workspace(self) -> WorkspaceModel:
        workspace = WorkspaceModel("测试工程")
        image = ImageData(np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3), "输入")
        source = workspace.add_source_image(image, path="C:/fake/input.png")
        op = workspace.graph.create_operator_node("op.copy", "复制")
        workspace.graph.add_connection(source.id, "image", op.id, "image")
        return workspace

    def test_project_roundtrip_embeds_images(self) -> None:
        workspace = self._workspace()
        path = save_project(self.unique("demo"), workspace)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        self.assertEqual(path.suffix, ".mi")
        self.assertTrue(path.exists())

        loaded = load_project(path)
        self.assertEqual(loaded.name, "测试工程")
        self.assertEqual(len(loaded.graph.nodes), 2)
        self.assertEqual(len(loaded.images), 1)
        image = next(iter(loaded.images.values()))
        self.assertEqual(image.array.dtype, np.uint8)
        self.assertEqual(image.array.shape, (3, 4, 3))
        source = loaded.graph.source_nodes()[0]
        self.assertIsNotNone(loaded.image_for_node(source))

    def test_project_preserves_float_and_alpha(self) -> None:
        workspace = WorkspaceModel("数值保真")
        float_img = ImageData(np.linspace(0, 1, 20 * 30, dtype=np.float32).reshape(20, 30), "float")
        alpha_img = ImageData(np.zeros((8, 9, 4), dtype=np.uint8), "alpha")
        workspace.add_source_image(float_img)
        workspace.add_source_image(alpha_img)
        path = save_project(self.unique("types"), workspace)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        loaded = load_project(path)
        arrays = [image.array for image in loaded.images.values()]
        self.assertTrue(any(array.dtype == np.float32 for array in arrays))
        self.assertTrue(any(array.shape == (8, 9, 4) for array in arrays))
        self.assertEqual(len(loaded.images), 2)

    def test_project_roundtrip_preserves_display_mode(self) -> None:
        workspace = self._workspace()
        self.assertEqual(workspace.display_mode, "imshow")
        workspace.display_mode = "mesh"
        path = save_project(self.unique("display_mode"), workspace)
        self.addCleanup(lambda: path.unlink(missing_ok=True))
        loaded = load_project(path)
        self.assertEqual(loaded.display_mode, "mesh")

        data = workspace.to_dict()
        self.assertEqual(data["display_mode"], "mesh")

    def test_display_mode_validation(self) -> None:
        workspace = WorkspaceModel("显示模式")
        self.assertEqual(workspace.display_mode, "imshow")
        workspace.display_mode = "MESH"
        self.assertEqual(workspace.display_mode, "mesh")
        workspace.display_mode = "imagesc"
        self.assertEqual(workspace.display_mode, "imagesc")
        workspace.display_mode = "unknown-mode"
        self.assertEqual(workspace.display_mode, "imshow")
        workspace.display_mode = ""
        self.assertEqual(workspace.display_mode, "imshow")

    def test_legacy_project_without_display_mode_defaults_to_imshow(self) -> None:
        legacy = self.unique("legacy_display_mode.mi")
        self.addCleanup(lambda: legacy.unlink(missing_ok=True))
        document = {
            "format": "mineimage-project",
            "format_version": 1,
            "workspace": {
                "name": "旧工程",
                "graph": {"nodes": [], "connections": []},
                "images": [],
            },
        }
        with zipfile.ZipFile(legacy, "w") as archive:
            archive.writestr("project.json", json.dumps(document, ensure_ascii=False))
        loaded = load_project(legacy)
        self.assertEqual(loaded.display_mode, "imshow")

    def test_invalid_project(self) -> None:
        bad = self.unique("bad.mi")
        bad.write_bytes(b"not a zip")
        with self.assertRaises(ProjectFormatError):
            load_project(bad)


class TestOperatorSpec(unittest.TestCase):
    def _spec(self) -> OperatorSpec:
        return OperatorSpec(
            id="test.op",
            name="测试算子",
            category=("基础调整", "测试"),
            inputs=(PortSpec("image", "输入"),),
            outputs=(PortSpec("image", "输出"),),
            params=(
                ParamSpec("ksize", "核大小", "int", default=3, min=1, max=31, odd=True),
                ParamSpec(
                    "mode",
                    "模式",
                    "enum",
                    default="fixed",
                    choices=(("fixed", "固定"), ("auto", "自动")),
                ),
                ParamSpec("threshold", "阈值", "float", default=0.5, min=0, max=1, step=0.01),
                ParamSpec("enabled", "启用", "bool", default=True),
                ParamSpec("roi", "区域", "roi"),
            ),
        )

    def test_defaults_and_coercion(self) -> None:
        spec = self._spec()
        params = normalize_params(spec, {})
        self.assertEqual(params["ksize"], 3)
        self.assertEqual(params["mode"], "fixed")
        self.assertIsNone(params["roi"])
        with self.assertRaises(ParamValidationError):
            coerce_param_value(spec.param("ksize"), 4)
        with self.assertRaises(ParamValidationError):
            coerce_param_value(spec.param("threshold"), 2)
        with self.assertRaises(ParamValidationError):
            normalize_params(spec, {"unknown": 1})

    def test_roi_and_points(self) -> None:
        spec = self._spec()
        self.assertEqual(coerce_param_value(spec.param("roi"), [1, 2, 3, 4]), [1.0, 2.0, 3.0, 4.0])
        with self.assertRaises(ParamValidationError):
            coerce_param_value(spec.param("roi"), [1, 2, 0, 4])

    def test_output_normalization_and_port_validation(self) -> None:
        spec = self._spec()
        array = np.zeros((2, 2), dtype=np.uint8)
        outputs = output_to_dict(spec, array)
        self.assertEqual(outputs["image"].array.shape, (2, 2))
        with self.assertRaises(OperatorError):
            output_to_dict(spec, {"bad_port": array})
        with self.assertRaises(OperatorError):
            validate_inputs(spec, {"image": None})
        # 彩色输入连接到只接受灰度的端口
        gray_spec = OperatorSpec(
            id="test.gray",
            name="灰度输入",
            inputs=(PortSpec("image", "输入", accepts=("L",)),),
        )
        with self.assertRaises(OperatorError):
            validate_inputs(gray_spec, {"image": ImageData(np.zeros((2, 2, 3), np.uint8))})


class TestDisplayMapping(unittest.TestCase):
    """imagesc 伪彩色 / mesh 强度图的数值语义。"""

    def test_imagesc_stretches_narrow_float_range(self) -> None:
        value = np.linspace(1000.0, 1002.0, 32 * 24, dtype=np.float32).reshape(32, 24)
        rgb = to_scaled_rgb(ImageData(value, "窄范围浮点"))
        self.assertEqual(rgb.shape, (32, 24, 3))
        self.assertEqual(rgb.dtype, np.uint8)
        # 拉伸后应覆盖色带两端（而不是显示成近乎纯色的图）
        first = rgb[0, 0].astype(int)
        last = rgb[-1, -1].astype(int)
        self.assertGreater(int(np.abs(first - last).sum()), 80)

    def test_imagesc_constant_image_and_tiny_shapes_are_safe(self) -> None:
        flat = to_scaled_rgb(ImageData(np.full((4, 5), 7, np.uint8)))
        self.assertEqual(flat.shape, (4, 5, 3))
        self.assertEqual(flat.dtype, np.uint8)
        # 1×1 图与空图也不能抛异常
        self.assertEqual(to_scaled_rgb(ImageData(np.zeros((1, 1, 3), np.uint8))).shape, (1, 1, 3))
        self.assertEqual(to_scaled_rgb(ImageData(np.zeros((0, 0), np.uint8))).shape, (0, 0, 3))

    def test_image_intensity_uses_bgr_luminance(self) -> None:
        array = np.zeros((2, 2, 3), np.uint8)
        array[:, :, 2] = 255  # BGR：R 通道
        intensity = image_intensity(ImageData(array, "红"))
        self.assertAlmostEqual(float(intensity[0, 0]), 0.299 * 255, delta=0.5)
        gray = image_intensity(ImageData(np.full((2, 2), 42, np.uint8)))
        self.assertAlmostEqual(float(gray[0, 0]), 42.0, places=3)
        with self.assertRaises(ValueError):
            image_intensity(np.zeros((2, 2, 2, 2), np.uint8))

    def test_intensity_grid_downsamples_and_preserves_flat_values(self) -> None:
        big = np.tile(np.linspace(0, 255, 256, dtype=np.float32), (256, 1))
        grid = intensity_grid(ImageData(big), 48)
        self.assertEqual(max(grid.shape), 48)
        flat = intensity_grid(ImageData(np.full((100, 100), 33, np.uint8)), 16)
        self.assertTrue(np.allclose(flat, 33.0))
        small = intensity_grid(ImageData(np.arange(12, dtype=np.float32).reshape(3, 4)), 48)
        self.assertEqual(small.shape, (3, 4))


if __name__ == "__main__":
    unittest.main()
