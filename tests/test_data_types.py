"""数据类型与结构化数据容器的单元测试（不启动 GUI 事件循环）。"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.data_types import (  # noqa: E402
    ComponentsData,
    ContoursData,
    DataType,
    EdgesData,
    NumberValue,
    coerce_data_type,
    data_type_of,
    is_data_value,
    render_preview,
)
from app.core.types import ImageData  # noqa: E402
from app.operators.spec import (  # noqa: E402
    OperatorError,
    OperatorSpec,
    PortSpec,
    output_to_dict,
    validate_inputs,
)


def structured_spec() -> OperatorSpec:
    """一个同时声明四种结构化输出、图像 + 轮廓输入的测试规范。"""

    return OperatorSpec(
        id="test.structured",
        name="结构化数据测试",
        description="仅用于数据类型单元测试。",
        inputs=(
            PortSpec("image", "输入图像", accepts=("L", "BGR")),
            PortSpec("contours", "输入轮廓", data_type=DataType.CONTOURS),
        ),
        outputs=(
            PortSpec("number", "数值", data_type=DataType.NUMBER),
            PortSpec("edges", "边缘", data_type=DataType.EDGES),
            PortSpec("contours", "轮廓", data_type=DataType.CONTOURS),
            PortSpec("components", "连通域", data_type=DataType.COMPONENTS),
        ),
    )


def two_component_labels() -> np.ndarray:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[0:2, 0:2] = 1  # 4 像素
    labels[5:8, 5:8] = 2  # 9 像素
    return labels


class DataTypeTests(unittest.TestCase):
    def test_labels_and_colors(self) -> None:
        expected_labels = {
            DataType.IMAGE: "图像",
            DataType.NUMBER: "数值",
            DataType.EDGES: "边缘特征",
            DataType.CONTOURS: "轮廓特征",
            DataType.COMPONENTS: "连通域",
        }
        expected_colors = {
            DataType.IMAGE: "#3b82f6",
            DataType.NUMBER: "#d97706",
            DataType.EDGES: "#7c3aed",
            DataType.CONTOURS: "#0d9488",
            DataType.COMPONENTS: "#e11d48",
        }
        for data_type, label in expected_labels.items():
            self.assertEqual(data_type.label, label)
            self.assertEqual(data_type.color, expected_colors[data_type])
        self.assertEqual(DataType.IMAGE.value, "image")
        self.assertEqual(DataType.NUMBER.value, "number")
        self.assertEqual(DataType.EDGES.value, "edges")
        self.assertEqual(DataType.CONTOURS.value, "contours")
        self.assertEqual(DataType.COMPONENTS.value, "components")

    def test_coerce_data_type_aliases(self) -> None:
        aliases = {
            "image": DataType.IMAGE,
            "IMG": DataType.IMAGE,
            "图像": DataType.IMAGE,
            "图片": DataType.IMAGE,
            "number": DataType.NUMBER,
            "numeric": DataType.NUMBER,
            "数值": DataType.NUMBER,
            "标量": DataType.NUMBER,
            "edge": DataType.EDGES,
            "edges": DataType.EDGES,
            "边缘特征": DataType.EDGES,
            "线段": DataType.EDGES,
            "contour": DataType.CONTOURS,
            "轮廓": DataType.CONTOURS,
            "轮廓特征": DataType.CONTOURS,
            "component": DataType.COMPONENTS,
            "connected_components": DataType.COMPONENTS,
            "连通域": DataType.COMPONENTS,
        }
        for text, expected in aliases.items():
            self.assertIs(coerce_data_type(text), expected, text)
        self.assertIs(coerce_data_type(DataType.EDGES), DataType.EDGES)

    def test_coerce_data_type_rejects_unknown(self) -> None:
        for invalid in ("", "unknown", "clips", "图像类"):
            with self.assertRaises(ValueError):
                coerce_data_type(invalid)


class PortSpecTests(unittest.TestCase):
    def test_default_port_is_image(self) -> None:
        port = PortSpec("image", "输入")
        self.assertIs(port.data_type, DataType.IMAGE)
        self.assertEqual(port.accepts, ("L", "BGR", "BGRA"))
        self.assertEqual(port.accept_text(), "L/BGR/BGRA")
        self.assertEqual(port.type_text(), "图像（L/BGR/BGRA）")
        self.assertTrue(port.accepts_value(ImageData(np.zeros((4, 4), np.uint8))))

    def test_non_image_port_clears_accepts(self) -> None:
        number_port = PortSpec(
            "number", "数值", accepts=("L", "BGR"), data_type=DataType.NUMBER
        )
        self.assertIs(number_port.data_type, DataType.NUMBER)
        self.assertEqual(number_port.accepts, ())
        self.assertEqual(number_port.accept_text(), "数值")
        self.assertEqual(number_port.type_text(), "数值")

        edges_port = PortSpec("edges", "线段", data_type="edges")
        self.assertIs(edges_port.data_type, DataType.EDGES)
        self.assertEqual(edges_port.accepts, ())
        self.assertEqual(edges_port.accept_text(), "边缘特征")

    def test_accepts_value_checks_type_and_color_space(self) -> None:
        gray_port = PortSpec("image", "灰度图", accepts=("L",))
        self.assertTrue(gray_port.accepts_value(ImageData(np.zeros((4, 4), np.uint8))))
        self.assertFalse(
            gray_port.accepts_value(ImageData(np.zeros((4, 4, 3), np.uint8)))
        )
        self.assertFalse(gray_port.accepts_value(NumberValue(1)))

        any_port = PortSpec("image", "任意图像", accepts=())
        self.assertEqual(any_port.accept_text(), "任意")
        self.assertTrue(
            any_port.accepts_value(ImageData(np.zeros((4, 4, 3), np.uint8)))
        )


class OutputNormalizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = structured_spec()

    def test_output_to_dict_normalizes_raw_values(self) -> None:
        outputs = output_to_dict(
            self.spec,
            {
                "number": 2.5,
                "edges": [[0, 0], [5, 5]],
                "contours": [[[0, 0], [4, 0], [4, 4]]],
                "components": two_component_labels(),
            },
        )
        self.assertIsInstance(outputs["number"], NumberValue)
        self.assertAlmostEqual(outputs["number"].value, 2.5)
        self.assertIsInstance(outputs["edges"], EdgesData)
        self.assertEqual(outputs["edges"].count, 1)
        self.assertIsInstance(outputs["contours"], ContoursData)
        self.assertEqual(outputs["contours"].count, 1)
        self.assertIsInstance(outputs["components"], ComponentsData)
        self.assertEqual(outputs["components"].count, 2)

    def test_output_to_dict_accepts_container_values(self) -> None:
        number = NumberValue(value=3, label="输入数值", unit="个")
        edges = EdgesData(segments=([0, 0], [3, 3]))
        contours = ContoursData(contours=([1, 1], [5, 5]))
        components = ComponentsData.from_labels(two_component_labels())
        outputs = output_to_dict(
            self.spec,
            {
                "number": number,
                "edges": edges,
                "contours": contours,
                "components": components,
            },
        )
        self.assertIs(outputs["number"], number)
        self.assertIs(outputs["edges"], edges)
        self.assertIs(outputs["contours"], contours)
        self.assertIs(outputs["components"], components)

    def test_output_to_dict_rejects_mismatched_types(self) -> None:
        with self.assertRaises(OperatorError):
            output_to_dict(self.spec, {"number": "not a number"})
        with self.assertRaises(OperatorError):
            output_to_dict(self.spec, {"edges": 12})
        with self.assertRaises(OperatorError):
            output_to_dict(self.spec, {"contours": object()})
        with self.assertRaises(OperatorError):
            output_to_dict(self.spec, {"components": "bad"})
        with self.assertRaises(OperatorError):
            output_to_dict(self.spec, {"unknown_port": 1})


class InputValidationTests(unittest.TestCase):
    def test_validate_inputs_accepts_matching_types(self) -> None:
        spec = structured_spec()
        validate_inputs(
            spec,
            {
                "image": ImageData(np.zeros((8, 8), np.uint8)),
                "contours": ContoursData(contours=([0, 0], [3, 3])),
            },
        )

    def test_validate_inputs_rejects_type_mismatch(self) -> None:
        spec = structured_spec()
        image = ImageData(np.zeros((8, 8), np.uint8))
        contours = ContoursData(contours=([0, 0], [3, 3]))
        with self.assertRaises(OperatorError):
            validate_inputs(
                spec, {"image": image, "contours": NumberValue(value=1)}
            )
        with self.assertRaises(OperatorError):
            validate_inputs(
                spec, {"image": contours, "contours": contours}
            )

    def test_validate_inputs_rejects_missing_and_bad_color_space(self) -> None:
        spec = structured_spec()
        with self.assertRaises(OperatorError):
            validate_inputs(
                spec, {"image": ImageData(np.zeros((8, 8), np.uint8))}
            )
        with self.assertRaises(OperatorError):
            validate_inputs(
                spec,
                {"contours": ContoursData(contours=([0, 0], [3, 3]))},
            )
        # 2 通道图像推断为 CH2，不属于 L/BGR 白名单。
        with self.assertRaises(OperatorError):
            validate_inputs(
                spec,
                {
                    "image": ImageData(np.zeros((8, 8, 2), np.uint8)),
                    "contours": ContoursData(contours=([0, 0], [3, 3])),
                },
            )


class RenderPreviewTests(unittest.TestCase):
    def test_render_preview_for_structured_data(self) -> None:
        edges = EdgesData(segments=([0, 0], [10, 10]), image_size=(20, 20))
        contours = ContoursData(
            contours=([2, 2], [10, 2], [10, 10]), image_size=(20, 20)
        )
        components = ComponentsData.from_labels(two_component_labels())
        for value in (edges, contours, components):
            preview = render_preview(value)
            self.assertIsInstance(preview, ImageData)
            self.assertEqual(preview.color_space, "BGR")
            self.assertGreater(preview.array.size, 0)

    def test_render_preview_returns_none_for_number(self) -> None:
        self.assertIsNone(render_preview(NumberValue(value=3, label="数值")))
        direct = ImageData(np.zeros((4, 4), np.uint8))
        self.assertIs(render_preview(direct), direct)


class ComponentsDataTests(unittest.TestCase):
    def test_from_labels_computes_statistics(self) -> None:
        components = ComponentsData.from_labels(two_component_labels())
        self.assertIsInstance(components, ComponentsData)
        self.assertEqual(components.count, 2)
        self.assertEqual(components.areas, (4, 9))
        self.assertEqual(components.largest_area, 9)
        self.assertEqual(components.total_area, 13)
        self.assertEqual(components.image_size, (10, 10))
        self.assertEqual(len(components.centroids), 2)
        self.assertAlmostEqual(components.centroids[0][0], 0.5)
        self.assertAlmostEqual(components.centroids[0][1], 0.5)
        self.assertIs(components.data_type, DataType.COMPONENTS)
        self.assertTrue(is_data_value(components))
        self.assertIs(data_type_of(components), DataType.COMPONENTS)

    def test_empty_labels(self) -> None:
        components = ComponentsData.from_labels(np.zeros((5, 5), dtype=np.int32))
        self.assertEqual(components.count, 0)
        self.assertEqual(components.areas, ())
        self.assertEqual(components.total_area, 0)
        self.assertEqual(components.largest_area, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
