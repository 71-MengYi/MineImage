"""输出预览节点：全部数据类型的显示与透传。"""

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
)
from app.core.types import ImageData  # noqa: E402
from app.engine.engine import NodeResult  # noqa: E402
from app.operators.loader import OperatorRegistry  # noqa: E402
from app.operators.spec import (  # noqa: E402
    normalize_params,
    output_to_dict,
    validate_inputs,
)

PASS_KEYS = ("image", "number", "edges", "contours", "components")


def sample_values() -> dict[str, object]:
    image = ImageData(np.full((24, 32, 3), 90, dtype=np.uint8), "input", "BGR")
    return {
        "image": image,
        "number": NumberValue(12.5, "均值", "px"),
        "edges": EdgesData(
            (np.array([[0, 0], [10, 10]], dtype=np.int32),),
            image_size=(32, 24),
            label="边缘",
        ),
        "contours": ContoursData(
            (np.array([[0, 0], [10, 0], [10, 10]], dtype=np.int32),),
            image_size=(32, 24),
            label="轮廓",
        ),
        "components": ComponentsData.from_labels(
            np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32),
            image_size=(3, 2),
            label="连通域",
        ),
    }


class OutputDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = OperatorRegistry.build()
        cls.entry = cls.registry.entry("output.display")
        assert cls.entry is not None
        cls.spec = cls.entry.spec
        cls.params = normalize_params(cls.spec, None)

    def run_operator(self, **inputs: object) -> dict[str, object]:
        resolved = {key: inputs.get(key) for key in PASS_KEYS}
        validate_inputs(self.spec, resolved)
        self.entry.operator.validate(resolved, self.params)
        return output_to_dict(
            self.spec,
            self.entry.operator.process(**resolved, **self.params),
        )

    # ------------------------------------------------------------------ 结构
    def test_registered_under_other_visualization(self) -> None:
        entry = self.registry.entry("output.display")
        self.assertIsNotNone(entry, "缺少输出预览节点 output.display")
        assert entry is not None
        self.assertTrue(entry.builtin)
        self.assertEqual(entry.spec.category, ("其他", "可视化"))
        self.assertEqual(entry.spec.tab, "其他")
        self.assertEqual(entry.spec.groups, ("可视化",))
        self.assertTrue(entry.spec.name)
        self.assertTrue(entry.spec.description)

    def test_ports_cover_every_known_data_type(self) -> None:
        self.assertEqual({port.data_type for port in self.spec.inputs}, set(DataType))
        self.assertEqual({port.data_type for port in self.spec.outputs}, set(DataType))
        self.assertTrue(all(port.optional for port in self.spec.inputs))
        for key in PASS_KEYS:
            self.assertIs(self.spec.output(key).data_type, self.spec.input(key).data_type)
        self.assertIs(self.spec.output("display").data_type, DataType.IMAGE)
        # 自动插入到既有连线时优先选择原样透传的图像端口，而不是显示面板。
        self.assertEqual(self.spec.outputs[0].key, "image")

    # ------------------------------------------------------------------ 行为
    def test_all_inputs_pass_through_unchanged(self) -> None:
        values = sample_values()
        outputs = self.run_operator(**values)
        for key, value in values.items():
            self.assertIs(outputs[key], value, f"{key} 端口必须原样透传同一对象")
        display = outputs["display"]
        self.assertIsInstance(display, ImageData)
        self.assertEqual(display.color_space, "BGR")
        self.assertEqual(display.array.dtype, np.uint8)
        self.assertGreater(display.array.size, 0)
        self.assertEqual(display.array.shape[2], 3)

    def test_display_is_first_output_for_node_preview(self) -> None:
        outputs = self.run_operator(**sample_values())
        self.assertEqual(next(iter(outputs)), "display")
        result = NodeResult(
            "node",
            outputs,
            1.0,
            1,
            port_labels={"display": "显示面板"},
        )
        self.assertIs(result.preview_image, outputs["display"])
        self.assertIn("显示面板", result.summary_text())

    def test_each_data_type_alone_produces_a_display(self) -> None:
        for key, value in sample_values().items():
            with self.subTest(key=key):
                outputs = self.run_operator(**{key: value})
                self.assertEqual(set(outputs), {"display", key})
                self.assertIs(outputs[key], value)
                self.assertIsInstance(outputs["display"], ImageData)
                self.assertGreater(outputs["display"].array.size, 0)

    def test_no_input_still_shows_hint_panel(self) -> None:
        outputs = self.run_operator()
        self.assertEqual(set(outputs), {"display"})
        display = outputs["display"]
        self.assertIsInstance(display, ImageData)
        self.assertGreater(display.array.size, 0)

    def test_structured_values_are_rendered_into_panel(self) -> None:
        plain = self.run_operator()
        with_edges = self.run_operator(edges=sample_values()["edges"])
        # 边缘预览是白线，面板尺寸变化即可说明结构化数据进入了显示路径。
        self.assertNotEqual(plain["display"].array.shape, with_edges["display"].array.shape)
        self.assertGreater(with_edges["display"].array.shape[0], plain["display"].array.shape[0])


if __name__ == "__main__":
    unittest.main(verbosity=2)
