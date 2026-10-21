"""输出预览节点：万能端口、原样透传，以及节点 / 参数面板的显示策略。"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtGui import QUndoStack
from PySide6.QtWidgets import QApplication, QGroupBox, QLabel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import AppSettings  # noqa: E402
from app.core.data_types import (  # noqa: E402
    ComponentsData,
    ContoursData,
    DataType,
    EdgesData,
    NumberValue,
    data_type_of,
)
from app.core.model import WorkspaceModel  # noqa: E402
from app.core.types import ImageData  # noqa: E402
from app.engine.engine import NodeResult  # noqa: E402
from app.graph.view import GraphView  # noqa: E402
from app.operators.loader import OperatorRegistry  # noqa: E402
from app.operators.spec import (  # noqa: E402
    OperatorError,
    normalize_params,
    output_to_dict,
    validate_inputs,
)
from app.ui.params import ParamPanel  # noqa: E402

_APP: QApplication | None = None


def sample_values() -> dict[str, object]:
    return {
        "image": ImageData(np.full((18, 24, 3), 90, dtype=np.uint8), "input", "BGR"),
        "number": NumberValue(12.5, "均值", "px"),
        "edges": EdgesData(
            (np.array([[0, 0], [10, 10]], dtype=np.int32),),
            image_size=(24, 18),
            label="边缘",
        ),
        "contours": ContoursData(
            (np.array([[0, 0], [10, 0], [10, 10]], dtype=np.int32),),
            image_size=(24, 18),
            label="轮廓",
        ),
        "components": ComponentsData.from_labels(
            np.array([[0, 0, 1], [0, 1, 1]], dtype=np.int32),
            image_size=(3, 2),
            label="连通域",
        ),
    }


def _result(node_id: str, value: object) -> NodeResult:
    return NodeResult(
        node_id,
        {"data": value},
        1.0,
        1,
        port_labels={"data": "任意数据"},
    )


def setUpModule() -> None:  # noqa: N802
    global _APP
    _APP = QApplication.instance() or QApplication([])


def tearDownModule() -> None:  # noqa: N802
    global _APP
    if _APP is not None:
        _APP.processEvents()
        _APP = None


class OutputDisplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = OperatorRegistry.build()
        cls.entry = cls.registry.entry("output.display")
        assert cls.entry is not None
        cls.spec = cls.entry.spec
        cls.params = normalize_params(cls.spec, None)

    def run_operator(self, value: object) -> dict[str, object]:
        validate_inputs(self.spec, {"data": value})
        self.entry.operator.validate({"data": value}, self.params)
        return output_to_dict(
            self.spec,
            self.entry.operator.process(data=value, **self.params),
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

    def test_one_universal_input_and_output(self) -> None:
        self.assertEqual(len(self.spec.inputs), 1)
        self.assertEqual(len(self.spec.outputs), 1)
        self.assertEqual(self.spec.inputs[0].key, "data")
        self.assertEqual(self.spec.outputs[0].key, "data")
        self.assertTrue(self.spec.inputs[0].any_type)
        self.assertTrue(self.spec.outputs[0].any_type)
        self.assertFalse(self.spec.inputs[0].optional)
        self.assertEqual(self.spec.inputs[0].type_text(), "任意数据")

    # ------------------------------------------------------------------ 行为
    def test_every_data_type_passes_through_unchanged(self) -> None:
        for key, value in sample_values().items():
            with self.subTest(kind=key):
                outputs = self.run_operator(value)
                self.assertEqual(set(outputs), {"data"})
                self.assertIs(outputs["data"], value)
                self.assertIs(data_type_of(outputs["data"]), data_type_of(value))

    def test_unknown_input_is_rejected(self) -> None:
        with self.assertRaises(OperatorError):
            validate_inputs(self.spec, {"data": object()})
        with self.assertRaises(OperatorError):
            validate_inputs(self.spec, {})

    def test_only_images_become_the_main_preview(self) -> None:
        values = sample_values()
        for key, value in values.items():
            with self.subTest(kind=key):
                result = _result("node", value)
                if key == "image":
                    self.assertIs(result.preview_image, value)
                else:
                    self.assertIsNone(result.preview_image)

    def test_structured_outputs_never_become_preview_images(self) -> None:
        values = sample_values()
        for key in ("number", "edges", "contours", "components"):
            with self.subTest(kind=key):
                result = _result("node", values[key])
                self.assertIsNone(result.preview_image)
        image_result = _result("node", values["image"])
        self.assertIs(image_result.preview_image, values["image"])

    # ------------------------------------------------------------------ 画布
    def _make_view(self) -> tuple[GraphView, str]:
        temp_dir = tempfile.TemporaryDirectory(prefix="mineimage-output-view-")

        def cleanup_temp_dir() -> None:
            try:
                temp_dir.cleanup()
            except OSError:
                # Windows 上 QSettings 可能仍持有 ini 文件句柄；清理失败不影响测试。
                pass

        self.addCleanup(cleanup_temp_dir)
        settings = AppSettings(Path(temp_dir.name) / "test.ini")
        workspace = WorkspaceModel("输出节点测试")
        view = GraphView(workspace, self.registry, QUndoStack(), settings)
        view.resize(900, 650)
        self.addCleanup(lambda: (view.close(), view.deleteLater()))
        node_id = view.add_operator_at("output.display")
        self.assertIsNotNone(node_id)
        return view, str(node_id)

    def test_node_card_number_and_array_content(self) -> None:
        view, node_id = self._make_view()
        item = view._nodes[node_id]

        item.set_result(_result(node_id, sample_values()["image"]))
        self.assertEqual(item._content_mode, "thumb")
        self.assertIsNotNone(item.thumbnail_pixmap)

        item.set_result(
            NodeResult(
                node_id,
                {"image": sample_values()["image"], "count": NumberValue(7, "数量")},
                1.0,
                1,
            )
        )
        self.assertEqual(item._content_mode, "thumb")
        self.assertIn("数量", item._footer_number)
        self.assertIn("7", item._footer_number)

        item.set_result(_result(node_id, sample_values()["number"]))
        self.assertEqual(item._content_mode, "number")
        self.assertIsNone(item.thumbnail_pixmap)
        self.assertIn("12.5", item._content_value)

        item.set_result(_result(node_id, sample_values()["contours"]))
        self.assertEqual(item._content_mode, "array")
        self.assertIsNone(item.thumbnail_pixmap)
        self.assertEqual(item._content_label, "数组数据")

    def test_ports_are_compatible_with_regular_typed_ports(self) -> None:
        view, node_id = self._make_view()
        gray_id = view.add_operator_at("basic.grayscale")
        self.assertIsNotNone(gray_id)
        assert gray_id is not None
        self.assertTrue(view.ports_compatible(gray_id, "image", node_id, "data"))
        self.assertTrue(view.ports_compatible(node_id, "data", gray_id, "image"))

    # ------------------------------------------------------------------ 参数面板
    def test_param_panel_shows_output_data_for_arrays(self) -> None:
        workspace = WorkspaceModel("输出节点参数测试")
        node = workspace.graph.create_operator_node("output.display", "输出预览", params={})
        panel = ParamPanel()
        self.addCleanup(lambda: panel.deleteLater())
        panel.load_node(node, self.entry.operator)
        panel.set_result(_result(node.id, sample_values()["contours"]))

        output_group = next(
            group for group in panel.findChildren(QGroupBox) if group.title() == "输出数据"
        )
        self.assertFalse(output_group.isHidden())
        texts = [label.text() for label in panel.findChildren(QLabel)]
        self.assertTrue(any("轮廓" in text for text in texts), texts)
        self.assertTrue(any("画布" in text for text in texts), texts)
        self.assertTrue(any("前几个点" in text for text in texts), texts)
        # 结构化数据只输出文字详情，不再渲染成 QLabel 预览图。
        self.assertTrue(all(label.pixmap().isNull() for label in panel.findChildren(QLabel)))

    def test_param_panel_hides_output_data_for_image_only_results(self) -> None:
        workspace = WorkspaceModel("输出节点参数测试")
        node = workspace.graph.create_operator_node("output.display", "输出预览", params={})
        panel = ParamPanel()
        self.addCleanup(lambda: panel.deleteLater())
        panel.load_node(node, self.entry.operator)
        panel.set_result(_result(node.id, sample_values()["image"]))
        output_group = next(
            group for group in panel.findChildren(QGroupBox) if group.title() == "输出数据"
        )
        self.assertTrue(output_group.isHidden())


if __name__ == "__main__":
    unittest.main(verbosity=2)
