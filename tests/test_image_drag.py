"""图像拖动：缩略图拖出、拖到源节点合并与整个源节点合并。"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt
from PySide6.QtGui import QDropEvent, QMouseEvent, QUndoStack
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from app.config import AppSettings
from app.core.constants import MIME_IMAGE_ITEM
from app.core.model import WorkspaceModel
from app.core.types import ImageData, ImageList
from app.engine.engine import NodeResult
import app.graph.view as graph_view_module
from app.graph import image_drag
from app.graph.image_drag import (
    ImageDragPayload,
    build_mime_data,
    lookup,
    payload_from_mime,
    register,
    release,
    token_from_mime,
)
from app.graph.view import GraphView
from app.operators.spec import OperatorSpec, PortSpec

_APP: QApplication | None = None

BLUR_SPEC = OperatorSpec(
    id="test.blur",
    name="模糊",
    category=("滤波与平滑", "平滑"),
    description="测试用模糊算子。",
    inputs=(PortSpec("image", "图像"),),
    outputs=(PortSpec("image", "图像"),),
)


class FakeRegistry:
    """仅实现 GraphView 需要的 spec 查询接口。"""

    def __init__(self, *specs: OperatorSpec) -> None:
        self._specs = {spec.id: spec for spec in specs}

    def spec(self, op_id: str) -> OperatorSpec | None:
        return self._specs.get(op_id)

    def get(self, op_id: str):
        return None

    def entry(self, op_id: str):
        return None


def _image(value: int = 100, width: int = 24, height: int = 18) -> ImageData:
    return ImageData(np.full((height, width, 3), value, dtype=np.uint8), f"img_{value}")


def setUpModule() -> None:  # noqa: N802
    global _APP
    _APP = QApplication.instance() or QApplication([])


def tearDownModule() -> None:  # noqa: N802
    global _APP
    if _APP is not None:
        _APP.processEvents()
        _APP = None


class ImageDragTestCase(unittest.TestCase):
    def setUp(self) -> None:
        image_drag.clear()
        self._temp_dir = tempfile.TemporaryDirectory(prefix="mineimage-image-drag-")
        self.settings = AppSettings(Path(self._temp_dir.name) / "test.ini")
        self.workspace = WorkspaceModel("图像拖动测试")
        self.graph = self.workspace.graph
        self.undo = QUndoStack()
        self.view = GraphView(
            self.workspace, FakeRegistry(BLUR_SPEC), self.undo, self.settings
        )
        self.view.resize(1000, 700)
        self.view.show()
        self._process()

    def tearDown(self) -> None:
        image_drag.clear()
        self.view.close()
        self.view.deleteLater()
        self._process()
        try:
            self._temp_dir.cleanup()
        except OSError:
            # Windows 上 QSettings 可能仍持有 ini 句柄，清理失败不影响测试。
            pass

    # ------------------------------------------------------------------ 工具
    def _process(self) -> None:
        if _APP is not None:
            _APP.processEvents()

    def _fit(self) -> None:
        self.view.fit_graph()
        self._process()

    def _add_source(self, title: str, count: int = 1, pos=(0.0, 0.0)):
        images = [
            _image(30 + index * 30, 24 + index * 4, 18 + index * 2)
            for index in range(count)
        ]
        paths = [f"C:/tmp/{title}_{index}.png" for index in range(count)]
        node = self.workspace.add_source_images(images, paths, pos=pos, title=title)
        self._process()
        return node

    def _add_operator(self, pos=(0.0, 0.0)):
        node = self.graph.create_operator_node("test.blur", "模糊", pos=pos)
        self._process()
        return node

    def _add_grid_operator(self, count: int = 3):
        node = self._add_operator(pos=(120.0, 90.0))
        images = tuple(
            _image(40 + index * 40, 26 + index * 4, 20 + index * 2)
            for index in range(count)
        )
        result = NodeResult(
            node.id, {"image": ImageList(images)}, 3.0, node.revision
        )
        self.view.notify_result(node.id, result)
        self._process()
        self._fit()
        return node

    def _source_image(self, node, index: int = 0) -> ImageData:
        return self.workspace.images_for_node(node)[index]

    def _payload(
        self,
        node,
        index: int = 0,
        *,
        name: str | None = None,
        path: str | None = None,
    ) -> ImageDragPayload:
        image = self._source_image(node, index)
        image_id = node.source_ids[index]
        return ImageDragPayload(
            image=image.copy(),
            name=image.name if name is None else name,
            path=self.workspace.image_paths.get(image_id) if path is None else path,
            source_node_id=node.id,
            source_index=index,
            source_title=node.display_title(),
        )

    def _tile_view_pos(self, item, index: int) -> QPoint:
        scene_pos = item.mapToScene(item.image_tile_rects()[index].center())
        return self.view.mapFromScene(scene_pos)

    def _send_drag_move(self, start: QPoint, delta: QPoint) -> QPoint:
        target = start + delta
        event = QMouseEvent(
            QEvent.Type.MouseMove,
            QPointF(target),
            QPointF(self.view.viewport().mapToGlobal(target)),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        QApplication.sendEvent(self.view.viewport(), event)
        self._process()
        return target

    # ------------------------------------------------------- 载荷注册表
    def test_payload_registry_register_lookup_release(self) -> None:
        payload = ImageDragPayload(
            image=_image(11),
            name="拖出图",
            path="C:/images/a.png",
            source_node_id="node-1",
            source_index=2,
            source_title="来源节点",
        )
        token = register(payload)
        self.assertTrue(token)
        self.assertIs(lookup(token), payload)
        mime = build_mime_data(token)
        self.assertTrue(mime.hasFormat(MIME_IMAGE_ITEM))
        self.assertEqual(token_from_mime(mime), token)
        self.assertIs(payload_from_mime(mime), payload)

        release(token)
        self.assertIsNone(lookup(token))
        release(token)  # 重复释放安全
        self.assertIsNone(payload_from_mime(mime))

    def test_payload_token_parser_supports_plain_token_and_invalid_data(self) -> None:
        mime = QMimeData()
        mime.setData(MIME_IMAGE_ITEM, b"plain-token")
        self.assertEqual(token_from_mime(mime), "plain-token")
        self.assertIsNone(payload_from_mime(mime))

        mime = QMimeData()
        mime.setData(MIME_IMAGE_ITEM, b"{not-json")
        self.assertEqual(token_from_mime(mime), "{not-json")
        self.assertIsNone(token_from_mime(QMimeData()))
        self.assertIsNone(lookup("missing-token"))

    def test_start_image_drag_builds_payload_and_releases_token(self) -> None:
        source = self._add_source("来源", 2, (0.0, 0.0))
        source_images = self.workspace.images_for_node(source)
        result = NodeResult(
            source.id,
            {"image": ImageList(tuple(source_images))},
            1.0,
            source.revision,
        )
        self.view.notify_result(source.id, result)
        self._process()
        self._fit()
        item = self.view._nodes[source.id]
        self.assertTrue(item.has_image_grid)

        captured: dict[str, object] = {}

        class FakeDrag:
            def __init__(self, parent=None) -> None:
                self.mime = None
                self.pixmap = None

            def setMimeData(self, mime) -> None:  # noqa: N802
                self.mime = mime

            def setPixmap(self, pixmap) -> None:  # noqa: N802
                self.pixmap = pixmap

            def setHotSpot(self, hotspot) -> None:  # noqa: N802
                self.hotspot = hotspot

            def exec(self, action) -> Qt.DropAction:  # noqa: A003
                captured["mime"] = self.mime
                captured["payload"] = image_drag.payload_from_mime(self.mime)
                captured["action"] = action
                return Qt.DropAction.CopyAction

        with patch.object(graph_view_module, "QDrag", FakeDrag):
            self.assertTrue(self.view.start_image_drag(item, 1))

        payload = captured["payload"]
        self.assertIsInstance(payload, ImageDragPayload)
        self.assertEqual(captured["action"], Qt.DropAction.CopyAction)
        self.assertEqual(payload.source_node_id, source.id)
        self.assertEqual(payload.source_index, 1)
        self.assertEqual(payload.source_title, source.display_title())
        self.assertEqual(payload.path, self.workspace.image_paths[source.source_ids[1]])
        self.assertIsNot(payload.image, source_images[1])
        self.assertTrue(np.array_equal(payload.image.array, source_images[1].array))
        # QDrag.exec 返回后 token 已释放，旧的 MIME 数据不能再次取到载荷。
        self.assertIsNotNone(captured["mime"])
        self.assertIsNone(image_drag.payload_from_mime(captured["mime"]))

    def test_start_image_drag_from_single_result_uses_its_image(self) -> None:
        node = self._add_operator(pos=(0.0, 0.0))
        image = _image(66, 28, 18)
        self.view.notify_result(
            node.id, NodeResult(node.id, {"image": image}, 1.0, node.revision)
        )
        self._process()
        self._fit()
        item = self.view._nodes[node.id]
        self.assertEqual(item._content_mode, "thumb")
        captured: dict[str, object] = {}

        class FakeDrag:
            def __init__(self, parent=None) -> None:
                self.mime = None
                self.pixmap = None

            def setMimeData(self, mime) -> None:  # noqa: N802
                self.mime = mime

            def setPixmap(self, pixmap) -> None:  # noqa: N802
                self.pixmap = pixmap

            def setHotSpot(self, hotspot) -> None:  # noqa: N802
                self.hotspot = hotspot

            def exec(self, action) -> Qt.DropAction:  # noqa: A003
                captured["payload"] = image_drag.payload_from_mime(self.mime)
                return Qt.DropAction.CopyAction

        with patch.object(graph_view_module, "QDrag", FakeDrag):
            self.assertTrue(self.view.start_image_drag(item, 0))
        payload = captured["payload"]
        self.assertIsInstance(payload, ImageDragPayload)
        self.assertEqual(payload.source_node_id, node.id)
        self.assertEqual(payload.source_index, 0)
        self.assertFalse(np.shares_memory(payload.image.array, image.array))
        self.assertTrue(np.array_equal(payload.image.array, image.array))

    # -------------------------------------------------- 缩略图拖动 / 点击
    def test_thumbnail_click_does_not_start_image_drag(self) -> None:
        node = self._add_grid_operator(3)
        item = self.view._nodes[node.id]
        self.view.set_selected_node(None)
        calls: list[tuple[object, int]] = []

        def fake_start(item_arg, index_arg) -> bool:
            calls.append((item_arg, index_arg))
            return True

        self.view.start_image_drag = fake_start  # type: ignore[method-assign]
        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            self._tile_view_pos(item, 1),
        )
        self._process()

        self.assertEqual(calls, [])
        self.assertEqual(item.selected_image_index, 1)
        self.assertFalse(item.isSelected())

    def test_thumbnail_drag_past_threshold_starts_image_drag(self) -> None:
        node = self._add_grid_operator(3)
        item = self.view._nodes[node.id]
        self.view.set_selected_node(None)
        calls: list[tuple[object, int]] = []

        def fake_start(item_arg, index_arg) -> bool:
            calls.append((item_arg, index_arg))
            return True

        self.view.start_image_drag = fake_start  # type: ignore[method-assign]
        start = self._tile_view_pos(item, 0)
        node_pos = QPointF(item.pos())
        threshold = QApplication.startDragDistance()
        self.assertGreater(threshold, 0)

        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            start,
        )
        target = self._send_drag_move(start, QPoint(threshold + 8, threshold + 6))
        self.assertEqual(len(calls), 1)
        self.assertIs(calls[0][0], item)
        self.assertEqual(calls[0][1], 0)
        self.assertEqual(item.pos(), node_pos)
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            target,
        )
        self._process()
        self.assertEqual(len(calls), 1)
        self.assertEqual(item.pos(), node_pos)

    def test_thumbnail_move_below_threshold_does_not_start_drag(self) -> None:
        node = self._add_grid_operator(2)
        item = self.view._nodes[node.id]
        self.view.set_selected_node(None)
        calls: list[tuple[object, int]] = []

        def fake_start(item_arg, index_arg) -> bool:
            calls.append((item_arg, index_arg))
            return True

        self.view.start_image_drag = fake_start  # type: ignore[method-assign]
        start = self._tile_view_pos(item, 0)
        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            start,
        )
        target = self._send_drag_move(start, QPoint(2, 1))
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            target,
        )
        self._process()
        self.assertEqual(calls, [])

    def test_single_thumbnail_drag_starts_image_drag(self) -> None:
        node = self._add_operator(pos=(0.0, 0.0))
        image = _image(77, 32, 20)
        self.view.notify_result(
            node.id, NodeResult(node.id, {"image": image}, 1.0, node.revision)
        )
        self._process()
        self._fit()
        item = self.view._nodes[node.id]
        self.assertEqual(item._content_mode, "thumb")
        self.assertIs(item.image_at(0), image)
        calls: list[tuple[object, int]] = []

        def fake_start(item_arg, index_arg) -> bool:
            calls.append((item_arg, index_arg))
            return True

        self.view.start_image_drag = fake_start  # type: ignore[method-assign]
        start = self.view.mapFromScene(item.mapToScene(item._thumb_rect.center()))
        node_pos = QPointF(item.pos())
        threshold = QApplication.startDragDistance()
        QTest.mousePress(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            start,
        )
        target = self._send_drag_move(start, QPoint(threshold + 8, 4))
        self.assertEqual(calls, [(item, 0)])
        self.assertEqual(item.pos(), node_pos)
        QTest.mouseRelease(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            target,
        )
        self._process()
        self.assertFalse(item._thumb_press_active)

    def test_single_thumbnail_click_selects_node(self) -> None:
        node = self._add_operator(pos=(0.0, 0.0))
        image = _image(78, 30, 22)
        self.view.notify_result(
            node.id, NodeResult(node.id, {"image": image}, 1.0, node.revision)
        )
        self._process()
        self._fit()
        item = self.view._nodes[node.id]
        self.view.set_selected_node(None)
        calls: list[tuple[object, int]] = []

        def fake_start(item_arg, index_arg) -> bool:
            calls.append((item_arg, index_arg))
            return True

        self.view.start_image_drag = fake_start  # type: ignore[method-assign]
        point = self.view.mapFromScene(item.mapToScene(item._thumb_rect.center()))
        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            point,
        )
        self._process()
        self.assertEqual(calls, [])
        self.assertTrue(item.isSelected())

    def test_thumbnail_double_click_keeps_existing_behavior(self) -> None:
        node = self._add_grid_operator(3)
        item = self.view._nodes[node.id]
        self.view.set_selected_node(None)
        activated: list[str] = []
        self.view.node_activated.connect(activated.append)
        QTest.mouseDClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            self._tile_view_pos(item, 2),
        )
        self._process()
        self.assertEqual(activated, [])
        self.assertEqual(item.selected_image_index, 2)
        self.assertFalse(item.isSelected())

    def test_node_header_click_and_double_click_keep_existing_behavior(self) -> None:
        node = self._add_grid_operator(3)
        item = self.view._nodes[node.id]
        self.view.set_selected_node(None)
        activated: list[str] = []
        self.view.node_activated.connect(activated.append)
        header_pos = self.view.mapFromScene(
            item.mapToScene(item._header_rect.center())
        )

        QTest.mouseClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            header_pos,
        )
        self._process()
        self.assertTrue(item.isSelected())

        QTest.mouseDClick(
            self.view.viewport(),
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
            header_pos,
        )
        self._process()
        self.assertEqual(activated, [node.id])

    # ------------------------------------------------ 拖到空白 / 非源节点
    def test_handle_image_drop_on_blank_creates_source_and_undo_redo(self) -> None:
        source = self._add_source("来源", 1, (0.0, 0.0))
        source_image = self._source_image(source)
        payload = self._payload(source)
        self._fit()
        created: list[str] = []
        selected: list[str] = []
        self.view.node_created.connect(created.append)
        self.view.selection_changed.connect(selected.append)
        before = set(self.graph.nodes)

        self.assertTrue(self.view.handle_image_drop(QPointF(520.0, 360.0), payload))
        self._process()
        new_ids = [node_id for node_id in self.graph.nodes if node_id not in before]
        self.assertEqual(len(new_ids), 1)
        created_node = self.graph.nodes[new_ids[0]]
        self.assertTrue(created_node.is_source)
        self.assertEqual(created_node.pos, (520.0, 360.0))
        self.assertEqual(created_node.title, payload.name)
        self.assertEqual(created_node.source_path, payload.path)
        self.assertEqual(len(created_node.source_ids), 1)
        self.assertEqual(created, [created_node.id])
        self.assertEqual(selected, [created_node.id])
        self.assertEqual(self.view.selected_node_id(), created_node.id)

        image_id = created_node.source_ids[0]
        new_image = self.workspace.images[image_id]
        self.assertIsNot(new_image, source_image)
        self.assertFalse(np.shares_memory(new_image.array, source_image.array))
        self.assertTrue(np.array_equal(new_image.array, source_image.array))
        self.assertEqual(self.workspace.image_paths[image_id], payload.path)

        # 修改副本不能影响原图。
        new_image.array[:, :, :] = 255
        self.assertFalse(np.array_equal(new_image.array, source_image.array))

        self.undo.undo()
        self._process()
        self.assertNotIn(created_node.id, self.graph.nodes)
        self.assertNotIn(created_node.id, self.view._nodes)
        self.assertIn(image_id, self.workspace.images)

        self.undo.redo()
        self._process()
        self.assertIn(created_node.id, self.graph.nodes)
        self.assertIn(created_node.id, self.view._nodes)
        self.assertEqual(created_node.source_ids, [image_id])

    def test_handle_image_drop_uses_source_title_when_name_missing(self) -> None:
        source = self._add_source("来源标题", 1, (0.0, 0.0))
        payload = self._payload(source, name="", path=None)
        payload = ImageDragPayload(
            image=payload.image,
            name="",
            path=None,
            source_node_id=payload.source_node_id,
            source_index=1,
            source_title="来源标题",
        )
        self._fit()
        self.assertTrue(self.view.handle_image_drop(QPointF(620.0, 420.0), payload))
        node = next(
            item for item in self.graph.nodes.values() if item.id != source.id
        )
        self.assertEqual(node.title, "来源标题 · 第 2 张")

    def test_handle_image_drop_on_non_source_creates_new_source(self) -> None:
        operator = self._add_operator(pos=(0.0, 0.0))
        source = self._add_source("源", 1, (0.0, 320.0))
        payload = self._payload(source)
        self._fit()
        operator_item = self.view._nodes[operator.id]
        before = set(self.graph.nodes)

        self.assertTrue(
            self.view.handle_image_drop(
                operator_item.sceneBoundingRect().center(), payload
            )
        )
        self._process()
        new_ids = [node_id for node_id in self.graph.nodes if node_id not in before]
        self.assertEqual(len(new_ids), 1)
        new_node = self.graph.nodes[new_ids[0]]
        self.assertTrue(new_node.is_source)
        self.assertIn(operator.id, self.graph.nodes)
        self.assertEqual(len(new_node.source_ids), 1)

    # ------------------------------------------------------ 拖到源节点
    def test_handle_image_drop_on_source_merges_and_undo_redo(self) -> None:
        source = self._add_source("甲", 2, (0.0, 0.0))
        target = self._add_source("乙", 1, (0.0, 320.0))
        payload = self._payload(source, 0)
        source_image = payload.image
        self._fit()
        target_item = self.view._nodes[target.id]
        signals: list[str] = []
        selected: list[str] = []
        self.view.source_images_changed.connect(signals.append)
        self.view.selection_changed.connect(selected.append)
        old_target_ids = list(target.source_ids)
        old_source_ids = list(source.source_ids)

        self.assertTrue(
            self.view.handle_image_drop(
                target_item.sceneBoundingRect().center(), payload
            )
        )
        self._process()
        self.assertEqual(len(target.source_ids), len(old_target_ids) + 1)
        new_image_id = target.source_ids[-1]
        new_image = self.workspace.images[new_image_id]
        self.assertIsNot(new_image, source_image)
        self.assertFalse(np.shares_memory(new_image.array, source_image.array))
        self.assertTrue(np.array_equal(new_image.array, source_image.array))
        self.assertEqual(signals, [target.id])
        self.assertEqual(selected, [target.id])
        self.assertEqual(self.view.selected_node_id(), target.id)
        self.assertEqual(source.source_ids, old_source_ids)
        self.assertEqual(len(self.workspace.images), 4)

        new_image.array[:, :, :] = 200
        self.assertFalse(np.array_equal(new_image.array, source_image.array))

        self.undo.undo()
        self._process()
        self.assertEqual(target.source_ids, old_target_ids)
        self.assertIn(new_image_id, self.workspace.images)

        self.undo.redo()
        self._process()
        self.assertEqual(target.source_ids, old_target_ids + [new_image_id])

        # 拖回它自己原来的源节点：不重复追加，只提示。
        messages: list[str] = []
        self.view.status_message.connect(messages.append)
        count_before = len(source.source_ids)
        same_payload = self._payload(source, 1)
        self.assertFalse(
            self.view.handle_image_drop(
                self.view._nodes[source.id].sceneBoundingRect().center(), same_payload
            )
        )
        self._process()
        self.assertEqual(len(source.source_ids), count_before)
        self.assertTrue(any("无需重复合并" in message for message in messages), messages)

    # -------------------------------------------------- 源节点整体合并
    def test_merge_source_nodes_undo_redo(self) -> None:
        source = self._add_source("甲", 2, (0.0, 0.0))
        target = self._add_source("乙", 1, (0.0, 320.0))
        source_ids = list(source.source_ids)
        target_ids = list(target.source_ids)
        self._fit()
        signals: list[str] = []
        selected: list[str] = []
        self.view.source_images_changed.connect(signals.append)
        self.view.selection_changed.connect(selected.append)

        self.assertTrue(self.view.merge_source_nodes(source.id, target.id))
        self._process()
        self.assertEqual(target.source_ids, target_ids + source_ids)
        self.assertNotIn(source.id, self.graph.nodes)
        self.assertNotIn(source.id, self.view._nodes)
        self.assertEqual(signals, [target.id])
        self.assertEqual(selected, [target.id])
        self.assertEqual(self.undo.count(), 1)
        # 直接引用已有图像条目，不复制数组。
        target_images = self.workspace.images_for_node(target)
        for image_id, image in zip(source_ids, target_images[-len(source_ids) :]):
            self.assertIn(image_id, target.source_ids)
            self.assertIs(image, self.workspace.images[image_id])

        self.undo.undo()
        self._process()
        self.assertIn(source.id, self.graph.nodes)
        self.assertIn(source.id, self.view._nodes)
        self.assertEqual(list(source.source_ids), source_ids)
        self.assertEqual(list(target.source_ids), target_ids)

        self.undo.redo()
        self._process()
        self.assertNotIn(source.id, self.graph.nodes)
        self.assertEqual(target.source_ids, target_ids + source_ids)

    def test_merge_source_nodes_rejects_outgoing_connection(self) -> None:
        source = self._add_source("甲", 1, (0.0, 0.0))
        target = self._add_source("乙", 1, (0.0, 320.0))
        operator = self._add_operator(pos=(360.0, 0.0))
        connection = self.graph.add_connection(
            source.id, "image", operator.id, "image"
        )
        self._fit()
        messages: list[str] = []
        self.view.status_message.connect(messages.append)

        self.assertFalse(self.view.merge_source_nodes(source.id, target.id))
        self._process()
        self.assertIn(source.id, self.graph.nodes)
        self.assertIn(target.id, self.graph.nodes)
        self.assertIn(connection.id, self.graph.connections)
        self.assertEqual(len(source.source_ids), 1)
        self.assertEqual(len(target.source_ids), 1)
        self.assertTrue(any("断开下游连线" in message for message in messages), messages)

    def test_dragging_single_source_node_onto_another_merges(self) -> None:
        source = self._add_source("甲", 2, (0.0, 0.0))
        target = self._add_source("乙", 1, (0.0, 320.0))
        source_ids = list(source.source_ids)
        target_ids = list(target.source_ids)
        self._fit()
        source_item = self.view._nodes[source.id]
        target_item = self.view._nodes[target.id]
        self.view.set_selected_node(None)
        source_item.setSelected(True)
        signals: list[str] = []
        selected: list[str] = []
        self.view.source_images_changed.connect(signals.append)
        self.view.selection_changed.connect(selected.append)

        self.view.begin_move_tracking()
        source_item.setPos(target_item.pos() + QPointF(20.0, 20.0))
        self._process()
        self.assertIs(self.view._node_merge_target, target_item)
        self.assertTrue(target_item.image_drop_target)

        self.view.finish_move_tracking()
        self._process()
        self.assertNotIn(source.id, self.graph.nodes)
        self.assertEqual(target.source_ids, target_ids + source_ids)
        self.assertEqual(signals, [target.id])
        self.assertEqual(selected, [target.id])
        self.assertFalse(target_item.image_drop_target)
        self.assertEqual(self.undo.count(), 1)

        self.undo.undo()
        self._process()
        self.assertIn(source.id, self.graph.nodes)
        self.assertEqual(list(source.source_ids), source_ids)
        self.assertEqual(list(target.source_ids), target_ids)

    def test_dragging_source_with_outgoing_connection_keeps_nodes(self) -> None:
        source = self._add_source("甲", 1, (0.0, 0.0))
        target = self._add_source("乙", 1, (0.0, 460.0))
        operator = self._add_operator(pos=(360.0, 0.0))
        connection = self.graph.add_connection(
            source.id, "image", operator.id, "image"
        )
        self._fit()
        source_item = self.view._nodes[source.id]
        target_item = self.view._nodes[target.id]
        self.view.set_selected_node(None)
        source_item.setSelected(True)
        messages: list[str] = []
        self.view.status_message.connect(messages.append)

        self.view.begin_move_tracking()
        source_item.setPos(target_item.pos() + QPointF(20.0, 20.0))
        self._process()
        self.assertIsNone(self.view._node_merge_target)
        self.assertIs(self.view._node_merge_refused, target_item)
        self.assertFalse(target_item.image_drop_target)

        self.view.finish_move_tracking()
        self._process()
        self.assertIn(source.id, self.graph.nodes)
        self.assertIn(target.id, self.graph.nodes)
        self.assertIn(connection.id, self.graph.connections)
        self.assertEqual(len(target.source_ids), 1)
        self.assertTrue(any("断开下游连线" in message for message in messages), messages)
        self.assertEqual(self.undo.count(), 1)

    def test_multi_select_drag_does_not_merge_source_nodes(self) -> None:
        first = self._add_source("甲", 1, (0.0, 0.0))
        second = self._add_source("乙", 1, (0.0, 320.0))
        self._fit()
        first_item = self.view._nodes[first.id]
        second_item = self.view._nodes[second.id]
        self.view.set_selected_node(None)
        first_item.setSelected(True)
        second_item.setSelected(True)

        self.view.begin_move_tracking()
        first_item.setPos(second_item.pos() + QPointF(20.0, 20.0))
        self.view.finish_move_tracking()
        self._process()
        self.assertIn(first.id, self.graph.nodes)
        self.assertIn(second.id, self.graph.nodes)
        self.assertEqual(len(first.source_ids), 1)

    # -------------------------------------------------- MIME 集成
    def test_mime_image_item_drop_creates_source(self) -> None:
        source = self._add_source("来源", 1, (0.0, 0.0))
        payload = self._payload(source)
        token = register(payload)
        try:
            mime = build_mime_data(token)
            point = self.view.mapFromScene(QPointF(600.0, 480.0))
            before = set(self.graph.nodes)
            event = QDropEvent(
                QPointF(point),
                Qt.DropAction.CopyAction,
                mime,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
            self.view.dropEvent(event)
            self._process()
        finally:
            release(token)
        self.assertTrue(event.isAccepted())
        new_ids = [node_id for node_id in self.graph.nodes if node_id not in before]
        self.assertEqual(len(new_ids), 1)
        self.assertTrue(self.graph.nodes[new_ids[0]].is_source)

    def test_mime_image_item_invalid_token_reports_status(self) -> None:
        self._fit()
        messages: list[str] = []
        self.view.status_message.connect(messages.append)
        mime = build_mime_data("missing-token")
        point = self.view.mapFromScene(QPointF(600.0, 480.0))
        before = set(self.graph.nodes)
        event = QDropEvent(
            QPointF(point),
            Qt.DropAction.CopyAction,
            mime,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        self.view.dropEvent(event)
        self._process()
        self.assertTrue(event.isAccepted())
        self.assertEqual(set(self.graph.nodes), before)
        self.assertTrue(any("失效" in message for message in messages), messages)


if __name__ == "__main__":
    unittest.main(verbosity=2)
