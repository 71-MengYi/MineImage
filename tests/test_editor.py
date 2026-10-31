"""编辑器探测 / 启动、配置项与「新建算子文件」基础能力的测试。

覆盖三个新模块：:mod:`app.core.editor`、:mod:`app.core.operator_files`、
:mod:`app.operators.template`，以及 :class:`app.config.AppSettings` 的新增配置项。
"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
import unittest
import uuid
from pathlib import Path
from unittest import mock

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import AppSettings  # noqa: E402
from app.core import editor  # noqa: E402
from app.core.editor import (  # noqa: E402
    AUTO_EDITOR,
    CUSTOM_EDITOR_PREFIX,
    SYSTEM_EDITOR,
    EditorInfo,
    detect_editors,
    launch_editor,
    resolve_editor,
)
from app.core.operator_files import (  # noqa: E402
    MoveOutcome,
    category_directory,
    create_category_dir,
    create_operator_file,
    delete_operator_file,
    move_category_dir,
    move_operator_file,
    normalize_category,
    rename_operator_file,
    sanitize_component,
    update_generated_operator_name,
)
from app.operators.loader import OperatorRegistry  # noqa: E402
from app.operators.template import (  # noqa: E402
    UI_CREATED_MARKER,
    render_operator_template,
)


class _TempDirTestCase(unittest.TestCase):
    """在仓库 ``.test_tmp/`` 下创建临时目录（测试结束自动删除）。"""

    def setUp(self) -> None:
        self.tmp = ROOT / ".test_tmp" / f"architect_{uuid.uuid4().hex[:10]}"
        self.tmp.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def make_file(self, name: str, content: str = "VALUE = 1\n") -> Path:
        path = self.tmp / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        return path


class EditorDetectionTests(_TempDirTestCase):
    def _detect(self, mapping: dict[str, str]) -> list[EditorInfo]:
        def fake_which(command: str) -> str | None:
            return mapping.get(command)

        with (
            mock.patch.object(editor.shutil, "which", side_effect=fake_which),
            mock.patch.object(editor, "_existing_file", return_value=None),
            mock.patch.object(editor, "_glob_paths", return_value=[]),
        ):
            return detect_editors()

    def test_detect_returns_known_editors_in_priority_order(self) -> None:
        found = self._detect(
            {
                "code": r"C:\Tools\Code.exe",
                "pycharm": r"C:\Tools\pycharm64.exe",
                "subl": r"C:\Tools\sublime_text.exe",
            }
        )
        self.assertEqual([item.key for item in found], ["vscode", "pycharm", "sublime"])
        self.assertEqual(found[0].name, "Visual Studio Code")
        self.assertEqual(found[0].executable, r"C:\Tools\Code.exe")

    def test_detect_dedupes_same_executable(self) -> None:
        found = self._detect({"code": r"C:\Tools\Code.exe", "cursor": r"C:\Tools\Code.exe"})
        self.assertEqual([item.key for item in found], ["vscode"])

    def test_detect_returns_empty_when_nothing_installed(self) -> None:
        self.assertEqual(self._detect({}), [])


class EditorResolveTests(unittest.TestCase):
    def test_auto_uses_first_detected_editor(self) -> None:
        detected = [EditorInfo("vscode", "Visual Studio Code", "/fake/code")]
        with mock.patch.object(editor, "detect_editors", return_value=detected):
            self.assertEqual(resolve_editor(AUTO_EDITOR), detected[0])
            self.assertEqual(resolve_editor(""), detected[0])

    def test_auto_returns_none_without_editors(self) -> None:
        with mock.patch.object(editor, "detect_editors", return_value=[]):
            self.assertIsNone(resolve_editor(AUTO_EDITOR))

    def test_system_editor_returns_none(self) -> None:
        with mock.patch.object(editor, "detect_editors", side_effect=AssertionError):
            self.assertIsNone(resolve_editor(SYSTEM_EDITOR))

    def test_key_lookup_uses_detection(self) -> None:
        detected = [
            EditorInfo("vscode", "Visual Studio Code", "/fake/code"),
            EditorInfo("pycharm", "PyCharm", "/fake/pycharm"),
        ]
        with mock.patch.object(editor, "detect_editors", return_value=detected):
            self.assertEqual(resolve_editor("pycharm"), detected[1])
            self.assertIsNone(resolve_editor("vscode-insiders"))

    def test_custom_path_returns_editor_info(self) -> None:
        executable = self._make_executable("MyEditor.exe")
        value = f"{CUSTOM_EDITOR_PREFIX}{executable}"
        info = resolve_editor(value)
        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(info.key, value)
        self.assertEqual(Path(info.executable), executable)
        self.assertEqual(info.name, executable.name)

    def test_custom_path_missing_returns_none(self) -> None:
        self.assertIsNone(resolve_editor(f"{CUSTOM_EDITOR_PREFIX}{self._missing()}"))

    def test_custom_path_accepts_quotes(self) -> None:
        executable = self._make_executable("Quoted Editor.exe")
        info = resolve_editor(f'{CUSTOM_EDITOR_PREFIX}"{executable}"')
        self.assertIsNotNone(info)
        assert info is not None
        self.assertEqual(Path(info.executable), executable)

    def _make_executable(self, name: str) -> Path:
        folder = ROOT / ".test_tmp" / f"architect_exe_{uuid.uuid4().hex[:10]}"
        folder.mkdir(parents=True, exist_ok=True)
        self.addCleanup(shutil.rmtree, folder, ignore_errors=True)
        path = folder / name
        path.write_text("", encoding="utf-8")
        return path

    def _missing(self) -> Path:
        return ROOT / ".test_tmp" / f"missing_{uuid.uuid4().hex[:10]}" / "nope.exe"


class EditorLaunchTests(_TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.target = self.make_file("operator.py", "VALUE = 1\n")
        self.executable = self.make_file("Editor.exe", "")

    def test_launch_custom_editor_starts_process(self) -> None:
        with mock.patch.object(editor.subprocess, "Popen") as popen:
            ok, message = launch_editor(self.target, f"{CUSTOM_EDITOR_PREFIX}{self.executable}")
        self.assertTrue(ok, message)
        popen.assert_called_once()
        command = popen.call_args.args[0]
        self.assertEqual(command, [str(self.executable), str(self.target)])
        self.assertIn(self.target.name, message)

    def test_launch_cmd_wrapper_uses_cmd_exe_start(self) -> None:
        wrapper = self.make_file("code.cmd", "@echo off\n")
        with (
            mock.patch.object(editor.sys, "platform", "win32"),
            mock.patch.object(editor.subprocess, "Popen") as popen,
        ):
            ok, message = launch_editor(self.target, f"{CUSTOM_EDITOR_PREFIX}{wrapper}")
        self.assertTrue(ok, message)
        command = popen.call_args.args[0]
        self.assertTrue(command[0].lower().endswith("cmd.exe"))
        self.assertEqual(command[1:4], ["/c", "start", ""])
        self.assertEqual(command[4:], [str(wrapper), str(self.target)])

    def test_launch_system_uses_desktop_services(self) -> None:
        with mock.patch.object(editor, "QDesktopServices") as desktop:
            desktop.openUrl.return_value = True
            ok, message = launch_editor(self.target, SYSTEM_EDITOR)
        self.assertTrue(ok, message)
        desktop.openUrl.assert_called_once()
        url = desktop.openUrl.call_args.args[0]
        self.assertEqual(Path(url.toLocalFile()), self.target.resolve())

    def test_launch_auto_falls_back_to_system_without_editor(self) -> None:
        with (
            mock.patch.object(editor, "detect_editors", return_value=[]),
            mock.patch.object(editor, "QDesktopServices") as desktop,
        ):
            desktop.openUrl.return_value = True
            ok, message = launch_editor(self.target, AUTO_EDITOR)
        self.assertTrue(ok, message)
        desktop.openUrl.assert_called_once()

    def test_launch_unknown_editor_reports_error(self) -> None:
        with mock.patch.object(editor.subprocess, "Popen") as popen:
            ok, message = launch_editor(self.target, "no-such-editor")
        self.assertFalse(ok)
        self.assertIn("未找到可用的编辑器", message)
        popen.assert_not_called()

    def test_launch_popen_failure_reports_error(self) -> None:
        with mock.patch.object(editor.subprocess, "Popen", side_effect=OSError("拒绝访问")):
            ok, message = launch_editor(self.target, f"{CUSTOM_EDITOR_PREFIX}{self.executable}")
        self.assertFalse(ok)
        self.assertIn("无法启动编辑器", message)

    def test_launch_missing_target_reports_error(self) -> None:
        ok, message = launch_editor(self.tmp / "missing.py", SYSTEM_EDITOR)
        self.assertFalse(ok)
        self.assertIn("不存在", message)


class SettingsEditorTests(_TempDirTestCase):
    def test_default_editor_defaults_to_auto(self) -> None:
        settings = AppSettings(self.tmp / "MineImage.ini")
        self.assertEqual(settings.default_editor, AUTO_EDITOR)

    def test_default_editor_roundtrip(self) -> None:
        config = self.tmp / "MineImage.ini"
        settings = AppSettings(config)
        settings.default_editor = f"{CUSTOM_EDITOR_PREFIX}C:/Tools/Code.exe"
        settings.sync()
        reloaded = AppSettings(config)
        self.assertEqual(reloaded.default_editor, f"{CUSTOM_EDITOR_PREFIX}C:/Tools/Code.exe")
        settings.default_editor = "   "
        self.assertEqual(settings.default_editor, AUTO_EDITOR)

    def test_watch_operator_dirs_defaults_to_true(self) -> None:
        settings = AppSettings(self.tmp / "MineImage.ini")
        self.assertTrue(settings.watch_operator_dirs)
        settings.watch_operator_dirs = False
        self.assertFalse(settings.watch_operator_dirs)


class OperatorFileHelperTests(_TempDirTestCase):
    def test_sanitize_component_cleans_invalid_characters(self) -> None:
        cleaned = sanitize_component(' 滤波: 测试? ')
        self.assertEqual(cleaned, "滤波_ 测试_")
        self.assertEqual(sanitize_component("   "), "新建算子")
        self.assertEqual(sanitize_component("_hidden"), "hidden")
        self.assertEqual(sanitize_component("con"), "con_")

    def test_normalize_category_accepts_str_and_tuple(self) -> None:
        self.assertEqual(normalize_category("基础调整/颜色"), ("基础调整", "颜色"))
        self.assertEqual(normalize_category((" 基础调整 ", "")), ("基础调整",))
        self.assertEqual(normalize_category(None), ())

    def test_category_directory_matches_normalized_parts(self) -> None:
        root = self.tmp / "operators"
        expected = root.resolve() / "基础调整" / "颜色"
        self.assertEqual(category_directory(root, ("基础调整", "颜色")), expected)
        self.assertEqual(category_directory(root, "基础调整/颜色"), expected)
        self.assertEqual(category_directory(root, ()), root.resolve())


class CreateOperatorFileTests(_TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.root = self.tmp / "operators"

    def _registry(self) -> OperatorRegistry:
        return OperatorRegistry.build([str(self.root)], include_builtin=False)

    def test_create_default_operator_is_loadable(self) -> None:
        created = create_operator_file(self.root, ())
        self.assertEqual(created.path.parent, self.root.resolve())
        self.assertEqual(created.path.name, "新建算子.py")
        self.assertEqual(created.display_name, "新建算子")
        self.assertEqual(created.category, ("其他",))
        self.assertRegex(created.operator_id, r"^user\.[a-z0-9_]+_[0-9a-f]{6}$")
        content = created.path.read_text(encoding="utf-8")
        self.assertIn(f"{UI_CREATED_MARKER} = True", content)

        registry = self._registry()
        self.assertEqual(registry.errors, [])
        entry = registry.entry(created.operator_id)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertTrue(entry.ui_created)
        self.assertEqual(entry.display_name, "新建算子")
        self.assertEqual(entry.spec.name, "新建算子")
        self.assertEqual(entry.spec.category, ("其他",))

    def test_create_in_category_and_dedupe_names(self) -> None:
        first = create_operator_file(self.root, ("基础调整", "颜色"))
        second = create_operator_file(self.root, "基础调整/颜色")
        self.assertEqual(first.path.parent, self.root.resolve() / "基础调整" / "颜色")
        self.assertEqual(second.path.parent, first.path.parent)
        self.assertEqual(second.path.name, "新建算子 2.py")
        self.assertNotEqual(first.operator_id, second.operator_id)
        self.assertEqual(second.category, ("基础调整", "颜色"))

        registry = self._registry()
        self.assertEqual(registry.spec(first.operator_id).category, ("基础调整", "颜色"))
        self.assertEqual(registry.spec(second.operator_id).name, "新建算子 2")

    def test_create_with_custom_name_sanitizes_file_name(self) -> None:
        created = create_operator_file(self.root, (), name='我的: 算子?')
        self.assertEqual(created.display_name, "我的_ 算子_")
        self.assertEqual(created.path.name, "我的_ 算子_.py")


class CreateCategoryDirTests(_TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.root = self.tmp / "operators"

    def test_create_top_level_and_nested_category(self) -> None:
        top = create_category_dir(self.root, None, "我的分类")
        self.assertTrue(top.is_dir())
        self.assertEqual(top, self.root.resolve() / "我的分类")
        nested = create_category_dir(self.root, ("我的分类",), "子分类")
        self.assertEqual(nested, self.root.resolve() / "我的分类" / "子分类")

    def test_duplicate_category_raises(self) -> None:
        create_category_dir(self.root, (), "重复分类")
        with self.assertRaises(FileExistsError):
            create_category_dir(self.root, (), "重复分类")

    def test_invalid_category_name_raises(self) -> None:
        for name in ("", "  ", "a/b", "a\\b", "bad:name", ".hidden", "_private", "CON"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                create_category_dir(self.root, (), name)


class RenameDeleteOperatorTests(_TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.root = self.tmp / "operators"

    def test_rename_updates_generated_template_name(self) -> None:
        created = create_operator_file(self.root, ("其他",))
        renamed = rename_operator_file(created.path, "改名后的算子")
        self.assertFalse(created.path.exists())
        self.assertTrue(renamed.is_file())
        self.assertEqual(renamed.name, "改名后的算子.py")
        content = renamed.read_text(encoding="utf-8")
        self.assertIn("name='改名后的算子',", content)

        registry = OperatorRegistry.build([str(self.root)], include_builtin=False)
        entry = registry.entry(created.operator_id)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.display_name, "改名后的算子")
        self.assertEqual(entry.spec.name, "改名后的算子")

    def test_rename_manual_operator_keeps_declared_name(self) -> None:
        manual = self.make_file(
            "operators/manual.py",
            textwrap.dedent(
                """\
                from app.operators.spec import Operator, OperatorSpec


                class Manual(Operator):
                    spec = OperatorSpec(id="manual.one", name="手写算子")

                    def process(self, image=None, **_):
                        return image.array if image is not None else None


                OPERATOR = Manual
                """
            ),
        )
        original = manual.read_text(encoding="utf-8")
        renamed = rename_operator_file(manual, "另一个名字")
        self.assertEqual(renamed.read_text(encoding="utf-8"), original)
        registry = OperatorRegistry.build([str(self.root)], include_builtin=False)
        entry = registry.entry("manual.one")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertFalse(entry.ui_created)
        self.assertEqual(entry.display_name, "手写算子")

    def test_rename_same_name_is_noop(self) -> None:
        created = create_operator_file(self.root, ())
        self.assertEqual(rename_operator_file(created.path, created.path.stem), created.path)

    def test_rename_rejects_invalid_names(self) -> None:
        created = create_operator_file(self.root, ())
        for name in (
            "",
            "  ",
            "a/b",
            "a\\b",
            "bad:name",
            " leading",
            "trailing ",
            ".start",
            "end.",
            "_hidden",
            "CON",
        ):
            with self.subTest(name=name), self.assertRaises(ValueError):
                rename_operator_file(created.path, name)

    def test_rename_to_existing_name_raises(self) -> None:
        first = create_operator_file(self.root, ())
        second = create_operator_file(self.root, ())
        with self.assertRaises(FileExistsError):
            rename_operator_file(first.path, second.path.stem)
        self.assertTrue(first.path.exists())

    def test_rename_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            rename_operator_file(self.tmp / "operators" / "missing.py", "新名字")

    def test_delete_removes_file(self) -> None:
        created = create_operator_file(self.root, ())
        delete_operator_file(created.path)
        self.assertFalse(created.path.exists())
        with self.assertRaises(FileNotFoundError):
            delete_operator_file(created.path)


class MoveOperatorFileTests(_TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.root = self.tmp / "operators"

    def _registry(self) -> OperatorRegistry:
        return OperatorRegistry.build([str(self.root)], include_builtin=False)

    def test_move_generated_operator_updates_file_and_category(self) -> None:
        created = create_operator_file(self.root, ("源分类", "子分类"))
        outcome = move_operator_file(
            self.root,
            created.path,
            ("目标分类", "深层"),
            current_category=("源分类", "子分类"),
        )
        self.assertIsInstance(outcome, MoveOutcome)
        expected = (
            self.root.resolve()
            / "目标分类"
            / "深层"
            / created.path.name
        )
        self.assertEqual(outcome.path, expected)
        self.assertFalse(created.path.exists())
        self.assertTrue(expected.is_file())
        content = expected.read_text(encoding="utf-8")
        self.assertIn("category=('目标分类', '深层')", content)
        self.assertNotIn("category=('源分类', '子分类')", content)
        self.assertEqual(outcome.warnings, ())

        entry = self._registry().entry(created.operator_id)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.spec.category, ("目标分类", "深层"))

    def test_move_manual_literal_category_is_synced(self) -> None:
        manual = self.make_file(
            "operators/源分类/manual.py",
            textwrap.dedent(
                """\
                from app.operators.spec import Operator, OperatorSpec


                class Manual(Operator):
                    spec = OperatorSpec(
                        id="manual.move_me",
                        name="手写算子",
                        category=("源分类",),
                    )

                    def process(self, image=None, **_):
                        return image.array if image is not None else None


                OPERATOR = Manual
                """
            ),
        )
        outcome = move_operator_file(
            self.root, manual, ("目标分类",), current_category=("源分类",)
        )
        self.assertTrue(outcome.path.is_file())
        content = outcome.path.read_text(encoding="utf-8")
        self.assertIn("category=('目标分类',)", content)
        entry = self._registry().entry("manual.move_me")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.spec.category, ("目标分类",))

    def test_move_operator_to_root_rewrites_empty_category(self) -> None:
        created = create_operator_file(self.root, ("源分类",))
        outcome = move_operator_file(
            self.root, created.path, (), current_category=("源分类",)
        )
        self.assertEqual(outcome.path, self.root.resolve() / created.path.name)
        content = outcome.path.read_text(encoding="utf-8")
        self.assertIn("category=(),", content)
        entry = self._registry().entry(created.operator_id)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.spec.category, ("其他",))

    def test_move_legacy_operator_without_category_allows_move(self) -> None:
        legacy = self.make_file(
            "operators/legacy.py",
            textwrap.dedent(
                """\
                from app.operators.spec import Operator, OperatorSpec


                class Legacy(Operator):
                    spec = OperatorSpec(id="legacy.one", name="旧算子")

                    def process(self, image=None, **_):
                        return image.array if image is not None else None


                OPERATOR = Legacy
                """
            ),
        )
        entry = self._registry().entry("legacy.one")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.spec.category, ("其他",))

        outcome = move_operator_file(
            self.root, legacy, ("新分类",), current_category=("其他",)
        )
        self.assertEqual(outcome.path, self.root.resolve() / "新分类" / legacy.name)
        self.assertTrue(outcome.path.is_file())
        moved_entry = self._registry().entry("legacy.one")
        self.assertIsNotNone(moved_entry)
        assert moved_entry is not None
        self.assertEqual(moved_entry.spec.category, ("新分类",))

    def test_move_operator_out_of_root_category_again(self) -> None:
        created = create_operator_file(self.root, ("源分类",))
        first = move_operator_file(
            self.root, created.path, (), current_category=("源分类",)
        )
        self.assertIn("category=(),", first.path.read_text(encoding="utf-8"))

        second = move_operator_file(
            self.root, first.path, ("目标分类",), current_category=("其他",)
        )
        self.assertEqual(
            second.path, self.root.resolve() / "目标分类" / created.path.name
        )
        content = second.path.read_text(encoding="utf-8")
        self.assertIn("category=('目标分类',)", content)
        entry = self._registry().entry(created.operator_id)
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertEqual(entry.spec.category, ("目标分类",))

    def test_move_same_directory_is_noop(self) -> None:
        created = create_operator_file(self.root, ("源分类",))
        before = created.path.read_bytes()
        outcome = move_operator_file(
            self.root, created.path, ("源分类",), current_category=("源分类",)
        )
        self.assertEqual(outcome.path, created.path.resolve())
        self.assertEqual(created.path.read_bytes(), before)

    def test_move_to_existing_name_raises_and_keeps_source(self) -> None:
        create_operator_file(self.root, ("目标分类",))
        created = create_operator_file(self.root, ("源分类",))
        with self.assertRaises(FileExistsError):
            move_operator_file(
                self.root, created.path, ("目标分类",), current_category=("源分类",)
            )
        self.assertTrue(created.path.is_file())

    def test_move_outside_root_raises(self) -> None:
        outside = self.make_file("outside.py", "VALUE = 1\n")
        with self.assertRaises(ValueError):
            move_operator_file(self.root, outside, ("目标分类",), current_category=())

    def test_move_dynamic_category_raises_before_moving(self) -> None:
        manual = self.make_file(
            "operators/源分类/dynamic.py",
            textwrap.dedent(
                """\
                from app.operators.spec import Operator, OperatorSpec

                CATEGORY = ("源分类",)


                class Manual(Operator):
                    spec = OperatorSpec(id="manual.dynamic", name="动态", category=CATEGORY)

                    def process(self, image=None, **_):
                        return image.array if image is not None else None


                OPERATOR = Manual
                """
            ),
        )
        with self.assertRaises(ValueError) as context:
            move_operator_file(
                self.root, manual, ("目标分类",), current_category=("源分类",)
            )
        self.assertIn("category", str(context.exception))
        self.assertTrue(manual.is_file())
        self.assertFalse((self.root / "目标分类" / manual.name).exists())

    def test_move_missing_file_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            move_operator_file(
                self.root,
                self.root / "missing.py",
                ("目标分类",),
                current_category=(),
            )


class MoveCategoryDirTests(_TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.root = self.tmp / "operators"

    def _registry(self) -> OperatorRegistry:
        return OperatorRegistry.build([str(self.root)], include_builtin=False)

    def test_move_category_moves_tree_and_remaps_categories(self) -> None:
        top = create_operator_file(self.root, ("旧分类",))
        nested = create_operator_file(self.root, ("旧分类", "子分类"))
        outcome = move_category_dir(self.root, self.root / "旧分类", ("新父分类",))
        expected = self.root.resolve() / "新父分类" / "旧分类"
        self.assertEqual(outcome.path, expected)
        self.assertFalse((self.root / "旧分类").exists())
        self.assertTrue((expected / top.path.name).is_file())
        self.assertTrue((expected / "子分类" / nested.path.name).is_file())
        self.assertEqual(outcome.warnings, ())

        registry = self._registry()
        top_entry = registry.entry(top.operator_id)
        nested_entry = registry.entry(nested.operator_id)
        self.assertIsNotNone(top_entry)
        self.assertIsNotNone(nested_entry)
        assert top_entry is not None and nested_entry is not None
        self.assertEqual(top_entry.spec.category, ("新父分类", "旧分类"))
        self.assertEqual(
            nested_entry.spec.category, ("新父分类", "旧分类", "子分类")
        )

    def test_move_empty_category(self) -> None:
        folder = create_category_dir(self.root, None, "空分类")
        outcome = move_category_dir(self.root, folder, ("父分类",))
        self.assertTrue(outcome.path.is_dir())
        self.assertTrue((self.root / "父分类" / "空分类").is_dir())

    def test_move_category_same_parent_is_noop(self) -> None:
        folder = create_category_dir(self.root, None, "分类")
        outcome = move_category_dir(self.root, folder, ())
        self.assertEqual(outcome.path, folder.resolve())

    def test_move_category_into_descendant_raises(self) -> None:
        create_category_dir(self.root, None, "父")
        create_category_dir(self.root, ("父",), "子")
        with self.assertRaises(ValueError):
            move_category_dir(self.root, self.root / "父", ("父", "子"))
        self.assertTrue((self.root / "父" / "子").is_dir())

    def test_move_category_to_existing_name_raises(self) -> None:
        create_category_dir(self.root, ("目标",), "分类")
        source = create_category_dir(self.root, None, "分类")
        with self.assertRaises(FileExistsError):
            move_category_dir(self.root, source, ("目标",))
        self.assertTrue(source.is_dir())

    def test_move_custom_and_hidden_directories_raise(self) -> None:
        custom = self.root / "custom"
        custom.mkdir(parents=True)
        hidden = self.root / "_hidden"
        hidden.mkdir(parents=True)
        for folder in (custom, hidden):
            with self.subTest(folder=folder.name), self.assertRaises(ValueError):
                move_category_dir(self.root, folder, ("目标",))

    def test_move_root_or_missing_directory_raises(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        with self.assertRaises(ValueError):
            move_category_dir(self.root, self.root, ("目标",))
        with self.assertRaises(FileNotFoundError):
            move_category_dir(self.root, self.root / "missing", ("目标",))

    def test_move_category_keeps_dynamic_category_file_and_warns(self) -> None:
        manual = self.make_file(
            "operators/旧分类/dynamic.py",
            textwrap.dedent(
                """\
                from app.operators.spec import Operator, OperatorSpec

                CATEGORY = ("旧分类",)


                class Manual(Operator):
                    spec = OperatorSpec(id="manual.dynamic", name="动态", category=CATEGORY)

                    def process(self, image=None, **_):
                        return image.array if image is not None else None


                OPERATOR = Manual
                """
            ),
        )
        outcome = move_category_dir(self.root, self.root / "旧分类", ("新父",))
        moved = outcome.path / manual.name
        self.assertTrue(moved.is_file())
        self.assertEqual(len(outcome.warnings), 1)
        self.assertIn("dynamic.py", outcome.warnings[0])
        self.assertIn("CATEGORY", moved.read_text(encoding="utf-8"))


class UpdateGeneratedNameTests(_TempDirTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.root = self.tmp / "operators"

    def test_updates_name_when_it_still_matches_old_stem(self) -> None:
        created = create_operator_file(self.root, (), name="旧名称")
        target = created.path.with_name("新名称.py")
        created.path.rename(target)
        update_generated_operator_name(target, "旧名称", "新名称")
        content = target.read_text(encoding="utf-8")
        self.assertIn("name='新名称',", content)
        self.assertNotIn("name='旧名称',", content)

    def test_idempotent_and_skips_when_name_changed(self) -> None:
        created = create_operator_file(self.root, (), name="旧名称")
        target = created.path.with_name("新名称.py")
        created.path.rename(target)
        update_generated_operator_name(target, "旧名称", "新名称")
        once = target.read_text(encoding="utf-8")
        update_generated_operator_name(target, "旧名称", "又一名称")
        self.assertEqual(target.read_text(encoding="utf-8"), once)

    def test_skips_files_without_marker(self) -> None:
        manual = self.make_file(
            "operators/manual.py",
            "from app.operators.spec import Operator, OperatorSpec\nVALUE = 1\n",
        )
        before = manual.read_text(encoding="utf-8")
        update_generated_operator_name(manual, "manual", "renamed")
        self.assertEqual(manual.read_text(encoding="utf-8"), before)

    def test_skips_missing_file(self) -> None:
        update_generated_operator_name(self.root / "missing.py", "a", "b")


class OperatorTemplateTests(_TempDirTestCase):
    def test_template_contains_marker_and_todos(self) -> None:
        text = render_operator_template(
            operator_id="user.demo_abc123",
            name="我的滤波",
            category=("基础调整", "颜色"),
        )
        self.assertIn(f"{UI_CREATED_MARKER} = True", text)
        self.assertIn("id='user.demo_abc123'", text)
        self.assertIn("name='我的滤波'", text)
        self.assertIn("category=('基础调整', '颜色')", text)
        self.assertIn('PortSpec("image", "输入"', text)
        self.assertIn("from app.operators.spec import ParamSpec", text)
        self.assertGreaterEqual(text.count("# TODO:"), 6)
        self.assertIn("class MyOperator(Operator):", text)

    def test_template_is_loadable(self) -> None:
        text = render_operator_template(
            operator_id="user.demo_def456",
            name="空白测试",
            category=("其他", "示例"),
        )
        path = self.tmp / "custom" / "demo.py"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        registry = OperatorRegistry.build([str(self.tmp)], include_builtin=False)
        self.assertEqual(registry.errors, [])
        entry = registry.entry("user.demo_def456")
        self.assertIsNotNone(entry)
        assert entry is not None
        self.assertTrue(entry.ui_created)
        self.assertEqual(entry.spec.name, "空白测试")
        self.assertEqual(entry.spec.category, ("其他", "示例"))
        self.assertEqual(entry.display_name, "demo")
        self.assertIsNone(entry.operator.process())
        self.assertIsNone(entry.operator.process(image=None))

    def test_template_defaults_and_invalid_class_name(self) -> None:
        text = render_operator_template(
            operator_id="user.blank_000001", name="   ", class_name="123 bad"
        )
        self.assertIn("name='新建算子'", text)
        self.assertIn("class MyOperator(Operator):", text)

    def test_template_requires_operator_id(self) -> None:
        with self.assertRaises(ValueError):
            render_operator_template(operator_id="", name="空白")


if __name__ == "__main__":
    unittest.main(verbosity=2)
