"""一个模块注册多个算子的加载测试（OPERATORS 约定）。"""

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

from app.core.constants import CATEGORY_ORDER  # noqa: E402
from app.operators.loader import OperatorRegistry  # noqa: E402


MODULE_TEMPLATE = """\
from app.operators.spec import Operator, OperatorSpec


class First(Operator):
    spec = OperatorSpec(id="{prefix}.first", name="第一个", category=("其他", "测试"))

    def process(self, image=None, **_):
        return image.array


class Second(Operator):
    spec = OperatorSpec(id="{prefix}.second", name="第二个")

    def process(self, image=None, **_):
        return image.array


OPERATORS = (First, Second)
"""


UI_CREATED_MODULE = """\
from app.operators.spec import Operator, OperatorSpec

MINEIMAGE_UI_CREATED = True


class UiMade(Operator):
    spec = OperatorSpec(id="ui.made", name="旧名字", category=("其他", "界面"))

    def process(self, image=None, **_):
        return image.array if image is not None else None


OPERATOR = UiMade
"""


class MultiOperatorModuleTests(unittest.TestCase):
    def _make_root(self) -> Path:
        root = ROOT / ".test_tmp" / f"loader_multi_{uuid.uuid4().hex[:10]}"
        self.addCleanup(shutil.rmtree, root, ignore_errors=True)
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _write(self, root: Path, files: dict[str, str]) -> None:
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(content), encoding="utf-8")

    def _build(
        self,
        files: dict[str, str],
        dirs: tuple[str, ...] = (),
    ) -> OperatorRegistry:
        root = self._make_root()
        self._write(root, files)
        for name in dirs:
            (root / name).mkdir(parents=True, exist_ok=True)
        return OperatorRegistry.build([str(root)], include_builtin=False)

    def test_every_operator_in_operators_is_registered(self) -> None:
        registry = self._build({"many.py": MODULE_TEMPLATE.format(prefix="many")})
        self.assertEqual(set(registry.entries), {"many.first", "many.second"})
        self.assertEqual(registry.errors, [])
        self.assertEqual(registry.spec("many.first").category, ("其他", "测试"))
        # 未声明分类时使用文件所在目录，顶层目录文件归入「其他」。
        self.assertEqual(registry.spec("many.second").category, ("其他",))

    def test_duplicate_id_inside_one_module_is_an_error(self) -> None:
        content = MODULE_TEMPLATE.format(prefix="same").replace(
            'id="same.second"', 'id="same.first"'
        )
        registry = self._build({"many.py": content})
        self.assertEqual(registry.entries, {})
        self.assertTrue(
            any("模块内算子 id 重复" in issue.message for issue in registry.errors)
        )

    def test_operator_and_operators_can_coexist(self) -> None:
        content = MODULE_TEMPLATE.format(prefix="coexist").replace(
            "OPERATORS = (First, Second)",
            textwrap.dedent(
                """\
                class Third(Operator):
                    spec = OperatorSpec(id="coexist.third", name="第三个")

                    def process(self, image=None, **_):
                        return image.array


                OPERATORS = (First, Second)
                OPERATOR = Third
                """
            ),
        )
        registry = self._build({"mixed.py": content})
        self.assertEqual(
            set(registry.entries), {"coexist.first", "coexist.second", "coexist.third"}
        )

    def test_module_without_operator_produces_warning(self) -> None:
        registry = self._build({"empty.py": "VALUE = 1\n"})
        self.assertEqual(registry.entries, {})
        self.assertEqual(registry.errors, [])
        self.assertTrue(any("未定义" in issue.message for issue in registry.issues))

    def test_known_categories_includes_empty_directories(self) -> None:
        registry = self._build(
            {"many.py": MODULE_TEMPLATE.format(prefix="many")},
            dirs=("滤波与平滑/我的分类", ".hidden/秘密", "__pycache__", "custom"),
        )
        categories = registry.known_categories()
        self.assertIn(("滤波与平滑", "我的分类"), categories)
        self.assertIn(("其他", "测试"), categories)
        self.assertFalse(any("__pycache__" in item for item in categories))
        self.assertFalse(any(part.startswith(".") for item in categories for part in item))
        # operators/custom 是算子组保存目录的保留名，不应成为空分类。
        self.assertNotIn(("custom",), categories)
        # 按 CATEGORY_ORDER 排序：「滤波与平滑」排在「其他」之前。
        first_filter = next(
            index for index, item in enumerate(categories) if item[0] == "滤波与平滑"
        )
        first_other = next(index for index, item in enumerate(categories) if item[0] == "其他")
        self.assertLess(first_filter, first_other)
        self.assertLess(
            CATEGORY_ORDER.index("滤波与平滑"), CATEGORY_ORDER.index("其他")
        )

    def test_ui_created_marker_and_display_name(self) -> None:
        registry = self._build(
            {
                "ui_made.py": UI_CREATED_MODULE,
                "normal.py": MODULE_TEMPLATE.format(prefix="normal"),
            }
        )
        self.assertEqual(registry.errors, [])
        ui_entry = registry.entry("ui.made")
        self.assertIsNotNone(ui_entry)
        assert ui_entry is not None
        self.assertTrue(ui_entry.ui_created)
        self.assertEqual(ui_entry.display_name, "ui_made")
        self.assertEqual(ui_entry.spec.name, "旧名字")

        normal_entry = registry.entry("normal.first")
        self.assertIsNotNone(normal_entry)
        assert normal_entry is not None
        self.assertFalse(normal_entry.ui_created)
        self.assertEqual(normal_entry.display_name, "第一个")

    def test_rebuild_reloads_same_second_same_size_rewrite(self) -> None:
        root = self._make_root()
        path = root / "probe.py"
        template = MODULE_TEMPLATE.format(prefix="probe")
        path.write_text(template, encoding="utf-8")
        original_stat = path.stat()
        first = OperatorRegistry.build([str(root)], include_builtin=False)
        self.assertEqual(first.spec("probe.first").name, "第一个")

        # 同长度内容（字节数不变）+ 同秒级 mtime：importlib 的 .pyc 校验会误判，
        # 重建注册表时必须仍加载新代码。
        changed = template.replace('name="第一个"', 'name="第一号"')
        self.assertEqual(len(changed.encode("utf-8")), len(template.encode("utf-8")))
        path.write_text(changed, encoding="utf-8")
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        second = OperatorRegistry.build([str(root)], include_builtin=False)
        self.assertEqual(second.spec("probe.first").name, "第一号")

    def test_rebuild_reloads_changed_module_with_same_mtime(self) -> None:
        root = self._make_root()
        path = root / "probe.py"
        template = MODULE_TEMPLATE.format(prefix="probe")
        path.write_text(template, encoding="utf-8")
        original_stat = path.stat()
        first = OperatorRegistry.build([str(root)], include_builtin=False)
        self.assertEqual(first.spec("probe.first").name, "第一个")

        # 内容改变但把 mtime 改回原值：按内容哈希也应该加载新模块。
        changed = template.replace('name="第一个"', 'name="改过的名字"')
        path.write_text(changed, encoding="utf-8")
        os.utime(path, ns=(original_stat.st_atime_ns, original_stat.st_mtime_ns))
        second = OperatorRegistry.build([str(root)], include_builtin=False)
        self.assertEqual(second.spec("probe.first").name, "改过的名字")

    def test_same_stem_and_content_in_different_directories_use_distinct_modules(
        self,
    ) -> None:
        """同 stem 同内容的不同文件不能被 sys.modules 缓存串到同一个模块。"""

        root = self._make_root()
        content = MODULE_TEMPLATE.format(prefix="dup")
        path_a = root / "catA" / "dup.py"
        path_b = root / "catB" / "dup.py"
        for path in (path_a, path_b):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content, encoding="utf-8")

        module_a = OperatorRegistry._import_module(path_a)  # noqa: SLF001 - 回归定位
        module_b = OperatorRegistry._import_module(path_b)  # noqa: SLF001 - 回归定位
        self.assertIsNot(module_a, module_b)
        self.assertEqual(Path(module_a.__file__).resolve(), path_a.resolve())
        self.assertEqual(Path(module_b.__file__).resolve(), path_b.resolve())

        registry = OperatorRegistry.build([str(root)], include_builtin=False)
        entry = registry.entry("dup.first")
        self.assertIsNotNone(entry)
        assert entry is not None
        # 两个文件声明了同一个 id：后扫描文件覆盖条目，但路径必须来自后者，
        # 不能因模块缓存把 spec（catA）和 path（catB）拼在一起。
        self.assertEqual(entry.path.resolve(), path_b.resolve())
        self.assertTrue(
            any("重复" in issue.message for issue in registry.issues),
            "同 id 时应给出重复提示",
        )

    def test_file_snapshot_tracks_user_operator_files(self) -> None:
        registry = self._build({"op.py": MODULE_TEMPLATE.format(prefix="snap")})
        path = registry.scan_roots[0] / "op.py"
        snapshot = registry.file_snapshot()
        self.assertEqual(set(snapshot), {str(path)})
        mtime_ns, size = snapshot[str(path)]
        self.assertEqual(size, path.stat().st_size)

        with path.open("a", encoding="utf-8") as handle:
            handle.write("\n# 修改后应被检测到\n")
        updated = registry.file_snapshot()
        self.assertNotEqual(updated[str(path)], (mtime_ns, size))

    def test_file_snapshot_can_include_builtin(self) -> None:
        user_root = self._make_root()
        self._write(user_root, {"user_op.py": MODULE_TEMPLATE.format(prefix="user")})
        builtin_root = self._make_root()
        self._write(builtin_root, {"builtin_op.py": MODULE_TEMPLATE.format(prefix="builtin")})

        with mock.patch("app.operators.loader.BUILTIN_DIR", builtin_root):
            registry = OperatorRegistry.build([str(user_root)], include_builtin=True)
        user_snapshot = registry.file_snapshot()
        full_snapshot = registry.file_snapshot(include_builtin=True)
        self.assertEqual(set(user_snapshot), {str(user_root / "user_op.py")})
        self.assertIn(str(builtin_root / "builtin_op.py"), full_snapshot)
        self.assertNotIn(str(builtin_root / "builtin_op.py"), user_snapshot)
        self.assertGreater(len(full_snapshot), len(user_snapshot))


if __name__ == "__main__":
    unittest.main(verbosity=2)
