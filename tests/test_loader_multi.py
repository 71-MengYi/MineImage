"""一个模块注册多个算子的加载测试（OPERATORS 约定）。"""

from __future__ import annotations

import os
import shutil
import sys
import textwrap
import unittest
import uuid
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


class MultiOperatorModuleTests(unittest.TestCase):
    def _build(self, files: dict[str, str]) -> OperatorRegistry:
        root = ROOT / ".test_tmp" / f"loader_multi_{uuid.uuid4().hex[:10]}"
        self.addCleanup(lambda: shutil.rmtree(root, ignore_errors=True))
        root.mkdir(parents=True, exist_ok=True)
        for name, content in files.items():
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(content), encoding="utf-8")
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
