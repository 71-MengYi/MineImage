"""算子自动扫描与注册。

约定
----
扫描目录下的每个 ``*.py`` 文件都是一个算子模块，模块级定义 ``OPERATOR`` 或
``OPERATORS``:

* ``OPERATOR = MyOperator``（类，由框架实例化），或 ``OPERATOR = MyOperator()``；
* ``OPERATORS = (A, B, C)``：一个模块打包多个算子，适合按 OpenCV 模块组织目录；
* 两者可以同时存在，按 ``OPERATOR`` → ``OPERATORS`` 的顺序注册，id 不能重复。

算子 id 全局唯一；``OperatorSpec.category`` 为空时，使用文件相对扫描根目录的
路径作为分类（支持任意层级）。用户算子目录中重复的 id 会覆盖内置算子，便于
本地定制。
"""

from __future__ import annotations

import hashlib
import importlib.util
import sys
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Iterable

from PySide6.QtCore import QObject, Signal

from app.core.constants import CATEGORY_ORDER
from app.operators.spec import Operator, OperatorSpec


@dataclass
class OperatorLoadIssue:
    path: Path
    message: str
    traceback: str = ""
    level: str = "error"  # error | warning

    def display(self) -> str:
        return f"[{self.level}] {self.path.name}: {self.message}"


@dataclass
class OperatorEntry:
    operator: Operator
    spec: OperatorSpec
    path: Path
    builtin: bool

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def tab(self) -> str:
        return self.spec.tab

    @property
    def subgroups(self) -> tuple[str, ...]:
        return self.spec.groups


@dataclass
class ScanResult:
    entries: dict[str, OperatorEntry]
    issues: list[OperatorLoadIssue]

    @property
    def errors(self) -> list[OperatorLoadIssue]:
        return [issue for issue in self.issues if issue.level == "error"]


def _category_sort_key(category: str, order: tuple[str, ...] = CATEGORY_ORDER) -> tuple[int, str]:
    try:
        return (order.index(category), category)
    except ValueError:
        return (len(order), category)


class OperatorRegistry(QObject):
    """一次扫描得到的不可变算子集合。重新扫描时由上层创建新实例。"""

    changed = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._entries: dict[str, OperatorEntry] = {}
        self.issues: list[OperatorLoadIssue] = []
        self.scan_roots: list[Path] = []

    # ------------------------------------------------------------------ 属性
    @property
    def entries(self) -> dict[str, OperatorEntry]:
        return dict(self._entries)

    @property
    def errors(self) -> list[OperatorLoadIssue]:
        return [issue for issue in self.issues if issue.level == "error"]

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, op_id: object) -> bool:
        return op_id in self._entries

    def get(self, op_id: str) -> Operator | None:
        entry = self._entries.get(op_id)
        return entry.operator if entry else None

    def entry(self, op_id: str) -> OperatorEntry | None:
        return self._entries.get(op_id)

    def spec(self, op_id: str) -> OperatorSpec | None:
        entry = self._entries.get(op_id)
        return entry.spec if entry else None

    # ------------------------------------------------------------------ 视图
    def tabs(self) -> dict[str, list[OperatorEntry]]:
        """按一级分类组织的算子列表，顺序稳定。"""
        grouped: dict[str, list[OperatorEntry]] = {}
        for entry in self._entries.values():
            grouped.setdefault(entry.tab, []).append(entry)
        ordered: dict[str, list[OperatorEntry]] = {}
        for tab in sorted(grouped, key=_category_sort_key):
            ordered[tab] = sorted(grouped[tab], key=lambda item: (item.spec.category[1:], item.name))
        return ordered

    def entries_for_tab(self, tab: str) -> list[OperatorEntry]:
        return self.tabs().get(tab, [])

    def all_entries(self) -> list[OperatorEntry]:
        result: list[OperatorEntry] = []
        for entries in self.tabs().values():
            result.extend(entries)
        return result

    # ------------------------------------------------------------------ 扫描
    @classmethod
    def build(
        cls,
        extra_dirs: Iterable[str | Path] = (),
        *,
        include_builtin: bool = True,
    ) -> "OperatorRegistry":
        registry = cls()
        roots: list[tuple[Path, bool]] = []
        if include_builtin:
            builtin = Path(__file__).resolve().parent / "builtin"
            if builtin.exists():
                roots.append((builtin, True))
        for raw in extra_dirs:
            if not raw:
                continue
            path = Path(raw).expanduser()
            if path.exists() and path.is_dir():
                roots.append((path.resolve(), False))
            else:
                registry.issues.append(
                    OperatorLoadIssue(path, "算子目录不存在或不是目录", level="warning")
                )
        registry.scan_roots = [path for path, _ in roots]
        for root, is_builtin in roots:
            registry._scan_root(root, is_builtin)
        registry.changed.emit()
        return registry

    def _scan_root(self, root: Path, builtin: bool) -> None:
        files = sorted(
            path
            for path in root.rglob("*.py")
            if not path.name.startswith("_") and "__pycache__" not in path.parts
        )
        for file_path in files:
            try:
                entries = self._load_file(file_path, root, builtin)
            except Exception as exc:  # noqa: BLE001 - 单个算子失败不能影响整体扫描
                self.issues.append(
                    OperatorLoadIssue(
                        file_path,
                        str(exc) or exc.__class__.__name__,
                        traceback.format_exc(),
                        level="error",
                    )
                )
                continue
            if not entries:
                self.issues.append(
                    OperatorLoadIssue(
                        file_path, "模块未定义 OPERATOR 或 OPERATORS，已跳过", level="warning"
                    )
                )
                continue
            for entry in entries:
                if entry.id in self._entries:
                    previous = self._entries[entry.id]
                    self.issues.append(
                        OperatorLoadIssue(
                            file_path,
                            f"算子 id “{entry.id}” 重复，已覆盖 {previous.path}",
                            level="warning",
                        )
                    )
                self._entries[entry.id] = entry

    def _load_file(self, file_path: Path, root: Path, builtin: bool) -> list[OperatorEntry]:
        module = self._import_module(file_path)
        raw_items: list[object] = []
        if hasattr(module, "OPERATOR"):
            raw_items.append(getattr(module, "OPERATOR"))
        if hasattr(module, "OPERATORS"):
            bundled = getattr(module, "OPERATORS")
            if not isinstance(bundled, (list, tuple)):
                raise TypeError("OPERATORS 必须是算子类/实例的列表或元组")
            raw_items.extend(bundled)
        if not raw_items:
            return []

        entries: list[OperatorEntry] = []
        seen: set[str] = set()
        for raw in raw_items:
            operator: Operator
            if isinstance(raw, type):
                if not issubclass(raw, Operator):
                    raise TypeError(f"OPERATOR 类必须继承 Operator，实际为 {raw!r}")
                operator = raw()
            elif isinstance(raw, Operator):
                operator = raw
            else:
                raise TypeError("OPERATOR / OPERATORS 的元素必须是 Operator 的子类或实例")

            spec = getattr(type(operator), "spec", None)
            if not isinstance(spec, OperatorSpec):
                raise TypeError("算子类缺少 OperatorSpec 类型的 spec 属性")
            if not spec.category:
                relative = file_path.relative_to(root)
                category = relative.parent.parts
                spec = replace(spec, category=category or ("其他",))
            if not spec.category:
                spec = replace(spec, category=("其他",))
            if operator.__class__.process is Operator.process:
                raise TypeError("算子必须实现 process()")
            if spec.id in seen:
                raise TypeError(f"模块内算子 id 重复：{spec.id}")
            seen.add(spec.id)
            entries.append(OperatorEntry(operator=operator, spec=spec, path=file_path, builtin=builtin))
        return entries

    @staticmethod
    def _import_module(file_path: Path) -> ModuleType:
        digest = hashlib.sha1(
            f"{file_path}|{file_path.stat().st_mtime_ns}".encode("utf-8")
        ).hexdigest()[:12]
        module_name = f"mineimage_user_ops.{file_path.stem}_{digest}"
        if module_name in sys.modules:
            return sys.modules[module_name]
        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"无法加载模块：{file_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        try:
            spec.loader.exec_module(module)
        except Exception:
            sys.modules.pop(module_name, None)
            raise
        return module

    # ------------------------------------------------------------------ 工具
    def category_tabs(self) -> list[str]:
        return list(self.tabs().keys())
