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
import os
import sys
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from types import ModuleType
from typing import Iterable

from PySide6.QtCore import QObject, Signal

from app.core.constants import CATEGORY_ORDER
from app.operators.spec import Operator, OperatorSpec
from app.operators.template import UI_CREATED_MARKER

#: 内置算子目录（测试可以替换它来模拟“内置 / 用户”两类扫描根）。
BUILTIN_DIR = Path(__file__).resolve().parent / "builtin"


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
    #: 是否由界面「新建算子」生成的模板（文件里带 MINEIMAGE_UI_CREATED 标记）。
    ui_created: bool = False

    @property
    def id(self) -> str:
        return self.spec.id

    @property
    def name(self) -> str:
        return self.spec.name

    @property
    def display_name(self) -> str:
        """界面展示名称：界面新建的算子跟随文件名，其余使用 ``spec.name``。"""
        if self.ui_created:
            return self.path.stem
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
        self._builtin_roots: set[Path] = set()
        #: 用户算子目录里出现过的分类路径（含尚无算子的空目录）。
        self._directory_categories: set[tuple[str, ...]] = set()

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
            ordered[tab] = sorted(
                grouped[tab], key=lambda item: (item.spec.category[1:], item.display_name)
            )
        return ordered

    def entries_for_tab(self, tab: str) -> list[OperatorEntry]:
        return self.tabs().get(tab, [])

    def all_entries(self) -> list[OperatorEntry]:
        result: list[OperatorEntry] = []
        for entries in self.tabs().values():
            result.extend(entries)
        return result

    def known_categories(self) -> list[tuple[str, ...]]:
        """全部已知的分类路径，含用户目录里还没有算子的空分类。

        由非内置扫描根目录下的子目录（排除 ``.`` 开头与 ``__pycache__``）和
        非内置算子的显式 ``spec.category`` 合并而成，按 ``CATEGORY_ORDER`` 排序，
        供算子库展示空分类、在分类上右键新建算子。
        """

        categories = set(self._directory_categories)
        for entry in self._entries.values():
            if not entry.builtin:
                categories.add(tuple(entry.spec.category))
        return sorted(categories, key=self._category_path_sort_key)

    @staticmethod
    def _category_path_sort_key(category: tuple[str, ...]) -> tuple[int, str, tuple[str, ...]]:
        tab = category[0] if category else "其他"
        return (*_category_sort_key(tab), category[1:])

    def file_snapshot(self, *, include_builtin: bool = False) -> dict[str, tuple[int, int]]:
        """参与扫描的 ``*.py`` 文件快照：``路径 -> (mtime_ns, size)``。

        执行引擎在每次请求前比对前后快照即可发现用户算子文件被外部编辑器改动；
        ``include_builtin=False`` 时只统计用户算子目录，读不到的文件会被跳过。
        """

        snapshot: dict[str, tuple[int, int]] = {}
        for root in self.scan_roots:
            if not include_builtin and root in self._builtin_roots:
                continue
            try:
                files = sorted(root.rglob("*.py"))
            except OSError:
                continue
            for file_path in files:
                if file_path.name.startswith("_") or "__pycache__" in file_path.parts:
                    continue
                try:
                    stat = file_path.stat()
                except OSError:
                    continue
                snapshot[str(file_path)] = (stat.st_mtime_ns, stat.st_size)
        return snapshot

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
            if BUILTIN_DIR.exists():
                roots.append((BUILTIN_DIR, True))
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
        registry._builtin_roots = {path for path, is_builtin in roots if is_builtin}
        for root, is_builtin in roots:
            registry._scan_root(root, is_builtin)
        registry.changed.emit()
        return registry

    def _scan_root(self, root: Path, builtin: bool) -> None:
        if not builtin:
            self._record_directories(root)
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

    def _record_directories(self, root: Path) -> None:
        """记录用户算子目录下的分类目录（包括暂无算子的空目录）。"""
        for current, subdirs, _files in os.walk(root):
            subdirs[:] = sorted(
                name for name in subdirs if not name.startswith(".") and name != "__pycache__"
            )
            current_path = Path(current)
            if current_path == root:
                continue
            try:
                relative = current_path.relative_to(root)
            except ValueError:
                continue
            # ``custom/`` 是「保存为新算子」的保留目录：算子组文件显式声明
            # 「用户自定义算子」分类，不应再额外出现一个空的 custom 分类。
            if relative.parts and relative.parts[0] == "custom":
                continue
            self._directory_categories.add(relative.parts)

    def _load_file(self, file_path: Path, root: Path, builtin: bool) -> list[OperatorEntry]:
        module = self._import_module(file_path, purge_bytecode=not builtin)
        ui_created = bool(getattr(module, UI_CREATED_MARKER, False))
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
            entries.append(
                OperatorEntry(
                    operator=operator,
                    spec=spec,
                    path=file_path,
                    builtin=builtin,
                    ui_created=ui_created,
                )
            )
        return entries

    @staticmethod
    def _purge_bytecode_cache(file_path: Path) -> None:
        """删除该源文件对应的 ``.pyc`` 缓存（失败时忽略）。

        Windows 上连续两次保存可能落在同一秒：importlib 的 ``.pyc`` 校验只看
        秒级 mtime + 文件大小，内容长度不变时会复用旧字节码。用户算子文件在
        每次扫描时先删缓存，保证重建注册表后立即执行新代码。
        """

        try:
            cache_path = importlib.util.cache_from_source(str(file_path))
        except (ValueError, NotImplementedError):
            return
        try:
            Path(cache_path).unlink()
        except OSError:
            return

    @staticmethod
    def _import_module(file_path: Path, *, purge_bytecode: bool = False) -> ModuleType:
        # 用「文件路径 + 内容」区分模块：
        # * 内容哈希解决 Windows 连续两次写入落在同一 mtime_ns、只靠 mtime 会
        #   复用旧模块的问题；
        # * 路径参与哈希，避免不同目录下同 stem 同内容的文件互相命中
        #   sys.modules，导致后扫描文件的 path 覆盖前一个文件。
        path_key = str(file_path).encode("utf-8", "surrogatepass")
        try:
            content_key = file_path.read_bytes()
        except OSError:
            content_key = f"|{file_path.stat().st_mtime_ns}".encode("utf-8")
        digest = hashlib.sha1(path_key + b"|" + content_key).hexdigest()[:12]
        module_name = f"mineimage_user_ops.{file_path.stem}_{digest}"
        if module_name in sys.modules:
            return sys.modules[module_name]
        if purge_bytecode:
            OperatorRegistry._purge_bytecode_cache(file_path)
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
