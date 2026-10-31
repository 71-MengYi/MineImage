"""用户算子文件与用户分类目录的新建 / 重命名 / 移动 / 删除等文件操作。

界面里「新建算子」「新建分类」「重命名」「拖动整理」「删除」都走这些函数，保证：

* 文件名与分类目录使用同一套清洗 / 校验规则（Windows 非法字符、首尾空格与点、
  下划线开头的文件会被 loader 忽略等）；
* 新建的文件写入带 ``MINEIMAGE_UI_CREATED`` 标记的空白模板，重命名文件时能同步
  模板里的 ``OperatorSpec.name``；
* 移动 / 重命名用户分类目录时，按 AST 安全同步目录内源码的字面量 ``category``；
* 写入使用“同目录临时文件 + 原子替换”，避免算子扫描读到半截文件。

所有错误信息都是可以直接展示给用户的中文，调用方只需 ``str(exc)``。
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
import shutil
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from app.operators.template import (
    DEFAULT_CATEGORY,
    DEFAULT_OPERATOR_NAME,
    UI_CREATED_MARKER,
    render_operator_template,
)

#: Windows 文件名非法字符；分类目录与算子文件名共用同一套规则。
_INVALID_NAME_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
#: 分类层级分隔符（与 OperatorSpec 的字符串分类写法一致）。
_CATEGORY_SPLIT = re.compile(r"[/>]")
#: 界面生成模板中的 ``name=...`` 行；仅用于同步重命名后的显示名称。
_NAME_LINE_PATTERN = re.compile(
    r"(?m)^(?P<indent>[ \t]*)name\s*=\s*(?P<quote>['\"])(?P<value>.*?)(?P=quote)\s*,\s*$"
)
#: 单个文件 / 目录名的最大长度（Windows 下更长的名字不利于阅读与备份）。
_MAX_COMPONENT_LENGTH = 64
#: Windows 保留设备名，不能用作文件 / 目录名。
_RESERVED_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{index}" for index in range(1, 10)}
    | {f"lpt{index}" for index in range(1, 10)}
)


@dataclass(frozen=True)
class CreatedOperator:
    """一次「新建算子」的结果。"""

    path: Path
    operator_id: str
    display_name: str
    category: tuple[str, ...]


@dataclass(frozen=True)
class MoveOutcome:
    """一次「拖动整理」的结果。

    ``path`` 是移动后的文件 / 目录；``warnings`` 是不会导致失败、但需要提示
    用户的说明（例如分类目录移动时遇到无法自动同步的动态 ``category``）。
    """

    path: Path
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class _CategoryLiteral:
    """源码中 ``OperatorSpec(category=...)`` 字面量的字节区间。"""

    start: int
    end: int
    category: tuple[str, ...]


# ---------------------------------------------------------------------------
# 名称清洗与校验
# ---------------------------------------------------------------------------
def sanitize_component(text: str, *, fallback: str = DEFAULT_OPERATOR_NAME) -> str:
    """把用户输入清洗成可用的文件 / 目录名（不做存在性检查）。

    非法字符替换为 ``_``，去掉首尾空格与点、开头的下划线；清洗后为空时使用
    ``fallback``。需要“报错而不是自动修正”的场景（如重命名）请用
    :func:`rename_operator_file` 的校验。
    """

    cleaned = _INVALID_NAME_CHARS.sub("_", str(text or "").strip())
    cleaned = cleaned.strip(" .").lstrip("_")
    if not cleaned:
        cleaned = fallback
    cleaned = cleaned[:_MAX_COMPONENT_LENGTH].rstrip(" .")
    if not cleaned:
        cleaned = fallback
    if cleaned.lower() in _RESERVED_NAMES:
        cleaned = f"{cleaned}_"
    return cleaned


def _validate_component(text: str) -> str:
    """校验用户输入的名称，非法时抛 :class:`ValueError`（中文说明）。"""
    raw = str(text or "")
    if not raw.strip():
        raise ValueError("名称不能为空")
    if raw != raw.strip():
        raise ValueError("名称首尾不能是空格")
    if "/" in raw or "\\" in raw:
        raise ValueError("名称不能包含路径分隔符 / 或 \\")
    illegal = _INVALID_NAME_CHARS.search(raw)
    if illegal:
        raise ValueError(f"名称不能包含字符：{illegal.group(0)}")
    if raw.startswith("."):
        raise ValueError("名称不能以点开头")
    if raw.endswith("."):
        raise ValueError("名称不能以点结尾")
    if raw.startswith("_"):
        raise ValueError("名称不能以下划线开头（下划线开头的算子文件会被忽略）")
    if raw.lower() in _RESERVED_NAMES:
        raise ValueError(f"名称不能使用 Windows 保留名：{raw}")
    if len(raw) > _MAX_COMPONENT_LENGTH:
        raise ValueError(f"名称不能超过 {_MAX_COMPONENT_LENGTH} 个字符")
    return raw


def normalize_category(category: str | Iterable[str] | None) -> tuple[str, ...]:
    """把分类统一成清洗后的层级元组；字符串支持 ``"一级/二级"`` 写法。"""
    if category is None:
        return ()
    if isinstance(category, str):
        parts = _CATEGORY_SPLIT.split(category)
    else:
        parts = [str(part) for part in category]
    normalized: list[str] = []
    for part in parts:
        text = str(part).strip()
        if not text:
            continue
        normalized.append(sanitize_component(text))
    return tuple(normalized)


def category_directory(root: Path | str, category: str | Iterable[str] | None) -> Path:
    """返回分类目录路径（只计算路径，不创建目录；空分类返回 ``root`` 本身）。"""
    base = Path(root).expanduser().resolve()
    parts = normalize_category(category)
    return base.joinpath(*parts) if parts else base


def unique_path(directory: Path | str, stem: str, suffix: str = ".py") -> Path:
    """在目录下找一个不冲突的文件路径：``名字.py``、``名字 2.py``、``名字 3.py`` ……"""
    folder = Path(directory)
    candidate = folder / f"{stem}{suffix}"
    counter = 2
    while candidate.exists():
        candidate = folder / f"{stem} {counter}{suffix}"
        counter += 1
    return candidate


# ---------------------------------------------------------------------------
# 拖动整理：路径校验与源码里的 category 字面量
# ---------------------------------------------------------------------------
def _resolve_user_root(root: Path | str) -> Path:
    return Path(root).expanduser().resolve()


def _relative_parts(root: Path, target: Path) -> tuple[str, ...]:
    """返回 ``target`` 相对 ``root`` 的路径分段；不在 root 内时抛中文错误。"""

    try:
        relative = target.resolve().relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"路径不在用户算子目录内：{target}") from exc
    return tuple(part for part in relative.parts if part not in ("", "."))


def _ensure_inside(root: Path, target: Path, label: str) -> None:
    try:
        target.resolve().relative_to(root)
    except (OSError, ValueError) as exc:
        raise ValueError(f"{label}不在用户算子目录内：{target}") from exc


def _literal_category_parts(node: ast.AST) -> tuple[str, ...] | None:
    """把 ``category=`` 后的字面量还原成分类元组；非字面量返回 ``None``。"""

    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        parts = re.split(r"[/>]", node.value)
        return tuple(part.strip() for part in parts if part.strip())
    if isinstance(node, (ast.Tuple, ast.List)):
        parts: list[str] = []
        for element in node.elts:
            if not (isinstance(element, ast.Constant) and isinstance(element.value, str)):
                return None
            text = element.value.strip()
            if text:
                parts.append(text)
        return tuple(parts)
    return None


def _line_start_offsets(data: bytes) -> list[int]:
    """每行第一个字节的绝对偏移，用于把 AST 行列换算成源码字节区间。"""

    starts = [0]
    for index, byte in enumerate(data):
        if byte == 0x0A:  # "\n"
            starts.append(index + 1)
    return starts


def _scan_category_literals(
    data: bytes,
) -> tuple[tuple[_CategoryLiteral, ...], bool]:
    """扫描源码里全部 ``OperatorSpec(category=...)``。

    返回 ``(字面量列表, 是否出现非字面量)``。不导入、不执行用户代码；
    无法按 UTF-8 解析（语法 / 编码错误）时视为“非字面量”，由调用方决定是否
    拒绝或只给出提示。
    """

    try:
        source = data.decode("utf-8")
        tree = ast.parse(source)
    except (UnicodeDecodeError, SyntaxError, ValueError):
        return (), True
    line_starts = _line_start_offsets(data)
    literals: list[_CategoryLiteral] = []
    dynamic = False
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        function = node.func
        if isinstance(function, ast.Name):
            function_name = function.id
        elif isinstance(function, ast.Attribute):
            function_name = function.attr
        else:
            continue
        if function_name != "OperatorSpec":
            continue
        for keyword in node.keywords:
            if keyword.arg != "category":
                continue
            value = keyword.value
            parts = _literal_category_parts(value)
            if parts is None:
                dynamic = True
                continue
            start_line = getattr(value, "lineno", None)
            end_line = getattr(value, "end_lineno", None) or start_line
            start_col = getattr(value, "col_offset", None)
            end_col = getattr(value, "end_col_offset", None)
            if None in (start_line, end_line, start_col, end_col):
                dynamic = True
                continue
            try:
                start = line_starts[int(start_line) - 1] + int(start_col)
                end = line_starts[int(end_line) - 1] + int(end_col)
            except IndexError:
                dynamic = True
                continue
            if not 0 <= start < end <= len(data):
                dynamic = True
                continue
            literals.append(_CategoryLiteral(start, end, parts))
    return tuple(literals), dynamic


def _rewrite_category_literals(
    data: bytes,
    literals: tuple[_CategoryLiteral, ...],
    mapper,
) -> bytes:
    """按 ``mapper(旧分类) -> 新分类 | None`` 就地替换字面量源码区间。"""

    replacements: list[tuple[int, int, bytes]] = []
    for literal in literals:
        new_category = mapper(literal.category)
        if new_category is None or tuple(new_category) == literal.category:
            continue
        try:
            text = repr(tuple(new_category)).encode("utf-8")
        except Exception:  # noqa: BLE001 - repr 不应失败，失败时保持源码不变
            continue
        replacements.append((literal.start, literal.end, text))
    for start, end, text in sorted(replacements, key=lambda item: item[0], reverse=True):
        data = data[:start] + text + data[end:]
    return data


def _has_reserved_parts(parts: tuple[str, ...]) -> bool:
    """隐藏目录 / 下划线目录 / ``__pycache__`` 不参与拖动整理。"""

    return any(
        part == "__pycache__" or part.startswith("_") or part.startswith(".")
        for part in parts
    )


# ---------------------------------------------------------------------------
# 新建
# ---------------------------------------------------------------------------
def _new_operator_id(stem: str, path: Path) -> str:
    """生成 ``user.<ascii slug>_<hash6>`` 形式的唯一算子 id。"""

    slug = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_") or "op"
    slug = slug[:32].strip("_") or "op"
    digest = hashlib.sha1(f"{path}|{uuid.uuid4().hex}".encode("utf-8")).hexdigest()[:6]
    return f"user.{slug}_{digest}"


def _write_bytes_atomic(path: Path, content: bytes) -> None:
    """先写同目录临时文件再原子替换，避免算子扫描读到半截文件。"""

    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}_", suffix=".tmp", dir=str(path.parent), delete=False
    )
    temp_path = Path(handle.name)
    try:
        handle.write(content)
        handle.close()
        os.replace(temp_path, path)
    except Exception:
        try:
            handle.close()
        except Exception:  # noqa: BLE001 - 清理临时文件时忽略二次错误
            pass
        temp_path.unlink(missing_ok=True)
        raise


def _write_text_atomic(path: Path, content: str) -> None:
    """把文本按 UTF-8 原子写入；用户算子源码统一使用该编码。"""

    _write_bytes_atomic(path, content.encode("utf-8"))


def create_operator_file(
    root: Path | str,
    category: str | Iterable[str] | None,
    name: str | None = None,
) -> CreatedOperator:
    """在 ``root`` 下的分类目录里新建一个空白算子文件。

    文件名默认「新建算子」，重名时自动追加序号；``category`` 为空时文件放在
    ``root`` 下并归入「其他」。返回实际写入的文件、算子 id 与显示名。
    """

    base = Path(root).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    parts = normalize_category(category)
    directory = base.joinpath(*parts) if parts else base
    directory.mkdir(parents=True, exist_ok=True)
    stem = sanitize_component(name if name is not None else DEFAULT_OPERATOR_NAME)
    path = unique_path(directory, stem)
    display_name = path.stem
    written_category = parts or DEFAULT_CATEGORY
    operator_id = _new_operator_id(display_name, path)
    content = render_operator_template(
        operator_id=operator_id,
        name=display_name,
        category=written_category,
    )
    _write_text_atomic(path, content)
    return CreatedOperator(
        path=path,
        operator_id=operator_id,
        display_name=display_name,
        category=written_category,
    )


def create_category_dir(
    root: Path | str,
    parent: str | Iterable[str] | None,
    name: str,
) -> Path:
    """在 ``parent`` 分类下新建分类目录；``parent`` 为空表示一级分类。"""

    base = Path(root).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    component = _validate_component(name)
    parent_parts = normalize_category(parent)
    directory = base.joinpath(*parent_parts, component)
    if directory.exists():
        raise FileExistsError(f"分类目录已存在：{directory}")
    directory.mkdir(parents=True, exist_ok=False)
    return directory


# ---------------------------------------------------------------------------
# 重命名 / 删除
# ---------------------------------------------------------------------------
def update_generated_operator_name(path: Path | str, old_stem: str, new_stem: str) -> None:
    """重命名文件后同步界面生成模板里的 ``name=...``。

    只有同时满足以下条件才会修改文件，其余情况安全地不做任何事：

    * 文件里带 ``MINEIMAGE_UI_CREATED`` 标记（由界面新建的模板）；
    * ``OperatorSpec.name`` 仍等于旧文件名 ``old_stem``（用户没手动改过名称）。
    """

    target = Path(path).expanduser()
    try:
        if not target.is_file():
            return
        with open(target, encoding="utf-8", newline="") as handle:
            content = handle.read()
    except (OSError, UnicodeDecodeError):
        return
    if UI_CREATED_MARKER not in content:
        return

    def _replace(match: re.Match[str]) -> str:
        if _literal_text(match) != old_stem:
            return match.group(0)
        return f"{match.group('indent')}name={new_stem!r},"

    updated = _NAME_LINE_PATTERN.sub(_replace, content)
    if updated == content:
        return
    try:
        with open(target, "w", encoding="utf-8", newline="") as handle:
            handle.write(updated)
    except OSError:
        # 同步失败不影响重命名本身；界面显示名仍然来自文件名。
        return


def _literal_text(match: re.Match[str]) -> str:
    """把 ``name='xxx'`` 的字符串字面量还原成真实文本；解析失败时返回原文。"""
    quote = match.group("quote")
    literal = f"{quote}{match.group('value')}{quote}"
    try:
        value = ast.literal_eval(literal)
    except (SyntaxError, ValueError):
        return match.group("value")
    return value if isinstance(value, str) else match.group("value")


def rename_operator_file(path: Path | str, new_stem: str) -> Path:
    """重命名算子文件并返回新路径。

    名称非法抛 :class:`ValueError`，目标同名抛 :class:`FileExistsError`，文件不存在
    抛 :class:`FileNotFoundError`；界面生成的模板会顺带同步 ``name``。
    """

    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"算子文件不存在：{source}")
    stem = _validate_component(new_stem)
    target = source.with_name(f"{stem}{source.suffix}")
    if target == source:
        return source
    if target.exists():
        raise FileExistsError(f"同名文件已存在：{target.name}")
    old_stem = source.stem
    source.rename(target)
    # 模板里的 name 只影响图以外场景的显示，同步失败时以文件名为准。
    update_generated_operator_name(target, old_stem, stem)
    return target


def delete_operator_file(path: Path | str) -> None:
    """删除算子文件；文件不存在时抛 :class:`FileNotFoundError`。"""

    target = Path(path).expanduser()
    if not target.is_file():
        raise FileNotFoundError(f"算子文件不存在：{target}")
    target.unlink()


# ---------------------------------------------------------------------------
# 拖动整理：移动算子文件 / 分类目录
# ---------------------------------------------------------------------------
def _category_move_updates(
    source: Path,
    old_prefix: tuple[str, ...],
    new_prefix: tuple[str, ...],
) -> tuple[dict[Path, bytes], dict[Path, bytes], list[str]]:
    """计算分类目录移动后需要改写的文件内容。

    返回 ``(原内容, 新内容, 提示)``；键是相对分类目录的路径。只处理
    ``OperatorSpec(category=...)`` 的字面量，不做任何导入或执行。
    """

    originals: dict[Path, bytes] = {}
    updates: dict[Path, bytes] = {}
    warnings: list[str] = []
    for file_path in sorted(source.rglob("*.py")):
        if "__pycache__" in file_path.parts or file_path.name.startswith("_"):
            continue
        relative = file_path.relative_to(source)
        try:
            data = file_path.read_bytes()
        except OSError as exc:
            warnings.append(f"{relative.as_posix()}：读取失败，未自动同步分类（{exc}）")
            continue
        literals, dynamic = _scan_category_literals(data)
        if dynamic:
            warnings.append(
                f"{relative.as_posix()}：无法解析或 category 不是字面量，未自动同步"
            )
            continue
        updated = _rewrite_category_literals(
            data,
            literals,
            lambda value: (
                new_prefix + value[len(old_prefix) :]
                if value[: len(old_prefix)] == old_prefix
                else None
            ),
        )
        if updated != data:
            originals[relative] = data
            updates[relative] = updated
    return originals, updates, warnings


def move_operator_file(
    root: Path | str,
    path: Path | str,
    target_category: str | Iterable[str] | None,
    *,
    current_category: str | Iterable[str] | None = None,
) -> MoveOutcome:
    """把用户算子文件移动到 ``root`` 下的目标分类目录。

    ``current_category`` 是该算子当前声明的分类（通常是
    ``OperatorEntry.spec.category``）。移动时会用 AST 找到源码里的
    ``OperatorSpec(category=...)`` 字面量并同步为新分类；非字面量无法自动
    同步时抛 :class:`ValueError` 且不移动文件。目标同名文件抛
    :class:`FileExistsError`。``target_category`` 为空表示移到根目录，由
    loader 归入「其他」。
    """

    base = _resolve_user_root(root)
    source = Path(path).expanduser()
    if not source.is_file():
        raise FileNotFoundError(f"算子文件不存在：{source}")
    source = source.resolve()
    _ensure_inside(base, source, "算子文件")
    if source.suffix.lower() != ".py":
        raise ValueError(f"只能移动 .py 算子文件：{source.name}")

    target_parts = normalize_category(target_category)
    old_parts = _relative_parts(base, source.parent)
    target_dir = base.joinpath(*target_parts) if target_parts else base
    if target_dir.resolve() == source.parent:
        return MoveOutcome(source)
    target = target_dir / source.name
    if target.exists():
        raise FileExistsError(f"目标分类下已有同名文件：{target.name}")

    declared = (
        normalize_category(current_category)
        if current_category is not None
        else old_parts
    )
    original = source.read_bytes()
    literals, dynamic = _scan_category_literals(original)
    if dynamic:
        raise ValueError(
            f"算子文件“{source.name}”的 category 不是字面量，"
            "无法随拖动自动同步分类。"
        )
    # loader 对根目录下没有声明 category（或声明为空元组）的算子会回退到
    # 「其他」，此时源码里可能没有字面量，也可能只有一个 ``category=()``。
    root_fallback_other = old_parts == () and declared == ("其他",)

    def _matches_current(value: tuple[str, ...]) -> bool:
        if value == declared:
            return True
        return root_fallback_other and value == ()

    updated = _rewrite_category_literals(
        original,
        literals,
        lambda value: target_parts if _matches_current(value) else None,
    )
    if (
        updated == original
        and declared != old_parts
        and literals
        and not any(_matches_current(literal.category) for literal in literals)
    ):
        raise ValueError(
            f"未找到与当前分类匹配的 category 字面量，无法同步：{source.name}"
        )

    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"无法创建目标分类目录：{exc}") from exc
    try:
        source.rename(target)
    except OSError as exc:
        raise OSError(f"移动算子文件失败：{exc}") from exc
    if updated != original:
        try:
            _write_bytes_atomic(target, updated)
        except Exception:
            # 分类同步失败时尽量退回原路径，避免文件“移动了但显示分类不对”。
            try:
                target.rename(source)
            except OSError:
                pass
            raise
    return MoveOutcome(target)


def _user_category_source(
    root: Path | str,
    path: Path | str,
) -> tuple[Path, tuple[str, ...]]:
    """校验“重命名 / 删除分类”的源目录，返回解析后的目录与相对路径。"""

    base = _resolve_user_root(root)
    source = Path(path).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"分类目录不存在：{source}")
    source = source.resolve()
    _ensure_inside(base, source, "分类目录")
    if source == base:
        raise ValueError("用户算子根目录不能作为分类操作。")
    parts = _relative_parts(base, source)
    if not parts:
        raise ValueError("用户算子根目录不能作为分类操作。")
    if parts[0] == "custom":
        raise ValueError("custom 是算子组保留目录，不能重命名或删除。")
    if _has_reserved_parts(parts):
        raise ValueError("隐藏或保留目录不能重命名或删除。")
    return source, parts


def _count_loadable_py(directory: Path) -> int:
    """统计目录内按 loader 规则可加载的 ``*.py`` 文件数。"""

    return sum(
        1
        for file_path in directory.rglob("*.py")
        if "__pycache__" not in file_path.parts
        and not file_path.name.startswith("_")
    )


def _move_category_directory(
    source: Path,
    target: Path,
    old_prefix: tuple[str, ...],
    new_prefix: tuple[str, ...],
    *,
    action: str,
) -> tuple[str, ...]:
    """移动 / 重命名分类目录并同步字面量 ``category``，返回 warnings。"""

    originals, updates, warnings = _category_move_updates(
        source, old_prefix, new_prefix
    )
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise OSError(f"无法创建目标分类目录：{exc}") from exc
    try:
        source.rename(target)
    except OSError as exc:
        raise OSError(f"{action}分类目录失败：{exc}") from exc

    written: list[Path] = []
    try:
        for relative, data in updates.items():
            _write_bytes_atomic(target / relative, data)
            written.append(relative)
    except Exception:
        # 目录移动成功但某个文件的分类同步失败：退回目录并恢复已改写内容。
        try:
            target.rename(source)
        except OSError:
            pass
        else:
            for relative in written:
                try:
                    _write_bytes_atomic(source / relative, originals[relative])
                except OSError:
                    pass
        raise
    return tuple(warnings)


def move_category_dir(
    root: Path | str,
    path: Path | str,
    target_parent: str | Iterable[str] | None,
) -> MoveOutcome:
    """把 ``root`` 下的用户分类目录移动到 ``target_parent`` 分类下。

    目录连同全部子分类 / 算子整体移动；内部 ``OperatorSpec(category=...)``
    字面量会按“旧相对路径前缀 → 新相对路径前缀”重映射。没有声明分类的算子
    由 loader 按新目录自动推导；非字面量分类不修改，仅写入 ``warnings``。
    目标同名、移入自身 / 后代、``custom`` 保留目录等会抛中文异常。
    """

    base = _resolve_user_root(root)
    source = Path(path).expanduser()
    if not source.is_dir():
        raise FileNotFoundError(f"分类目录不存在：{source}")
    source = source.resolve()
    _ensure_inside(base, source, "分类目录")
    if source == base:
        raise ValueError("用户算子根目录不能移动。")
    source_parts = _relative_parts(base, source)
    if not source_parts:
        raise ValueError("用户算子根目录不能移动。")
    if source_parts[0] == "custom":
        raise ValueError("custom 是算子组保留目录，不能拖动整理。")
    if _has_reserved_parts(source_parts):
        raise ValueError("隐藏或保留目录不能拖动整理。")

    target_parts = normalize_category(target_parent)
    if target_parts[:1] == ("custom",):
        raise ValueError("custom 是算子组保留目录，不能作为拖动目标。")
    target_dir = base.joinpath(*target_parts) if target_parts else base
    target_dir_resolved = target_dir.resolve()
    if target_dir_resolved == source or source in target_dir_resolved.parents:
        raise ValueError("不能把分类移动到它自己或子分类下。")
    new_path = target_dir / source.name
    if new_path.resolve() == source:
        return MoveOutcome(source)
    if new_path.exists():
        raise FileExistsError(f"目标位置已存在同名分类或文件：{new_path.name}")

    new_parts = target_parts + (source.name,)
    warnings = _move_category_directory(
        source, new_path, source_parts, new_parts, action="移动"
    )
    return MoveOutcome(new_path, warnings)


def rename_category_dir(
    root: Path | str,
    path: Path | str,
    new_name: str,
) -> MoveOutcome:
    """重命名 ``root`` 下的用户分类目录（保持在同一父级）。

    目录内 ``OperatorSpec(category=...)`` 字面量会按“旧相对路径前缀 → 新相对
    路径前缀”重映射；没有声明分类的算子由 loader 按新目录自动推导；非字面量
    分类不修改，仅写入 ``warnings``。目标同名、非法名称、``custom`` 保留目录等
    会抛中文异常。
    """

    source, source_parts = _user_category_source(root, path)
    name = _validate_component(new_name)
    if name == source.name:
        return MoveOutcome(source)
    target = source.with_name(name)
    if target.exists():
        raise FileExistsError(f"目标位置已存在同名分类或文件：{name}")
    new_parts = source_parts[:-1] + (name,)
    warnings = _move_category_directory(
        source, target, source_parts, new_parts, action="重命名"
    )
    return MoveOutcome(target, warnings)


def delete_category_dir(root: Path | str, path: Path | str) -> int:
    """递归删除 ``root`` 下的用户分类目录，返回删除的可加载算子数量。

    目录内全部子分类 / 算子 / 缓存都会被删除；``root`` 本身、``custom`` 保留
    目录、隐藏 / 下划线目录与 root 外路径都会抛中文错误。
    """

    source, _parts = _user_category_source(root, path)
    try:
        count = _count_loadable_py(source)
    except OSError as exc:
        raise OSError(f"无法读取分类目录：{exc}") from exc
    try:
        shutil.rmtree(source)
    except OSError as exc:
        raise OSError(f"删除分类目录失败：{exc}") from exc
    return count


__all__ = [
    "CreatedOperator",
    "MoveOutcome",
    "category_directory",
    "create_category_dir",
    "create_operator_file",
    "delete_category_dir",
    "delete_operator_file",
    "move_category_dir",
    "move_operator_file",
    "normalize_category",
    "rename_category_dir",
    "rename_operator_file",
    "sanitize_component",
    "unique_path",
    "update_generated_operator_name",
]
