"""用户算子文件的新建 / 重命名 / 删除等文件操作。

界面里「新建算子」「在分类下新建」「重命名」「删除」都走这些函数，保证：

* 文件名与分类目录使用同一套清洗 / 校验规则（Windows 非法字符、首尾空格与点、
  下划线开头的文件会被 loader 忽略等）；
* 新建的文件写入带 ``MINEIMAGE_UI_CREATED`` 标记的空白模板，重命名文件时能同步
  模板里的 ``OperatorSpec.name``；
* 写入使用“同目录临时文件 + 原子替换”，避免算子扫描读到半截文件。

所有错误信息都是可以直接展示给用户的中文，调用方只需 ``str(exc)``。
"""

from __future__ import annotations

import ast
import hashlib
import os
import re
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
# 新建
# ---------------------------------------------------------------------------
def _new_operator_id(stem: str, path: Path) -> str:
    """生成 ``user.<ascii slug>_<hash6>`` 形式的唯一算子 id。"""
    slug = re.sub(r"[^a-z0-9]+", "_", stem.lower()).strip("_") or "op"
    slug = slug[:32].strip("_") or "op"
    digest = hashlib.sha1(f"{path}|{uuid.uuid4().hex}".encode("utf-8")).hexdigest()[:6]
    return f"user.{slug}_{digest}"


def _write_text_atomic(path: Path, content: str) -> None:
    """先写同目录临时文件再原子替换，避免算子扫描读到半截文件。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}_", suffix=".py.tmp", dir=str(path.parent), delete=False
    )
    temp_path = Path(handle.name)
    try:
        handle.write(content.encode("utf-8"))
        handle.close()
        os.replace(temp_path, path)
    except Exception:
        try:
            handle.close()
        except Exception:  # noqa: BLE001 - 清理临时文件时忽略二次错误
            pass
        temp_path.unlink(missing_ok=True)
        raise


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


__all__ = [
    "CreatedOperator",
    "category_directory",
    "create_category_dir",
    "create_operator_file",
    "delete_operator_file",
    "normalize_category",
    "rename_operator_file",
    "sanitize_component",
    "unique_path",
    "update_generated_operator_name",
]
