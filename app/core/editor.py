"""外部编辑器探测与启动。

界面里的「新建算子 / 在编辑器中打开」需要知道用什么程序打开 ``.py`` 文件。
设置项 ``operators/editor`` 的取值约定：

* ``auto``：自动探测本机已安装的编辑器，取第一个可用的（推荐默认值）；
* ``system``：交给操作系统的默认程序打开；
* ``path:<可执行文件路径>``：用户手动指定的编辑器；
* 其它字符串：:func:`detect_editors` 返回的 ``key``（如 ``vscode`` / ``pycharm``）。

探测只使用 ``shutil.which`` 与常见安装路径，不下载、不安装任何软件，也不会
修改系统设置；未找到编辑器时由调用方回退到系统默认程序。
"""

from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices

#: 自动选择本机可用的编辑器
AUTO_EDITOR = "auto"
#: 使用操作系统默认程序打开
SYSTEM_EDITOR = "system"
#: 手动指定编辑器可执行文件（后面接完整路径）
CUSTOM_EDITOR_PREFIX = "path:"

#: 编辑器中文提示里统一使用的设置项名称。
SETTINGS_EDITOR_HINT = "「设置 → 算子目录 → 默认编辑器」"


@dataclass(frozen=True)
class EditorInfo:
    """一个可用的外部编辑器。"""

    key: str
    name: str
    executable: str


@dataclass(frozen=True)
class _EditorCandidate:
    """一条编辑器探测规则：命中 ``commands`` / ``paths`` / ``patterns`` 任意一项即可。"""

    key: str
    name: str
    commands: tuple[str, ...] = ()
    paths: tuple[str, ...] = ()
    patterns: tuple[str, ...] = ()


# ---------------------------------------------------------------------------
# 各平台的常见安装位置（只列官方安装器 / 包管理器的公开路径）
# ---------------------------------------------------------------------------
def _windows_candidates() -> tuple[_EditorCandidate, ...]:
    local = os.environ.get("LOCALAPPDATA", r"%LOCALAPPDATA%")
    program_files = os.environ.get("ProgramFiles", r"%ProgramFiles%")
    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"%ProgramFiles(x86)%")
    toolbox = os.path.join(local, "JetBrains", "Toolbox", "apps")
    return (
        _EditorCandidate(
            "vscode",
            "Visual Studio Code",
            commands=("code",),
            paths=(
                os.path.join(local, "Programs", "Microsoft VS Code", "Code.exe"),
                os.path.join(program_files, "Microsoft VS Code", "Code.exe"),
                os.path.join(program_files_x86, "Microsoft VS Code", "Code.exe"),
            ),
        ),
        _EditorCandidate(
            "vscodium",
            "VSCodium",
            commands=("codium",),
            paths=(
                os.path.join(local, "Programs", "VSCodium", "VSCodium.exe"),
                os.path.join(program_files, "VSCodium", "VSCodium.exe"),
            ),
        ),
        _EditorCandidate(
            "cursor",
            "Cursor",
            commands=("cursor",),
            paths=(
                os.path.join(local, "Programs", "cursor", "Cursor.exe"),
                os.path.join(local, "Programs", "Cursor", "Cursor.exe"),
                os.path.join(program_files, "Cursor", "Cursor.exe"),
            ),
        ),
        _EditorCandidate(
            "pycharm",
            "PyCharm",
            commands=("pycharm", "pycharm64"),
            patterns=(
                os.path.join(local, "Programs", "PyCharm*", "bin", "pycharm64.exe"),
                os.path.join(program_files, "JetBrains", "PyCharm *", "bin", "pycharm64.exe"),
                os.path.join(toolbox, "PyCharm-*", "ch-*", "bin", "pycharm64.exe"),
            ),
        ),
        _EditorCandidate(
            "sublime",
            "Sublime Text",
            commands=("subl", "sublime_text"),
            paths=(
                os.path.join(program_files, "Sublime Text", "sublime_text.exe"),
                os.path.join(program_files, "Sublime Text 3", "sublime_text.exe"),
            ),
        ),
        _EditorCandidate(
            "notepadpp",
            "Notepad++",
            commands=("notepad++",),
            paths=(
                os.path.join(program_files, "Notepad++", "notepad++.exe"),
                os.path.join(program_files_x86, "Notepad++", "notepad++.exe"),
            ),
        ),
    )


def _macos_candidates() -> tuple[_EditorCandidate, ...]:
    applications = "/Applications"
    toolbox = os.path.expanduser("~/Library/Application Support/JetBrains/Toolbox/apps")
    return (
        _EditorCandidate(
            "vscode",
            "Visual Studio Code",
            commands=("code",),
            paths=(
                f"{applications}/Visual Studio Code.app/Contents/Resources/app/bin/code",
            ),
        ),
        _EditorCandidate(
            "vscodium",
            "VSCodium",
            commands=("codium",),
            paths=(f"{applications}/VSCodium.app/Contents/Resources/app/bin/codium",),
        ),
        _EditorCandidate(
            "cursor",
            "Cursor",
            commands=("cursor",),
            paths=(f"{applications}/Cursor.app/Contents/MacOS/Cursor",),
        ),
        _EditorCandidate(
            "pycharm",
            "PyCharm",
            commands=("pycharm",),
            paths=(
                f"{applications}/PyCharm.app/Contents/MacOS/pycharm",
                f"{applications}/PyCharm CE.app/Contents/MacOS/pycharm",
            ),
            patterns=(os.path.join(toolbox, "PyCharm-*", "ch-*", "bin", "pycharm"),),
        ),
        _EditorCandidate(
            "sublime",
            "Sublime Text",
            commands=("subl",),
            paths=(f"{applications}/Sublime Text.app/Contents/SharedSupport/bin/subl",),
        ),
    )


def _linux_candidates() -> tuple[_EditorCandidate, ...]:
    home = os.path.expanduser("~")
    toolbox = os.path.join(home, ".local", "share", "JetBrains", "Toolbox", "apps")
    return (
        _EditorCandidate(
            "vscode",
            "Visual Studio Code",
            commands=("code",),
            paths=(
                "/usr/share/code/code",
                "/usr/bin/code",
                "/snap/bin/code",
            ),
        ),
        _EditorCandidate(
            "vscodium",
            "VSCodium",
            commands=("codium",),
            paths=("/usr/bin/codium", "/snap/bin/codium"),
        ),
        _EditorCandidate(
            "cursor",
            "Cursor",
            commands=("cursor",),
            paths=("/usr/bin/cursor", os.path.join(home, ".local", "bin", "cursor")),
        ),
        _EditorCandidate(
            "pycharm",
            "PyCharm",
            commands=("pycharm", "pycharm.sh"),
            patterns=(
                os.path.join(toolbox, "PyCharm-*", "ch-*", "bin", "pycharm.sh"),
                "/opt/pycharm*/bin/pycharm.sh",
                "/snap/bin/pycharm*",
            ),
        ),
        _EditorCandidate(
            "sublime",
            "Sublime Text",
            commands=("subl", "sublime_text"),
            paths=("/usr/bin/subl",),
        ),
    )


def _candidate_specs() -> tuple[_EditorCandidate, ...]:
    """按当前平台返回候选编辑器（顺序即 ``auto`` 的优先级）。"""
    if sys.platform.startswith("win"):
        return _windows_candidates()
    if sys.platform == "darwin":
        return _macos_candidates()
    return _linux_candidates()


# ---------------------------------------------------------------------------
# 探测与解析
# ---------------------------------------------------------------------------
def _expand(raw: str) -> str:
    """展开 ``~`` 与 ``%VAR%`` / ``$VAR``，失败时原样返回。"""
    try:
        return os.path.expandvars(os.path.expanduser(raw))
    except (OSError, ValueError):
        return raw


def _existing_file(raw: str) -> str | None:
    """返回存在的文件路径；不存在 / 无权限时返回 ``None``。"""
    candidate = _expand(raw)
    if not candidate:
        return None
    try:
        if Path(candidate).is_file():
            return candidate
    except OSError:
        return None
    return None


def _glob_paths(pattern: str) -> list[str]:
    try:
        matches = glob.glob(_expand(pattern))
    except (OSError, ValueError):
        return []
    return sorted(match for match in matches if _existing_file(match))


def _resolve_candidate(candidate: _EditorCandidate) -> str | None:
    for command in candidate.commands:
        found = shutil.which(command)
        if found:
            return found
    for raw in candidate.paths:
        found = _existing_file(raw)
        if found:
            return found
    for pattern in candidate.patterns:
        matches = _glob_paths(pattern)
        if matches:
            return matches[0]
    return None


def _dedupe_key(executable: str) -> str:
    """同一可执行文件即使被多条规则命中，也只保留第一次出现的结果。"""
    return os.path.normcase(os.path.abspath(_expand(executable)))


def detect_editors() -> list[EditorInfo]:
    """探测本机已安装的编辑器。

    返回顺序即 ``auto`` 的优先级（VS Code → VSCodium → Cursor → PyCharm →
    Sublime Text → Notepad++）；同一个可执行文件只会出现一次。
    """

    editors: list[EditorInfo] = []
    seen: set[str] = set()
    for candidate in _candidate_specs():
        executable = _resolve_candidate(candidate)
        if not executable:
            continue
        marker = _dedupe_key(executable)
        if marker in seen:
            continue
        seen.add(marker)
        editors.append(
            EditorInfo(key=candidate.key, name=candidate.name, executable=executable)
        )
    return editors


def resolve_editor(value: str) -> EditorInfo | None:
    """把设置里的编辑器取值解析成 :class:`EditorInfo`。

    ``system`` 与无法解析的取值返回 ``None``；``auto`` 返回探测到的第一个编辑器
    （没有时也是 ``None``，由 :func:`launch_editor` 回退到系统默认程序）。
    """

    text = str(value or "").strip()
    if not text or text == AUTO_EDITOR:
        editors = detect_editors()
        return editors[0] if editors else None
    if text == SYSTEM_EDITOR:
        return None
    if text.startswith(CUSTOM_EDITOR_PREFIX):
        raw = text[len(CUSTOM_EDITOR_PREFIX) :].strip().strip('"').strip("'")
        executable = _existing_file(raw) if raw else None
        if not executable:
            return None
        return EditorInfo(
            key=text,
            name=Path(executable).name or "自定义编辑器",
            executable=executable,
        )
    for editor in detect_editors():
        if editor.key == text:
            return editor
    return None


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
def _command_for(editor: EditorInfo, target: Path) -> list[str]:
    """组装启动命令；Windows 的 ``.cmd`` / ``.bat`` 包装器交给 cmd.exe 执行。"""

    executable = editor.executable
    suffix = Path(executable).suffix.lower()
    if sys.platform.startswith("win") and suffix in {".cmd", ".bat"}:
        comspec = os.environ.get("COMSPEC") or "cmd.exe"
        # start 让 bat/cmd 在新进程中启动，不等待编辑器退出。
        return [comspec, "/c", "start", "", executable, str(target)]
    return [executable, str(target)]


def _popen(command: list[str]) -> None:
    kwargs: dict[str, object] = {"close_fds": True}
    if sys.platform.startswith("win"):
        # 通过 cmd.exe 启动批处理包装器时不要弹出控制台窗口。
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    subprocess.Popen(command, **kwargs)  # noqa: S603 - 命令来自探测结果或用户设置


def _open_with_system(target: Path) -> tuple[bool, str]:
    try:
        opened = QDesktopServices.openUrl(QUrl.fromLocalFile(str(target.resolve())))
    except Exception as exc:  # noqa: BLE001 - 系统关联失败时给出可操作提示
        return False, f"无法用系统默认程序打开：{exc}"
    if opened:
        return True, f"已用系统默认程序打开：{target.name}"
    return False, (
        f"系统默认程序没有打开“{target.name}”。"
        f"请在{SETTINGS_EDITOR_HINT}中手动指定一个编辑器路径。"
    )


def launch_editor(path: Path | str, value: str) -> tuple[bool, str]:
    """用指定编辑器打开文件，返回 ``(是否成功, 中文说明)``。

    只负责启动，不等待编辑器退出；失败时返回可操作的中文提示，不抛异常。
    """

    target = Path(path).expanduser()
    try:
        exists = target.is_file()
    except OSError:
        exists = False
    if not exists:
        return False, f"要打开的文件不存在：{target}"

    text = str(value or "").strip()
    editor = resolve_editor(text)
    if editor is None:
        if text in ("", AUTO_EDITOR, SYSTEM_EDITOR):
            return _open_with_system(target)
        return False, (
            "未找到可用的编辑器。"
            f"请在{SETTINGS_EDITOR_HINT}中选择系统默认程序，或手动指定编辑器可执行文件的完整路径。"
        )
    try:
        _popen(_command_for(editor, target))
    except OSError as exc:
        return False, (
            f"无法启动编辑器“{editor.name}”：{exc}。"
            f"请在{SETTINGS_EDITOR_HINT}中改为系统默认程序，或重新指定编辑器路径。"
        )
    return True, f"已用 {editor.name} 打开：{target.name}"


__all__ = [
    "AUTO_EDITOR",
    "CUSTOM_EDITOR_PREFIX",
    "SYSTEM_EDITOR",
    "EditorInfo",
    "detect_editors",
    "launch_editor",
    "resolve_editor",
]
