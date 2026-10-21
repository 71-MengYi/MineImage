"""PyInstaller 运行时钩子：修正被某些文件系统/杀软追加 NUL 填充的 .py 资源。

少数 Windows 环境（以及受限 CI 沙箱）在复制小文件时会把目标文件补齐到 4 KiB
块并在尾部留下 NUL 字节。Python 编译这些资源时会抛
``SyntaxError: source code string cannot contain null bytes``。本钩子在应用导入前
扫描 ``sys._MEIPASS`` 下的 ``*.py``，仅当 NUL 只出现在文件尾部时截断它们；正常
文件不会被改写。
"""

from __future__ import annotations

import sys
from pathlib import Path


def _normalize(root: Path) -> None:
    for path in root.rglob("*.py"):
        try:
            data = path.read_bytes()
            if b"\x00" not in data:
                continue
            cleaned = data.rstrip(b"\x00")
            if b"\x00" in cleaned:
                continue
            path.write_bytes(cleaned)
        except OSError:
            continue


_meipass = getattr(sys, "_MEIPASS", None)
if _meipass:
    try:
        _normalize(Path(_meipass))
    except Exception:  # noqa: BLE001 - 启动钩子绝不能阻断应用
        pass
