# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onefile 打包配置。

构建：``python -m PyInstaller --noconfirm --clean packaging/MineImage.spec``
"""

import os
from pathlib import Path

SPEC_DIR = Path(SPECPATH)  # noqa: F821 - 由 PyInstaller 注入
ROOT = SPEC_DIR.parent

# 内置算子以源码形式随包分发，供 loader 在运行时扫描；只收集 *.py，
# 避免把开发机上的 __pycache__/*.pyc 一并打进 exe。
builtin_dir = ROOT / "app" / "operators" / "builtin"
datas = [(str(path), "app/operators/builtin") for path in sorted(builtin_dir.glob("*.py"))]
datas += [
    (str(ROOT / "assets"), "assets"),
    (str(ROOT / "docs"), "docs"),
    (str(ROOT / "operators" / "README.md"), "operators"),
    (str(ROOT / "operators" / "_template.py"), "operators"),
]
license_path = ROOT / "LICENSE"
if license_path.exists():
    datas.append((str(license_path), "."))

icon_path = ROOT / "assets" / "icon.ico"
icon = str(icon_path) if icon_path.exists() else None

a = Analysis(
    [str(SPEC_DIR / "launcher.py")],
    pathex=[str(ROOT)],
    binaries=[],
    datas=datas,
    hiddenimports=["cv2", "PySide6.QtSvg"],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[str(ROOT / "packaging" / "rthook_normalize_resources.py")],
    excludes=["tkinter", "matplotlib", "pytest"],
    noarchive=False,
    optimize=1,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name="MineImage",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=icon,
)
