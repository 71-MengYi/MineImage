# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller onefile 打包配置。

构建：``python -m PyInstaller --noconfirm --clean packaging/MineImage.spec``
"""

import os
from pathlib import Path

SPEC_DIR = Path(SPECPATH)  # noqa: F821 - 由 PyInstaller 注入
ROOT = SPEC_DIR.parent

# 内置算子以源码形式随包分发，供 loader 在运行时扫描；只收集 *.py，
# 避免把开发机上的 __pycache__/*.pyc 一并打进 exe。目录算子放在
# app/operators/builtin/catalog/ 子包中，需要递归收集并保留相对目录。
builtin_dir = ROOT / "app" / "operators" / "builtin"
builtin_datas = []
for source in sorted(builtin_dir.rglob("*.py")):
    if "__pycache__" in source.parts:
        continue
    relative = source.relative_to(builtin_dir)
    target = Path("app/operators/builtin") / relative.parent
    builtin_datas.append((str(source), str(target)))
datas = builtin_datas
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
    hiddenimports=[
        "cv2",
        "PySide6.QtSvg",
        "PySide6.QtOpenGL",
        # 目录算子在运行时按文件路径加载，但共享工具通过包导入访问。
        "app.operators.builtin.catalog",
        "app.operators.builtin.catalog._cv",
    ],
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
