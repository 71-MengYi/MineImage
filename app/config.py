"""应用设置：持久化到 .ini 文件（QSettings::IniFormat）。

设置统一由 :class:`AppSettings` 管理，UI 只通过属性读写，避免散落的 key。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

from PySide6.QtCore import QSettings, QStandardPaths

_CONFIG_FILE_NAME = "MineImage.ini"


def application_root() -> Path:
    """源码运行 / 打包运行时的应用根目录（可写目录，exe 所在目录）。"""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resource_root() -> Path:
    """只读资源根目录：源码树根目录，或 PyInstaller 临时解包目录。"""
    if getattr(sys, "frozen", False):
        meipass = getattr(sys, "_MEIPASS", None)
        if meipass:
            return Path(meipass)
    return application_root()


def default_config_path() -> Path:
    """返回便携式配置文件路径：``<应用目录>/config/MineImage.ini``。

    可通过环境变量 ``MINEIMAGE_CONFIG`` 指定其它 .ini 位置。若应用目录不可写
    （例如安装到 Program Files），回退到系统配置目录。
    """
    override = os.environ.get("MINEIMAGE_CONFIG", "").strip()
    if override:
        return Path(override).expanduser()
    portable = application_root() / "config" / _CONFIG_FILE_NAME
    try:
        portable.parent.mkdir(parents=True, exist_ok=True)
        probe = portable.parent / ".write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
        return portable
    except OSError:
        base = QStandardPaths.writableLocation(QStandardPaths.StandardLocation.AppConfigLocation)
        if not base:
            base = str(Path.home() / ".mineimage")
        return Path(base) / _CONFIG_FILE_NAME


def default_user_operator_dir() -> Path:
    """内置的“用户算子”目录；放到这里的 .py 文件会自动出现在算子库中。"""
    return application_root() / "operators"


class AppSettings:
    """对 QSettings 的强类型封装。

    配置文件默认为 ``<应用目录>/config/MineImage.ini``（便携式）。
    单元测试可以传入临时路径。
    """

    def __init__(self, path: str | Path | None = None) -> None:
        self.path = Path(path) if path is not None else default_config_path()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._qs = QSettings(str(self.path), QSettings.Format.IniFormat)
        self._qs.setFallbacksEnabled(False)

    # ------------------------------------------------------------------ 通用
    def sync(self) -> None:
        self._qs.sync()

    def value(self, key: str, default: Any = None, type: Any = None) -> Any:
        if type is not None:
            return self._qs.value(key, default, type=type)
        return self._qs.value(key, default)

    # --------------------------------------------------------------- 通用设置
    @property
    def undo_limit(self) -> int:
        return max(1, int(self.value("general/undo_limit", 100)))

    @undo_limit.setter
    def undo_limit(self, value: int) -> None:
        self._qs.setValue("general/undo_limit", max(1, int(value)))

    @property
    def show_node_thumbnails(self) -> bool:
        return self._as_bool(self.value("general/node_thumbnails", True))

    @show_node_thumbnails.setter
    def show_node_thumbnails(self, value: bool) -> None:
        self._qs.setValue("general/node_thumbnails", bool(value))

    @property
    def auto_fit_on_open(self) -> bool:
        return self._as_bool(self.value("preview/auto_fit_on_open", True))

    @auto_fit_on_open.setter
    def auto_fit_on_open(self, value: bool) -> None:
        self._qs.setValue("preview/auto_fit_on_open", bool(value))

    @property
    def downsample_preview(self) -> bool:
        """是否对预览显示做降采样（不改变计算得到的像素）。"""
        return self._as_bool(self.value("preview/downsample", False))

    @downsample_preview.setter
    def downsample_preview(self, value: bool) -> None:
        self._qs.setValue("preview/downsample", bool(value))

    @property
    def downsample_max_dim(self) -> int:
        return max(256, int(self.value("preview/downsample_max_dim", 2048)))

    @downsample_max_dim.setter
    def downsample_max_dim(self, value: int) -> None:
        self._qs.setValue("preview/downsample_max_dim", max(256, int(value)))

    @property
    def cursor_pixel_info(self) -> bool:
        return self._as_bool(self.value("preview/cursor_pixel_info", True))

    @cursor_pixel_info.setter
    def cursor_pixel_info(self, value: bool) -> None:
        self._qs.setValue("preview/cursor_pixel_info", bool(value))

    @property
    def node_thumbnail_size(self) -> int:
        return max(32, int(self.value("general/node_thumbnail_size", 72)))

    @node_thumbnail_size.setter
    def node_thumbnail_size(self, value: int) -> None:
        self._qs.setValue("general/node_thumbnail_size", max(32, int(value)))

    # --------------------------------------------------------------- 算子目录
    @property
    def extra_operator_dirs(self) -> list[str]:
        raw = self.value("operators/extra_dirs", [])
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw] if raw else []
        result: list[str] = []
        for item in raw:
            text = str(item).strip()
            if text:
                result.append(text)
        return result

    @extra_operator_dirs.setter
    def extra_operator_dirs(self, value: list[str]) -> None:
        cleaned = [str(item) for item in value if str(item).strip()]
        self._qs.setValue("operators/extra_dirs", cleaned)

    @property
    def builtin_operators_enabled(self) -> bool:
        return self._as_bool(self.value("operators/builtin_enabled", True))

    @builtin_operators_enabled.setter
    def builtin_operators_enabled(self, value: bool) -> None:
        self._qs.setValue("operators/builtin_enabled", bool(value))

    @property
    def watch_operator_dirs(self) -> bool:
        return self._as_bool(self.value("operators/watch", False))

    @watch_operator_dirs.setter
    def watch_operator_dirs(self, value: bool) -> None:
        self._qs.setValue("operators/watch", bool(value))

    # ------------------------------------------------------------------ 窗口
    @property
    def window_geometry(self) -> bytes | None:
        value = self._qs.value("window/geometry")
        return bytes(value) if value is not None else None

    @window_geometry.setter
    def window_geometry(self, value: bytes) -> None:
        self._qs.setValue("window/geometry", value)

    @property
    def window_state(self) -> bytes | None:
        value = self._qs.value("window/state")
        return bytes(value) if value is not None else None

    @window_state.setter
    def window_state(self, value: bytes) -> None:
        self._qs.setValue("window/state", value)

    @property
    def workspace_splitter_sizes(self) -> list[int]:
        # 默认三栏比例为 算子库 : 画布 : 显示 = 1:2:4。
        raw = self.value("window/workspace_sizes", [200, 400, 800])
        try:
            sizes = [int(x) for x in raw]
        except (TypeError, ValueError):
            return [200, 400, 800]
        return sizes if len(sizes) == 3 else [200, 400, 800]

    @workspace_splitter_sizes.setter
    def workspace_splitter_sizes(self, value: list[int]) -> None:
        self._qs.setValue("window/workspace_sizes", [int(x) for x in value])

    @property
    def recent_files(self) -> list[str]:
        raw = self.value("files/recent", [])
        if raw is None:
            return []
        if isinstance(raw, str):
            return [raw]
        return [str(x) for x in raw]

    def add_recent_file(self, path: str, limit: int = 10) -> None:
        items = [p for p in self.recent_files if p != path]
        items.insert(0, path)
        self._qs.setValue("files/recent", items[:limit])

    # ------------------------------------------------------------------ 工具
    @staticmethod
    def _as_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)
