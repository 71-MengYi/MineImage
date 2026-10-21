"""OpenCV 官方 ONNX 模型：下载地址、本地缓存与自动下载。

OpenCV 自 4.5.4 起提供 `cv2.FaceDetectorYN` 等 DNN 检测器，但模型文件（如
YuNet 的 `face_detection_yunet_2023mar.onnx`）不随 `opencv-python` 一起安装。
OpenCV 5.0 又移除了 `cv2.CascadeClassifier`，因此人脸检测等算子必须显式提供
ONNX 模型文件。

本模块集中维护模型清单（官方 OpenCV Zoo 下载地址）与下载逻辑：

* 默认缓存目录为 `<应用目录>/models`，可用环境变量
  `MINEIMAGE_MODEL_DIR` 覆盖（便携版 / 打包版建议放在 exe 同级的 models）；
* `ensure_model()` 在模型缺失时可自动下载（可用
  `MINEIMAGE_AUTO_DOWNLOAD_MODELS=0` 全局关闭）；
* 下载先写入同目录的 `.part` 临时文件，校验大小、HTML 错误页与 SHA-256 后
  再原子替换，避免留下半截文件；
* 本模块只依赖标准库，可安全地在后台线程 / 无 GUI 测试中调用。
"""

from __future__ import annotations

import hashlib
import os
import sys
import tempfile
import threading
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from app.core.constants import APP_VERSION

#: 下载 / 读取网络数据的默认超时（秒）。大模型下载慢时可显式传更大的 timeout。
DEFAULT_TIMEOUT = 60.0

#: 每次从网络读取的字节数。
CHUNK_SIZE = 256 * 1024

#: OpenCV Zoo 模型仓库首页。
OPENCV_ZOO_URL = "https://github.com/opencv/opencv_zoo"

#: OpenCV Zoo 人脸检测 YuNet 模型页面。
OPENCV_ZOO_YUNET_URL = "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet"

#: OpenCV Zoo 人脸识别 SFace 模型页面。
OPENCV_ZOO_SFACE_URL = "https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface"

ProgressCallback = Callable[[int, int], None]
"""下载进度回调：`(已下载字节, 总字节)`；总字节未知时为 0。"""

CancelCallback = Callable[[], bool]
"""返回 True 时中止下载。"""


class ModelError(RuntimeError):
    """模型相关的通用错误。"""


class ModelNotFoundError(ModelError):
    """模型文件不存在，且未启用自动下载 / 自动下载失败。"""


class ModelDownloadError(ModelError):
    """模型下载或校验失败。"""


@dataclass(frozen=True)
class ModelSpec:
    """一个可下载模型的元数据。"""

    key: str
    name: str
    filename: str
    url: str
    mirrors: tuple[str, ...] = ()
    homepage: str = ""
    description: str = ""
    min_size: int = 1024
    sha256: str = ""
    size: int = 0

    @property
    def urls(self) -> tuple[str, ...]:
        return (self.url, *self.mirrors)


#: 内置模型清单。地址来自 OpenCV Zoo，均为官方原始文件地址。
MODELS: dict[str, ModelSpec] = {
    "face_detection_yunet": ModelSpec(
        key="face_detection_yunet",
        name="YuNet 人脸检测模型",
        filename="face_detection_yunet_2023mar.onnx",
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
        ),
        # GitHub API 返回的 LFS download_url；raw 地址重定向失败时作为备用。
        mirrors=(
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
            "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ),
        homepage=OPENCV_ZOO_YUNET_URL,
        description=(
            "OpenCV Zoo 官方 YuNet 正面人脸检测模型（约 230 KB）。"
            "OpenCV 5.0 移除了 Haar 级联后，「人脸检测（YuNet）」算子使用该模型。"
        ),
        min_size=100_000,
        size=232_589,
    ),
    "face_recognition_sface": ModelSpec(
        key="face_recognition_sface",
        name="SFace 人脸识别模型",
        filename="face_recognition_sface_2021dec.onnx",
        url=(
            "https://github.com/opencv/opencv_zoo/raw/main/"
            "models/face_recognition_sface/face_recognition_sface_2021dec.onnx"
        ),
        mirrors=(
            "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
            "models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        ),
        homepage=OPENCV_ZOO_SFACE_URL,
        description=(
            "OpenCV Zoo 官方 SFace 人脸特征模型（约 37 MB），"
            "用于 `cv2.FaceRecognizerSF` 的人脸特征提取与比对。"
        ),
        min_size=1_000_000,
        size=38_696_353,
    ),
}

# 进程内按目标路径串行下载，避免算子执行线程与界面下载按钮同时写同一个文件。
_PATH_LOCKS: dict[str, threading.Lock] = {}
_PATH_LOCKS_GUARD = threading.Lock()


def _lock_for(path: Path) -> threading.Lock:
    key = str(path)
    with _PATH_LOCKS_GUARD:
        lock = _PATH_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _PATH_LOCKS[key] = lock
        return lock


def _application_root() -> Path:
    """源码运行 / 打包运行时的应用根目录（与 `app.config.application_root` 一致）。"""

    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


#: 默认模型目录的进程级缓存（应用根目录不可写时回退到用户目录）。
_DEFAULT_MODEL_DIR: Path | None = None


def _default_model_dir() -> Path:
    global _DEFAULT_MODEL_DIR
    if _DEFAULT_MODEL_DIR is not None:
        return _DEFAULT_MODEL_DIR
    root = _application_root()
    try:
        root.mkdir(parents=True, exist_ok=True)
        # 与 app.config.default_config_path 相同的可写性探测：安装在
        # Program Files 等只读目录时，把模型缓存放到用户目录。
        probe = root / ".models_write_probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink(missing_ok=True)
    except OSError:
        root = Path.home() / ".mineimage"
    _DEFAULT_MODEL_DIR = root / "models"
    return _DEFAULT_MODEL_DIR


def model_dir() -> Path:
    """模型缓存目录；`MINEIMAGE_MODEL_DIR` 环境变量优先。

    默认是 `<应用目录>/models`；若应用目录不可写（例如打包版安装在
    `Program Files`），回退到 `~/.mineimage/models`。
    """

    override = os.environ.get("MINEIMAGE_MODEL_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    return _default_model_dir()


def auto_download_enabled() -> bool:
    """是否允许自动下载模型；默认开启，设置 `MINEIMAGE_AUTO_DOWNLOAD_MODELS=0` 关闭。"""

    raw = os.environ.get("MINEIMAGE_AUTO_DOWNLOAD_MODELS", "1")
    return str(raw).strip().lower() not in {"0", "false", "no", "off", ""}


def resolve_model(key_or_spec: str | ModelSpec) -> ModelSpec:
    """按 key 或直接传入 :class:`ModelSpec` 取模型定义。"""

    if isinstance(key_or_spec, ModelSpec):
        return key_or_spec
    key = str(key_or_spec)
    spec = MODELS.get(key)
    if spec is None:
        raise ModelError(
            f"未知模型：{key}。可用模型：{'、'.join(sorted(MODELS)) or '（无）'}"
        )
    return spec


def model_path(key_or_spec: str | ModelSpec) -> Path:
    """模型在本机的缓存路径（不保证文件存在）。"""

    spec = resolve_model(key_or_spec)
    return model_dir() / spec.filename


def model_size(key_or_spec: str | ModelSpec) -> int:
    """已缓存的模型字节数；文件不存在时返回 0。"""

    path = model_path(key_or_spec)
    try:
        return path.stat().st_size if path.is_file() else 0
    except OSError:
        return 0


def is_model_ready(key_or_spec: str | ModelSpec) -> bool:
    spec = resolve_model(key_or_spec)
    size = model_size(spec)
    return size > 0 and size >= max(int(spec.min_size), 1)


def model_url(key_or_spec: str | ModelSpec) -> str:
    """模型官方下载地址。"""

    return resolve_model(key_or_spec).url


def download_url(key_or_spec: str | ModelSpec) -> str:
    """兼容别名：模型官方下载地址。"""

    return model_url(key_or_spec)


def format_size(size: int) -> str:
    """把字节数格式化为人类可读文本。"""

    value = float(max(int(size), 0))
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024.0 or unit == "GB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} GB"


def missing_model_message(key_or_spec: str | ModelSpec) -> str:
    """模型缺失时给用户看的中文提示（含下载地址与两种解决方式）。"""

    spec = resolve_model(key_or_spec)
    path = model_path(spec)
    return (
        f"缺少{spec.name}：{path}\n"
        f"下载地址：{spec.url}\n"
        "可点击参数面板中的「下载模型」自动下载，或手动下载后把文件放到上述目录，"
        "也可以在算子参数里选择已下载的模型文件。"
    )


def model_status_text(key_or_spec: str | ModelSpec) -> str:
    """一行状态文本，供参数面板 / 设置界面显示。"""

    spec = resolve_model(key_or_spec)
    size = model_size(spec)
    if size > 0 and size >= max(int(spec.min_size), 1):
        return f"已就绪 · {format_size(size)}"
    if size > 0:
        return f"文件异常（{format_size(size)}，可能未下载完整）"
    return "未下载"


def _emit(progress: ProgressCallback | None, downloaded: int, total: int) -> None:
    if progress is not None:
        progress(downloaded, total)


def _check_cancel(cancel: CancelCallback | None) -> None:
    if cancel is not None and cancel():
        raise ModelDownloadError("已取消模型下载。")


def _looks_like_html(path: Path) -> bool:
    try:
        with path.open("rb") as handle:
            head = handle.read(512).lstrip().lower()
    except OSError:
        return False
    return head.startswith(b"<!doctype") or head.startswith(b"<html") or b"<head>" in head[:256]


def _sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_one(
    spec: ModelSpec,
    url: str,
    target: Path,
    *,
    progress: ProgressCallback | None,
    cancel: CancelCallback | None,
    timeout: float,
) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "User-Agent": f"MineImage/{APP_VERSION} (OpenCV model downloader)",
            "Accept": "application/octet-stream,*/*",
        },
    )
    fd, temp_name = tempfile.mkstemp(
        prefix=f"{target.name}.",
        suffix=".part",
        dir=str(target.parent),
    )
    temp_path = Path(temp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                length = response.headers.get("Content-Length")
                total = int(length) if length and str(length).isdigit() else int(spec.size or 0)
                downloaded = 0
                _emit(progress, 0, total)
                while True:
                    _check_cancel(cancel)
                    chunk = response.read(CHUNK_SIZE)
                    if not chunk:
                        break
                    handle.write(chunk)
                    downloaded += len(chunk)
                    _emit(progress, downloaded, total)
            handle.flush()
            os.fsync(handle.fileno())
        _check_cancel(cancel)
        size = temp_path.stat().st_size
        if size < max(int(spec.min_size), 1):
            raise ModelDownloadError(
                f"下载内容过小（{format_size(size)}），可能下载到了错误页面。"
            )
        if _looks_like_html(temp_path):
            raise ModelDownloadError("下载内容看起来是 HTML 页面而不是模型文件。")
        if spec.sha256:
            actual = _sha256_of(temp_path)
            if actual.lower() != spec.sha256.lower():
                raise ModelDownloadError("模型文件校验失败（SHA-256 不匹配）。")
        _check_cancel(cancel)
        os.replace(temp_path, target)
        temp_path = None  # 已成功替换，finally 中不再删除
    except ModelDownloadError:
        raise
    except urllib.error.HTTPError as exc:
        raise ModelDownloadError(f"下载失败：HTTP {exc.code} {exc.reason}") from exc
    except urllib.error.URLError as exc:
        raise ModelDownloadError(f"下载失败：{exc.reason}") from exc
    except TimeoutError as exc:
        raise ModelDownloadError("下载超时，请检查网络后重试。") from exc
    except OSError as exc:
        raise ModelDownloadError(f"下载失败：{exc}") from exc
    finally:
        if temp_path is not None:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass


def download_model(
    key_or_spec: str | ModelSpec,
    *,
    target: str | Path | None = None,
    force: bool = False,
    progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Path:
    """下载模型到本地缓存（或指定的 `target`）。

    * 已存在且大小合法时直接返回路径；`force=True` 时重新下载；
    * 所有官方地址都失败时抛出 :class:`ModelDownloadError`，错误信息包含下载地址；
    * `progress` / `cancel` 可在后台线程中使用。
    """

    spec = resolve_model(key_or_spec)
    target_path = Path(target) if target is not None else model_path(spec)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    lock = _lock_for(target_path)
    with lock:
        if not force and target_path.is_file() and target_path.stat().st_size >= max(int(spec.min_size), 1):
            return target_path
        errors: list[str] = []
        for url in spec.urls:
            _check_cancel(cancel)
            try:
                _download_one(
                    spec,
                    url,
                    target_path,
                    progress=progress,
                    cancel=cancel,
                    timeout=timeout,
                )
                return target_path
            except ModelDownloadError as exc:
                errors.append(str(exc))
        detail = "；".join(dict.fromkeys(errors)) or "未知错误"
        raise ModelDownloadError(
            f"模型下载失败：{spec.name}。{detail}\n"
            f"手动下载地址：{spec.url}\n"
            f"下载后放到：{target_path}"
        )


def ensure_model(
    key_or_spec: str | ModelSpec,
    *,
    auto: bool | None = None,
    force: bool = False,
    progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Path:
    """确保模型存在：已缓存直接返回；缺失时按 `auto` / 全局开关决定是否下载。"""

    spec = resolve_model(key_or_spec)
    path = model_path(spec)
    if not force and path.is_file() and path.stat().st_size >= max(int(spec.min_size), 1):
        return path
    if auto is None:
        auto = auto_download_enabled()
    if not auto:
        raise ModelNotFoundError(missing_model_message(spec))
    try:
        return download_model(
            spec,
            target=path,
            force=force,
            progress=progress,
            cancel=cancel,
            timeout=timeout,
        )
    except ModelDownloadError as exc:
        raise ModelDownloadError(f"{exc}\n可稍后重试，或按上述地址手动下载。") from exc
