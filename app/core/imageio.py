"""图像文件读写（支持 Unicode 路径，尽量保持原始 dtype 与通道）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from app.core.types import ImageData, infer_color_space


class ImageIOError(RuntimeError):
    pass


def load_image(path: str | Path) -> ImageData:
    """读取图像文件为 :class:`ImageData`。

    优先使用 OpenCV（``IMREAD_UNCHANGED``，保留 16bit / alpha），失败后回退 Pillow。
    ``.npy`` 文件会直接读取数组，便于 float32 数据交换。
    """
    file_path = Path(path)
    if not file_path.exists():
        raise ImageIOError(f"文件不存在：{file_path}")
    if file_path.suffix.lower() == ".npy":
        try:
            array = np.load(str(file_path), allow_pickle=False)
        except Exception as exc:  # noqa: BLE001
            raise ImageIOError(f"读取 NumPy 图像失败：{exc}") from exc
        return ImageData(np.asarray(array), file_path.stem)

    try:
        data = file_path.read_bytes()
    except OSError as exc:
        raise ImageIOError(f"无法读取文件：{exc}") from exc

    array = None
    if data:
        buffer = np.frombuffer(data, dtype=np.uint8)
        array = cv2.imdecode(buffer, cv2.IMREAD_UNCHANGED)
    if array is None:
        array = _load_with_pillow_bgr(file_path)
    if array is None:
        raise ImageIOError(f"不支持的图像格式或文件损坏：{file_path.name}")

    array = np.asarray(array)
    if array.ndim == 3 and array.shape[2] == 1:
        array = array[:, :, 0]
    array = np.ascontiguousarray(array)
    return ImageData(array, file_path.name, infer_color_space(array))


def _load_with_pillow_bgr(path: Path) -> np.ndarray | None:
    """Pillow 回退读取，并转换为内部的 BGR / BGRA 约定。"""
    try:
        from PIL import Image
    except Exception:  # noqa: BLE001
        return None
    try:
        with Image.open(path) as pil:
            mode = pil.mode
            if mode in ("L", "I", "F", "I;16", "I;16B", "I;16L"):
                return np.array(pil)
            if mode == "P":
                pil = pil.convert("RGBA")
                mode = "RGBA"
            if mode == "LA":
                pil = pil.convert("RGBA")
                mode = "RGBA"
            if mode == "RGBA":
                array = np.array(pil)
                return array[:, :, [2, 1, 0, 3]]
            if mode in ("RGB", "CMYK", "YCbCr", "HSV"):
                array = np.array(pil.convert("RGB"))
                return array[:, :, ::-1]
            array = np.array(pil.convert("RGBA"))
            return array[:, :, [2, 1, 0, 3]]
    except Exception:  # noqa: BLE001
        return None


def save_image(path: str | Path, image: ImageData) -> None:
    """导出图像；扩展名决定编码格式。``.npy`` 保存原始数组。"""
    file_path = Path(path)
    suffix = file_path.suffix.lower()
    if suffix == ".npy":
        np.save(str(file_path), image.array)
        return
    encode_ext = suffix if suffix else ".png"
    success, buffer = cv2.imencode(encode_ext, image.array)
    if not success:
        raise ImageIOError(f"无法编码为 {encode_ext}：{file_path.name}")
    file_path.write_bytes(buffer.tobytes())
