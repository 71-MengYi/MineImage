"""节点内图像缩略图拖拽：载荷数据与进程内 token 注册表。

QDrag 的 MIME 数据只携带一个短期 token，真正的图像副本、来源节点等信息保存在
模块级注册表中。源节点在 QDrag.exec 返回后释放 token；同一进程内的其它
标签页 / 画布只要拿到 MIME 数据就能在本进程里查回完整载荷。
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QMimeData

from app.core.constants import MIME_IMAGE_ITEM
from app.core.types import ImageData


@dataclass(frozen=True)
class ImageDragPayload:
    """从节点图像矩阵中拖出的一张图像副本。"""

    image: ImageData
    name: str
    path: str | None
    source_node_id: str
    source_index: int
    source_title: str


#: token -> payload；仅 GUI 线程使用，跨画布 / 标签页共享。
_PAYLOADS: dict[str, ImageDragPayload] = {}


def register(payload: ImageDragPayload) -> str:
    """登记载荷并返回一次性 token。"""

    token = uuid.uuid4().hex
    _PAYLOADS[token] = payload
    return token


def lookup(token: str | None) -> ImageDragPayload | None:
    """按 token 查询载荷；token 无效时返回 None。"""

    if not token:
        return None
    return _PAYLOADS.get(str(token))


def release(token: str | None) -> None:
    """释放 token；重复释放或 token 为空都安全。"""

    if not token:
        return
    _PAYLOADS.pop(str(token), None)


def clear() -> None:
    """清空全部载荷（测试 / 工作区关闭时使用）。"""

    _PAYLOADS.clear()


def build_mime_data(token: str) -> QMimeData:
    """构造携带 token 的 QMimeData（MIME_IMAGE_ITEM）。"""

    mime = QMimeData()
    payload = json.dumps({"token": str(token)}, ensure_ascii=False).encode("utf-8")
    mime.setData(MIME_IMAGE_ITEM, payload)
    return mime


def token_from_mime(mime: Any) -> str | None:
    """从拖拽 MIME 数据中解析 token；兼容 JSON 对象与纯 token 两种写法。"""

    if mime is None or not mime.hasFormat(MIME_IMAGE_ITEM):
        return None
    try:
        raw = bytes(mime.data(MIME_IMAGE_ITEM)).decode("utf-8", "replace").strip()
    except Exception:  # noqa: BLE001 - 非法 MIME 数据按无效处理
        return None
    if not raw:
        return None
    try:
        data = json.loads(raw)
    except (TypeError, ValueError):
        return raw
    if isinstance(data, dict):
        token = data.get("token") or data.get("id")
    elif isinstance(data, str):
        token = data
    else:
        token = None
    return str(token) if token else None


def payload_from_mime(mime: Any) -> ImageDragPayload | None:
    """从拖拽 MIME 数据取回载荷；token 无效时返回 None。"""

    return lookup(token_from_mime(mime))


def mime_for_payload(payload: ImageDragPayload) -> QMimeData:
    """登记载荷并返回对应 MIME 数据。

    便捷函数；调用方若需要显式释放 token，请改用 register + build_mime_data。
    """

    return build_mime_data(register(payload))


__all__ = [
    "ImageDragPayload",
    "build_mime_data",
    "clear",
    "lookup",
    "mime_for_payload",
    "payload_from_mime",
    "register",
    "release",
    "token_from_mime",
]
