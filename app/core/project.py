"""``.mi`` 工程文件：zip 容器，JSON 描述 DAG，内嵌源图像数组。

布局::

    project.json
    images/<image_id>.npz

源图像以压缩 npz 保存，因此 uint16 / float32 / alpha 都能无损往返。
"""

from __future__ import annotations

import io
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from app.core.constants import APP_VERSION, PROJECT_EXTENSION, PROJECT_IMAGE_DIR
from app.core.model import GraphModel, NodeModel, WorkspaceModel
from app.core.types import ImageData

FORMAT_NAME = "mineimage-project"
FORMAT_VERSION = 1


class ProjectFormatError(RuntimeError):
    pass


def normalize_project_path(path: str | Path) -> Path:
    target = Path(path)
    if target.suffix.lower() != PROJECT_EXTENSION:
        target = target.with_suffix(PROJECT_EXTENSION)
    return target


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_project(path: str | Path, workspace: WorkspaceModel) -> Path:
    """原子地写出 ``.mi`` 工程文件，返回最终路径。"""
    target = normalize_project_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    workspace.prune_unused_images()

    images_payload: list[dict[str, Any]] = []
    image_arrays: dict[str, ImageData] = {}
    for image_id, image in workspace.images.items():
        images_payload.append(
            {
                "id": image_id,
                "name": image.name,
                "path": workspace.image_paths.get(image_id),
                "array_file": f"{PROJECT_IMAGE_DIR}/{image_id}.npz",
                "color_space": image.color_space,
                "dtype": image.dtype_name,
            }
        )
        image_arrays[image_id] = image

    document = {
        "format": FORMAT_NAME,
        "format_version": FORMAT_VERSION,
        "app_version": APP_VERSION,
        "workspace": {
            "name": workspace.name,
            "graph": _jsonable(workspace.graph.to_dict()),
            "images": images_payload,
        },
    }

    handle = tempfile.NamedTemporaryFile(
        prefix=".mineimage_", suffix=".mi.tmp", dir=str(target.parent), delete=False
    )
    temp_path = Path(handle.name)
    handle.close()
    try:
        with zipfile.ZipFile(temp_path, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            archive.writestr(
                "project.json",
                json.dumps(document, ensure_ascii=False, indent=2),
            )
            for image_id, image in image_arrays.items():
                buffer = io.BytesIO()
                np.savez_compressed(buffer, array=image.array)
                archive.writestr(f"{PROJECT_IMAGE_DIR}/{image_id}.npz", buffer.getvalue())
        os.replace(temp_path, target)
    except Exception:
        temp_path.unlink(missing_ok=True)
        raise
    return target


def load_project(path: str | Path) -> WorkspaceModel:
    source = Path(path)
    if not source.exists():
        raise ProjectFormatError(f"工程文件不存在：{source}")
    try:
        archive = zipfile.ZipFile(source, "r")
    except zipfile.BadZipFile as exc:
        raise ProjectFormatError(f"不是有效的 .mi 工程文件：{exc}") from exc
    except OSError as exc:
        raise ProjectFormatError(f"无法读取工程文件：{exc}") from exc

    with archive:
        try:
            raw_document = archive.read("project.json")
        except KeyError as exc:
            raise ProjectFormatError("工程文件缺少 project.json") from exc
        try:
            document = json.loads(raw_document.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ProjectFormatError(f"工程描述 JSON 损坏：{exc}") from exc
        if document.get("format") != FORMAT_NAME:
            raise ProjectFormatError("不是 MineImage 工程文件")
        if int(document.get("format_version", 0)) > FORMAT_VERSION:
            raise ProjectFormatError("工程由更高版本的 MineImage 创建，当前版本无法打开")

        workspace_data = document.get("workspace") or {}
        workspace = WorkspaceModel(str(workspace_data.get("name") or source.stem))
        for image_meta in workspace_data.get("images") or []:
            image_id = str(image_meta.get("id") or "")
            if not image_id:
                continue
            array_file = str(image_meta.get("array_file") or f"{PROJECT_IMAGE_DIR}/{image_id}.npz")
            try:
                payload = archive.read(array_file)
            except KeyError as exc:
                raise ProjectFormatError(f"工程缺少图像数据：{array_file}") from exc
            try:
                with np.load(io.BytesIO(payload), allow_pickle=False) as npz:
                    array = np.array(npz["array"], copy=True)
            except Exception as exc:  # noqa: BLE001
                raise ProjectFormatError(f"图像数据损坏：{array_file}（{exc}）") from exc
            image = ImageData(
                array,
                str(image_meta.get("name") or image_id),
                image_meta.get("color_space"),
            )
            workspace.images[image_id] = image
            original_path = image_meta.get("path")
            workspace.image_paths[image_id] = str(original_path) if original_path else None

        graph_data = workspace_data.get("graph") or {}
        try:
            graph = GraphModel.from_dict(graph_data)
        except Exception as exc:  # noqa: BLE001
            raise ProjectFormatError(f"流程图数据损坏：{exc}") from exc
        missing = [
            node.id
            for node in graph.nodes.values()
            if node.is_source and (not node.source_image_id or node.source_image_id not in workspace.images)
        ]
        if missing:
            raise ProjectFormatError(f"流程节点引用了缺失的图像：{', '.join(missing)}")
        old_graph = workspace.graph
        old_graph.setParent(None)
        old_graph.deleteLater()
        workspace.graph = graph
        graph.setParent(workspace)
        return workspace
