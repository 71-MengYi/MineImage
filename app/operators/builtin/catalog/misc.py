"""OpenCV 通用调用与运行时信息算子。

这两个算子让工作台在 160+ 个精选目录算子之外，仍能直接触达 ``cv2`` 的其余
函数：``misc.cv2_call`` 按名称调用任意 OpenCV 函数，``misc.cv2_info`` 输出
当前 OpenCV 构建信息。参数只接受 JSON，不执行任意 Python 代码。
"""

from __future__ import annotations

import json
from typing import Any

import cv2
import numpy as np

from app.operators.builtin.catalog import _cv
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec


#: 会在后台线程阻塞或操作窗口 / 文件系统的函数，不允许通过通用调用执行。
_BLOCKED_FUNCTIONS = {
    "imshow", "imread", "imwrite", "imdecode", "imencode",
    "waitKey", "waitKeyEx", "namedWindow", "destroyWindow", "destroyAllWindows",
    "startWindowThread", "setWindowProperty", "setMouseCallback",
    "selectROI", "selectROIs", "createTrackbar",
    "VideoCapture", "VideoWriter",
}


class Cv2Call(Operator):
    """按名称调用任意 OpenCV 函数。"""

    spec = OperatorSpec(
        id="misc.cv2_call",
        name="OpenCV 通用调用",
        category=("其他", "OpenCV 通用"),
        description=(
            "按函数名调用当前 OpenCV 中的任意可调用函数，用于覆盖目录里尚未单独"
            "封装为节点的 cv2 API。第一个图像按位置传入函数；args_json 支持 JSON "
            "数组（位置参数）或对象（关键字参数），并用 \"$image\" / \"$image2\" "
            "占位符引用输入图像。返回数组时输出图像；返回数值时生成文本面板。"
        ),
        keywords=("opencv", "cv2", "通用", "调用", "function", "escape hatch"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec("image2", "第二图像（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "输出图像"),
        ),
        params=(
            ParamSpec(
                "function",
                "函数名",
                "string",
                default="GaussianBlur",
                tooltip="不带 cv2. 前缀，例如 GaussianBlur、bilateralFilter、Canny。",
            ),
            ParamSpec(
                "args_json",
                "参数 JSON",
                "string",
                default="[[5, 5], 0]",
                multiline=True,
                tooltip=(
                    "JSON 数组表示位置参数，JSON 对象表示关键字参数；"
                    "字符串 \"$image\" / \"$image2\" 会替换为输入数组。"
                ),
            ),
        ),
    )

    def _resolve_arguments(self, payload: Any, image: np.ndarray, second: np.ndarray | None):
        def resolve(value: Any) -> Any:
            if value == "$image":
                return image
            if value == "$image2":
                if second is None:
                    raise OperatorError("参数引用了 $image2，但第二图像端口未连接。")
                return second
            if isinstance(value, list):
                return [resolve(item) for item in value]
            if isinstance(value, dict):
                return {key: resolve(item) for key, item in value.items()}
            return value

        if isinstance(payload, list):
            return tuple(resolve(payload)), {}
        if isinstance(payload, dict):
            return (), resolve(payload)
        raise OperatorError("参数 JSON 必须是数组或对象。")

    def process(
        self,
        image=None,
        image2=None,
        function: str = "GaussianBlur",
        args_json: str = "[[5, 5], 0]",
        **_: Any,
    ):
        name = str(function or "").strip()
        if not name or name.startswith("_") or "." in name:
            raise OperatorError("函数名不能为空、不能以下划线开头，也不能包含点号。")
        if name in _BLOCKED_FUNCTIONS:
            raise OperatorError(f"“{name}”涉及窗口 / 文件 IO，不能在后台执行；请使用专门节点。")
        target = getattr(cv2, name, None)
        if target is None:
            raise OperatorError(f"当前 OpenCV 中找不到函数“{name}”，请检查名称。")
        if not callable(target):
            raise OperatorError(f"“{name}”不是可调用函数，请换一个函数名。")
        try:
            payload = json.loads(str(args_json or "[]"))
        except json.JSONDecodeError as exc:
            raise OperatorError(f"参数 JSON 解析失败：{exc.msg}（第 {exc.lineno} 行）。") from exc

        source = _cv.source(image, "输入图像")
        second = None if image2 is None else _cv.source(image2, "第二图像")
        args, kwargs = self._resolve_arguments(payload, source, second)
        try:
            result = target(source, *args, **kwargs)
        except OperatorError:
            raise
        except Exception as exc:  # noqa: BLE001 - OpenCV 的错误信息需要转成中文提示
            raise OperatorError(
                f"调用 {name} 失败：{exc.__class__.__name__}: {exc}。"
                "请检查参数类型与数量是否匹配该函数的签名。"
            ) from exc

        # 多返回值函数：取第一个图像结果；没有图像结果时取第一个标量。
        if isinstance(result, tuple):
            images = [item for item in result if isinstance(item, np.ndarray) and item.ndim in (2, 3)]
            if images:
                result = images[0]
            elif result:
                result = result[0]
            else:
                raise OperatorError(f"{name} 没有返回结果。")
        if isinstance(result, np.ndarray):
            if result.ndim in (2, 3) and result.size > 0:
                return result
            preview = np.asarray(result).ravel()[:8]
            return _cv.text_panel(
                420,
                140,
                [f"{name} 返回 {result.shape} 数组", np.array2string(preview, precision=4)],
            )
        if isinstance(result, (int, float, bool, np.number)):
            return _cv.text_panel(320, 80, [f"{name} 返回：{result}"])
        if result is None:
            raise OperatorError(f"{name} 返回了 None，未产生可显示的图像。")
        return _cv.text_panel(420, 120, [f"{name} 返回：{type(result).__name__}", str(result)[:120]])


class Cv2Info(Operator):
    """输出当前 OpenCV 版本与构建信息。"""

    spec = OperatorSpec(
        id="misc.cv2_info",
        name="OpenCV 构建信息",
        category=("其他", "OpenCV 通用"),
        description=(
            "把当前 OpenCV 版本、构建配置与关键模块可用性渲染成文本面板，"
            "方便确认运行环境是否包含 5.0 的新模块（dnn、stereo、3d 等）。"
        ),
        keywords=("opencv", "版本", "构建", "信息", "version", "build"),
        inputs=(),
        outputs=(PortSpec("image", "信息面板"),),
    )

    def process(self, **_: Any):
        lines = [
            f"OpenCV 版本：{cv2.__version__}",
            "Python 绑定：cv2",
        ]
        build = ""
        try:
            build = cv2.getBuildInformation()
        except Exception:  # noqa: BLE001 - 某些精简构建没有 getBuildInformation
            build = ""
        wanted = ("Version control", "Parallel framework", "FFMPEG", "Intel", "OpenCL", "Python")
        for raw in build.splitlines():
            text = raw.strip()
            if any(text.startswith(prefix) for prefix in wanted):
                lines.append(text[:64])
        modules = [
            ("dnn", hasattr(cv2, "dnn")),
            ("photo", hasattr(cv2, "inpaint")),
            ("features2d", hasattr(cv2, "SIFT_create")),
            ("objdetect", hasattr(cv2, "CascadeClassifier")),
            ("stitching", hasattr(cv2, "Stitcher_create")),
            ("3d/stereo (5.0)", hasattr(cv2, "stereo") or hasattr(cv2, "StereoMatcher")),
        ]
        lines.append("模块：" + "  ".join(f"{name}={'✓' if ok else '×'}" for name, ok in modules))
        return _cv.text_panel(620, 190, lines)


OPERATORS = (Cv2Call, Cv2Info)
