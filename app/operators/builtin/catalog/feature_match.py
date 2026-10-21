"""特征匹配算子（features2d / FlannBasedMatcher 覆盖）。

``feature.match`` 提供一张可直接预览的匹配连线图，并输出匹配点数；支持
SIFT / ORB / AKAZE / BRISK 检测器与 BF / FLANN 两种匹配器，便于覆盖
OpenCV features2d 模块的 DescriptorMatcher 能力。
"""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from app.core.data_types import DataType, NumberValue
from app.operators.builtin.catalog import _cv
from app.operators.spec import Operator, OperatorError, OperatorSpec, ParamSpec, PortSpec

#: 检测器名称 -> (创建函数, 描述子距离)
_DETECTORS = {
    "sift": ("SIFT_create", cv2.NORM_L2),
    "orb": ("ORB_create", cv2.NORM_HAMMING),
    "akaze": ("AKAZE_create", cv2.NORM_HAMMING),
    "brisk": ("BRISK_create", cv2.NORM_HAMMING),
}


class FeatureMatch(Operator):
    """两幅图像的特征点匹配与连线可视化。"""

    spec = OperatorSpec(
        id="feature.match",
        name="特征匹配",
        category=("特征与分析", "匹配"),
        description=(
            "在两幅图像上提取 SIFT / ORB / AKAZE / BRISK 特征，用 BF 或 FLANN "
            "匹配器做 KNN 匹配，并按 Lowe 比例测试筛选；输出带连线的匹配图与匹配点数。"
            "第二幅图未连接时自动用第一幅图水平平移生成，便于单图试跑。"
        ),
        keywords=("特征", "匹配", "match", "sift", "orb", "flann", "bfmatcher", "features2d"),
        inputs=(
            PortSpec("image", "输入图像"),
            PortSpec("image2", "第二图像（可选）", optional=True),
        ),
        outputs=(
            PortSpec("image", "匹配连线图", accepts=("BGR",)),
            PortSpec("count", "匹配点数", data_type=DataType.NUMBER),
        ),
        params=(
            ParamSpec(
                "detector",
                "特征检测器",
                "enum",
                choices=(
                    ("sift", "SIFT（尺度不变）"),
                    ("orb", "ORB（快速）"),
                    ("akaze", "AKAZE"),
                    ("brisk", "BRISK"),
                ),
                default="sift",
            ),
            ParamSpec(
                "matcher",
                "匹配器",
                "enum",
                choices=(("bf", "暴力匹配 BF"), ("flann", "FLANN 近似")),
                default="bf",
            ),
            ParamSpec(
                "ratio",
                "比例测试阈值",
                "float",
                default=0.75,
                min=0.1,
                max=1.0,
                step=0.05,
                tooltip="越小越严格，通常取 0.7–0.8。",
            ),
            ParamSpec(
                "max_features",
                "最大特征点数",
                "int",
                default=800,
                min=50,
                max=10000,
                step=50,
            ),
            ParamSpec(
                "cross_check",
                "双向交叉校验",
                "bool",
                default=False,
                tooltip="仅 BF 匹配器有效；开启时忽略比例测试。",
            ),
        ),
    )

    def process(
        self,
        image=None,
        image2=None,
        detector: str = "sift",
        matcher: str = "bf",
        ratio: float = 0.75,
        max_features: int = 800,
        cross_check: bool = False,
        **_: Any,
    ):
        first = _cv.source_gray8(image, "输入图像")
        if image2 is None:
            second = np.roll(first, shift=max(1, first.shape[1] // 20), axis=1)
        else:
            second = _cv.source_gray8(image2, "第二图像")
            _cv.same_size(first, second, "第二图像")

        if detector not in _DETECTORS:
            raise OperatorError(f"不支持的特征检测器：{detector}")
        creator_name, norm = _DETECTORS[detector]
        creator = getattr(cv2, creator_name, None)
        if creator is None:
            raise OperatorError(f"当前 OpenCV 没有 {creator_name}（需要 contrib 构建）。")
        descriptor = creator()
        if hasattr(descriptor, "setMaxFeatures"):
            descriptor.setMaxFeatures(int(max_features))

        keypoints1, descriptors1 = descriptor.detectAndCompute(first, None)
        keypoints2, descriptors2 = descriptor.detectAndCompute(second, None)
        if descriptors1 is None or descriptors2 is None or len(keypoints1) < 2 or len(keypoints2) < 2:
            return {
                "image": cv2.cvtColor(np.hstack([first, second]), cv2.COLOR_GRAY2BGR),
                "count": NumberValue(0.0, "匹配点数"),
            }

        if matcher == "flann":
            if norm == cv2.NORM_HAMMING:
                index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
            else:
                index_params = dict(algorithm=1, trees=5)
            search_params = dict(checks=50)
            matcher_impl = cv2.FlannBasedMatcher(index_params, search_params)
        else:
            matcher_impl = cv2.BFMatcher(norm, crossCheck=bool(cross_check))

        good: list[Any] = []
        try:
            if cross_check and matcher == "bf":
                good = list(matcher_impl.match(descriptors1, descriptors2))
            else:
                raw = matcher_impl.knnMatch(descriptors1, descriptors2, k=2)
                for pair in raw:
                    if len(pair) < 2:
                        continue
                    best, second_best = pair
                    if best.distance < float(ratio) * max(second_best.distance, 1e-6):
                        good.append(best)
        except cv2.error as exc:
            raise OperatorError(f"特征匹配失败：{exc}") from exc
        good.sort(key=lambda item: item.distance)

        canvas = cv2.drawMatches(
            first,
            keypoints1,
            second,
            keypoints2,
            good,
            None,
            matchColor=(0, 220, 0),
            singlePointColor=(220, 120, 0),
            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS,
        )
        return {
            "image": canvas,
            "count": NumberValue(float(len(good)), "匹配点数"),
        }


OPERATORS = (FeatureMatch,)
