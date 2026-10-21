"""连线路径的共享几何工具。

端口改为上下方向后，连线统一使用“出口向下、入口自上而下”的三次贝塞尔曲线。
画布中的正式连线与拖拽中的临时连线都复用这里，避免两处路径形状不一致。
"""

from __future__ import annotations

from PySide6.QtCore import QPointF
from PySide6.QtGui import QPainterPath

#: 纵向曲线的最小控制点偏移，保证上下端点接近时仍能看出连接方向。
VERTICAL_MIN_BEND = 64.0


def connection_path(
    start: QPointF,
    end: QPointF,
    min_bend: float = VERTICAL_MIN_BEND,
) -> QPainterPath:
    """生成从上/下端口出发的纵向三次贝塞尔曲线。

    * ``start`` 通常是输出端口（节点下边缘），曲线先向下；
    * ``end`` 通常是输入端口（节点上边缘），曲线从上方进入。
    """

    start = QPointF(start)
    end = QPointF(end)
    bend = max(float(min_bend), abs(end.y() - start.y()) * 0.55)
    path = QPainterPath(start)
    path.cubicTo(
        QPointF(start.x(), start.y() + bend),
        QPointF(end.x(), end.y() - bend),
        end,
    )
    return path


__all__ = ["connection_path", "VERTICAL_MIN_BEND"]
