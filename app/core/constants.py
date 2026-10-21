"""跨模块共享的常量。"""

from __future__ import annotations

APP_NAME = "MineImage"
APP_DISPLAY_NAME = "MineImage 图像处理实验工作台"
APP_VERSION = "0.1.0"

#: 拖拽算子时使用的自定义 MIME 类型，payload 为 UTF-8 JSON: {"id": "<operator id>"}
MIME_OPERATOR = "application/x-mineimage-operator"
#: 从外部拖入图像文件时使用
MIME_IMAGE_PATHS = "application/x-mineimage-image-paths"

#: 源图像节点的 pseudo operator id
SOURCE_OP_ID = "__source__"

#: 工程文件扩展名
PROJECT_EXTENSION = ".mi"
PROJECT_FILTER = "MineImage 工程 (*.mi)"

IMAGE_FILTER = (
    "图像文件 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.webp *.ppm *.pgm *.npy);;"
    "所有文件 (*)"
)

#: 一级分类的默认顺序；扫描到的未知分类排在后面（按名称排序）。
CATEGORY_ORDER = (
    "基础调整",
    "滤波与平滑",
    "阈值与分割",
    "边缘检测",
    "形态学",
    "几何变换",
    "通道与色彩",
    "交互处理",
    "其他",
)

#: 参数变更预览的默认防抖时长（毫秒）
PARAM_DEBOUNCE_MS = 120
#: 单击节点预览与双击进入参数面板的判定延迟（毫秒）
SINGLE_CLICK_DELAY_MS = 220

#: 项目中源图像在 zip 内的目录
PROJECT_IMAGE_DIR = "images"
