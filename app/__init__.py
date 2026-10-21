"""MineImage - 节点式图像处理实验工作台。"""

import logging

__version__ = "0.1.0"
APP_NAME = "MineImage"
APP_DISPLAY_NAME = "MineImage 图像处理实验工作台"
ORGANIZATION = "MineImage"

# 包内模块统一使用 logging；未配置任何处理器时（例如单元测试）不输出到
# stderr，真正运行时会由 app.main 把根日志接入文件与 SQLite。
logging.getLogger(__name__).addHandler(logging.NullHandler())
