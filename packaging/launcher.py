"""PyInstaller 启动脚本（onefile 入口）。"""

from app.main import main

if __name__ == "__main__":
    raise SystemExit(main())
