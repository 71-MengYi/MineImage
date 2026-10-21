"""从 assets/icon.svg 生成 Windows .ico 与 application PNG。"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QImage, QPainter
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import QApplication


def render(svg_path: Path, png_path: Path, size: int = 256) -> None:
    renderer = QSvgRenderer(str(svg_path))
    if not renderer.isValid():
        raise SystemExit(f"无法加载 SVG：{svg_path}")
    image = QImage(QSize(size, size), QImage.Format.Format_ARGB32)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    renderer.render(painter)
    painter.end()
    image.save(str(png_path), "PNG")
    try:
        from PIL import Image

        with Image.open(png_path) as pil:
            pil.save(png_path.with_suffix(".ico"), sizes=[(16, 16), (32, 32), (48, 48), (64, 64), (128, 128), (256, 256)])
    except Exception as exc:  # noqa: BLE001
        print(f"生成 .ico 失败（仍可使用 PNG）：{exc}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    app = QApplication.instance() or QApplication([])
    render(root / "assets" / "icon.svg", root / "assets" / "icon.png")
    print("图标已生成：assets/icon.png / assets/icon.ico")


if __name__ == "__main__":
    main()
