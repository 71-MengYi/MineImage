"""生成 MineImage 演示用示例图像（无需外部素材）。"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "sample_images"


def gradient_canvas(width: int = 1280, height: int = 800) -> np.ndarray:
    x = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    y = np.linspace(0, 1, height, dtype=np.float32)[:, None]
    blue = (60 + 120 * (1 - y) + 30 * x).clip(0, 255)
    green = (80 + 100 * x + 40 * y).clip(0, 255)
    red = (120 + 80 * y + 30 * (1 - x)).clip(0, 255)
    canvas = np.dstack([blue, green, red]).astype(np.uint8)
    rng = np.random.default_rng(7)
    noise = rng.normal(0, 4, canvas.shape).astype(np.int16)
    return np.clip(canvas.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def make_color_sample() -> np.ndarray:
    image = gradient_canvas()
    height, width = image.shape[:2]
    cv2.circle(image, (width // 4, height // 2), 150, (40, 180, 250), -1, lineType=cv2.LINE_AA)
    cv2.rectangle(image, (width // 2, 160), (width - 160, height - 160), (240, 180, 60), -1)
    cv2.line(image, (80, height - 80), (width - 80, 80), (255, 255, 255), 6, cv2.LINE_AA)
    cv2.putText(
        image,
        "MineImage",
        (170, 190),
        cv2.FONT_HERSHEY_SIMPLEX,
        2.4,
        (255, 255, 255),
        5,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "node-based image lab",
        (175, 240),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (240, 240, 240),
        2,
        cv2.LINE_AA,
    )
    return image


def make_noise(width: int = 960, height: int = 640) -> np.ndarray:
    rng = np.random.default_rng(2024)
    base = gradient_canvas(width, height).astype(np.int16)
    noise = rng.normal(0, 38, base.shape).astype(np.int16)
    image = np.clip(base + noise, 0, 255).astype(np.uint8)
    for i in range(18):
        center = (rng.integers(60, width - 60), rng.integers(60, height - 60))
        radius = int(rng.integers(18, 70))
        color = tuple(int(v) for v in rng.integers(30, 255, 3))
        cv2.circle(image, center, radius, color, -1, cv2.LINE_AA)
    return image


def make_16bit() -> np.ndarray:
    gray = cv2.cvtColor(make_color_sample(), cv2.COLOR_BGR2GRAY).astype(np.uint16)
    return np.clip(gray.astype(np.float32) * 257 + 8000, 0, 65535).astype(np.uint16)


def make_float() -> np.ndarray:
    gray = cv2.cvtColor(make_color_sample(), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    yy, xx = np.mgrid[0 : gray.shape[0], 0 : gray.shape[1]]
    vignette = 1.0 - 0.35 * (((xx - gray.shape[1] / 2) / gray.shape[1]) ** 2 + ((yy - gray.shape[0] / 2) / gray.shape[0]) ** 2)
    return np.clip(gray * vignette, 0, 1).astype(np.float32)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    color = make_color_sample()
    cv2.imwrite(str(OUT / "sample_color.png"), color)
    gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(str(OUT / "sample_gray.png"), gray)
    cv2.imwrite(str(OUT / "sample_noise.png"), make_noise())
    cv2.imwrite(str(OUT / "sample_16bit.png"), make_16bit())
    np.save(str(OUT / "sample_float32.npy"), make_float())
    print(f"示例图像已生成：{OUT}")
    for path in sorted(OUT.iterdir()):
        print(f"  {path.name}  {path.stat().st_size / 1024:.1f} KiB")


if __name__ == "__main__":
    main()
