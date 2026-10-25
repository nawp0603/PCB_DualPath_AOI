#!/usr/bin/env python3
"""
Shared plotting & layout utilities for Thesis Scribe visualization scripts.
Uses OpenCV + NumPy + Pillow (FreeType Windows fonts) to render 300-DPI
publication-quality figures with full Vietnamese Unicode support.

STRICT ISOLATION RULE:
  Only writes inside `aoi_final_plan/thesis_artifacts/`.
  Reads `src/` and `results/` in strict READ-ONLY mode.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# Paths
THESIS_DIR = Path(__file__).resolve().parent
FIGURES_DIR = THESIS_DIR / "figures"
PROJECT_ROOT = THESIS_DIR.parent.parent

# Fonts (Windows system fonts with full Vietnamese Unicode glyphs)
FONT_REG_PATH = "C:/Windows/Fonts/segoeui.ttf"
FONT_BOLD_PATH = "C:/Windows/Fonts/segoeuib.ttf"
FONT_ITALIC_PATH = "C:/Windows/Fonts/segoeuii.ttf"
FONT_MONO_PATH = "C:/Windows/Fonts/consola.ttf"
FONT_MONO_BOLD_PATH = "C:/Windows/Fonts/consolab.ttf"


def get_font(size: int = 20, bold: bool = False, italic: bool = False,
             mono: bool = False) -> ImageFont.FreeTypeFont:
    path = FONT_REG_PATH
    if mono:
        path = FONT_MONO_BOLD_PATH if bold else FONT_MONO_PATH
    elif bold:
        path = FONT_BOLD_PATH
    elif italic:
        path = FONT_ITALIC_PATH
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()


def bgr_to_pil(img_bgr: np.ndarray) -> Image.Image:
    if img_bgr.ndim == 2:
        return Image.fromarray(img_bgr).convert("RGB")
    return Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))


def pil_to_bgr(img_pil: Image.Image) -> np.ndarray:
    rgb = np.array(img_pil.convert("RGB"), dtype=np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def colorize_heatmap(arr: np.ndarray, vmin: float = 0.0, vmax: float = 10.0,
                     colormap: int = cv2.COLORMAP_TURBO) -> np.ndarray:
    """Convert float 2D array to BGR uint8 heatmap using OpenCV colormap."""
    norm = np.clip((arr.astype(np.float32) - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    u8 = np.round(norm * 255.0).astype(np.uint8)
    return cv2.applyColorMap(u8, colormap)


def make_colorbar_vertical(height: int, width: int, vmin: float, vmax: float,
                           label: str, colormap: int = cv2.COLORMAP_TURBO,
                           ticks: int = 5) -> Image.Image:
    """Create a vertical colorbar with tick labels and title as a PIL image."""
    total_w = width + 115
    canvas = Image.new("RGB", (total_w, height + 40), (255, 255, 255))
    draw = ImageDraw.Draw(canvas)

    grad = np.linspace(1.0, 0.0, height, dtype=np.float32)[:, None]
    grad = np.repeat(grad, width, axis=1)
    bar_bgr = colorize_heatmap(grad, 0.0, 1.0, colormap=colormap)
    bar_pil = bgr_to_pil(bar_bgr)
    y_top = 30
    canvas.paste(bar_pil, (8, y_top))
    draw.rectangle([8, y_top, 8 + width, y_top + height], outline=(40, 40, 40), width=2)

    f_tick = get_font(15, mono=True)
    f_lbl = get_font(15, bold=True)
    for i in range(ticks):
        frac = i / max(ticks - 1, 1)
        y = int(y_top + frac * height)
        val = vmax - frac * (vmax - vmin)
        draw.line([(8 + width, y), (8 + width + 6, y)], fill=(40, 40, 40), width=2)
        draw.text((8 + width + 10, y - 9), f"{val:.1f}", fill=(30, 30, 30), font=f_tick)

    draw.text((8, 2), label, fill=(20, 20, 20), font=f_lbl)
    return canvas


def draw_rounded_box(draw: ImageDraw.ImageDraw, box: Sequence[int],
                     fill: tuple[int, int, int] | None = None,
                     outline: tuple[int, int, int] | None = None,
                     width: int = 2, radius: int = 10) -> None:
    draw.rounded_rectangle(list(box), radius=radius, fill=fill,
                           outline=outline, width=width)


def add_header_and_footer(canvas: Image.Image, title: str, subtitle: str,
                          provenance: str) -> None:
    """Draw academic header banner and provenance footer on canvas."""
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    # Header bar
    draw.rectangle([0, 0, w, 92], fill=(18, 35, 64))
    draw.rectangle([0, 88, w, 92], fill=(217, 119, 6))
    f_title = get_font(28, bold=True)
    f_sub = get_font(18, italic=False)
    draw.text((28, 14), title, fill=(255, 255, 255), font=f_title)
    draw.text((28, 52), subtitle, fill=(203, 213, 225), font=f_sub)

    # Footer bar
    draw.rectangle([0, h - 38, w, h], fill=(241, 245, 249))
    draw.line([(0, h - 38), (w, h - 38)], fill=(203, 213, 225), width=1)
    f_foot = get_font(14, mono=True)
    draw.text((24, h - 28), provenance, fill=(71, 85, 105), font=f_foot)


def crop_with_border(img_bgr: np.ndarray, cx: float, cy: float, half: int,
                     border_color: tuple[int, int, int] = (0, 200, 255),
                     out_size: tuple[int, int] = (320, 320)) -> np.ndarray:
    """Safe crop around (cx, cy) resized to out_size with colored border."""
    h, w = img_bgr.shape[:2]
    x1 = max(0, int(round(cx - half)))
    y1 = max(0, int(round(cy - half)))
    x2 = min(w, int(round(cx + half)))
    y2 = min(h, int(round(cy + half)))
    patch = img_bgr[y1:y2, x1:x2]
    if patch.size == 0:
        patch = np.zeros((out_size[1], out_size[0], 3), dtype=np.uint8)
    else:
        patch = cv2.resize(patch, out_size, interpolation=cv2.INTER_NEAREST)
    cv2.rectangle(patch, (0, 0), (out_size[0] - 1, out_size[1] - 1), border_color, 4)
    return patch


def save_figure(canvas: Image.Image, filename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = FIGURES_DIR / filename
    canvas.save(str(out_path), format="PNG", dpi=(300, 300))
    print(f"[saved] {out_path} ({canvas.size[0]}x{canvas.size[1]} px @ 300 DPI)")
    return out_path
