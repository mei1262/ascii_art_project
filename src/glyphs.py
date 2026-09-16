"""Shared glyph kernels for line_art_3 / line_art_4. Not a standalone converter."""
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

STRUCTURE_CHARS = list(" .'`^\",:;~-_+<>\\/|()[]{}=")
FULL_CHARS = [chr(code) for code in range(32, 127)]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def load_mono_font(font_size):
    candidates = [
        os.path.join(SCRIPT_DIR, "consolas.ttf"),
        r"C:\Windows\Fonts\consola.ttf",
        r"C:\Windows\Fonts\cour.ttf",
        "/System/Library/Fonts/Supplemental/Courier New.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    ]
    for path in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size)
            except Exception:
                continue
    return ImageFont.load_default()


def build_glyph_kernels(chars, cell_w, cell_h):
    font = load_mono_font(max(8, cell_h - 2))
    kernels = np.zeros((len(chars), cell_h, cell_w), dtype=np.float32)
    for i, ch in enumerate(chars):
        if ch == " ":
            continue
        canvas = Image.new("L", (cell_w, cell_h), 0)
        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), ch, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (cell_w - tw) // 2 - bbox[0]
        y = (cell_h - th) // 2 - bbox[1]
        draw.text((x, y), ch, font=font, fill=255, font_mode="1")
        kernels[i] = np.array(canvas, dtype=np.float32) / 255.0
    return kernels


def extract_tiles(image, rows, cols, cell_h, cell_w, offset_y=0, offset_x=0):
    view = image[offset_y : offset_y + rows * cell_h, offset_x : offset_x + cols * cell_w]
    return view.reshape(rows, cell_h, cols, cell_w).transpose(0, 2, 1, 3).reshape(-1, cell_h, cell_w)


def match_tiles(tiles, kernels, density_weight=0.85):
    n_tiles = tiles.shape[0]
    flat_t = tiles.reshape(n_tiles, -1)
    flat_k = kernels.reshape(kernels.shape[0], -1)
    t_norm = np.linalg.norm(flat_t, axis=1, keepdims=True)
    k_norm = np.linalg.norm(flat_k, axis=1, keepdims=True)
    t_unit = flat_t / (t_norm + 1e-6)
    k_unit = flat_k / (k_norm + 1e-6)
    cosine = t_unit @ k_unit.T
    t_den = flat_t.mean(axis=1, keepdims=True)
    k_den = flat_k.mean(axis=1, keepdims=True)
    return cosine - density_weight * np.abs(t_den - k_den.T)


def render_ascii_image(ascii_text, cell_w=18, cell_h=30):
    lines = ascii_text.split("\n")
    rows = max(1, len(lines))
    cols = max((len(line) for line in lines), default=1)
    font = load_mono_font(max(8, cell_h - 2))
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch != " ":
                draw.text((x * cell_w, y * cell_h), ch, font=font, fill=(255, 255, 255))
    return canvas


def save_ascii_png(ascii_text, path, cell_w=18, cell_h=30):
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    render_ascii_image(ascii_text, cell_w=cell_w, cell_h=cell_h).save(path)
    return path
