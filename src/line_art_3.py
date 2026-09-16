"""
line_art_3：更贴原线的结构 ASCII。供 line_art_demo 复用。

不是禁用 <>，而是按格子外形选字：像 / 就用 /，像 < 才用 <。
分数接近时优先更简单的笔画，避免细线被 []+_ 抢走。
匹配用字体核 + 几何核的归一化互相关。
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw

from line_art import (
    load_gray_for_lines,
    photo_to_line,
    resize_line_art,
    save_line_gallery,
    to_white_on_black,
)
from main import DEFAULT_INPUT, DEFAULT_OUTPUT, save_batch_html
from glyphs import STRUCTURE_CHARS, extract_tiles, load_mono_font, match_tiles, save_ascii_png

DEFAULT_OUT = os.path.join(DEFAULT_OUTPUT, "line_art_3")
# Courier New 在 line-height:1 时字格约 0.60 宽 / 1.00 高，不是正方形。
CELL_W = 18
CELL_H = 30
FONT_ASPECT = CELL_W / CELL_H
CHARS = list(STRUCTURE_CHARS)
if CHARS[0] != " ":
    CHARS = [" "] + [c for c in CHARS if c != " "]
SPACE = CHARS.index(" ")

COMPLEXITY = {
    " ": 0,
    ".": 1,
    "'": 1,
    "`": 1,
    ",": 1,
    "^": 2,
    "-": 3,
    "|": 3,
    "/": 3,
    "\\": 3,
    "~": 3,
    "_": 4,
    ":": 4,
    ";": 4,
    "(": 5,
    ")": 5,
    "<": 6,
    ">": 6,
    "[": 7,
    "]": 7,
    "{": 8,
    "}": 8,
    "+": 9,
    "=": 9,
    '"': 9,
}


def _ink_line(canvas, p1, p2):
    cv2.line(canvas, p1, p2, 255, 1, lineType=cv2.LINE_8)


def geometric_glyph(ch, cell_w=CELL_W, cell_h=CELL_H):
    g = np.zeros((cell_h, cell_w), dtype=np.uint8)
    sx, sy = cell_w - 1, cell_h - 1
    mx, my = cell_w // 2, cell_h // 2
    qx, qy = max(1, cell_w // 4), max(1, cell_h // 4)
    if ch == " ":
        return g
    if ch == ".":
        cv2.circle(g, (mx, cell_h - 4), 1, 255, -1)
    elif ch == "'":
        _ink_line(g, (mx, 2), (mx, qy + 1))
    elif ch == "`":
        _ink_line(g, (qx, 2), (mx, qy + 1))
    elif ch == ",":
        _ink_line(g, (mx, cell_h - 6), (mx - 2, cell_h - 2))
    elif ch == "^":
        _ink_line(g, (2, my), (mx, 2))
        _ink_line(g, (mx, 2), (sx - 2, my))
    elif ch == '"':
        _ink_line(g, (mx - 3, 2), (mx - 3, qy + 2))
        _ink_line(g, (mx + 3, 2), (mx + 3, qy + 2))
    elif ch == ":":
        cv2.circle(g, (mx, qy + 1), 1, 255, -1)
        cv2.circle(g, (mx, cell_h - qy - 2), 1, 255, -1)
    elif ch == ";":
        cv2.circle(g, (mx, qy + 1), 1, 255, -1)
        _ink_line(g, (mx, cell_h - 7), (mx - 2, cell_h - 2))
    elif ch == "~":
        pts = np.array(
            [[2, my], [qx, my - 3], [mx, my], [mx + qx, my + 3], [sx - 2, my]],
            dtype=np.int32,
        )
        cv2.polylines(g, [pts], False, 255, 1)
    elif ch == "-":
        _ink_line(g, (1, my), (sx - 1, my))
    elif ch == "_":
        _ink_line(g, (1, sy - 2), (sx - 1, sy - 2))
    elif ch == "|":
        _ink_line(g, (mx, 1), (mx, sy - 1))
    elif ch == "/":
        _ink_line(g, (2, sy - 2), (sx - 2, 2))
    elif ch == "\\":
        _ink_line(g, (2, 2), (sx - 2, sy - 2))
    elif ch == "+":
        _ink_line(g, (1, my), (sx - 1, my))
        _ink_line(g, (mx, 1), (mx, sy - 1))
    elif ch == "=":
        _ink_line(g, (2, my - 3), (sx - 2, my - 3))
        _ink_line(g, (2, my + 3), (sx - 2, my + 3))
    elif ch == "<":
        _ink_line(g, (sx - 3, 2), (2, my))
        _ink_line(g, (2, my), (sx - 3, sy - 2))
    elif ch == ">":
        _ink_line(g, (3, 2), (sx - 2, my))
        _ink_line(g, (sx - 2, my), (3, sy - 2))
    elif ch == "(":
        cv2.ellipse(g, (mx + 3, my), (max(2, mx - 2), max(3, my - 1)), 0, 50, 310, 255, 1)
    elif ch == ")":
        cv2.ellipse(g, (mx - 3, my), (max(2, mx - 2), max(3, my - 1)), 0, 230, 490, 255, 1)
    elif ch == "[":
        _ink_line(g, (mx - 2, 2), (mx - 2, sy - 2))
        _ink_line(g, (mx - 2, 2), (mx + 4, 2))
        _ink_line(g, (mx - 2, sy - 2), (mx + 4, sy - 2))
    elif ch == "]":
        _ink_line(g, (mx + 2, 2), (mx + 2, sy - 2))
        _ink_line(g, (mx - 4, 2), (mx + 2, 2))
        _ink_line(g, (mx - 4, sy - 2), (mx + 2, sy - 2))
    elif ch == "{":
        _ink_line(g, (mx + 2, 2), (mx - 1, 2))
        _ink_line(g, (mx - 1, 2), (mx - 1, my - 2))
        _ink_line(g, (mx - 1, my - 2), (mx - 4, my))
        _ink_line(g, (mx - 4, my), (mx - 1, my + 2))
        _ink_line(g, (mx - 1, my + 2), (mx - 1, sy - 2))
        _ink_line(g, (mx - 1, sy - 2), (mx + 2, sy - 2))
    elif ch == "}":
        _ink_line(g, (mx - 2, 2), (mx + 1, 2))
        _ink_line(g, (mx + 1, 2), (mx + 1, my - 2))
        _ink_line(g, (mx + 1, my - 2), (mx + 4, my))
        _ink_line(g, (mx + 4, my), (mx + 1, my + 2))
        _ink_line(g, (mx + 1, my + 2), (mx + 1, sy - 2))
        _ink_line(g, (mx + 1, sy - 2), (mx - 2, sy - 2))
    return g


def font_kernels():
    font = load_mono_font(CELL_H - 2)
    kernels = np.zeros((len(CHARS), CELL_H, CELL_W), dtype=np.float32)
    for i, ch in enumerate(CHARS):
        if ch == " ":
            continue
        canvas = Image.new("L", (CELL_W, CELL_H), 0)
        draw = ImageDraw.Draw(canvas)
        bbox = draw.textbbox((0, 0), ch, font=font)
        tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
        x = (CELL_W - tw) // 2 - bbox[0]
        y = (CELL_H - th) // 2 - bbox[1]
        draw.text((x, y), ch, font=font, fill=255, font_mode="1")
        kernels[i] = np.array(canvas, dtype=np.float32) / 255.0
    return kernels


def geo_kernels():
    """只放在格子正中。平移变体会让邻格抢同一根线，字形叠在一起。"""
    bases = [geometric_glyph(ch).astype(np.float32) / 255.0 for ch in CHARS]
    ids = np.arange(len(CHARS), dtype=np.int32)
    return np.stack(bases, axis=0), ids


def pick_chars(scores, margin=0.07):
    complexity = np.array([COMPLEXITY.get(ch, 9) for ch in CHARS], dtype=np.float32)
    best = scores.max(axis=1, keepdims=True)
    near = scores >= (best - margin)
    masked_c = np.where(near, complexity[None, :], 99.0)
    min_c = masked_c.min(axis=1, keepdims=True)
    prefer = (masked_c <= min_c) & near
    chosen_scores = np.where(prefer, scores, -1e9)
    return chosen_scores.argmax(axis=1)


def _collapse_variant_scores(raw_scores, ids, n_chars):
    out = np.full((raw_scores.shape[0], n_chars), -1e9, dtype=np.float32)
    for k, char_i in enumerate(ids):
        out[:, char_i] = np.maximum(out[:, char_i], raw_scores[:, k])
    return out


def tile_com(tiles):
    """每格墨水的质心（格子坐标）和总墨量。"""
    n, cell_h, cell_w = tiles.shape
    ys = np.arange(cell_h, dtype=np.float32)[:, None]
    xs = np.arange(cell_w, dtype=np.float32)[None, :]
    mass = tiles.reshape(n, -1).sum(axis=1)
    cy = (tiles * ys).reshape(n, -1).sum(axis=1) / (mass + 1e-8)
    cx = (tiles * xs).reshape(n, -1).sum(axis=1) / (mass + 1e-8)
    return cy, cx, mass


def exclusive_cells(tiles, rows, cols):
    """
    格子是互斥划分：落在边界上的线只归能量更高的一侧。
    相等时归右/下，避免两格都画同一笔。
    """
    n, cell_h, cell_w = tiles.shape
    energy = tiles.reshape(n, -1).mean(axis=1).reshape(rows, cols)
    cy, cx, _mass = tile_com(tiles)
    cy = cy.reshape(rows, cols)
    cx = cx.reshape(rows, cols)
    keep = np.ones((rows, cols), dtype=bool)
    keep[energy < 1e-6] = False
    bx, by = cell_w * 0.28, cell_h * 0.28
    lo_x, hi_x = bx, cell_w - 1 - bx
    lo_y, hi_y = by, cell_h - 1 - by
    for r in range(rows):
        for c in range(cols):
            if not keep[r, c]:
                continue
            e = energy[r, c]
            give = False
            if cx[r, c] > hi_x and c + 1 < cols and energy[r, c + 1] >= e * 0.85:
                give = True
            elif cy[r, c] > hi_y and r + 1 < rows and energy[r + 1, c] >= e * 0.85:
                give = True
            elif cx[r, c] < lo_x and c > 0 and energy[r, c - 1] > e * 1.15:
                give = True
            elif cy[r, c] < lo_y and r > 0 and energy[r - 1, c] > e * 1.15:
                give = True
            if give:
                keep[r, c] = False
    return keep.reshape(-1)


def match_line_page(edges, cols, rows, font_k=None, geo_pack=None):
    if font_k is None:
        font_k = font_kernels()
    if geo_pack is None:
        geo_pack = geo_kernels()
    geo_k, geo_ids = geo_pack
    tiles = extract_tiles(edges, rows, cols, CELL_H, CELL_W, 0, 0)
    font_s = match_tiles(tiles, font_k, 1.15)
    geo_s = _collapse_variant_scores(match_tiles(tiles, geo_k, 1.15), geo_ids, len(CHARS))
    scores = np.maximum(font_s, geo_s)
    scores[:, SPACE] = -1.0
    energy = tiles.reshape(rows * cols, -1).mean(axis=1)
    chosen = pick_chars(scores, margin=0.12)
    chosen_score = scores.max(axis=1)
    chosen[(energy < 0.018) | (chosen_score < 0.10)] = SPACE
    chosen[~exclusive_cells(tiles, rows, cols)] = SPACE
    grid = np.array(CHARS, dtype="<U1")[chosen].reshape(rows, cols)
    return "\n".join("".join(row) for row in grid)


def grid_rows(height, width, cols):
    return max(1, int(round(cols * (height / float(width)) * FONT_ASPECT)))


def convert_gray(gray, cols=80, method="auto", font_k=None, geo_k=None):
    height, width = gray.shape
    rows = grid_rows(height, width, cols)
    line_wb = photo_to_line(gray, method=method, stroke_width=1)
    page = resize_line_art(line_wb, cols * CELL_W, rows * CELL_H, stroke_width=1)
    edges = to_white_on_black(page).astype(np.float32) / 255.0
    ascii_text = match_line_page(edges, cols, rows, font_k=font_k, geo_pack=geo_k)
    return ascii_text, line_wb, rows


def convert_from_line(gray, cols=80, font_k=None, geo_k=None):
    """原线稿直接进匹配：只缩到字符格。不抽线、不二值、不骨架。"""
    height, width = gray.shape
    rows = grid_rows(height, width, cols)
    page = cv2.resize(gray, (cols * CELL_W, rows * CELL_H), interpolation=cv2.INTER_AREA)
    ink = page.astype(np.float32)
    if float(ink.mean()) > 127.0:
        ink = 255.0 - ink
    ink = np.clip(ink - ink.min(), 0, 255) / 255.0
    preview = page if float(page.mean()) > 127.0 else (255 - page)
    ascii_text = match_line_page(ink, cols, rows, font_k=font_k, geo_pack=geo_k)
    return ascii_text, preview.astype(np.uint8), rows


def batch_extract(
    input_folder=DEFAULT_INPUT,
    output_folder=DEFAULT_OUT,
    cols=80,
    method="auto",
    from_lines=False,
):
    if not os.path.exists(input_folder):
        print(f"找不到输入文件夹: {input_folder}")
        return []
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    saved = []
    arts = {}
    font_k = font_kernels()
    geo_pack = geo_kernels()
    print(f"line_art_3：{len(files)} 张  列={cols}  from_lines={from_lines}")
    for idx, name in enumerate(files, 1):
        src = os.path.join(input_folder, name)
        try:
            if from_lines:
                gray = np.array(Image.open(src).convert("L"), dtype=np.uint8)
                ascii_text, line_wb, rows = convert_from_line(
                    gray, cols=cols, font_k=font_k, geo_k=geo_pack
                )
                orig_w, orig_h = gray.shape[1], gray.shape[0]
            else:
                gray = np.array(load_gray_for_lines(src), dtype=np.uint8)
                ascii_text, line_wb, rows = convert_gray(
                    gray, cols=cols, method=method, font_k=font_k, geo_k=geo_pack
                )
                orig_w, orig_h = gray.shape[1], gray.shape[0]
            base, _ = os.path.splitext(name)
            out = os.path.join(output_folder, f"{base}_line.png")
            cv2.imwrite(out, line_wb)
            saved.append((name, out))
            arts[name] = ascii_text
            save_ascii_png(
                ascii_text,
                os.path.join(output_folder, f"{base}_ascii.png"),
                cell_w=CELL_W,
                cell_h=CELL_H,
            )
            print(f"[{idx}/{len(files)}] {name}  {orig_w}x{orig_h} -> {cols}x{rows}")
        except Exception as exc:
            print(f"[{idx}/{len(files)}] 失败 {name}: {exc}")
    stem = "all_paper_lineart" if from_lines else "all_line_art_3"
    if saved:
        gallery = os.path.join(DEFAULT_OUTPUT, f"{stem}.html")
        title = "论文原线稿（未抽线）" if from_lines else "line_art_3 线稿"
        save_line_gallery(saved, gallery, title=title)
        print(f"线稿总览: {os.path.abspath(gallery)}")
    if arts:
        ascii_html = os.path.join(DEFAULT_OUTPUT, f"{stem}_ascii.html")
        save_batch_html(
            arts,
            ascii_html,
            dark_mode=True,
            font_size=max(6, min(12, 640 // max(cols, 1))),
            line_height=1.0,
        )
        print(f"ASCII 总览: {os.path.abspath(ascii_html)}")
    return saved


def parse_args():
    parser = argparse.ArgumentParser(description="line_art_3：按线形匹配字符")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument(
        "--method", default="auto", choices=["auto", "structure", "xdog", "canny", "sketch"]
    )
    parser.add_argument("--from-lines", action="store_true", help="原线稿直接匹配，不抽线不二值")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    batch_extract(args.input, args.output, args.width, args.method, args.from_lines)
