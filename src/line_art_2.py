"""
line_art_2：不改 line_art.py。

轮廓在字形 CNN 里容易被配成 []<>+_ 这种块状字，看起来更粗、也不贴线。
这里仍用原来的抽线，再按局部走向只画 / | - \\，ASCII 也只从这几个细笔画里选。
"""
import argparse
import math
import os
import sys

import cv2
import numpy as np

from line_art import binarize_ink, load_gray_for_lines, photo_to_line, restroke, save_line_gallery
from line_clean import prune_spurs
from glyphs import save_ascii_png
from main import DEFAULT_INPUT, DEFAULT_OUTPUT, save_batch_html
from PIL import Image

DEFAULT_OUT = os.path.join(DEFAULT_OUTPUT, "line_art_2")
_N8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))
_DIRS = (
    (1, 0),
    (1, 1),
    (0, 1),
    (-1, 1),
    (-1, 0),
    (-1, -1),
    (0, -1),
    (1, -1),
)
# 只保留细线字符，不用 []<>+_{}
THIN = "-\\|/"


def trace_skeleton(ink):
    live = set(zip(*np.where(ink > 0)))
    if not live:
        return []

    def nbrs(y, x):
        found = []
        for dy, dx in _N8:
            q = (y + dy, x + dx)
            if q in live:
                found.append(q)
        return found

    polylines = []
    while live:
        start = None
        for y, x in live:
            if len(nbrs(y, x)) <= 1:
                start = (y, x)
                break
        if start is None:
            start = next(iter(live))
        y, x = start
        path = []
        prev = None
        while True:
            path.append((x, y))
            cand = [p for p in nbrs(y, x) if p != prev]
            live.discard((y, x))
            if not cand:
                break
            if len(cand) >= 2 and len(path) > 1:
                live.add((y, x))
                break
            prev = (y, x)
            y, x = cand[0]
        if len(path) >= 2:
            polylines.append(path)
        else:
            for px, py in path:
                live.discard((py, px))
    return polylines


def snap_pair(p0, p1):
    x0, y0 = p0
    x1, y1 = p1
    dx, dy = x1 - x0, y1 - y0
    length = math.hypot(dx, dy)
    if length < 1.0:
        return None
    k = int(round(math.atan2(dy, dx) / (math.pi / 4.0))) % 8
    ux, uy = _DIRS[k]
    scale = length / math.hypot(ux, uy)
    return (int(round(x0)), int(round(y0))), (
        int(round(x0 + ux * scale)),
        int(round(y0 + uy * scale)),
    )


def refine_to_stroke_lines(line_wb, stroke_width=1):
    """保留原骨架走向，只把短折卡到 8 向细线。"""
    ink = binarize_ink(line_wb)
    if cv2.countNonZero(ink) == 0:
        return np.full_like(line_wb, 255)
    ink = prune_spurs(ink, max_len=8)
    canvas = np.zeros_like(ink)
    for path in trace_skeleton(ink):
        arr = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
        simple = cv2.approxPolyDP(arr, 1.2, False)
        pts = [tuple(int(v) for v in p[0]) for p in simple]
        if len(pts) < 2:
            pts = path
        for a, b in zip(pts[:-1], pts[1:]):
            pair = snap_pair(a, b)
            if pair is None:
                continue
            (x0, y0), (x1, y1) = pair
            cv2.line(canvas, (x0, y0), (x1, y1), 255, 1, lineType=cv2.LINE_8)
    if cv2.countNonZero(canvas) == 0:
        canvas = ink
    return 255 - restroke(canvas, stroke_width)


def photo_to_line_2(gray_u8, method="auto", stroke_width=1):
    raw = photo_to_line(gray_u8, method=method, stroke_width=1)
    return refine_to_stroke_lines(raw, stroke_width=stroke_width)


def frame_bgr_to_line(frame_bgr, method="auto", stroke_width=1, max_side=0):
    from line_art import gray_from_bgr

    gray = gray_from_bgr(frame_bgr)
    if max_side and max(gray.shape) > max_side:
        height, width = gray.shape
        scale = max_side / float(max(height, width))
        gray = cv2.resize(
            gray,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return photo_to_line_2(gray, method=method, stroke_width=stroke_width)


def _cell_char(cell):
    ink = cell < 128
    if float(ink.mean()) < 0.035:
        return " "
    ys, xs = np.where(ink)
    if xs.size < 3:
        return " "
    pts = np.stack([xs.astype(np.float32), ys.astype(np.float32)], axis=1)
    pts -= pts.mean(axis=0)
    cov = pts.T @ pts
    _, vecs = np.linalg.eigh(cov)
    vx, vy = float(vecs[0, 1]), float(vecs[1, 1])
    ang = math.atan2(vy, vx) % math.pi
    # 0° -, 45° \, 90° |, 135° /
    k = int((ang + math.pi / 8.0) // (math.pi / 4.0)) % 4
    ch = THIN[k]
    if ch == "-" and ink.std() > 0.28:
        return "~"
    return ch


def line_to_thin_ascii(line_wb, cols=80, font_aspect=0.75, cell=16):
    height, width = line_wb.shape
    rows = max(1, int(cols * (height / width) * font_aspect))
    page = cv2.resize(line_wb, (cols * cell, rows * cell), interpolation=cv2.INTER_AREA)
    _, page = cv2.threshold(page, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    lines = []
    for r in range(rows):
        row = []
        for c in range(cols):
            tile = page[r * cell : (r + 1) * cell, c * cell : (c + 1) * cell]
            row.append(_cell_char(tile))
        lines.append("".join(row))
    return "\n".join(lines)


def batch_extract(
    input_folder=DEFAULT_INPUT,
    output_folder=DEFAULT_OUT,
    method="auto",
    stroke_width=1,
    cols=80,
    from_lines=False,
):
    if not os.path.exists(input_folder):
        print(f"找不到输入文件夹: {input_folder}")
        return []
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    saved = []
    arts = {}
    print(f"line_art_2：{len(files)} 张，方法 {method}  from_lines={from_lines}")
    for idx, name in enumerate(files, 1):
        src = os.path.join(input_folder, name)
        try:
            if from_lines:
                gray = np.array(Image.open(src).convert("L"), dtype=np.uint8)
                lines = gray
            else:
                gray = np.array(load_gray_for_lines(src), dtype=np.uint8)
                lines = photo_to_line_2(gray, method=method, stroke_width=stroke_width)
            base, _ = os.path.splitext(name)
            out = os.path.join(output_folder, f"{base}_line.png")
            cv2.imwrite(out, lines)
            saved.append((name, out))
            ascii_text = line_to_thin_ascii(lines, cols=cols)
            arts[name] = ascii_text
            save_ascii_png(
                ascii_text,
                os.path.join(output_folder, f"{base}_ascii.png"),
                cell_w=16,
                cell_h=16,
            )
            print(f"[{idx}/{len(files)}] {name}")
        except Exception as exc:
            print(f"[{idx}/{len(files)}] 失败 {name}: {exc}")
    if saved:
        gallery = os.path.join(DEFAULT_OUTPUT, "all_line_art_2.html")
        save_line_gallery(saved, gallery, title="line_art_2 细线稿")
        print(f"线稿总览: {os.path.abspath(gallery)}")
    if arts:
        ascii_html = os.path.join(DEFAULT_OUTPUT, "all_line_art_2_ascii.html")
        save_batch_html(arts, ascii_html, dark_mode=True, font_size=max(6, min(12, 640 // max(cols, 1))))
        print(f"细笔画 ASCII: {os.path.abspath(ascii_html)}")
    return saved


def parse_args():
    parser = argparse.ArgumentParser(description="line_art_2：边缘只用 /|-~ 细笔画")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument(
        "--method",
        default="auto",
        choices=["auto", "structure", "xdog", "canny", "sketch"],
    )
    parser.add_argument("--stroke-width", type=int, default=1)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument("--from-lines", action="store_true", help="输入已是线稿，不再抽线")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    batch_extract(
        args.input, args.output, args.method, args.stroke_width, args.width, args.from_lines
    )
