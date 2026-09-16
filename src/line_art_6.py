"""
line_art_6: structure ASCII with a richer stroke alphabet.

Circles (o O 0) are detected first. Wide horizontals stay - ~ _
instead of a picket of | l I. Remaining cells use glyph match.
"""
import argparse
import os
import sys

import cv2
import numpy as np
from PIL import Image

from line_art import load_gray_for_lines, photo_to_line, resize_line_art, save_line_gallery, to_white_on_black
from line_art_3 import CELL_H, CELL_W, _collapse_variant_scores, exclusive_cells, grid_rows, tile_com
from line_art_4 import (
    CHARS,
    IDX,
    SPACE,
    VIDEO_EXTS,
    bin_angle,
    bump,
    finalize_mp4,
    font_kernels,
    geo_kernels,
    glyph_fill,
    horiz_char,
    is_stroke,
    pick_chars,
    render_ascii_bgr,
    snap_horiz,
    stroke_span,
    tile_pca,
)
from line_art_demo import ENERGY_MIN, _to_ink
from glyphs import extract_tiles, match_tiles, save_ascii_png
from main import DEFAULT_INPUT, DEFAULT_OUTPUT, save_batch_html

DEFAULT_OUT = os.path.join(DEFAULT_OUTPUT, "line_art_6")
RENDER_CELL_W = 18
RENDER_CELL_H = 30
RICH_STROKES = set(
    "/\\|-~_"
    "iIlL1!jft7"
    "YVKxX*+=%"
    "()[]{}<>"
    "^`',.:;\""
    "oO0"
)
VERT = "iIlL1!jft|"
HORIZ = "-~_="
SLASH = "/7"
BACK = "\\"
TICKS = ".'`,:;^\""
CROSS = "+xX*%"
ROUNDS = "oO0"


def mask_to_rich(scores):
    out = scores.copy()
    for i, ch in enumerate(CHARS):
        if ch not in RICH_STROKES:
            out[i] = -1e9
    return out


def orientation_adjust(row_scores, ang, ecc, tile_den, fill, stroke, fam):
    s = mask_to_rich(row_scores)
    s -= np.clip(fill - tile_den - 0.04, 0.0, None) * 2.4
    if stroke:
        s[fill >= 0.22] -= 0.45
        bump(s, ROUNDS, -0.28)
    else:
        bump(s, ROUNDS, 0.10)
    if ecc >= 1.45:
        bump(s, TICKS, -0.22)
        bump(s, CROSS, -0.12)
        bump(s, ROUNDS, -0.20)
    if fam == "slash":
        bump(s, BACK + HORIZ + VERT + ROUNDS, -0.26)
        bump(s, SLASH, 0.22)
    elif fam == "back":
        bump(s, SLASH + HORIZ + VERT + ROUNDS, -0.26)
        bump(s, BACK, 0.22)
    elif fam == "h":
        bump(s, SLASH + BACK + VERT + ROUNDS + "[]<>()", -0.34)
        bump(s, HORIZ, 0.26)
    elif fam == "v":
        bump(s, HORIZ + "/", -0.20)
        bump(s, VERT, 0.14)
    return s


def _ink_mask(tile, thresh=0.12):
    return (tile >= thresh).astype(np.uint8)


def _circle_metrics(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(contour))
    if area < 10:
        return None
    peri = float(cv2.arcLength(contour, True))
    if peri < 8:
        return None
    circularity = 4.0 * np.pi * area / (peri * peri)
    (cx, cy), radius = cv2.minEnclosingCircle(contour)
    if radius < 2.8 or circularity < 0.58:
        return None
    fill = area / (np.pi * radius * radius + 1e-6)
    return float(cx), float(cy), float(radius), float(circularity), float(fill)


def _circle_char(radius, fill):
    ratio = radius / (0.5 * min(CELL_W, CELL_H))
    if ratio < 0.55:
        return "o"
    if fill < 0.55:
        return "0"
    if ratio < 1.20:
        return "O"
    return "0"


def detect_circles(tiles, rows, cols):
    """Cells whose neighborhood holds a circle; only the center cell is marked."""
    found = {}
    for i in range(rows * cols):
        row, col = divmod(i, cols)
        r0, r1 = max(0, row - 1), min(rows, row + 2)
        c0, c1 = max(0, col - 1), min(cols, col + 2)
        bands = []
        for rr in range(r0, r1):
            bands.append(np.concatenate([tiles[rr * cols + cc] for cc in range(c0, c1)], axis=1))
        pad = np.concatenate(bands, axis=0)
        mask = _ink_mask(pad)
        info = _circle_metrics(mask)
        if info is None:
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
            info = _circle_metrics(closed)
        if info is None:
            continue
        cx, cy, radius, circularity, fill = info
        min_r = 0.28 * min(CELL_W, CELL_H)
        max_r = 1.35 * max(CELL_W, CELL_H)
        if radius < min_r or radius > max_r:
            continue
        grow = r0 * CELL_H + cy
        gcol = c0 * CELL_W + cx
        owner_r = int(grow // CELL_H)
        owner_c = int(gcol // CELL_W)
        if not (0 <= owner_r < rows and 0 <= owner_c < cols):
            continue
        owner = owner_r * cols + owner_c
        char = _circle_char(radius, fill)
        prev = found.get(owner)
        if prev is None or circularity > prev[1]:
            found[owner] = (char, circularity)
    return {idx: char for idx, (char, _) in found.items()}


def is_wide_horiz(tile):
    xspan, yspan = stroke_span(tile)
    return bool(xspan >= 0.42 and yspan <= 0.40 and xspan >= yspan * 1.5)


def round_blob_char(tile):
    """Compact, not-too-elongated ink: pupil / small circle that Hough misses."""
    _ang, ecc, _mass = tile_pca(tile, thresh=0.10)
    xspan, yspan = stroke_span(tile, thresh=0.10)
    if not np.isfinite(ecc) or ecc >= 1.75:
        return None
    if min(xspan, yspan) < 0.32 or max(xspan, yspan) < 0.42:
        return None
    if max(xspan, yspan) / (min(xspan, yspan) + 1e-6) > 1.55:
        return None
    fill = float((tile >= 0.10).mean())
    if fill < 0.035 or fill > 0.55:
        return None
    if fill < 0.12:
        return "o"
    if fill < 0.28:
        return "O"
    return "0"


def match_line_page(edges, cols, rows, font_k=None, geo_pack=None):
    if font_k is None:
        font_k = font_kernels()
    if geo_pack is None:
        geo_pack = geo_kernels()
    geo_k, geo_ids = geo_pack
    tiles = extract_tiles(edges, rows, cols, CELL_H, CELL_W, 0, 0)
    font_s = match_tiles(tiles, font_k, 1.45)
    geo_s = _collapse_variant_scores(match_tiles(tiles, geo_k, 1.15), geo_ids, len(CHARS))
    scores = np.maximum(font_s, geo_s)
    scores[:, SPACE] = -1.0
    energy = tiles.reshape(rows * cols, -1).mean(axis=1)
    keep = exclusive_cells(tiles, rows, cols)
    fill = glyph_fill(font_k)
    flat = np.zeros(rows * cols, dtype=bool)
    circles = detect_circles(tiles, rows, cols)

    chosen = np.full(rows * cols, SPACE, dtype=np.int32)
    for i in range(rows * cols):
        if energy[i] < ENERGY_MIN or not keep[i]:
            continue
        if i in circles:
            chosen[i] = IDX[circles[i]]
            continue
        blob = round_blob_char(tiles[i])
        if blob is not None:
            chosen[i] = IDX[blob]
            continue
        if is_wide_horiz(tiles[i]):
            cy_i = float(tile_com(tiles[i][None, ...])[0][0])
            chosen[i] = IDX[horiz_char(cy_i, CELL_H)]
            flat[i] = True
            continue
        ang, ecc, mass = tile_pca(tiles[i])
        fam, _ = bin_angle(ang)
        xspan, yspan = stroke_span(tiles[i])
        if xspan >= yspan * 1.35:
            fam = "h"
        stroke = is_stroke(tiles[i], ecc)
        adj = orientation_adjust(scores[i], ang, ecc, mass, fill, stroke, fam)
        pick = int(pick_chars(adj[None, :], margin=0.10)[0])
        if adj[pick] < 0.08:
            continue
        chosen[i] = pick
        if CHARS[pick] in HORIZ:
            flat[i] = True

    grid = np.array(CHARS, dtype="<U1")[chosen].reshape(rows, cols)
    grid = snap_horiz(grid, flat.reshape(rows, cols), rows, cols)
    return "\n".join("".join(row) for row in grid)


def convert_from_line(gray, cols=80, font_k=None, geo_k=None):
    height, width = gray.shape
    rows = grid_rows(height, width, cols)
    page = cv2.resize(gray, (cols * CELL_W, rows * CELL_H), interpolation=cv2.INTER_AREA)
    ink = _to_ink(page)
    preview = page if float(page.mean()) > 127.0 else (255 - page)
    ascii_text = match_line_page(ink, cols, rows, font_k=font_k, geo_pack=geo_k)
    return ascii_text, preview.astype(np.uint8), rows


def convert_gray(gray, cols=80, method="auto", font_k=None, geo_k=None):
    height, width = gray.shape
    rows = grid_rows(height, width, cols)
    line_wb = photo_to_line(gray, method=method, stroke_width=1)
    page = resize_line_art(line_wb, cols * CELL_W, rows * CELL_H, stroke_width=1)
    edges = _to_ink(to_white_on_black(page))
    ascii_text = match_line_page(edges, cols, rows, font_k=font_k, geo_pack=geo_k)
    return ascii_text, line_wb, rows


def convert_line_video(input_path, output_folder=DEFAULT_OUT, cols=80, video_out=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {input_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    os.makedirs(output_folder, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    if not video_out:
        video_out = os.path.join(output_folder, f"{stem}_ascii.mp4")
    tmp_path = video_out + ".tmp.mp4"
    font_k = font_kernels()
    geo_k = geo_kernels()
    writer = None
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ascii_text, _, rows = convert_from_line(gray, cols=cols, font_k=font_k, geo_k=geo_k)
        vis = render_ascii_bgr(ascii_text, cell_w=RENDER_CELL_W, cell_h=RENDER_CELL_H)
        if writer is None:
            writer = cv2.VideoWriter(
                tmp_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                max(1.0, float(src_fps)),
                (vis.shape[1], vis.shape[0]),
                True,
            )
            if not writer.isOpened():
                raise RuntimeError(f"cannot write video: {tmp_path}")
        writer.write(vis)
        kept += 1
        if kept == 1 or kept % 20 == 0:
            print(f"  ascii frame {kept}/{total or '?'}  {cols}x{rows}", flush=True)
    cap.release()
    if writer is not None:
        writer.release()
        finalize_mp4(tmp_path, video_out)
    print(f"line_art_6 video: {kept} frames")
    return video_out


def batch_extract(
    input_folder=DEFAULT_INPUT,
    output_folder=DEFAULT_OUT,
    cols=80,
    method="auto",
    from_lines=False,
):
    if not os.path.exists(input_folder):
        print(f"input not found: {input_folder}")
        return []
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    saved = []
    arts = {}
    font_k = font_kernels()
    geo_pack = geo_kernels()
    print(f"line_art_6: {len(files)} images  cols={cols}  from_lines={from_lines}")
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
                cell_w=RENDER_CELL_W,
                cell_h=RENDER_CELL_H,
            )
            print(f"[{idx}/{len(files)}] {name}  {orig_w}x{orig_h} -> {cols}x{rows}")
        except Exception as exc:
            print(f"[{idx}/{len(files)}] failed {name}: {exc}")
    stem = "all_line_art_6"
    if saved:
        gallery = os.path.join(DEFAULT_OUTPUT, f"{stem}.html")
        save_line_gallery(saved, gallery, title="line_art_6")
        print(f"line gallery: {os.path.abspath(gallery)}")
    if arts:
        ascii_html = os.path.join(DEFAULT_OUTPUT, f"{stem}_ascii.html")
        save_batch_html(
            arts,
            ascii_html,
            dark_mode=True,
            font_size=max(6, min(12, 640 // max(cols, 1))),
            line_height=1.0,
        )
        print(f"ASCII gallery: {os.path.abspath(ascii_html)}")
    return saved


def parse_args():
    parser = argparse.ArgumentParser(
        description="line_art_6: rich stroke alphabet, no isolation"
    )
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument(
        "--method", default="auto", choices=["auto", "structure", "xdog", "canny", "sketch"]
    )
    parser.add_argument("--from-lines", action="store_true", help="input is already line art")
    parser.add_argument("--video-out", default="", help="ASCII video output path")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    if os.path.isfile(args.input) and args.input.lower().endswith(VIDEO_EXTS):
        convert_line_video(args.input, args.output, args.width, args.video_out or None)
    else:
        batch_extract(args.input, args.output, args.width, args.method, args.from_lines)
