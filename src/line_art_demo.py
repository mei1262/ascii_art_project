"""
line_art_demo: first usable structure-ASCII converter.

Fixes staircased diagonals from v4 without flooding / \\.
Force / \\ only on long, clearly tilted strokes. Keep - | for flat bars.
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
    TICKS,
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
from glyphs import extract_tiles, match_tiles, save_ascii_png
from main import DEFAULT_INPUT, DEFAULT_OUTPUT, save_batch_html

DEFAULT_OUT = os.path.join(DEFAULT_OUTPUT, "line_art_demo")
STROKE_ONLY = set("/\\|-~_()<>")
ENERGY_MIN = 0.016
DIAG_ENERGY = 0.022


def is_flat_horiz(tile, ang, ecc):
    xspan, yspan = stroke_span(tile)
    if xspan >= 0.45 and yspan <= 0.36 and xspan >= yspan * 1.7:
        return True
    fam, dists = bin_angle(ang)
    return bool(np.isfinite(ang) and fam == "h" and ecc >= 1.25 and dists["h"] <= 16)


def is_flat_vert(tile, ang, ecc):
    xspan, yspan = stroke_span(tile)
    if yspan >= 0.45 and xspan <= 0.36 and yspan >= xspan * 1.7:
        return True
    fam, dists = bin_angle(ang)
    return bool(np.isfinite(ang) and fam == "v" and ecc >= 1.25 and dists["v"] <= 16)


def diag_family(tile, ang, ecc):
    """Only a long, clearly tilted stroke becomes / or \\."""
    xspan, yspan = stroke_span(tile, thresh=0.10)
    fam, dists = bin_angle(ang)
    if fam not in ("slash", "back"):
        return None
    if dists[fam] > 18:
        return None
    if ecc < 1.35:
        return None
    if min(xspan, yspan) < 0.32 or max(xspan, yspan) < 0.48:
        return None
    return fam


def mask_letters(scores):
    out = scores.copy()
    for i, ch in enumerate(CHARS):
        if ch not in STROKE_ONLY:
            out[i] = -1e9
    return out


def orientation_adjust(row_scores, ang, ecc, tile_den, fill, stroke, fam):
    s = mask_letters(row_scores)
    s -= np.clip(fill - tile_den - 0.04, 0.0, None) * 2.4
    if stroke:
        s[fill >= 0.16] -= 0.70
    if ecc >= 1.3:
        bump(s, TICKS, -0.40)
    if fam == "slash":
        bump(s, "\\", -0.35)
        bump(s, "|-", -0.16)
        bump(s, "/", 0.20)
    elif fam == "back":
        bump(s, "/", -0.35)
        bump(s, "|-", -0.16)
        bump(s, "\\", 0.20)
    elif fam == "h":
        bump(s, "/\\|[]<>()", -0.40)
        bump(s, "-~_", 0.22)
    elif fam == "v":
        bump(s, "-_=/", -0.28)
        bump(s, "|", 0.20)
    return s


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
    cy, _cx, _mass = tile_com(tiles)
    keep = exclusive_cells(tiles, rows, cols)
    fill = glyph_fill(font_k)
    flat = np.zeros(rows * cols, dtype=bool)

    chosen = np.full(rows * cols, SPACE, dtype=np.int32)
    for i in range(rows * cols):
        if energy[i] < ENERGY_MIN or not keep[i]:
            continue
        ang, ecc, mass = tile_pca(tiles[i])
        fam, _ = bin_angle(ang)
        diag = diag_family(tiles[i], ang, ecc)
        if diag and energy[i] >= DIAG_ENERGY:
            chosen[i] = IDX["/" if diag == "slash" else "\\"]
            continue
        if is_flat_horiz(tiles[i], ang, ecc):
            chosen[i] = IDX[horiz_char(cy[i], CELL_H)]
            flat[i] = True
            continue
        if is_flat_vert(tiles[i], ang, ecc):
            chosen[i] = IDX["|"]
            continue
        stroke = is_stroke(tiles[i], ecc)
        adj = orientation_adjust(scores[i], ang, ecc, mass, fill, stroke, fam)
        pick = int(pick_chars(adj[None, :], margin=0.10)[0])
        if adj[pick] < 0.08:
            continue
        chosen[i] = pick

    grid = np.array(CHARS, dtype="<U1")[chosen].reshape(rows, cols)
    grid = snap_horiz(grid, flat.reshape(rows, cols), rows, cols)
    return "\n".join("".join(row) for row in grid)


def _to_ink(gray):
    page = gray.astype(np.float32)
    if float(page.mean()) > 127.0:
        page = 255.0 - page
    ink = np.clip(page - page.min(), 0, 255)
    return ink / 255.0


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
        vis = render_ascii_bgr(ascii_text)
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
    print(f"line_art_demo video: {kept} frames")
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
    print(f"line_art_demo: {len(files)} images  cols={cols}  from_lines={from_lines}")
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
            print(f"[{idx}/{len(files)}] failed {name}: {exc}")
    stem = "all_line_art_demo"
    if saved:
        gallery = os.path.join(DEFAULT_OUTPUT, f"{stem}.html")
        save_line_gallery(saved, gallery, title="line_art_demo")
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
    parser = argparse.ArgumentParser(description="line_art_demo: usable structure ASCII")
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
