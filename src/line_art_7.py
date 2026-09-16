"""
line-art-7-direction-first-other-last

Simple single-axis stroke: same glyph as model 5.
Curve / corner / dense / competing axes: DeepAA 411 (with current fold).

dir-lock applies to every cell. There is no forced bypass.
"""
import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from line_art import load_gray_for_lines, photo_to_line, resize_line_art, save_line_gallery, to_white_on_black
from line_art_3 import CELL_H, CELL_W, exclusive_cells, grid_rows
from line_art_4 import (
    VIDEO_EXTS,
    bin_angle,
    finalize_mp4,
    font_kernels,
    geo_kernels,
    render_ascii_bgr,
    stroke_span,
    tile_pca,
)
from line_art_demo import (
    DIAG_ENERGY,
    ENERGY_MIN,
    _to_ink,
    diag_family,
    match_line_page as match_model5,
)
from deepaa_cnn import (
    INK_MIN,
    MERGIN,
    cell_ink_ratio,
    crop_window,
    load_aa_font,
    load_model,
    to_structure_char,
    to_train_domain,
)
from glyphs import extract_tiles, load_mono_font
from main import DEFAULT_INPUT, DEFAULT_OUTPUT, save_batch_html

STRATEGY_NAME = "line-art-7-direction-first-other-last"
DEFAULT_OUT = os.path.join(DEFAULT_OUTPUT, STRATEGY_NAME)

DIRECTION_STRENGTH = 1.0
DIR_LOCK = 0.45


def _to_grid(text, rows, cols):
    grid = np.full((rows, cols), " ", dtype="<U1")
    for r, line in enumerate(text.split("\n")[:rows]):
        for c, ch in enumerate(line[:cols]):
            grid[r, c] = ch
    return grid


def complexity_penalty(tile, ang, ecc, dists, fam):
    """0 = a clean stick, 1 = curve / corner / blob. Soft, not a hard cut."""
    xspan, yspan = stroke_span(tile)
    fill = float((tile >= 0.10).mean())
    aspect = max(xspan, yspan) / (min(xspan, yspan) + 1e-6)
    diag = fam in ("slash", "back") and np.isfinite(ang) and dists[fam] <= 20
    both = float(np.clip((min(xspan, yspan) - 0.18) / 0.32, 0.0, 1.0))
    squat = float(np.clip((2.1 - aspect) / 1.1, 0.0, 1.0))
    fat = float(np.clip((fill - 0.06) / 0.22, 0.0, 1.0))
    if diag:
        both = 0.0
        squat = 0.0
    if not np.isfinite(ang):
        compete = 1.0
        skinny = 0.0
    else:
        ordered = sorted(dists.values())
        compete = float(np.clip((16.0 - (ordered[1] - ordered[0])) / 16.0, 0.0, 1.0))
        skinny = float(np.clip((ecc - 1.0) / 1.8, 0.0, 1.0))
        if diag:
            compete = min(compete, 0.20)
    return float(np.clip(max(both, squat * 0.85, fat, compete) * (1.0 - 0.25 * skinny), 0.0, 1.0))


def direction_confidence(tile, energy):
    """Continuous 0..1. Clean / \\ stay high; curves stay lower."""
    ang, ecc, _mass = tile_pca(tile)
    fam, dists = bin_angle(ang)
    if diag_family(tile, ang, ecc) and energy >= DIAG_ENERGY:
        return 0.95
    if np.isfinite(ang):
        align = max(0.0, 1.0 - float(sorted(dists.values())[0]) / 28.0)
        elong = float(np.clip((ecc - 1.0) / 1.8, 0.0, 1.0))
        if fam in ("slash", "back") and dists[fam] <= 16 and ecc >= 1.15:
            align = max(align, 0.88)
    else:
        align, elong = 0.0, 0.0
    en = float(np.clip(energy / 0.05, 0.0, 1.0))
    raw = 0.48 * align + 0.32 * elong + 0.20 * en
    return raw * (1.0 - 0.72 * complexity_penalty(tile, ang, ecc, dists, fam))


def save_mixed_png(ascii_text, path, cell_w=CELL_W, cell_h=CELL_H):
    """ASCII (including \\) via Consolas so it is not drawn as yen."""
    lines = ascii_text.split("\n")
    rows = max(1, len(lines))
    cols = max((len(line) for line in lines), default=1)
    size = max(8, cell_h - 2)
    latin = load_mono_font(size)
    cjk = load_aa_font(size)
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch in (" ", "\u3000"):
                continue
            font = latin if ch.isascii() else cjk
            draw.text((x * cell_w, y * cell_h), ch, font=font, fill=(255, 255, 255))
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    canvas.save(path)
    return path


def direction_mask(edges, cols, rows, direction_strength, dir_lock):
    tiles = extract_tiles(edges, rows, cols, CELL_H, CELL_W, 0, 0)
    energy = tiles.reshape(rows * cols, -1).mean(axis=1)
    keep = exclusive_cells(tiles, rows, cols)
    mask = np.zeros(rows * cols, dtype=bool)
    for i in range(rows * cols):
        if energy[i] < ENERGY_MIN or not keep[i]:
            continue
        conf = direction_confidence(tiles[i], energy[i])
        if conf * max(0.0, direction_strength) >= dir_lock:
            mask[i] = True
    return mask.reshape(rows, cols)


def predict_deepaa(model, chars, device, patches):
    out = []
    if not patches:
        return out
    with torch.no_grad():
        for i in range(0, len(patches), 64):
            batch = np.stack(patches[i : i + 64]).astype(np.float32) / 255.0
            pred = model(torch.from_numpy(batch)[:, None].to(device)).argmax(1).cpu().numpy()
            for p in pred:
                folded = to_structure_char(chars[int(p)])
                out.append(folded if folded else " ")
    return out


def fill_deepaa(grid, use5, aa_gray, edges, cols, rows, aa_model):
    model, aa_chars, device, _w, _dw = aa_model
    padded = np.pad(aa_gray, MERGIN, constant_values=255)
    tiles = extract_tiles(edges, rows, cols, CELL_H, CELL_W, 0, 0)
    energy = tiles.reshape(rows * cols, -1).mean(axis=1)
    need = []
    for r in range(rows):
        for c in range(cols):
            if use5[r, c]:
                continue
            if energy[r * cols + c] < ENERGY_MIN:
                grid[r, c] = " "
                continue
            need.append((r, c))
    patches = []
    keep = []
    for r, c in need:
        patch = crop_window(padded, c * CELL_W, r * CELL_H, slide=0)
        if cell_ink_ratio(patch, CELL_W) < INK_MIN:
            grid[r, c] = " "
            continue
        patches.append(patch)
        keep.append((r, c))
    preds = predict_deepaa(model, aa_chars, device, patches)
    for (r, c), ch in zip(keep, preds):
        grid[r, c] = ch
    return grid


def blend_pages(
    edges,
    aa_gray,
    cols,
    rows,
    font_k,
    geo_pack,
    aa_model,
    direction_strength=DIRECTION_STRENGTH,
    dir_lock=DIR_LOCK,
):
    text5 = match_model5(edges, cols, rows, font_k=font_k, geo_pack=geo_pack)
    grid = _to_grid(text5, rows, cols)
    use5 = direction_mask(edges, cols, rows, direction_strength, dir_lock)
    n_ink = int(((grid != " ") | use5).sum())
    n5 = int((use5 & (grid != " ")).sum())
    for r in range(rows):
        for c in range(cols):
            if not use5[r, c]:
                grid[r, c] = " "
    grid = fill_deepaa(grid, use5, aa_gray, edges, cols, rows, aa_model)
    n_aa = int(((~use5) & (grid != " ")).sum())
    print(
        f"  model5 {n5}/{n_ink or 1} ({100.0 * n5 / max(n_ink, 1):.0f}%)  "
        f"deepaa {n_aa}",
        flush=True,
    )
    return "\n".join("".join(row) for row in grid)


def convert_from_line(
    gray,
    cols=160,
    font_k=None,
    geo_k=None,
    aa_model=None,
    direction_strength=DIRECTION_STRENGTH,
    dir_lock=DIR_LOCK,
):
    height, width = gray.shape
    rows = grid_rows(height, width, cols)
    page = cv2.resize(gray, (cols * CELL_W, rows * CELL_H), interpolation=cv2.INTER_AREA)
    ink = _to_ink(page)
    preview = page if float(page.mean()) > 127.0 else (255 - page)
    ascii_text = blend_pages(
        ink, to_train_domain(page), cols, rows, font_k, geo_k, aa_model,
        direction_strength, dir_lock,
    )
    return ascii_text, preview.astype(np.uint8), rows


def convert_gray(
    gray,
    cols=160,
    method="auto",
    font_k=None,
    geo_k=None,
    aa_model=None,
    direction_strength=DIRECTION_STRENGTH,
    dir_lock=DIR_LOCK,
):
    height, width = gray.shape
    rows = grid_rows(height, width, cols)
    line_wb = photo_to_line(gray, method=method, stroke_width=1)
    page = resize_line_art(line_wb, cols * CELL_W, rows * CELL_H, stroke_width=1)
    edges = _to_ink(to_white_on_black(page))
    ascii_text = blend_pages(
        edges, to_train_domain(page), cols, rows, font_k, geo_k, aa_model,
        direction_strength, dir_lock,
    )
    return ascii_text, line_wb, rows


def convert_line_video(
    input_path,
    output_folder=DEFAULT_OUT,
    cols=160,
    video_out=None,
    direction_strength=DIRECTION_STRENGTH,
    dir_lock=DIR_LOCK,
):
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
    aa_model = load_model(struct=False)
    writer = None
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ascii_text, _, rows = convert_from_line(
            gray,
            cols=cols,
            font_k=font_k,
            geo_k=geo_k,
            aa_model=aa_model,
            direction_strength=direction_strength,
            dir_lock=dir_lock,
        )
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
    print(f"{STRATEGY_NAME} video: {kept} frames")
    return video_out


def batch_extract(
    input_folder=DEFAULT_INPUT,
    output_folder=DEFAULT_OUT,
    cols=160,
    method="auto",
    from_lines=False,
    direction_strength=DIRECTION_STRENGTH,
    dir_lock=DIR_LOCK,
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
    aa_model = load_model(struct=False)
    print(
        f"{STRATEGY_NAME}: {len(files)} images  cols={cols}  from_lines={from_lines}  "
        f"dir={direction_strength} lock={dir_lock}"
    )
    for idx, name in enumerate(files, 1):
        src = os.path.join(input_folder, name)
        try:
            if from_lines:
                gray = np.array(Image.open(src).convert("L"), dtype=np.uint8)
                ascii_text, line_wb, rows = convert_from_line(
                    gray,
                    cols=cols,
                    font_k=font_k,
                    geo_k=geo_pack,
                    aa_model=aa_model,
                    direction_strength=direction_strength,
                    dir_lock=dir_lock,
                )
                orig_w, orig_h = gray.shape[1], gray.shape[0]
            else:
                gray = np.array(load_gray_for_lines(src), dtype=np.uint8)
                ascii_text, line_wb, rows = convert_gray(
                    gray,
                    cols=cols,
                    method=method,
                    font_k=font_k,
                    geo_k=geo_pack,
                    aa_model=aa_model,
                    direction_strength=direction_strength,
                    dir_lock=dir_lock,
                )
                orig_w, orig_h = gray.shape[1], gray.shape[0]
            base, _ = os.path.splitext(name)
            out = os.path.join(output_folder, f"{base}_line.png")
            cv2.imwrite(out, line_wb)
            saved.append((name, out))
            arts[name] = ascii_text
            save_mixed_png(
                ascii_text,
                os.path.join(output_folder, f"{base}_ascii.png"),
                cell_w=CELL_W,
                cell_h=CELL_H,
            )
            print(f"[{idx}/{len(files)}] {name}  {orig_w}x{orig_h} -> {cols}x{rows}")
        except Exception as exc:
            print(f"[{idx}/{len(files)}] failed {name}: {exc}")
    stem = "all_" + STRATEGY_NAME.replace("-", "_")
    if saved:
        gallery = os.path.join(DEFAULT_OUTPUT, f"{stem}.html")
        save_line_gallery(saved, gallery, title=STRATEGY_NAME)
        print(f"line gallery: {os.path.abspath(gallery)}")
    if arts:
        ascii_html = os.path.join(DEFAULT_OUTPUT, f"{stem}_ascii.html")
        save_batch_html(
            arts,
            ascii_html,
            dark_mode=True,
            font_size=max(6, min(12, 640 // max(cols, 1))),
            line_height=1.0,
            font_family='Consolas, "Courier New", "MS Gothic", "Yu Gothic", monospace',
        )
        print(f"ASCII gallery: {os.path.abspath(ascii_html)}")
    return saved


def parse_args():
    parser = argparse.ArgumentParser(description=STRATEGY_NAME)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--width", type=int, default=160)
    parser.add_argument(
        "--method", default="auto", choices=["auto", "structure", "xdog", "canny", "sketch"]
    )
    parser.add_argument("--from-lines", action="store_true")
    parser.add_argument("--video-out", default="")
    parser.add_argument(
        "--direction-strength",
        type=float,
        default=DIRECTION_STRENGTH,
        help="multiply direction confidence; higher = more model-5 cells",
    )
    parser.add_argument(
        "--dir-lock",
        type=float,
        default=DIR_LOCK,
        help="0.2 more model 5, 0.7 more DeepAA. Threshold on 0-1 direction score.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    kw = dict(direction_strength=args.direction_strength, dir_lock=args.dir_lock)
    if os.path.isfile(args.input) and args.input.lower().endswith(VIDEO_EXTS):
        convert_line_video(args.input, args.output, args.width, args.video_out or None, **kw)
    else:
        batch_extract(args.input, args.output, args.width, args.method, args.from_lines, **kw)
