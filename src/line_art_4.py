"""
line_art_4：标准可打印 ASCII（32–126）+ 走向。

粗线仍按线条走向选字，不用 M/@ 这类填满的字去凑密度。
"""
import argparse
import os
import sys

import subprocess

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
from line_art_3 import (
    CELL_H,
    CELL_W,
    exclusive_cells,
    geometric_glyph,
    grid_rows,
    tile_com,
    _collapse_variant_scores,
)
from main import DEFAULT_INPUT, DEFAULT_OUTPUT, save_batch_html
from glyphs import FULL_CHARS, build_glyph_kernels, extract_tiles, load_mono_font, match_tiles, save_ascii_png

DEFAULT_OUT = os.path.join(DEFAULT_OUTPUT, "line_art_4")
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
CHARS = list(FULL_CHARS)
SPACE = CHARS.index(" ")
IDX = {ch: i for i, ch in enumerate(CHARS)}

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
    "!": 5,
    "?": 5,
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
    "*": 8,
    "#": 12,
    "@": 12,
    "%": 12,
    "&": 12,
    "$": 12,
    "M": 12,
    "W": 12,
    "Q": 11,
    "N": 11,
    "B": 11,
    "0": 11,
    "8": 11,
    "g": 11,
}

BACK_ANG = np.degrees(np.arctan2(CELL_H, CELL_W))
SLASH_ANG = 180.0 - BACK_ANG

HORIZ = ["-", "~", "_"]
TICKS = list(".'`^,:;\"")


def font_kernels():
    return build_glyph_kernels(CHARS, CELL_W, CELL_H)


def geo_kernels():
    bases = [geometric_glyph(ch).astype(np.float32) / 255.0 for ch in CHARS]
    return np.stack(bases, axis=0), np.arange(len(CHARS), dtype=np.int32)


def pick_chars(scores, margin=0.10):
    complexity = np.array([COMPLEXITY.get(ch, 7) for ch in CHARS], dtype=np.float32)
    best = scores.max(axis=1, keepdims=True)
    near = scores >= (best - margin)
    masked_c = np.where(near, complexity[None, :], 99.0)
    min_c = masked_c.min(axis=1, keepdims=True)
    prefer = (masked_c <= min_c) & near
    chosen_scores = np.where(prefer, scores, -1e9)
    return chosen_scores.argmax(axis=1)


def tile_pca(tile, thresh=0.05):
    """主轴角 [0,180)：0=水平，90=竖直；y 向下时 ~59° 是 \\，~121° 是 /。"""
    ys, xs = np.nonzero(tile > thresh)
    mass = float(tile.mean())
    if xs.size < 4:
        return np.nan, 0.0, mass
    w = tile[ys, xs].astype(np.float64)
    w = w / (w.sum() + 1e-8)
    mx = float(np.dot(w, xs))
    my = float(np.dot(w, ys))
    dx = xs - mx
    dy = ys - my
    cxx = float(np.dot(w, dx * dx))
    cyy = float(np.dot(w, dy * dy))
    cxy = float(np.dot(w, dx * dy))
    trace = cxx + cyy
    det = cxx * cyy - cxy * cxy
    disc = max(0.0, 0.25 * trace * trace - det)
    l1 = 0.5 * trace + np.sqrt(disc)
    l2 = 0.5 * trace - np.sqrt(disc)
    ecc = float(l1 / (l2 + 1e-8))
    if abs(cxy) > 1e-10:
        vx, vy = cxy, l1 - cxx
    elif cxx >= cyy:
        vx, vy = 1.0, 0.0
    else:
        vx, vy = 0.0, 1.0
    ang = np.degrees(np.arctan2(vy, vx))
    if ang < 0:
        ang += 180.0
    return float(ang), ecc, mass


def circ_dist(a, b):
    return abs((a - b + 90.0) % 180.0 - 90.0)


def bin_angle(ang):
    if not np.isfinite(ang):
        return "blob", {"h": 90, "v": 90, "back": 90, "slash": 90}
    dists = {
        "h": min(ang, 180.0 - ang),
        "v": abs(ang - 90.0),
        "back": circ_dist(ang, BACK_ANG),
        "slash": circ_dist(ang, SLASH_ANG),
    }
    return min(dists, key=dists.get), dists


def bump(scores, chars, delta):
    for ch in chars:
        j = IDX.get(ch)
        if j is not None:
            scores[j] += delta


def stroke_span(tile, thresh=0.05):
    ys, xs = np.nonzero(tile > thresh)
    if xs.size == 0:
        return 0.0, 0.0
    h, w = tile.shape
    return (xs.max() - xs.min() + 1) / w, (ys.max() - ys.min() + 1) / h


def horiz_char(cy, cell_h):
    t = cy / max(cell_h, 1)
    if t > 0.68:
        return "_"
    if t < 0.30:
        return "~"
    return "-"


def glyph_fill(font_k):
    return font_k.reshape(len(CHARS), -1).mean(axis=1)


def is_flat_horiz(tile, ang, ecc):
    """横条：含较粗的线。用宽高比，不要求细。"""
    xspan, yspan = stroke_span(tile)
    if xspan >= 0.40 and xspan >= yspan * 1.18:
        return True
    fam, dists = bin_angle(ang)
    return bool(np.isfinite(ang) and fam == "h" and ecc >= 1.20 and dists["h"] <= 26)


def is_flat_vert(tile, ang, ecc):
    xspan, yspan = stroke_span(tile)
    if yspan >= 0.42 and yspan >= xspan * 1.18:
        return True
    fam, dists = bin_angle(ang)
    return bool(np.isfinite(ang) and fam == "v" and ecc >= 1.20 and dists["v"] <= 26)


def is_stroke(tile, ecc):
    """还是一条线（可粗），不是色块。粗对角线的包围盒接近整格，所以也看 PCA。"""
    xspan, yspan = stroke_span(tile)
    aspect = max(xspan, yspan) / (min(xspan, yspan) + 1e-6)
    if aspect >= 1.28 and max(xspan, yspan) >= 0.35:
        return True
    return ecc >= 1.18


def orientation_adjust(row_scores, ang, ecc, tile_den, fill, stroke):
    s = row_scores.copy()
    s -= np.clip(fill - tile_den - 0.04, 0.0, None) * 2.4
    if stroke:
        s[fill >= 0.16] -= 0.55
    fam, _dists = bin_angle(ang)
    if ecc >= 1.4:
        bump(s, TICKS, -0.36)
    if np.isfinite(ang) and ecc >= 1.15:
        if fam == "slash":
            bump(s, "\\", -0.40)
            bump(s, "/", 0.08)
        elif fam == "back":
            bump(s, "/", -0.40)
            bump(s, "\\", 0.08)
        elif fam == "h":
            bump(s, "()/\\|[]<>", -0.45)
            bump(s, "-~_", 0.22)
        elif fam == "v":
            bump(s, "-_=", -0.16)
            bump(s, "|", 0.08)
    return s, fam


def snap_horiz(grid, flat, rows, cols):
    """横走的格子收成 -~_；夹在两条横线中间的 ()/ 也收掉。"""
    out = grid.copy()
    hset = set(HORIZ)
    messy = set("()/\\|[]<>{}+" + "".join(TICKS) + "@#%&$MWQBN08")
    for r in range(rows):
        for c in range(cols):
            if flat[r, c] and out[r, c] not in hset and out[r, c] != " ":
                out[r, c] = "-"
            if c > 0 and c < cols - 1 and out[r, c] in messy:
                if out[r, c - 1] in hset and out[r, c + 1] in hset:
                    out[r, c] = "-"
    return out


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
        if energy[i] < 0.018 or not keep[i]:
            continue
        ang, ecc, mass = tile_pca(tiles[i])
        if is_flat_horiz(tiles[i], ang, ecc):
            chosen[i] = IDX[horiz_char(cy[i], CELL_H)]
            flat[i] = True
            continue
        if is_flat_vert(tiles[i], ang, ecc):
            chosen[i] = IDX["|"]
            continue
        stroke = is_stroke(tiles[i], ecc)
        adj, _fam = orientation_adjust(scores[i], ang, ecc, mass, fill, stroke)
        pick = int(pick_chars(adj[None, :], margin=0.10)[0])
        if adj[pick] < 0.10:
            continue
        chosen[i] = pick

    grid = np.array(CHARS, dtype="<U1")[chosen].reshape(rows, cols)
    grid = snap_horiz(grid, flat.reshape(rows, cols), rows, cols)
    return "\n".join("".join(row) for row in grid)


def convert_gray(gray, cols=80, method="auto", font_k=None, geo_k=None):
    height, width = gray.shape
    rows = grid_rows(height, width, cols)
    line_wb = photo_to_line(gray, method=method, stroke_width=1)
    page = resize_line_art(line_wb, cols * CELL_W, rows * CELL_H, stroke_width=1)
    edges = to_white_on_black(page).astype(np.float32) / 255.0
    ascii_text = match_line_page(edges, cols, rows, font_k=font_k, geo_pack=geo_k)
    return ascii_text, line_wb, rows


def convert_from_line(gray, cols=80, font_k=None, geo_k=None):
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


def render_ascii_bgr(ascii_text, cell_w=CELL_W, cell_h=CELL_H):
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
    frame = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)
    height, width = frame.shape[:2]
    if width % 2 or height % 2:
        frame = cv2.copyMakeBorder(
            frame, 0, height % 2, 0, width % 2, cv2.BORDER_CONSTANT, value=(0, 0, 0)
        )
    return frame


def finalize_mp4(tmp_path, output_path):
    try:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-i",
                tmp_path,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                output_path,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if os.path.isfile(tmp_path):
            os.remove(tmp_path)
    except Exception:
        if os.path.isfile(tmp_path):
            if os.path.isfile(output_path):
                os.remove(output_path)
            os.replace(tmp_path, output_path)
    print(f"ASCII 视频: {os.path.abspath(output_path)}")
    return output_path


def convert_line_video(input_path, output_folder=DEFAULT_OUT, cols=80, video_out=None):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {input_path}")
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
                raise RuntimeError(f"无法写视频: {tmp_path}")
        writer.write(vis)
        kept += 1
        if kept == 1 or kept % 20 == 0:
            print(f"  ascii frame {kept}/{total or '?'}  {cols}x{rows}", flush=True)
    cap.release()
    if writer is not None:
        writer.release()
        finalize_mp4(tmp_path, video_out)
    print(f"line_art_4 视频：{kept} 帧")
    return video_out


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
    print(f"line_art_4：{len(files)} 张  列={cols}  from_lines={from_lines}")
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
    stem = "all_paper_lineart_4" if from_lines else "all_line_art_4"
    if saved:
        gallery = os.path.join(DEFAULT_OUTPUT, f"{stem}.html")
        title = "line_art_4 论文原线稿" if from_lines else "line_art_4 线稿"
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
    parser = argparse.ArgumentParser(description="line_art_4：按走向把平滑线连起来")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--width", type=int, default=80)
    parser.add_argument(
        "--method", default="auto", choices=["auto", "structure", "xdog", "canny", "sketch"]
    )
    parser.add_argument("--from-lines", action="store_true", help="原线稿直接匹配，不抽线不二值")
    parser.add_argument("--video-out", default="", help="线稿视频转 ASCII 的输出路径")
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
