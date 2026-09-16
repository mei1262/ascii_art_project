"""
line-art-8-puredeepaa

Official DeepAA full weights with official output.py decode:
variable-width sliding windows, original Japanese 411 chars (no fold),
char_dict layout, and the previous-char space penalty.

Keeps batch-friendly extras: slide=0 (not 18), and official one-window-per-char
decode (jump by glyph width) instead of scoring every pixel.
"""
import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
from PIL import Image

from line_art import load_gray_for_lines, photo_to_line, save_line_gallery
from line_art_4 import VIDEO_EXTS, finalize_mp4
from deepaa_official_full import (
    decode_official,
    gray_official_array,
    load_char_dict,
    load_official_full,
)
from main import DEFAULT_INPUT, DEFAULT_OUTPUT, save_batch_html

STRATEGY_NAME = "line-art-8-puredeepaa"
DEFAULT_OUT = os.path.join(DEFAULT_OUTPUT, STRATEGY_NAME)


def line_gray_from_path(path, from_lines, method):
    if from_lines:
        arr = np.array(Image.open(path))
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return arr.astype(np.uint8)
    gray = np.array(load_gray_for_lines(path), dtype=np.uint8)
    return photo_to_line(gray, method=method, stroke_width=1)


def convert_from_line(gray, aa_model, char_dict, new_width=0, slide=0, chunk=128):
    model, chars, device, _w, _dw = aa_model
    work, used_w, used_h = gray_official_array(gray, new_width)
    text, png, rows = decode_official(
        model, device, work, chars, char_dict, slide=slide, chunk=chunk
    )
    preview = work if float(work.mean()) > 127.0 else (255 - work)
    return text, preview.astype(np.uint8), png, rows, used_w, used_h


def convert_line_video(
    input_path,
    output_folder=DEFAULT_OUT,
    new_width=0,
    video_out=None,
    slide=0,
    chunk=128,
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
    aa_model = load_official_full()
    char_dict = load_char_dict()
    writer = None
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = frame[:, :, 0] if frame.ndim == 3 else frame
        text, _, png, rows, used_w, used_h = convert_from_line(
            gray, aa_model, char_dict, new_width=new_width, slide=slide, chunk=chunk
        )
        vis = cv2.cvtColor(np.array(png.convert("RGB")), cv2.COLOR_RGB2BGR)
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
            print(f"  ascii frame {kept}/{total or '?'}  {used_w}x{used_h} rows={rows}", flush=True)
    cap.release()
    if writer is not None:
        writer.release()
        finalize_mp4(tmp_path, video_out)
    print(f"{STRATEGY_NAME} video: {kept} frames")
    return video_out


def batch_extract(
    input_folder=DEFAULT_INPUT,
    output_folder=DEFAULT_OUT,
    new_width=0,
    method="auto",
    from_lines=False,
    slide=0,
    chunk=128,
):
    if not os.path.exists(input_folder):
        print(f"input not found: {input_folder}")
        return []
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    saved = []
    arts = {}
    aa_model = load_official_full()
    char_dict = load_char_dict()
    print(
        f"{STRATEGY_NAME}: {len(files)} images  pixel_width={new_width or 'original'}  "
        f"from_lines={from_lines}  slide={slide}  chunk={chunk}"
    )
    for idx, name in enumerate(files, 1):
        src = os.path.join(input_folder, name)
        try:
            gray = line_gray_from_path(src, from_lines, method)
            orig_h, orig_w = gray.shape
            text, preview, png, rows, used_w, used_h = convert_from_line(
                gray, aa_model, char_dict, new_width=new_width, slide=slide, chunk=chunk
            )
            base, _ = os.path.splitext(name)
            line_path = os.path.join(output_folder, f"{base}_line.png")
            cv2.imwrite(line_path, preview)
            png.save(os.path.join(output_folder, f"{base}_ascii.png"))
            with open(os.path.join(output_folder, f"{base}_ascii.txt"), "w", encoding="utf-8") as f:
                f.write(text + "\n")
            saved.append((name, line_path))
            arts[name] = text
            print(
                f"[{idx}/{len(files)}] {name}  {orig_w}x{orig_h} -> {used_w}x{used_h}  "
                f"rows={rows}  chars={sum(len(line) for line in text.splitlines())}"
            )
        except Exception as exc:
            print(f"[{idx}/{len(files)}] failed {name}: {exc}")
    tag = os.path.basename(os.path.normpath(output_folder))
    stem = "all_" + STRATEGY_NAME.replace("-", "_")
    if tag != STRATEGY_NAME:
        stem = f"{stem}_{tag}"
    if saved:
        gallery = os.path.join(DEFAULT_OUTPUT, f"{stem}.html")
        save_line_gallery(saved, gallery, title=f"{STRATEGY_NAME} {tag}")
        print(f"line gallery: {os.path.abspath(gallery)}")
    if arts:
        ascii_html = os.path.join(DEFAULT_OUTPUT, f"{stem}_ascii.html")
        save_batch_html(
            arts,
            ascii_html,
            dark_mode=False,
            font_size=12,
            line_height=1.125,
            font_family='"MS Gothic", "MS PGothic", "Yu Gothic", monospace',
        )
        print(f"ASCII gallery: {os.path.abspath(ascii_html)}")
    return saved


def parse_args():
    parser = argparse.ArgumentParser(description=STRATEGY_NAME)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Official pixel width. 0 = keep original (output.py new_width=0). Not column count.",
    )
    parser.add_argument(
        "--method", default="auto", choices=["auto", "structure", "xdog", "canny", "sketch"]
    )
    parser.add_argument("--from-lines", action="store_true")
    parser.add_argument("--video-out", default="")
    parser.add_argument("--slide", type=int, default=0, help="Official vertical slide. Default 0, not 0-17.")
    parser.add_argument(
        "--chunk",
        type=int,
        default=128,
        help="Unused. Decode jumps by char width; kept for compatibility.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    if os.path.isfile(args.input) and args.input.lower().endswith(VIDEO_EXTS):
        convert_line_video(
            args.input,
            args.output,
            args.width,
            args.video_out or None,
            slide=args.slide,
            chunk=args.chunk,
        )
    else:
        batch_extract(
            args.input,
            args.output,
            args.width,
            args.method,
            args.from_lines,
            slide=args.slide,
            chunk=args.chunk,
        )
