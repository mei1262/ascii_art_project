"""
line-art-8-deepaaweight

Same mix as model 7: simple single-axis strokes stay model 5,
curves / corners / dense cells go to DeepAA.

The DeepAA branch uses the official full Keras weights (weight.hdf5),
converted to PyTorch. Grid, dir-lock, and fold are unchanged.
"""
import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
from PIL import Image

from line_art import load_gray_for_lines, save_line_gallery
from line_art_4 import VIDEO_EXTS, finalize_mp4, font_kernels, geo_kernels, render_ascii_bgr
from line_art_7 import (
    CELL_H,
    CELL_W,
    DIR_LOCK,
    DIRECTION_STRENGTH,
    convert_from_line,
    convert_gray,
    save_mixed_png,
)
from deepaa_official_full import load_official_full
from main import DEFAULT_INPUT, DEFAULT_OUTPUT, save_batch_html

STRATEGY_NAME = "line-art-8-deepaaweight"
DEFAULT_OUT = os.path.join(DEFAULT_OUTPUT, STRATEGY_NAME)


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
    aa_model = load_official_full()
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
    aa_model = load_official_full()
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
