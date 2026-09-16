"""
line-art-9-deepaa-light

Official DeepAA convolution (frozen, from weight.hdf5) + FC512 head.
Official output.py decode: variable width, no fold, char_dict layout,
space penalty, one window per character.

~10.6M params instead of the official 87.3M FC4096 net.
"""
import argparse
import json
import os
import subprocess
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
from PIL import Image

from line_art import load_gray_for_lines, photo_to_line, save_line_gallery
from line_art_4 import VIDEO_EXTS, finalize_mp4
from deepaa_official_full import decode_official, gray_official_array, load_char_dict
from deepaa_smallhead import load_smallhead
from main import DEFAULT_INPUT, DEFAULT_OUTPUT, save_batch_html

STRATEGY_NAME = "line-art-9-deepaa-light"
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


def _save_ascii_frame(png, out_path):
    out_path = os.path.abspath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path[:-4] + ".part.png" if out_path.lower().endswith(".png") else out_path + ".part.png"
    rgb = np.array(png.convert("RGB"))
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if not cv2.imwrite(tmp, bgr):
        raise RuntimeError(f"cannot write {tmp}")
    os.replace(tmp, out_path)


def _frame_png(work_dir, idx):
    return os.path.join(work_dir, f"f{idx:06d}.png")


def _count_done(work_dir):
    n = 0
    while os.path.isfile(_frame_png(work_dir, n)):
        n += 1
    return n


def _write_progress(work_dir, meta):
    path = os.path.join(work_dir, "progress.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _stitch_frames(work_dir, n_frames, fps, video_out):
    if n_frames <= 0:
        raise RuntimeError("no ASCII frames to stitch")
    os.makedirs(os.path.dirname(os.path.abspath(video_out)) or ".", exist_ok=True)
    pattern = os.path.join(work_dir, "f%06d.png")
    tmp_path = video_out + ".tmp.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-framerate",
        str(max(1.0, float(fps))),
        "-start_number",
        "0",
        "-i",
        pattern,
        "-frames:v",
        str(n_frames),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        tmp_path,
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        finalize_mp4(tmp_path, video_out)
        return video_out
    except Exception:
        first = cv2.imread(_frame_png(work_dir, 0))
        if first is None:
            raise RuntimeError(f"cannot stitch {work_dir}")
        writer = cv2.VideoWriter(
            tmp_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            max(1.0, float(fps)),
            (first.shape[1], first.shape[0]),
            True,
        )
        if not writer.isOpened():
            raise RuntimeError(f"cannot write video: {tmp_path}")
        for i in range(n_frames):
            vis = cv2.imread(_frame_png(work_dir, i))
            if vis is None:
                writer.release()
                raise RuntimeError(f"missing ASCII frame f{i:06d}.png")
            writer.write(vis)
        writer.release()
        finalize_mp4(tmp_path, video_out)
        return video_out


def convert_line_video(
    input_path,
    output_folder=DEFAULT_OUT,
    new_width=0,
    video_out=None,
    slide=0,
    chunk=128,
    restart=False,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {input_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    os.makedirs(output_folder, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    if not video_out:
        video_out = os.path.join(DEFAULT_OUTPUT, "videos", "ascii", f"{stem}_ascii.mp4")
    video_out = os.path.abspath(video_out.strip())
    work_dir = os.path.splitext(video_out)[0] + "_frames"
    if restart and os.path.isdir(work_dir):
        for name in os.listdir(work_dir):
            path = os.path.join(work_dir, name)
            if os.path.isfile(path):
                os.remove(path)
    os.makedirs(work_dir, exist_ok=True)
    done = _count_done(work_dir)
    finished = total > 0 and done >= total
    print(
        f"{STRATEGY_NAME}: video  {total} frames  resume {done}  "
        f"pixel_width={new_width or 'original'}"
    )
    if not finished:
        aa_model = load_smallhead()
        char_dict = load_char_dict()
        idx = 0
        try:
            while True:
                ok, frame = cap.read()
                if not ok:
                    finished = True
                    break
                if idx < done:
                    idx += 1
                    continue
                gray = frame[:, :, 0] if frame.ndim == 3 else frame
                text, _, png, rows, used_w, used_h = convert_from_line(
                    gray, aa_model, char_dict, new_width=new_width, slide=slide, chunk=chunk
                )
                _save_ascii_frame(png, _frame_png(work_dir, idx))
                idx += 1
                done = idx
                _write_progress(
                    work_dir,
                    {
                        "source": os.path.abspath(input_path),
                        "video_out": os.path.abspath(video_out),
                        "done": done,
                        "total": total,
                        "fps": src_fps,
                        "new_width": new_width,
                    },
                )
                if done == 1 or done % 5 == 0 or done == total:
                    print(
                        f"  ascii frame {done}/{total or '?'}  {used_w}x{used_h} rows={rows}",
                        flush=True,
                    )
                if total and done >= total:
                    finished = True
                    break
        except KeyboardInterrupt:
            cap.release()
            print(
                f"stopped at {done}/{total or '?'}  "
                f"run the same command to continue",
                flush=True,
            )
            return video_out
    cap.release()
    done = _count_done(work_dir)
    if not finished:
        print(f"incomplete: {done}/{total}  run the same command to continue")
        return video_out
    print(f"stitching {done} frames -> {video_out}", flush=True)
    _stitch_frames(work_dir, done, src_fps, video_out)
    print(f"{STRATEGY_NAME} video: {done} frames")
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
    aa_model = load_smallhead()
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
    parser.add_argument(
        "--restart",
        action="store_true",
        help="Ignore saved ASCII frames and start the video over.",
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
            restart=args.restart,
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
