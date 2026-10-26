"""
line-art-12-color

Colors an 11-official layout. Not the main product.

Each character's rectangle (advance width x row height) is filled with a
color sampled from the original frame. The glyph itself uses that same color,
darker. Needs the line-art video, because the rectangles come from the decode.

Take:
  original color video  (--input)
  line-art video        (--line)
"""
import line_art_11 as official
import argparse
import os
import subprocess
import sys

import cv2
import numpy as np

STRATEGY_NAME = "line-art-12-color"
GLYPH_DARK = 0.42
COLOR_KEEP = 0.55
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(os.path.dirname(SCRIPT_DIR), "output", STRATEGY_NAME)


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
    print(f"color video: {os.path.abspath(output_path)}")
    return output_path


def _fit(orig_bgr, shape):
    h, w = shape[:2]
    if orig_bgr.shape[0] != h or orig_bgr.shape[1] != w:
        return cv2.resize(orig_bgr, (w, h), interpolation=cv2.INTER_AREA)
    return orig_bgr


def _region_color(orig, x0, y0, x1, y1):
    height, width = orig.shape[:2]
    xa, xb = max(0, int(x0)), min(width, int(x1))
    ya, yb = max(0, int(y0)), min(height, int(y1))
    if xb <= xa or yb <= ya:
        return np.zeros(3, np.float32)
    patch = orig[ya:yb, xa:xb].reshape(-1, 3).astype(np.float32)
    luma = patch.mean(axis=1)
    keep = patch[luma > 45.0]
    src = keep if keep.shape[0] >= 8 else patch
    return src.mean(axis=0)


def predict_lines(gray, aa_model, char_dict, new_width=0, slide=0, ink_skip=None):
    if ink_skip is None:
        ink_skip = official.INK_SKIP_DEFAULT
    model, chars, device = aa_model
    work, used_w, used_h = official.gray_official_array(gray, new_width)
    space = official.space_index(chars)
    ideo = official._ideo_space_index(chars, space)
    widths = official.char_widths(chars, char_dict)
    img = official.pad_official(work)
    img = np.concatenate([np.ones((1 + slide, img.shape[1]), dtype=np.float32), img], axis=0)
    num_line = (img.shape[0] - official.WIN) // official.CELL_H
    stats = {"skip": 0, "infer": 0}
    if official.ROW_BATCH and num_line > 0:
        rows_np = np.stack(
            [img[h * official.CELL_H : h * official.CELL_H + official.WIN] for h in range(num_line)]
        )
        predicts = official._decode_rows_batched(
            model, device, rows_np, chars, space, widths, ink_skip, stats, ideo
        )
    else:
        predicts = [
            official._decode_row(
                model,
                device,
                img[h * official.CELL_H : h * official.CELL_H + official.WIN],
                chars,
                space,
                widths,
                ink_skip,
                stats,
                ideo,
            )
            for h in range(num_line)
        ]
    return predicts, img.shape, used_w, used_h, stats


def render_char_rects(predicts, orig_bgr, char_dict, img_shape, slide, out_w, out_h):
    """Fill each character box, then paint the glyph in a darker shade of that color."""
    orig = _fit(orig_bgr, (out_h, out_w, 3))
    canvas = np.zeros((img_shape[0], img_shape[1], 3), dtype=np.uint8)
    for h, line in enumerate(predicts):
        x = 0
        y = h * official.CELL_H
        for char in line:
            glyph = np.asarray(char_dict[char])
            cw = int(glyph.shape[1])
            rh = official.CELL_H
            fill = _region_color(orig, x, y - slide, x + cw, y - slide + rh)
            fill = fill * COLOR_KEEP + 255.0 * (1.0 - COLOR_KEEP)
            fill_u8 = np.clip(fill + 0.5, 0, 255).astype(np.uint8)
            dark_u8 = np.clip(fill * GLYPH_DARK + 0.5, 0, 255).astype(np.uint8)
            y1 = min(img_shape[0], y + rh)
            x1 = min(img_shape[1], x + cw)
            if y1 > y and x1 > x:
                canvas[y:y1, x:x1] = fill_u8
                gh = min(official.GLYPH_H, y1 - y, glyph.shape[0])
                gw = min(cw, x1 - x, glyph.shape[1])
                ink = glyph[:gh, :gw]
                if np.any(ink):
                    patch = canvas[y : y + gh, x : x + gw]
                    patch[ink] = dark_u8
                    canvas[y : y + gh, x : x + gw] = patch
            x += cw
    return canvas[slide : slide + out_h, 0:out_w]


def _frame_png(work_dir, idx):
    return os.path.join(work_dir, f"f{idx:06d}.png")


def _count_done(work_dir):
    n = 0
    while os.path.isfile(_frame_png(work_dir, n)):
        n += 1
    return n


def _save_bgr(path, bgr):
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path[:-4] + ".part.png" if path.lower().endswith(".png") else path + ".part.png"
    if not cv2.imwrite(tmp, bgr):
        raise RuntimeError(f"cannot write {tmp}")
    os.replace(tmp, path)


def _stitch_frames(work_dir, n_frames, fps, video_out):
    if n_frames <= 0:
        raise RuntimeError("no color frames to stitch")
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
                raise RuntimeError(f"missing color frame f{i:06d}.png")
            writer.write(vis)
        writer.release()
        finalize_mp4(tmp_path, video_out)
        return video_out


def _is_video(path):
    return os.path.isfile(path) and path.lower().endswith(VIDEO_EXTS)


def _open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    return cap, fps, total


def _seek(cap, idx):
    if idx <= 0:
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    if pos == idx:
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(idx):
        ok, _frame = cap.read()
        if not ok:
            break


def colorize_video(
    orig_path,
    line_path,
    output_folder=DEFAULT_OUT,
    video_out=None,
    max_frames=0,
    start=0,
    restart=False,
    new_width=0,
    slide=0,
    ink_skip=None,
):
    if not os.path.isfile(orig_path):
        raise RuntimeError(f"original not found: {orig_path}")
    if not os.path.isfile(line_path):
        raise RuntimeError(f"line video not found: {line_path}")
    if ink_skip is None:
        ink_skip = official.INK_SKIP_DEFAULT
    orig_cap, orig_fps, orig_total = _open_video(orig_path)
    line_cap, line_fps, line_total = _open_video(line_path)
    full = min(n for n in (orig_total, line_total) if n > 0) if (orig_total or line_total) else 0
    begin = max(0, int(start))
    if max_frames and max_frames > 0:
        total = begin + int(max_frames)
        if full:
            total = min(full, total)
    else:
        total = full
    os.makedirs(output_folder, exist_ok=True)
    stem = os.path.splitext(os.path.basename(orig_path))[0]
    if not video_out:
        video_out = os.path.join(output_folder, f"{stem}_rect.mp4")
    video_out = os.path.abspath(video_out.strip())
    work_dir = os.path.splitext(video_out)[0] + "_frames"
    if restart and os.path.isdir(work_dir):
        for name in os.listdir(work_dir):
            path = os.path.join(work_dir, name)
            if os.path.isfile(path):
                os.remove(path)
    os.makedirs(work_dir, exist_ok=True)
    done = _count_done(work_dir)
    if done < begin:
        done = begin
    if total and done > total:
        done = total
    fps = line_fps or orig_fps or 24.0
    print(
        f"{STRATEGY_NAME}: {total or '?'} frames  resume {done}  "
        f"glyph_dark={GLYPH_DARK}  ink_skip={ink_skip}",
        flush=True,
    )
    finished = total > 0 and done >= total
    if not finished:
        print(f"{STRATEGY_NAME}: loading 11-official weights", flush=True)
        aa_model = official.load_net()
        char_dict = official.load_char_dict()
        _seek(orig_cap, done)
        _seek(line_cap, done)
        idx = done
        try:
            while True:
                if total and idx >= total:
                    finished = True
                    break
                ok_o, orig = orig_cap.read()
                ok_l, line = line_cap.read()
                if not ok_o or not ok_l:
                    finished = True
                    break
                gray = line[:, :, 0] if line.ndim == 3 else line
                predicts, img_shape, used_w, used_h, stats = predict_lines(
                    gray, aa_model, char_dict, new_width=new_width, slide=slide, ink_skip=ink_skip
                )
                colored = render_char_rects(
                    predicts, orig, char_dict, img_shape, slide, used_w, used_h
                )
                _save_bgr(_frame_png(work_dir, idx), colored)
                idx += 1
                done = idx
                if done == 1 or done % 5 == 0 or (total and done == total):
                    nwin = stats["skip"] + stats["infer"]
                    skip_msg = ""
                    if ink_skip > 0 and nwin:
                        skip_msg = f"  empty-skip {stats['skip']}/{nwin} ({100.0 * stats['skip'] / nwin:.1f}%)"
                    print(f"  color frame {done}/{total or '?'}  {used_w}x{used_h}{skip_msg}", flush=True)
                if total and done >= total:
                    finished = True
                    break
        except KeyboardInterrupt:
            orig_cap.release()
            line_cap.release()
            print(f"stopped at {done}/{total or '?'}  run the same command to continue", flush=True)
            return video_out
    orig_cap.release()
    line_cap.release()
    if begin:
        print(f"frames {begin}..{idx - 1} -> {work_dir}", flush=True)
        return video_out
    done = _count_done(work_dir)
    if total and done > total:
        done = total
    if total and done < total and not max_frames:
        print(f"incomplete: {done}/{total}  run the same command to continue")
        return video_out
    print(f"stitching {done} frames -> {video_out}", flush=True)
    _stitch_frames(work_dir, done, fps, video_out)
    print(f"{STRATEGY_NAME} video: {done} frames")
    return video_out


def parse_args():
    parser = argparse.ArgumentParser(description=STRATEGY_NAME)
    parser.add_argument("--input", required=True, help="Original color video")
    parser.add_argument("--line", required=True, help="Line-art video, same frames as the color source")
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--video-out", default="")
    parser.add_argument("--width", type=int, default=0, help="Pixel width. 0 = line-art width.")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after N frames. 0 = all.")
    parser.add_argument("--start", type=int, default=0, help="First source frame index.")
    parser.add_argument("--ink-skip", type=float, default=official.INK_SKIP_DEFAULT)
    parser.add_argument("--restart", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    print(f"{STRATEGY_NAME}: max_frames={args.max_frames or 'all'}  ink_skip={args.ink_skip}", flush=True)
    if not (_is_video(args.input) and _is_video(args.line)):
        raise SystemExit("need a color video (--input) and a line-art video (--line)")
    colorize_video(
        args.input,
        args.line,
        args.output,
        args.video_out or None,
        max_frames=args.max_frames,
        start=args.start,
        restart=args.restart,
        new_width=args.width,
        ink_skip=args.ink_skip,
    )
