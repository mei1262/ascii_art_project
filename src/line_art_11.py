"""
line-art-11-deepaa-light-row-batch

Standalone inference: empty-window skip + one batched forward per row-step.
Does not import other project scripts.

Take these with the .py:
  weight_smallhead.pt
  char_dict.pkl
Looked up in (first hit wins):
  ./models/deepaa_official/
  ../models/deepaa_official/     (when this file lives in src/)
  ./deepaa_official/             (next to this file)
"""
import argparse
import json
import os
import pickle
import subprocess
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

STRATEGY_NAME = "line-art-11-deepaa-light-row-batch"
ROW_BATCH = True
INK_SKIP_DEFAULT = 0.0008
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
WIN = 64
CELL_H = 18
GLYPH_H = 16
FC_SIZE = 512
NUM_CLASSES = 411

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def official_dir():
    cands = [
        os.path.join(SCRIPT_DIR, "models", "deepaa_official"),
        os.path.join(os.path.dirname(SCRIPT_DIR), "models", "deepaa_official"),
        os.path.join(SCRIPT_DIR, "deepaa_official"),
    ]
    for path in cands:
        if os.path.isfile(os.path.join(path, "weight_smallhead.pt")):
            return path
    return cands[1]


OFFICIAL_DIR = official_dir()
PT_PATH = os.path.join(OFFICIAL_DIR, "weight_smallhead.pt")
CHAR_DICT_PATH = os.path.join(OFFICIAL_DIR, "char_dict.pkl")
DEFAULT_OUT = os.path.join(os.path.dirname(SCRIPT_DIR), "output", STRATEGY_NAME)


class SmallHead(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, fc_size=FC_SIZE):
        super().__init__()

        def conv(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1),
                nn.BatchNorm2d(cout, eps=0.001, momentum=0.99),
                nn.ReLU(inplace=True),
            )

        self.features = nn.Sequential(
            conv(1, 64),
            conv(64, 64),
            nn.MaxPool2d(2),
            conv(64, 128),
            conv(128, 128),
            nn.MaxPool2d(2),
            conv(128, 256),
            conv(256, 256),
            conv(256, 256),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Linear(256 * 8 * 8, fc_size),
            nn.BatchNorm1d(fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_size, fc_size),
            nn.BatchNorm1d(fc_size),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 2, 3, 1).contiguous().flatten(1)
        return self.classifier(x)


def load_net(device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(PT_PATH):
        raise FileNotFoundError(
            f"missing {PT_PATH}\n"
            "put weight_smallhead.pt (and char_dict.pkl) in models/deepaa_official/"
        )
    try:
        pack = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        pack = torch.load(PT_PATH, map_location="cpu")
    chars = pack["chars"]
    if isinstance(chars, str):
        chars = list(chars)
    model = SmallHead(len(chars), int(pack.get("fc_size", FC_SIZE)))
    model.load_state_dict(pack["state_dict"], strict=True)
    model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(
        f"{STRATEGY_NAME}: FC{pack.get('fc_size', FC_SIZE)}  "
        f"params={n/1e6:.2f}M  device={device}  weights={PT_PATH}",
        flush=True,
    )
    return model, chars, device


def load_char_dict():
    if not os.path.isfile(CHAR_DICT_PATH):
        raise FileNotFoundError(f"missing {CHAR_DICT_PATH}")
    with open(CHAR_DICT_PATH, "rb") as f:
        return pickle.load(f, encoding="latin1")


def space_index(chars):
    for i, ch in enumerate(chars):
        if ch == " ":
            return i
    raise ValueError("charset has no ASCII space")


def char_widths(chars, char_dict, default=11):
    widths = np.empty(len(chars), dtype=np.int32)
    for i, ch in enumerate(chars):
        glyph = char_dict.get(ch)
        widths[i] = max(1, int(glyph.shape[1]) if glyph is not None else default)
    return widths


def gray_official_array(arr, new_width=0):
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    arr = arr.astype(np.uint8)
    orig_h, orig_w = arr.shape
    if new_width <= 0:
        new_width = orig_w
    new_height = int(orig_h * new_width / orig_w)
    if new_width != orig_w or new_height != orig_h:
        arr = np.array(Image.fromarray(arr).resize((new_width, new_height), Image.LANCZOS))
    return arr, new_width, new_height


def pad_official(gray):
    mergin = (WIN - CELL_H) // 2
    h, w = gray.shape
    canvas = np.ones((h + 2 * mergin + CELL_H, w + 2 * mergin + CELL_H), dtype=np.uint8) * 255
    canvas[mergin : mergin + h, mergin : mergin + w] = gray
    return canvas.astype(np.float32) / 255.0


def _ideo_space_index(chars, space):
    for i, ch in enumerate(chars):
        if ch == "\u3000":
            return i
    return space


def window_ink_ratio(patch, dark=0.88):
    if patch.size == 0:
        return 0.0
    return float((patch < dark).mean())


@torch.no_grad()
def _decode_row(model, device, row, chars, space, widths, ink_skip=0.0, skip_stats=None, ideo=None):
    row_t = torch.from_numpy(np.ascontiguousarray(row))[None, None].to(device)
    width = row.shape[1]
    w = 0
    penalty = True
    line = []
    max_w = width - WIN
    if ideo is None:
        ideo = space
    while w <= max_w:
        if ink_skip > 0 and window_ink_ratio(row[:, w : w + WIN]) < ink_skip:
            idx = ideo
            if skip_stats is not None:
                skip_stats["skip"] = skip_stats.get("skip", 0) + 1
        else:
            logits = model(row_t[:, :, :, w : w + WIN])[0]
            if penalty:
                logits = logits.clone()
                logits[space] = -1e9
            idx = int(logits.argmax())
            if skip_stats is not None:
                skip_stats["infer"] = skip_stats.get("infer", 0) + 1
        penalty = idx == space
        line.append(chars[idx])
        w += int(widths[idx])
    return line


def render_official(predicts, img_shape, char_dict, slide, out_w, out_h):
    canvas = np.ones(img_shape, dtype=np.uint8) * 255
    for h, line in enumerate(predicts):
        w = 0
        for char in line:
            glyph = char_dict[char]
            cw = int(glyph.shape[1])
            canvas[h * CELL_H : h * CELL_H + GLYPH_H, w : w + cw] = (
                255 - glyph.astype(np.uint8) * 255
            )
            w += cw
    return Image.fromarray(canvas).crop((0, slide, out_w, out_h + slide))


@torch.no_grad()
def _decode_rows_batched(model, device, rows_np, chars, space, widths, ink_skip=0.0, skip_stats=None, ideo=None):
    num_line, _win_h, width = rows_np.shape
    if ideo is None:
        ideo = space
    max_w = width - WIN
    rows_t = torch.from_numpy(np.ascontiguousarray(rows_np))[:, None].to(device)
    ws = [0] * num_line
    alive = [True] * num_line
    penalty = [True] * num_line
    lines = [[] for _ in range(num_line)]
    while True:
        infer_h = []
        for h in range(num_line):
            if not alive[h]:
                continue
            if ws[h] > max_w:
                alive[h] = False
                continue
            w = ws[h]
            if ink_skip > 0 and window_ink_ratio(rows_np[h, :, w : w + WIN]) < ink_skip:
                idx = ideo
                lines[h].append(chars[idx])
                penalty[h] = idx == space
                ws[h] = w + int(widths[idx])
                if skip_stats is not None:
                    skip_stats["skip"] = skip_stats.get("skip", 0) + 1
            else:
                infer_h.append(h)
        if not infer_h:
            if any(alive):
                continue
            break
        batch = torch.stack([rows_t[h, :, :, ws[h] : ws[h] + WIN] for h in infer_h], dim=0)
        logits = model(batch).clone()
        pen = torch.tensor([penalty[h] for h in infer_h], device=logits.device, dtype=torch.bool)
        logits[pen, space] = -1e9
        idxs = logits.argmax(dim=1)
        for i, h in enumerate(infer_h):
            idx = int(idxs[i].item())
            lines[h].append(chars[idx])
            penalty[h] = idx == space
            ws[h] += int(widths[idx])
            if skip_stats is not None:
                skip_stats["infer"] = skip_stats.get("infer", 0) + 1
    return lines


@torch.no_grad()
def decode_official(model, device, gray, chars, char_dict, slide=0, ink_skip=0.0, skip_stats=None, row_batch=False):
    space = space_index(chars)
    ideo = _ideo_space_index(chars, space)
    widths = char_widths(chars, char_dict)
    img = pad_official(gray)
    img = np.concatenate([np.ones((1 + slide, img.shape[1]), dtype=np.float32), img], axis=0)
    num_line = (img.shape[0] - WIN) // CELL_H
    if row_batch and num_line > 0:
        rows_np = np.stack([img[h * CELL_H : h * CELL_H + WIN] for h in range(num_line)])
        predicts = _decode_rows_batched(
            model, device, rows_np, chars, space, widths, ink_skip, skip_stats, ideo
        )
    else:
        predicts = [
            _decode_row(
                model,
                device,
                img[h * CELL_H : h * CELL_H + WIN],
                chars,
                space,
                widths,
                ink_skip,
                skip_stats,
                ideo,
            )
            for h in range(num_line)
        ]
    text = "\n".join("".join(line) for line in predicts)
    png = render_official(predicts, img.shape, char_dict, slide, gray.shape[1], gray.shape[0])
    return text, png, num_line


def convert_from_line(gray, aa_model, char_dict, new_width=0, slide=0, ink_skip=0.0, row_batch=False):
    model, chars, device = aa_model
    work, used_w, used_h = gray_official_array(gray, new_width)
    stats = {"skip": 0, "infer": 0}
    text, png, rows = decode_official(
        model, device, work, chars, char_dict, slide=slide, ink_skip=ink_skip, skip_stats=stats, row_batch=row_batch
    )
    preview = work if float(work.mean()) > 127.0 else (255 - work)
    return text, preview.astype(np.uint8), png, rows, used_w, used_h, stats


def load_line_gray(path):
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8)


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
    print(f"ASCII video: {os.path.abspath(output_path)}")
    return output_path


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


def convert_line_video(input_path, output_folder=DEFAULT_OUT, new_width=0, video_out=None, slide=0, restart=False, ink_skip=INK_SKIP_DEFAULT):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {input_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    os.makedirs(output_folder, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    if not video_out:
        video_out = os.path.join(output_folder, f"{stem}_ascii.mp4")
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
        f"pixel_width={new_width or 'original'}  row_batch={ROW_BATCH}  ink_skip={ink_skip}"
    )
    if not finished:
        aa_model = load_net()
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
                text, _, png, rows, used_w, used_h, stats = convert_from_line(
                    gray, aa_model, char_dict, new_width=new_width, slide=slide, ink_skip=ink_skip, row_batch=ROW_BATCH
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
                    nwin = stats["skip"] + stats["infer"]
                    skip_msg = ""
                    if ink_skip > 0 and nwin:
                        skip_msg = f"  empty-skip {stats['skip']}/{nwin} ({100.0 * stats['skip'] / nwin:.1f}%)"
                    print(f"  ascii frame {done}/{total or '?'}  {used_w}x{used_h} rows={rows}{skip_msg}", flush=True)
                if total and done >= total:
                    finished = True
                    break
        except KeyboardInterrupt:
            cap.release()
            print(f"stopped at {done}/{total or '?'}  run the same command to continue", flush=True)
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


def batch_extract(input_folder, output_folder=DEFAULT_OUT, new_width=0, slide=0, ink_skip=INK_SKIP_DEFAULT):
    if not os.path.exists(input_folder):
        print(f"input not found: {input_folder}")
        return []
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    saved = []
    aa_model = load_net()
    char_dict = load_char_dict()
    print(
        f"{STRATEGY_NAME}: {len(files)} images  pixel_width={new_width or 'original'}  "
        f"ink_skip={ink_skip}  row_batch={ROW_BATCH}  (expects line art)"
    )
    for idx, name in enumerate(files, 1):
        src = os.path.join(input_folder, name)
        try:
            gray = load_line_gray(src)
            orig_h, orig_w = gray.shape
            text, preview, png, rows, used_w, used_h, stats = convert_from_line(
                gray, aa_model, char_dict, new_width=new_width, slide=slide, ink_skip=ink_skip, row_batch=ROW_BATCH
            )
            base, _ = os.path.splitext(name)
            line_path = os.path.join(output_folder, f"{base}_line.png")
            cv2.imwrite(line_path, preview)
            png.save(os.path.join(output_folder, f"{base}_ascii.png"))
            with open(os.path.join(output_folder, f"{base}_ascii.txt"), "w", encoding="utf-8") as f:
                f.write(text + "\n")
            saved.append(line_path)
            nwin = stats["skip"] + stats["infer"]
            skip_msg = ""
            if ink_skip > 0 and nwin:
                skip_msg = f"  empty-skip {stats['skip']}/{nwin} ({100.0 * stats['skip'] / nwin:.1f}%)"
            print(
                f"[{idx}/{len(files)}] {name}  {orig_w}x{orig_h} -> {used_w}x{used_h}  "
                f"rows={rows}  chars={sum(len(line) for line in text.splitlines())}{skip_msg}"
            )
        except Exception as exc:
            print(f"[{idx}/{len(files)}] failed {name}: {exc}")
    return saved


def parse_args():
    parser = argparse.ArgumentParser(description=STRATEGY_NAME)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--width", type=int, default=0, help="Pixel width. 0 = original.")
    parser.add_argument("--from-lines", action="store_true", help="Kept for compatibility; images are loaded as line art.")
    parser.add_argument("--video-out", default="")
    parser.add_argument("--slide", type=int, default=0)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--ink-skip", type=float, default=INK_SKIP_DEFAULT)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    print(f"{STRATEGY_NAME}: row_batch={ROW_BATCH}  ink_skip={args.ink_skip}", flush=True)
    if os.path.isfile(args.input) and args.input.lower().endswith(VIDEO_EXTS):
        convert_line_video(
            args.input,
            args.output,
            args.width,
            args.video_out or None,
            slide=args.slide,
            restart=args.restart,
            ink_skip=args.ink_skip,
        )
    else:
        batch_extract(args.input, args.output, args.width, slide=args.slide, ink_skip=args.ink_skip)
