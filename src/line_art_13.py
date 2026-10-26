"""
line-art-13-hold

11-official decode, plus a long-hold lock. Does not recolor.

Reads a line-art video (same input as model 11). When the line frame stays
close to the last inferred frame for --hold-sec, later frames copy that ASCII
instead of running the network. Output frame count and fps stay with the source.
A lookalike shorter than --hold-sec is still inferred, same as 11-official.

Needs src/line_art_11.py and the small-head weights next to the project.
"""
import argparse
import json
import os
import sys

import cv2
import numpy as np
from PIL import Image

import line_art_11 as official

STRATEGY_NAME = "line-art-13-hold"
HOLD_MAE_DEFAULT = 6.0
HOLD_SEC_DEFAULT = 0.5
PROBE_SIDE = 96
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(os.path.dirname(SCRIPT_DIR), "output", STRATEGY_NAME)


def probe_line(gray, side=PROBE_SIDE):
    if gray.ndim == 3:
        gray = gray[:, :, 0]
    h, w = gray.shape[:2]
    longest = max(h, w)
    if longest > side:
        scale = side / float(longest)
        size = (max(8, int(round(w * scale))), max(8, int(round(h * scale))))
        gray = cv2.resize(gray, size, interpolation=cv2.INTER_AREA)
    return gray


def frame_mae(a, b):
    return float(np.mean(np.abs(a.astype(np.int16) - b.astype(np.int16))))


class LineHold:
    """Decide which line frames need 11-official, and which repeat the key ASCII."""

    def __init__(self, min_frames, mae_max):
        self.min_frames = max(2, int(min_frames))
        self.mae_max = float(mae_max)
        self.key_probe = None
        self.locked = False
        self.pending = []
        self.last_mae = None

    def set_key(self, probe, locked=False):
        self.key_probe = probe
        self.locked = bool(locked)
        self.pending = []

    def push(self, gray, probe):
        if self.mae_max <= 0 or self.key_probe is None:
            self.set_key(probe, locked=False)
            self.last_mae = 0.0
            return [("infer", gray)]
        self.last_mae = frame_mae(self.key_probe, probe)
        if self.last_mae <= self.mae_max:
            if self.locked:
                return [("copy", 1)]
            self.pending.append(gray)
            if 1 + len(self.pending) >= self.min_frames:
                n = len(self.pending)
                self.pending = []
                self.locked = True
                return [("copy", n)]
            return []
        burst = self.pending + [gray]
        self.pending = []
        self.locked = False
        self.key_probe = None
        return [("infer", g) for g in burst]

    def finish(self):
        burst = self.pending
        self.pending = []
        self.locked = False
        return [("infer", g) for g in burst]


def _min_frames(fps, hold_sec):
    if hold_sec <= 0:
        return 1
    return max(2, int(round(float(fps) * float(hold_sec))))


def _read_progress(work_dir):
    path = os.path.join(work_dir, "progress.json")
    if not os.path.isfile(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}
    return meta if isinstance(meta, dict) else {}


def _write_progress(work_dir, meta):
    path = os.path.join(work_dir, "progress.json")
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def _seek(cap, idx):
    if idx <= 0:
        return True
    cap.set(cv2.CAP_PROP_POS_FRAMES, float(idx))
    pos = int(cap.get(cv2.CAP_PROP_POS_FRAMES) or 0)
    if pos == idx:
        return True
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    for _ in range(idx):
        ok, _frame = cap.read()
        if not ok:
            return False
    return True


def _line_gray(frame):
    if frame.ndim == 3:
        return frame[:, :, 0]
    return frame


def convert_line_video(
    input_path,
    output_folder=DEFAULT_OUT,
    new_width=0,
    video_out=None,
    slide=0,
    restart=False,
    ink_skip=official.INK_SKIP_DEFAULT,
    hold_mae=HOLD_MAE_DEFAULT,
    hold_sec=HOLD_SEC_DEFAULT,
):
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"cannot open video: {input_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 8.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    os.makedirs(output_folder, exist_ok=True)
    stem = os.path.splitext(os.path.basename(input_path))[0]
    if not video_out:
        video_out = os.path.join(output_folder, f"{stem}_hold.mp4")
    video_out = os.path.abspath(video_out.strip())
    work_dir = os.path.splitext(video_out)[0] + "_frames"
    if restart and os.path.isdir(work_dir):
        for name in os.listdir(work_dir):
            path = os.path.join(work_dir, name)
            if os.path.isfile(path):
                os.remove(path)
    os.makedirs(work_dir, exist_ok=True)
    done = official._count_done(work_dir)
    mae_max = 0.0 if hold_sec <= 0 else float(hold_mae)
    gate = LineHold(_min_frames(src_fps, hold_sec), mae_max)
    key_png = None
    held_n = 0
    fresh_n = 0
    meta = _read_progress(work_dir)
    if done > 0 and _seek(cap, done - 1):
        ok, prev = cap.read()
        key_path = official._frame_png(work_dir, done - 1)
        if ok and os.path.isfile(key_path):
            key_png = Image.open(key_path).convert("RGB")
            locked = int(meta.get("done", -1)) == done and bool(meta.get("locked"))
            probe_path = os.path.join(work_dir, "key_probe.png")
            if locked and os.path.isfile(probe_path):
                saved = cv2.imread(probe_path, cv2.IMREAD_GRAYSCALE)
                gate.set_key(saved if saved is not None else probe_line(_line_gray(prev)), locked=True)
            else:
                gate.set_key(probe_line(_line_gray(prev)), locked=False)
            if int(meta.get("done", -1)) == done:
                held_n = int(meta.get("held", 0))
                fresh_n = int(meta.get("fresh", 0))
    else:
        _seek(cap, 0)
    print(
        f"{STRATEGY_NAME}: video {total or '?'} frames  resume {done}  "
        f"pixel_width={new_width or 'original'}  ink_skip={ink_skip}  "
        f"hold_mae={mae_max}  hold_sec={hold_sec}  min_frames={gate.min_frames}  locked={gate.locked}",
        flush=True,
    )
    aa_model = None
    char_dict = None
    stats = {"held": held_n, "fresh": fresh_n}
    finished = total > 0 and done >= total

    def ensure_model():
        nonlocal aa_model, char_dict
        if aa_model is None:
            print(f"{STRATEGY_NAME}: loading 11-official weights", flush=True)
            aa_model = official.load_net()
            char_dict = official.load_char_dict()
        return aa_model, char_dict

    def commit_png(png, held, probe=None):
        nonlocal done, key_png
        official._save_ascii_frame(png, official._frame_png(work_dir, done))
        key_png = png
        done += 1
        if held:
            stats["held"] += 1
        else:
            stats["fresh"] += 1
            if probe is not None:
                cv2.imwrite(os.path.join(work_dir, "key_probe.png"), probe)
        _write_progress(
            work_dir,
            {
                "source": os.path.abspath(input_path),
                "video_out": video_out,
                "done": done,
                "total": total,
                "fps": src_fps,
                "new_width": new_width,
                "locked": gate.locked,
                "held": stats["held"],
                "fresh": stats["fresh"],
                "hold_mae": mae_max,
                "hold_sec": hold_sec,
                "last_mae": gate.last_mae,
            },
        )

    def emit(actions):
        info = None
        for kind, payload in actions:
            if kind == "copy":
                if key_png is None:
                    raise RuntimeError("hold copy without a key frame")
                for _ in range(int(payload)):
                    commit_png(key_png, True)
                continue
            model, glyphs = ensure_model()
            _text, _, png, rows, used_w, used_h, win = official.convert_from_line(
                payload,
                model,
                glyphs,
                new_width=new_width,
                slide=slide,
                ink_skip=ink_skip,
                row_batch=official.ROW_BATCH,
            )
            probe = probe_line(payload)
            gate.set_key(probe, locked=False)
            commit_png(png, False, probe)
            nwin = win["skip"] + win["infer"]
            skip_msg = ""
            if ink_skip > 0 and nwin:
                skip_msg = f"  empty-skip {win['skip']}/{nwin} ({100.0 * win['skip'] / nwin:.1f}%)"
            info = (rows, used_w, used_h, skip_msg)
        return info

    if not finished:
        seen = done
        try:
            while True:
                if total and seen >= total:
                    finished = True
                    break
                ok, frame = cap.read()
                if not ok:
                    finished = True
                    break
                gray = _line_gray(frame)
                actions = gate.push(gray.copy(), probe_line(gray))
                info = emit(actions)
                seen += 1
                if info and (done == 1 or done % 5 == 0 or done == total):
                    rows, used_w, used_h, skip_msg = info
                    mae_txt = f"{gate.last_mae:.2f}" if gate.last_mae is not None else "-"
                    print(
                        f"  ascii frame {done}/{total or '?'}  {used_w}x{used_h} rows={rows}  "
                        f"mae {mae_txt}  locked={gate.locked}  held {stats['held']}  fresh {stats['fresh']}{skip_msg}",
                        flush=True,
                    )
                elif done % 10 == 0 and gate.locked:
                    mae_txt = f"{gate.last_mae:.2f}" if gate.last_mae is not None else "-"
                    print(
                        f"  hold frame {done}/{total or '?'}  mae {mae_txt}  "
                        f"held {stats['held']}  fresh {stats['fresh']}",
                        flush=True,
                    )
                if total and done >= total:
                    finished = True
                    break
            if finished or not total:
                emit(gate.finish())
        except KeyboardInterrupt:
            cap.release()
            print(f"stopped at {done}/{total or '?'}  run the same command to continue", flush=True)
            return video_out
    cap.release()
    done = official._count_done(work_dir)
    if total and done < total:
        print(f"incomplete: {done}/{total}  run the same command to continue")
        return video_out
    print(
        f"stitching {done} frames -> {video_out}  held {stats['held']}  fresh {stats['fresh']}",
        flush=True,
    )
    official._stitch_frames(work_dir, done, src_fps, video_out)
    print(f"{STRATEGY_NAME} video: {done} frames  held {stats['held']}  fresh {stats['fresh']}")
    return video_out


def parse_args():
    parser = argparse.ArgumentParser(description=STRATEGY_NAME)
    parser.add_argument("--input", required=True, help="Line-art video or image folder, same as 11-official")
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--width", type=int, default=0, help="Pixel width. 0 = original.")
    parser.add_argument("--from-lines", action="store_true", help="Kept for compatibility; images are loaded as line art.")
    parser.add_argument("--video-out", default="")
    parser.add_argument("--slide", type=int, default=0)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--ink-skip", type=float, default=official.INK_SKIP_DEFAULT)
    parser.add_argument("--hold-mae", type=float, default=HOLD_MAE_DEFAULT, help="Downscaled line MAE. Higher locks more.")
    parser.add_argument("--hold-sec", type=float, default=HOLD_SEC_DEFAULT, help="Seconds of similarity before locking. 0 disables.")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    print(
        f"{STRATEGY_NAME}: base=11-official  hold_mae={args.hold_mae}  hold_sec={args.hold_sec}  "
        f"ink_skip={args.ink_skip}",
        flush=True,
    )
    if os.path.isfile(args.input) and args.input.lower().endswith(official.VIDEO_EXTS):
        convert_line_video(
            args.input,
            args.output,
            args.width,
            args.video_out or None,
            slide=args.slide,
            restart=args.restart,
            ink_skip=args.ink_skip,
            hold_mae=args.hold_mae,
            hold_sec=args.hold_sec,
        )
    else:
        official.batch_extract(args.input, args.output, args.width, slide=args.slide, ink_skip=args.ink_skip)
