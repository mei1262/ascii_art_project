"""
Train a DeepAA CNN on the real 500-image AA dataset.

Data:
  data/image_500/{name}line_resize.png
  data/deepaa/data_500.csv   (file_name, x, y, char, label)

Each sample is a 64x64 crop on the estimated line image; the label is the
center character from hand-made AA (not synthetic font glyphs).

The character set is the original DeepAA Japanese 411 classes
(labels with frequency >= 10 in data_500.csv, including spaces).

Train:
  python src/deepaa_cnn.py --train
  python src/deepaa_cnn.py --train --epochs 8 --batch 64

Convert existing line drawings (do not extract lines again):
  python src/deepaa_cnn.py --from-lines --input input/paper_lineart --width 80
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from glyphs import FULL_CHARS
from main import DEFAULT_OUTPUT, PROJECT_ROOT, save_batch_html

WIN = 64
CELL_H = 18
RENDER_CELL_W = 18
RENDER_CELL_H = 30
PAD = 30
MERGIN = (WIN - CELL_H) // 2 + PAD
MIN_FREQ = 10
INK_MIN = 0.06
IMG_DIR = os.path.join(PROJECT_ROOT, "data", "image_500")
CSV_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data", "deepaa", "data_500.csv"),
    os.path.join(PROJECT_ROOT, "data", "data_500.csv"),
]
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "own")
WEIGHT_PATH = os.path.join(MODEL_DIR, "deepaa_cnn.pt")
META_PATH = os.path.join(MODEL_DIR, "deepaa_cnn.json")
STRUCT_CSV = os.path.join(PROJECT_ROOT, "data", "struct_nonempty.csv")
STRUCT_META = os.path.join(PROJECT_ROOT, "data", "struct_nonempty.json")
STRUCT_PREVIEW = os.path.join(PROJECT_ROOT, "data", "struct_nonempty_preview")
CHARSET_PATH = os.path.join(PROJECT_ROOT, "data", "deepaa_411.json")
EMPTY_CHARS = {" ", "\u3000"}

# Fold Japanese/fullwidth twins into printable ASCII 32-126 (Xu 2010 / line_art_4).
# Do not collapse l/i/j/L into |. 二/三/ﾆ are dropped; ニ and ≡ stay.
# Brackets fold to [ ].
KEEP_STRUCT = {"ニ", "≡"}
CHAR_ALIASES = {
    "\u3000": " ",
    "／": "/",
    "ノ": "/",
    "ﾉ": "/",
    "ヽ": "/",
    "丿": "/",
    "ヘ": "/",
    "ﾍ": "/",
    "へ": "/",
    "く": "/",
    "ゝ": "/",
    "＼": "\\",
    "｜": "|",
    "│": "|",
    "┃": "|",
    "￤": "|",
    "ｌ": "l",
    "ｉ": "i",
    "Ｉ": "I",
    "＿": "_",
    "￣": "-",
    "―": "-",
    "ー": "-",
    "ｰ": "-",
    "‐": "-",
    "─": "-",
    "━": "-",
    "－": "-",
    "一": "-",
    "～": "~",
    "⌒": "~",
    "〜": "~",
    "。": ".",
    "．": ".",
    "｡": ".",
    "・": ".",
    "･": ".",
    "丶": ".",
    "、": ",",
    "､": ",",
    "，": ",",
    "´": "'",
    "｀": "'",
    "′": "'",
    "‘": "'",
    "’": "'",
    "¨": '"',
    "〃": '"',
    "″": '"',
    "“": '"',
    "”": '"',
    "：": ":",
    "；": ";",
    "＾": "^",
    "＋": "+",
    "十": "+",
    "┼": "+",
    "〈": "<",
    "＜": "<",
    "≪": "<",
    "《": "<",
    "〉": ">",
    "＞": ">",
    "≫": ">",
    "》": ">",
    "（": "(",
    "）": ")",
    "〔": "[",
    "｢": "[",
    "「": "[",
    "［": "[",
    "【": "[",
    "〕": "]",
    "｣": "]",
    "」": "]",
    "］": "]",
    "】": "]",
    "｛": "{",
    "｝": "}",
    "＝": "=",
    "７": "7",
    "└": "L",
    "┗": "L",
    "┐": "7",
    "┓": "7",
    "┘": "j",
    "┛": "j",
    "┌": "r",
    "┏": "r",
    "┴": "+",
    "┬": "+",
    "├": "+",
    "┤": "+",
    "∧": "^",
    "∨": "v",
    "Ｖ": "V",
    "Ｙ": "Y",
    "∠": "/",
    "≧": ">",
    "≦": "<",
}


def to_structure_char(ch):
    if ch in KEEP_STRUCT:
        return ch
    if ch in FULL_CHARS:
        return ch
    return CHAR_ALIASES.get(ch)


def model_paths(struct=False):
    if struct:
        return (
            os.path.join(MODEL_DIR, "deepaa_struct.pt"),
            os.path.join(MODEL_DIR, "deepaa_struct.json"),
            os.path.join(MODEL_DIR, "deepaa_struct_last.pt"),
        )
    return (
        WEIGHT_PATH,
        META_PATH,
        os.path.join(MODEL_DIR, "deepaa_cnn_last.pt"),
    )


class DeepAACNN(nn.Module):
    """Paper net: C64-C64-P-C128-C128-P-C256-C256-C256-P-FC-FC-O."""

    def __init__(self, num_classes, fc_size=512):
        super().__init__()
        self.features = nn.Sequential(
            self._conv(1, 64),
            self._conv(64, 64),
            nn.MaxPool2d(2),
            self._conv(64, 128),
            self._conv(128, 128),
            nn.MaxPool2d(2),
            self._conv(128, 256),
            self._conv(256, 256),
            self._conv(256, 256),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
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
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    @staticmethod
    def _conv(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def find_csv():
    for path in CSV_CANDIDATES:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError("data_500.csv not found; put it in data/deepaa/ or data/")


def mapped_frame():
    df = pd.read_csv(find_csv(), encoding="cp932")
    df = df.copy()
    df["char"] = df["char"].map(to_structure_char)
    return df.dropna(subset=["char"]).reset_index(drop=True)


def build_struct_dataset(min_freq=MIN_FREQ, preview_per_class=8):
    """Keep only nonempty AA cells: line-art 64x64 window + printable ASCII label."""
    os.makedirs(os.path.dirname(STRUCT_CSV), exist_ok=True)
    full = mapped_frame()
    char_widths, default_width = estimate_widths(full)
    nonempty = full[~full["char"].isin(EMPTY_CHARS)].copy()
    freq = nonempty["char"].value_counts()
    keep = set(freq[freq >= min_freq].index)
    chars = [ch for ch in FULL_CHARS if ch in keep]
    nonempty = nonempty[nonempty["char"].isin(keep)].reset_index(drop=True)
    nonempty[["file_name", "x", "y", "char"]].to_csv(STRUCT_CSV, index=False, encoding="utf-8")
    meta = {
        "n": int(len(nonempty)),
        "n_classes": len(chars),
        "chars": chars,
        "dropped_empty": int(full["char"].isin(EMPTY_CHARS).sum()),
        "min_freq": min_freq,
        "default_width": int(default_width),
        "char_widths": {ch: int(char_widths.get(ch, default_width)) for ch in chars},
        "window": WIN,
        "note": "non-space cells from DeepAA line_resize + mapped printable ASCII",
    }
    with open(STRUCT_META, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(
        f"wrote {STRUCT_CSV}  rows={len(nonempty)}  classes={len(chars)} "
        f"{''.join(chars)!r}"
    )
    print(f"dropped empty={meta['dropped_empty']}  default_width={default_width}")
    _write_struct_preview(nonempty, per_class=preview_per_class)
    return nonempty, chars, char_widths, default_width


def _write_struct_preview(df, per_class=8):
    os.makedirs(STRUCT_PREVIEW, exist_ok=True)
    cache = load_image_cache(df["file_name"].unique(), IMG_DIR)
    written = 0
    for ch, group in df.groupby("char"):
        hits = group[group["file_name"].isin(cache)]
        if hits.empty:
            continue
        pick = hits.sample(n=min(per_class, len(hits)), random_state=0)
        folder = os.path.join(STRUCT_PREVIEW, f"u{ord(ch):04x}")
        os.makedirs(folder, exist_ok=True)
        for i, row in enumerate(pick.itertuples(index=False)):
            patch = crop_window(cache[row.file_name], row.x, row.y, slide=0)
            Image.fromarray(patch).save(os.path.join(folder, f"{i:02d}.png"))
            written += 1
    print(f"preview patches: {written} in {STRUCT_PREVIEW}")


def original_charset(min_freq=MIN_FREQ):
    """Japanese 411-class set from the original DeepAA CSV (freq >= 10)."""
    if os.path.isfile(CHARSET_PATH):
        with open(CHARSET_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        chars = [ch for ch in (meta.get("chars") or []) if ch is not None]
        if chars:
            return chars
    df = pd.read_csv(find_csv(), encoding="cp932")
    freq = df["char"].value_counts()
    chars = [ch for ch in freq[freq >= min_freq].index.tolist() if isinstance(ch, str)]
    char_widths, default_width = estimate_widths(df[df["char"].isin(chars)])
    os.makedirs(os.path.dirname(CHARSET_PATH), exist_ok=True)
    with open(CHARSET_PATH, "w", encoding="utf-8") as f:
        json.dump(
            {
                "source": "DeepAA data_500.csv original labels",
                "n_classes": len(chars),
                "min_freq": min_freq,
                "default_width": int(default_width),
                "char_widths": {ch: int(char_widths.get(ch, default_width)) for ch in chars},
                "chars": chars,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"wrote original DeepAA charset {CHARSET_PATH}  classes={len(chars)}")
    return chars


def load_index(min_freq=MIN_FREQ, struct=False):
    if struct:
        if not os.path.isfile(STRUCT_CSV) or not os.path.isfile(STRUCT_META):
            build_struct_dataset(min_freq=min_freq)
        df = pd.read_csv(STRUCT_CSV, encoding="utf-8", dtype={"char": str})
        with open(STRUCT_META, encoding="utf-8") as f:
            meta = json.load(f)
        chars = [ch for ch in (meta.get("chars") or []) if ch not in EMPTY_CHARS]
        if not chars:
            freq = df["char"].value_counts()
            keep = set(freq[freq >= min_freq].index) - EMPTY_CHARS
            chars = [ch for ch in FULL_CHARS if ch in keep]
        df = df[df["char"].isin(chars)].reset_index(drop=True)
        char_to_idx = {ch: i for i, ch in enumerate(chars)}
        df = df.copy()
        df["cls"] = df["char"].map(char_to_idx).astype(np.int64)
        print(
            f"struct nonempty dataset: {len(df)} windows  "
            f"classes {len(chars)} {''.join(chars)!r}"
        )
        return df, chars
    chars = original_charset(min_freq=min_freq)
    df = pd.read_csv(find_csv(), encoding="cp932")
    df = df[df["char"].isin(chars)].reset_index(drop=True)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    df = df.copy()
    df["cls"] = df["char"].map(char_to_idx).astype(np.int64)
    print(
        f"DeepAA original dataset: {len(df)} windows  "
        f"classes {len(chars)} (original Japanese labels)"
    )
    return df, chars


def estimate_widths(df, default=8):
    sub = df[["file_name", "y", "x", "char"]].sort_values(["file_name", "y", "x"])
    names = sub["file_name"].to_numpy()
    ys = sub["y"].to_numpy()
    xs = sub["x"].to_numpy()
    chs = sub["char"].to_numpy()
    buckets = {}
    for i in range(len(sub) - 1):
        if names[i] == names[i + 1] and ys[i] == ys[i + 1]:
            width = int(xs[i + 1] - xs[i])
            if width > 0:
                buckets.setdefault(chs[i], []).append(width)
    all_w = [w for vals in buckets.values() for w in vals]
    fallback = int(np.median(all_w)) if all_w else default
    med = {}
    for ch in df["char"].unique():
        vals = buckets.get(ch)
        med[ch] = int(np.median(vals)) if vals else fallback
    return med, fallback


def crop_window(padded, x, y, slide=0):
    xx = int(x) + PAD
    yy = int(y) + PAD
    if slide:
        xx += int(np.random.randint(-slide, slide + 1))
        yy += int(np.random.randint(-slide, slide + 1))
    patch = padded[yy : yy + WIN, xx : xx + WIN]
    if patch.shape != (WIN, WIN):
        canvas = np.full((WIN, WIN), 255, dtype=np.uint8)
        h, w = patch.shape
        canvas[:h, :w] = patch
        return canvas
    return patch.copy()


def load_image_cache(file_names, img_dir, tail="line_resize"):
    cache = {}
    missing = 0
    for name in file_names:
        path = os.path.join(img_dir, f"{name}{tail}.png")
        if not os.path.isfile(path):
            missing += 1
            continue
        gray = np.array(Image.open(path).convert("L"), dtype=np.uint8)
        cache[name] = np.pad(gray, MERGIN, constant_values=255)
    if missing:
        print(f"missing images for {missing} file_name values")
    return cache


class DeepAAWindows(Dataset):
    def __init__(self, file_ids, xs, ys, labels, images, train=True, slide=1):
        self.file_ids = file_ids.astype(np.int32)
        self.xs = xs.astype(np.int32)
        self.ys = ys.astype(np.int32)
        self.labels = labels.astype(np.int64)
        self.images = images
        self.train = train
        self.slide = slide if train else 0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        patch = crop_window(
            self.images[int(self.file_ids[idx])],
            int(self.xs[idx]),
            int(self.ys[idx]),
            self.slide,
        )
        if self.train and np.random.rand() < 0.3:
            noise = np.random.normal(0, 4, patch.shape)
            patch = np.clip(patch.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        x = torch.from_numpy(patch.astype(np.float32) / 255.0).unsqueeze(0)
        y = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        return x, y


def make_split_sets(df, cache, seed=42):
    names = sorted(cache.keys())
    name_to_id = {name: i for i, name in enumerate(names)}
    images = [cache[name] for name in names]
    df = df[df["file_name"].isin(name_to_id)].reset_index(drop=True)
    files = np.array(sorted(df["file_name"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(files)
    n_val = max(1, int(0.1 * len(files)))
    val_files = set(files[:n_val].tolist())
    train_df = df[~df["file_name"].isin(val_files)]
    val_df = df[df["file_name"].isin(val_files)]
    if train_df.empty or val_df.empty:
        raise RuntimeError("empty train/val split; check image_500 vs CSV filenames")

    def pack(frame, train):
        return DeepAAWindows(
            frame["file_name"].map(name_to_id).to_numpy(),
            frame["x"].to_numpy(),
            frame["y"].to_numpy(),
            frame["cls"].to_numpy(),
            images,
            train=train,
            slide=1 if train else 0,
        )

    return pack(train_df, True), pack(val_df, False), len(train_df), len(val_df)


def train_model(
    epochs=8,
    batch=64,
    lr=1e-3,
    fc_size=512,
    max_samples=0,
    min_freq=MIN_FREQ,
    struct=False,
):
    os.makedirs(MODEL_DIR, exist_ok=True)
    weight_path, meta_path, last_path = model_paths(struct)
    df, chars = load_index(min_freq=min_freq, struct=struct)
    if struct and os.path.isfile(STRUCT_META):
        with open(STRUCT_META, encoding="utf-8") as f:
            meta = json.load(f)
        default_width = int(meta.get("default_width", 11))
        char_widths = meta.get("char_widths") or {}
    else:
        char_widths, default_width = estimate_widths(df)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    print(f"samples {len(df)}  classes {len(chars)}  images {IMG_DIR}")
    cache = load_image_cache(df["file_name"].unique(), IMG_DIR)
    train_set, val_set, n_train, n_val = make_split_sets(df, cache)
    print(f"train windows {n_train}  val windows {n_val}  cached images {len(cache)}")

    train_loader = DataLoader(
        train_set,
        batch_size=batch,
        shuffle=True,
        num_workers=0,
        drop_last=n_train >= batch,
    )
    val_loader = DataLoader(val_set, batch_size=batch, shuffle=False, num_workers=0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device {device}")
    if device.type == "cpu" and n_train > 20000:
        print("full-set CPU training is slow; checkpoints are written after each epoch")
    model = DeepAACNN(len(chars), fc_size=fc_size).to(device)
    best_acc = -1.0
    best_epoch = 0
    epochs_done = 0
    if os.path.isfile(weight_path):
        try:
            pack = torch.load(weight_path, map_location=device, weights_only=False)
        except TypeError:
            pack = torch.load(weight_path, map_location=device)
        prev_chars = pack.get("chars") or []
        if isinstance(prev_chars, str):
            prev_chars = list(prev_chars)
        if int(pack.get("fc_size", 512)) == fc_size and list(prev_chars) == list(chars):
            model.load_state_dict(pack["state_dict"])
            if os.path.isfile(meta_path):
                with open(meta_path, encoding="utf-8") as f:
                    prev_meta = json.load(f)
                best_acc = float(prev_meta.get("val_accuracy", -1.0))
                best_epoch = int(prev_meta.get("best_epoch", 0))
                epochs_done = int(prev_meta.get("epochs_done", 0))
            print(
                f"resumed {weight_path}  prev_best={best_acc * 100:.2f}% "
                f"(epoch {best_epoch})  epochs_done={epochs_done}"
            )
        else:
            print("existing weights do not match this run, training from scratch")
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    counts = np.bincount(train_set.labels, minlength=len(chars)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    class_w = 1.0 / np.sqrt(counts)
    class_w *= len(class_w) / class_w.sum()
    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(class_w, dtype=torch.float32, device=device)
    )
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    n_train_batches = max(1, len(train_loader))

    for epoch in range(1, epochs + 1):
        model.train()
        running, seen = 0.0, 0
        for step, (xb, yb) in enumerate(train_loader, 1):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            opt.step()
            running += float(loss.item()) * len(yb)
            seen += len(yb)
            if step == 1 or step % 200 == 0 or step == n_train_batches:
                print(
                    f"  epoch {epochs_done + epoch}  batch {step}/{n_train_batches}  "
                    f"loss={running / max(1, seen):.4f}",
                    flush=True,
                )
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += int((pred == yb).sum())
                total += len(yb)
        acc = correct / max(1, total)
        global_epoch = epochs_done + epoch
        epoch_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        pack = {
            "state_dict": epoch_state,
            "chars": chars,
            "fc_size": fc_size,
            "char_widths": char_widths,
            "default_width": default_width,
            "struct": bool(struct),
        }
        torch.save(pack, last_path)
        if acc > best_acc:
            best_acc = acc
            best_epoch = global_epoch
            best_state = epoch_state
            pack["state_dict"] = best_state
            torch.save(pack, weight_path)
            saved = "  saved-best"
        else:
            saved = "  kept-best"
        print(
            f"epoch {global_epoch}  this={acc * 100:.2f}%  loss={running / max(1, seen):.4f}  "
            f"best={best_acc * 100:.2f}% (epoch {best_epoch}){saved}",
            flush=True,
        )
        meta = {
            "method": "DeepAA 64x64 CNN (structure chars)" if struct else "DeepAA 64x64 CNN",
            "val_accuracy": float(best_acc),
            "last_val_accuracy": float(acc),
            "best_epoch": int(best_epoch),
            "last_epoch": int(global_epoch),
            "epochs_done": int(global_epoch),
            "n_classes": len(chars),
            "n_train": int(n_train),
            "n_val": int(n_val),
            "window": WIN,
            "min_freq": min_freq,
            "default_width": int(default_width),
            "struct": bool(struct),
            "chars": chars,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

    model.load_state_dict(best_state)
    torch.save(
        {
            "state_dict": best_state,
            "chars": chars,
            "fc_size": fc_size,
            "char_widths": char_widths,
            "default_width": default_width,
            "struct": bool(struct),
        },
        weight_path,
    )
    print(f"best val acc {best_acc * 100:.2f}% (epoch {best_epoch})")
    print(f"weights: {weight_path}")
    return model, chars


def _load_pack(device, struct=False):
    weight_path, _, _ = model_paths(struct)
    if not os.path.isfile(weight_path):
        hint = "python src/deepaa_cnn.py --train --struct" if struct else "python src/deepaa_cnn.py --train"
        raise FileNotFoundError(f"no weights yet; run {hint}")
    try:
        return torch.load(weight_path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(weight_path, map_location=device)


def load_model(device=None, struct=False):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pack = _load_pack(device, struct=struct)
    chars = pack["chars"]
    if isinstance(chars, str):
        chars = list(chars)
    model = DeepAACNN(len(chars), fc_size=int(pack.get("fc_size", 512))).to(device)
    model.load_state_dict(pack["state_dict"])
    model.eval()
    char_widths = pack.get("char_widths") or {}
    default_width = int(pack.get("default_width", 8))
    return model, chars, device, char_widths, default_width


def load_aa_font(font_size):
    candidates = [
        (r"C:\Windows\Fonts\msgothic.ttc", 0),
        (r"C:\Windows\Fonts\YuGothR.ttc", 0),
        (r"C:\Windows\Fonts\simsun.ttc", 0),
    ]
    for path, index in candidates:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, font_size, index=index)
            except Exception:
                continue
    return ImageFont.load_default()


def save_aa_png(ascii_text, path, cell_w=RENDER_CELL_W, cell_h=RENDER_CELL_H):
    lines = ascii_text.split("\n")
    rows = max(1, len(lines))
    cols = max((len(line) for line in lines), default=1)
    font = load_aa_font(max(8, cell_h - 2))
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (0, 0, 0))
    draw = ImageDraw.Draw(canvas)
    for y, line in enumerate(lines):
        for x, ch in enumerate(line):
            if ch not in (" ", "\u3000"):
                draw.text((x * cell_w, y * cell_h), ch, font=font, fill=(255, 255, 255))
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)
    canvas.save(path)
    return path


def to_train_domain(gray):
    """Paper scans are gray-on-gray; DeepAA windows are white(255) with sparse black ink."""
    g = gray.astype(np.float32)
    if float(g.mean()) < 127:
        g = 255.0 - g
    lo, hi = np.percentile(g, (2.0, 96.0))
    if hi > lo + 5:
        g = np.clip((g - lo) / (hi - lo), 0.0, 1.0) * 255.0
    return g.astype(np.uint8)


def _resize_for_cols(gray, target_cols, default_width):
    if not target_cols:
        return gray
    new_w = max(WIN, int(target_cols * max(4, default_width)))
    new_h = max(WIN, int(round(gray.shape[0] * new_w / max(1, gray.shape[1]))))
    return np.array(Image.fromarray(gray).resize((new_w, new_h), Image.Resampling.LANCZOS))


def _space_indices(chars):
    return [i for i, ch in enumerate(chars) if ch in (" ", "\u3000") or ch.isspace()]


def cell_ink_ratio(patch, cell_w):
    """Ink fraction of the center character cell, not the whole 64x64 context."""
    cell_w = max(4, int(cell_w))
    oy = (WIN - CELL_H) // 2
    ox = max(0, (WIN - cell_w) // 2)
    cell = patch[oy : oy + CELL_H, ox : ox + cell_w]
    if cell.size == 0:
        return 0.0
    return float((cell < 200).mean())


def convert_line_image(
    path,
    model,
    chars,
    device,
    char_widths,
    default_width=8,
    step_y=18,
    width=0,
    step_x=0,
    ink_min=INK_MIN,
):
    gray = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    gray = to_train_domain(gray)
    gray = _resize_for_cols(gray, width, default_width)
    padded = np.pad(gray, MERGIN, constant_values=255)
    height, img_w = gray.shape
    step_x = max(4, int(step_x or default_width))
    xs = list(range(0, max(1, img_w), step_x))
    rows = []
    with torch.no_grad():
        for y in range(0, max(1, height - 2), step_y):
            patches = np.stack([crop_window(padded, x, y, slide=0) for x in xs])
            ink = np.array([cell_ink_ratio(p, step_x) for p in patches])
            line = [" "] * len(xs)
            use = ink >= ink_min
            if use.any():
                tensor = torch.from_numpy(patches[use].astype(np.float32) / 255.0)[:, None].to(device)
                pred = model(tensor).argmax(1).cpu().numpy()
                idx = np.flatnonzero(use)
                for slot, cls in zip(idx, pred):
                    folded = to_structure_char(chars[int(cls)])
                    line[int(slot)] = folded if folded else " "
            rows.append("".join(line).rstrip())
    return "\n".join(rows)


def batch_convert(input_folder, output_html, step_y=18, width=0, step_x=0, struct=False, ink_min=INK_MIN):
    model, chars, device, char_widths, default_width = load_model(struct=struct)
    files = [
        f
        for f in os.listdir(input_folder)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    ]
    arts = {}
    tag = "struct" if struct else "DeepAA fold"
    print(f"{tag} CNN: {len(files)} images  device {device}  ink_min={ink_min}")
    for idx, name in enumerate(files, 1):
        src = os.path.join(input_folder, name)
        try:
            arts[name] = convert_line_image(
                src,
                model,
                chars,
                device,
                char_widths,
                default_width,
                step_y,
                width,
                step_x,
                ink_min,
            )
            print(f"[{idx}/{len(files)}] {name}")
        except Exception as exc:
            print(f"[{idx}/{len(files)}] failed {name}: {exc}")
    if arts:
        os.makedirs(os.path.dirname(output_html) or ".", exist_ok=True)
        font = (
            '"Courier New", Consolas, "Liberation Mono", "Menlo", monospace'
            if struct
            else '"MS Gothic", "Yu Gothic", "NSimSun", "Courier New", monospace'
        )
        save_batch_html(
            arts,
            output_html,
            dark_mode=True,
            font_size=max(6, min(12, 640 // max(width or 80, 1))),
            line_height=1.0,
            font_family=font,
        )
        print(f"ASCII gallery: {os.path.abspath(output_html)}")
        png_dir = os.path.join(
            os.path.dirname(output_html) or ".",
            "deepaa_struct" if struct else "deepaa",
        )
        for name, ascii_text in arts.items():
            base, _ = os.path.splitext(name)
            save_aa_png(
                ascii_text,
                os.path.join(png_dir, f"{base}_ascii.png"),
                cell_w=RENDER_CELL_W,
                cell_h=RENDER_CELL_H,
            )
        print(f"ASCII png: {os.path.abspath(png_dir)}  cell={RENDER_CELL_W}x{RENDER_CELL_H}")


def parse_args():
    parser = argparse.ArgumentParser(description="DeepAA real-data CNN")
    parser.add_argument("--train", action="store_true")
    parser.add_argument(
        "--struct",
        action="store_true",
        help="optional ASCII fold; default is original Japanese 411 labels",
    )
    parser.add_argument(
        "--build-dataset",
        action="store_true",
        help="write the original DeepAA Japanese 411 charset",
    )
    parser.add_argument("--from-lines", action="store_true", help="input is already line art")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--fc", type=int, default=512)
    parser.add_argument("--paper", action="store_true", help="use paper FC=4096")
    parser.add_argument("--max-samples", type=int, default=0, help="0 = use all cells")
    parser.add_argument("--min-freq", type=int, default=MIN_FREQ)
    parser.add_argument("--convert", nargs="?", const="INPUT", default=None)
    parser.add_argument("--input", default=os.path.join(PROJECT_ROOT, "input", "paper_lineart"))
    parser.add_argument(
        "--output", default=os.path.join(DEFAULT_OUTPUT, "all_deepaa_ascii.html")
    )
    parser.add_argument("--width", type=int, default=40, help="target columns; 0 = native scale")
    parser.add_argument("--step-x", type=int, default=0, help="0 = median char width from CSV")
    parser.add_argument("--step-y", type=int, default=18)
    parser.add_argument("--ink-min", type=float, default=INK_MIN, help="skip cells with less ink than this")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    fc_size = 4096 if args.paper else args.fc
    did_anything = False
    if args.build_dataset:
        original_charset(min_freq=args.min_freq)
        did_anything = True
    if args.train:
        train_model(
            epochs=args.epochs,
            batch=args.batch,
            fc_size=fc_size,
            max_samples=args.max_samples,
            min_freq=args.min_freq,
            struct=args.struct,
        )
        did_anything = True
    convert = args.from_lines or args.convert is not None
    if convert:
        output = args.output
        if args.struct and os.path.basename(output) == "all_deepaa_ascii.html":
            output = os.path.join(DEFAULT_OUTPUT, "all_deepaa_struct_ascii.html")
        if args.convert and args.convert != "INPUT" and os.path.isfile(args.convert):
            model, chars, device, char_widths, default_width = load_model(struct=args.struct)
            print(
                convert_line_image(
                    args.convert,
                    model,
                    chars,
                    device,
                    char_widths,
                    default_width,
                    args.step_y,
                    args.width,
                    args.step_x,
                    args.ink_min,
                )
            )
        else:
            folder = args.input
            if args.convert and args.convert != "INPUT" and os.path.isdir(args.convert):
                folder = args.convert
            if not os.path.isdir(folder):
                folder = os.path.join(PROJECT_ROOT, "input")
            batch_convert(
                folder,
                output,
                args.step_y,
                args.width,
                args.step_x,
                struct=args.struct,
                ink_min=args.ink_min,
            )
        did_anything = True
    if not did_anything:
        print("usage:")
        print("  python src/deepaa_cnn.py --build-dataset")
        print("  python src/deepaa_cnn.py --train --epochs 8 --batch 64")
        print("  python src/deepaa_cnn.py --from-lines --input input/paper_lineart --width 40")
