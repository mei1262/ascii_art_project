"""
Official DeepAA conv (frozen) + small FC512 head.

Same 411-class order as official output.py. NHWC flatten so the
copied conv features match the Keras full net.
"""
import argparse
import json
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from deepaa_cnn import (
    IMG_DIR,
    find_csv,
    load_image_cache,
    make_split_sets,
)
from deepaa_official_full import (
    DeepAAOfficialFull,
    NUM_CLASSES,
    OFFICIAL_DIR,
    load_official_full,
    official_chars,
)

PT_PATH = os.path.join(OFFICIAL_DIR, "weight_smallhead.pt")
META_PATH = os.path.join(OFFICIAL_DIR, "weight_smallhead.json")
LAST_PATH = os.path.join(OFFICIAL_DIR, "weight_smallhead_last.pt")
FC_SIZE = 512


class DeepAASmallHead(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, fc_size=FC_SIZE):
        super().__init__()
        conv = DeepAAOfficialFull._conv
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

    def freeze_conv(self):
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 2, 3, 1).contiguous().flatten(1)
        return self.classifier(x)


def _copy_official_conv(model):
    teacher, _, _, _, _ = load_official_full(device="cpu")
    model.features.load_state_dict(teacher.features.state_dict())
    model.freeze_conv()
    return model


def load_official_index(max_samples=0):
    chars = official_chars()
    df = pd.read_csv(find_csv(), encoding="cp932")
    df = df[df["char"].isin(chars)].reset_index(drop=True)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    df = df.copy()
    df["cls"] = df["char"].map(char_to_idx).astype(np.int64)
    if max_samples and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)
    print(f"small-head dataset: {len(df)} windows  classes {len(chars)}  official order")
    return df, chars


def train_smallhead(epochs=2, batch=64, lr=1e-3, max_samples=20000):
    os.makedirs(OFFICIAL_DIR, exist_ok=True)
    df, chars = load_official_index(max_samples=max_samples)
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
    print(f"device {device}  frozen official conv + FC{FC_SIZE}")
    model = DeepAASmallHead(len(chars), FC_SIZE)
    _copy_official_conv(model)
    model.to(device)
    best_acc = -1.0
    best_epoch = 0
    epochs_done = 0
    if os.path.isfile(PT_PATH):
        try:
            pack = torch.load(PT_PATH, map_location=device, weights_only=False)
        except TypeError:
            pack = torch.load(PT_PATH, map_location=device)
        if list(pack.get("chars") or []) == list(chars) and int(pack.get("fc_size", 0)) == FC_SIZE:
            model.load_state_dict(pack["state_dict"], strict=False)
            _copy_official_conv(model)
            model.to(device)
            if os.path.isfile(META_PATH):
                with open(META_PATH, encoding="utf-8") as f:
                    prev = json.load(f)
                best_acc = float(prev.get("val_accuracy", -1.0))
                best_epoch = int(prev.get("best_epoch", 0))
                epochs_done = int(prev.get("epochs_done", 0))
            print(
                f"resumed {PT_PATH}  prev_best={best_acc * 100:.2f}% "
                f"(epoch {best_epoch})  epochs_done={epochs_done}"
            )
        else:
            print("existing small-head weights do not match, training head from scratch")
    model.freeze_conv()
    opt = torch.optim.Adam(model.classifier.parameters(), lr=lr)
    counts = np.bincount(train_set.labels, minlength=len(chars)).astype(np.float64)
    counts = np.maximum(counts, 1.0)
    class_w = 1.0 / np.sqrt(counts)
    class_w *= len(class_w) / class_w.sum()
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(class_w, dtype=torch.float32, device=device))
    best_cls = {k: v.detach().cpu().clone() for k, v in model.classifier.state_dict().items()}
    n_train_batches = max(1, len(train_loader))

    for epoch in range(1, epochs + 1):
        model.features.eval()
        model.classifier.train()
        running, seen = 0.0, 0
        for step, (xb, yb) in enumerate(train_loader, 1):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
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
        pack = {
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "chars": chars,
            "fc_size": FC_SIZE,
            "frozen_conv": True,
            "nhwc_flatten": True,
        }
        torch.save(pack, LAST_PATH)
        saved = "  kept-best"
        if acc > best_acc:
            best_acc = acc
            best_epoch = global_epoch
            best_cls = {k: v.detach().cpu().clone() for k, v in model.classifier.state_dict().items()}
            torch.save(pack, PT_PATH)
            saved = "  saved-best"
        print(
            f"epoch {global_epoch}  this={acc * 100:.2f}%  loss={running / max(1, seen):.4f}  "
            f"best={best_acc * 100:.2f}% (epoch {best_epoch}){saved}",
            flush=True,
        )
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "method": "official DeepAA conv + FC512 head",
                    "val_accuracy": float(best_acc),
                    "last_val_accuracy": float(acc),
                    "best_epoch": int(best_epoch),
                    "last_epoch": int(global_epoch),
                    "epochs_done": int(global_epoch),
                    "n_classes": len(chars),
                    "n_train": int(n_train),
                    "n_val": int(n_val),
                    "fc_size": FC_SIZE,
                    "frozen_conv": True,
                },
                f,
                ensure_ascii=False,
                indent=2,
            )
    model.classifier.load_state_dict(best_cls)
    torch.save(
        {
            "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            "chars": chars,
            "fc_size": FC_SIZE,
            "frozen_conv": True,
            "nhwc_flatten": True,
        },
        PT_PATH,
    )
    print(f"best val acc {best_acc * 100:.2f}% (epoch {best_epoch})")
    print(f"weights: {PT_PATH}")


def load_smallhead(device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.isfile(PT_PATH):
        raise FileNotFoundError(
            "no small-head weights; run "
            "python src/deepaa_smallhead.py --train --epochs 2 --batch 64 --max-samples 20000"
        )
    try:
        pack = torch.load(PT_PATH, map_location="cpu", weights_only=False)
    except TypeError:
        pack = torch.load(PT_PATH, map_location="cpu")
    chars = pack["chars"]
    if isinstance(chars, str):
        chars = list(chars)
    model = DeepAASmallHead(len(chars), int(pack.get("fc_size", FC_SIZE)))
    model.load_state_dict(pack["state_dict"], strict=True)
    model.freeze_conv()
    model.to(device).eval()
    n = sum(p.numel() for p in model.parameters())
    print(f"official conv + FC{pack.get('fc_size', FC_SIZE)}  params={n/1e6:.2f}M  device={device}", flush=True)
    return model, chars, device, {}, 11


def parse_args():
    parser = argparse.ArgumentParser(description="Official conv + small FC head")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--max-samples", type=int, default=20000)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    if args.train:
        train_smallhead(epochs=args.epochs, batch=args.batch, max_samples=args.max_samples)
    else:
        print("usage: python src/deepaa_smallhead.py --train --epochs 2 --batch 64 --max-samples 20000")
