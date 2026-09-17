"""
Official DeepAA full net (Keras weight.hdf5 -> PyTorch).

Paper architecture: C64-C64-P-C128-C128-P-C256-C256-C256-P-FC4096-FC4096-411.
Keras is channels-last, so FC sees NHWC-flattened features.
"""
import os
import pickle
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

from main import PROJECT_ROOT

OFFICIAL_DIR = os.path.join(PROJECT_ROOT, "models", "deepaa_official")
HDF5_CANDIDATES = [
    os.path.join(PROJECT_ROOT, "data", "weight.hdf5"),
    os.path.join(OFFICIAL_DIR, "weight.hdf5"),
]
CHAR_LIST_PATH = os.path.join(OFFICIAL_DIR, "char_list.csv")
CHAR_DICT_PATH = os.path.join(OFFICIAL_DIR, "char_dict.pkl")
PT_PATH = os.path.join(OFFICIAL_DIR, "weight_full.pt")
FC_SIZE = 4096
NUM_CLASSES = 411
WIN = 64
CELL_H = 18
GLYPH_H = 16


class DeepAAOfficialFull(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, fc_size=FC_SIZE):
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
            nn.Linear(256 * 8 * 8, fc_size),
            nn.BatchNorm1d(fc_size, eps=0.001, momentum=0.99),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_size, fc_size),
            nn.BatchNorm1d(fc_size, eps=0.001, momentum=0.99),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(fc_size, num_classes),
        )

    @staticmethod
    def _conv(cin, cout):
        return nn.Sequential(
            nn.Conv2d(cin, cout, 3, padding=1),
            nn.BatchNorm2d(cout, eps=0.001, momentum=0.99),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.permute(0, 2, 3, 1).contiguous().flatten(1)
        return self.classifier(x)


def official_chars():
    df = pd.read_csv(CHAR_LIST_PATH, encoding="cp932")
    chars = df.loc[df["frequency"] >= 10, "char"].tolist()
    if len(chars) != NUM_CLASSES:
        raise ValueError(f"expected {NUM_CLASSES} official classes, got {len(chars)}")
    return chars


def _find_hdf5():
    for path in HDF5_CANDIDATES:
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(
        "official weight.hdf5 not found; put it in data/ or models/deepaa_official/"
    )


def _keras_conv(layer, kernel, bias):
    layer.weight.data.copy_(torch.from_numpy(np.transpose(kernel, (3, 2, 0, 1))))
    layer.bias.data.copy_(torch.from_numpy(bias))


def _keras_bn(layer, gamma, beta, mean, var):
    layer.weight.data.copy_(torch.from_numpy(gamma))
    layer.bias.data.copy_(torch.from_numpy(beta))
    layer.running_mean.copy_(torch.from_numpy(mean))
    layer.running_var.copy_(torch.from_numpy(var))


def convert_hdf5(hdf5_path=None, out_path=PT_PATH):
    hdf5_path = hdf5_path or _find_hdf5()
    print(f"converting {hdf5_path}", flush=True)
    model = DeepAAOfficialFull()
    with h5py.File(hdf5_path, "r") as f:
        w = f["model_weights"]

        def arr(*parts):
            node = w
            for part in parts:
                node = node[part]
            return np.array(node)

        convs = [
            (model.features[0][0], "conv2d_1"),
            (model.features[1][0], "conv2d_2"),
            (model.features[3][0], "conv2d_3"),
            (model.features[4][0], "conv2d_4"),
            (model.features[6][0], "conv2d_5"),
            (model.features[7][0], "conv2d_6"),
            (model.features[8][0], "conv2d_7"),
        ]
        bns = [
            (model.features[0][1], "batch_normalization_1"),
            (model.features[1][1], "batch_normalization_2"),
            (model.features[3][1], "batch_normalization_3"),
            (model.features[4][1], "batch_normalization_4"),
            (model.features[6][1], "batch_normalization_5"),
            (model.features[7][1], "batch_normalization_6"),
            (model.features[8][1], "batch_normalization_7"),
        ]
        for layer, name in convs:
            _keras_conv(layer, arr(name, name, "kernel:0"), arr(name, name, "bias:0"))
        for layer, name in bns:
            _keras_bn(
                layer,
                arr(name, name, "gamma:0"),
                arr(name, name, "beta:0"),
                arr(name, name, "moving_mean:0"),
                arr(name, name, "moving_variance:0"),
            )
        model.classifier[0].weight.data.copy_(
            torch.from_numpy(arr("dense_1", "dense_1", "kernel:0").T)
        )
        model.classifier[0].bias.data.copy_(torch.from_numpy(arr("dense_1", "dense_1", "bias:0")))
        _keras_bn(
            model.classifier[1],
            arr("batch_normalization_8", "batch_normalization_8", "gamma:0"),
            arr("batch_normalization_8", "batch_normalization_8", "beta:0"),
            arr("batch_normalization_8", "batch_normalization_8", "moving_mean:0"),
            arr("batch_normalization_8", "batch_normalization_8", "moving_variance:0"),
        )
        model.classifier[4].weight.data.copy_(
            torch.from_numpy(arr("dense_2", "dense_2", "kernel:0").T)
        )
        model.classifier[4].bias.data.copy_(torch.from_numpy(arr("dense_2", "dense_2", "bias:0")))
        _keras_bn(
            model.classifier[5],
            arr("batch_normalization_9", "batch_normalization_9", "gamma:0"),
            arr("batch_normalization_9", "batch_normalization_9", "beta:0"),
            arr("batch_normalization_9", "batch_normalization_9", "moving_mean:0"),
            arr("batch_normalization_9", "batch_normalization_9", "moving_variance:0"),
        )
        model.classifier[8].weight.data.copy_(
            torch.from_numpy(arr("predictions", "predictions", "kernel:0").T)
        )
        model.classifier[8].bias.data.copy_(
            torch.from_numpy(arr("predictions", "predictions", "bias:0"))
        )
    chars = official_chars()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    pack = {
        "state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "chars": chars,
        "fc_size": FC_SIZE,
        "nhwc_flatten": True,
        "source": os.path.abspath(hdf5_path),
    }
    torch.save(pack, out_path)
    print(f"wrote {out_path}", flush=True)
    return pack


def load_official_full(device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if os.path.isfile(PT_PATH):
        try:
            pack = torch.load(PT_PATH, map_location="cpu", weights_only=False)
        except TypeError:
            pack = torch.load(PT_PATH, map_location="cpu")
    else:
        pack = convert_hdf5()
    chars = pack["chars"]
    if isinstance(chars, str):
        chars = list(chars)
    model = DeepAAOfficialFull(len(chars), fc_size=int(pack.get("fc_size", FC_SIZE)))
    model.load_state_dict(pack["state_dict"])
    model.to(device).eval()
    print(f"official DeepAA full  FC={pack.get('fc_size', FC_SIZE)}  device={device}", flush=True)
    return model, chars, device, {}, 11


def load_char_dict():
    with open(CHAR_DICT_PATH, "rb") as f:
        return pickle.load(f, encoding="latin1")


def space_index(chars):
    for i, ch in enumerate(chars):
        if ch == " ":
            return i
    raise ValueError("official charset has no ASCII space")


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
        arr = np.array(
            Image.fromarray(arr).resize((new_width, new_height), Image.LANCZOS)
        )
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
    """Fraction of 64x64 pixels darker than `dark`. White paper is ~1.0."""
    if patch.size == 0:
        return 0.0
    return float((patch < dark).mean())


@torch.no_grad()
def _decode_row(model, device, row, chars, space, widths, ink_skip=0.0, skip_stats=None, ideo=None):
    """Official output.py: one window, one char, then jump by that char's width."""
    row_t = torch.from_numpy(np.ascontiguousarray(row))[None, None].to(device)
    width = row.shape[1]
    w = 0
    penalty = True
    line = []
    max_w = width - WIN
    if ideo is None:
        ideo = space
    while w <= max_w:
        patch = row[:, w : w + WIN]
        ink = window_ink_ratio(patch) if ink_skip > 0 else 1.0
        if ink_skip > 0 and ink < ink_skip:
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
def _decode_rows_batched(
    model,
    device,
    rows_np,
    chars,
    space,
    widths,
    ink_skip=0.0,
    skip_stats=None,
    ideo=None,
):
    """Same greedy decode as _decode_row, one batched forward per step across rows.

    Rows are independent. Every unfinished row contributes its current 64x64 window;
    the net runs once; each row then jumps by its own glyph width.
    """
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
        pen = torch.tensor(
            [penalty[h] for h in infer_h], device=logits.device, dtype=torch.bool
        )
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
def decode_official(
    model,
    device,
    gray,
    chars,
    char_dict,
    slide=0,
    chunk=128,
    ink_skip=0.0,
    skip_stats=None,
    row_batch=False,
):
    """
    Official output.py decode: 18px rows, variable char width, no fold.
    Space class is banned when the previous char was not a space.
    Only one vertical slide (default 0). chunk is unused; decode is one window per char.
    ink_skip: skip the net when 64x64 ink fraction is below this. 0 disables (model 9).
    row_batch: batch current windows from all rows in one forward (model 11). Same greedy
    rule per row as the sequential decoder.
    """
    space = space_index(chars)
    ideo = _ideo_space_index(chars, space)
    widths = char_widths(chars, char_dict)
    img = pad_official(gray)
    img = np.concatenate(
        [np.ones((1 + slide, img.shape[1]), dtype=np.float32), img], axis=0
    )
    num_line = (img.shape[0] - WIN) // CELL_H
    if row_batch and num_line > 0:
        rows_np = np.stack(
            [img[h * CELL_H : h * CELL_H + WIN] for h in range(num_line)]
        )
        predicts = _decode_rows_batched(
            model,
            device,
            rows_np,
            chars,
            space,
            widths,
            ink_skip=ink_skip,
            skip_stats=skip_stats,
            ideo=ideo,
        )
    else:
        predicts = []
        for h in range(num_line):
            row = img[h * CELL_H : h * CELL_H + WIN]
            predicts.append(
                _decode_row(
                    model,
                    device,
                    row,
                    chars,
                    space,
                    widths,
                    ink_skip=ink_skip,
                    skip_stats=skip_stats,
                    ideo=ideo,
                )
            )
    text = "\n".join("".join(line) for line in predicts)
    png = render_official(predicts, img.shape, char_dict, slide, gray.shape[1], gray.shape[0])
    return text, png, num_line


if __name__ == "__main__":
    convert_hdf5()
