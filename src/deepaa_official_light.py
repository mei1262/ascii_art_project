"""
Run the official DeepAA light Keras weights (no TensorFlow).

Architecture from model_light.json:
  64x64x1 -> Conv16-P-Conv32-P-Conv64-P-Conv128-P -> Flatten 2048 -> Softmax 411

Decode matches official output.py: 18px row stride, variable character width.
"""
import argparse
import json
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

from main import DEFAULT_OUTPUT, PROJECT_ROOT, save_batch_html

WIN = 64
CELL_H = 18
OFFICIAL_DIR = os.path.join(PROJECT_ROOT, "models", "deepaa_official")
WEIGHT_PATH = os.path.join(OFFICIAL_DIR, "weight_light.hdf5")
CHAR_LIST_PATH = os.path.join(OFFICIAL_DIR, "char_list.csv")
CHAR_DICT_PATH = os.path.join(OFFICIAL_DIR, "char_dict.pkl")
LOCAL_411 = os.path.join(PROJECT_ROOT, "data", "deepaa_411.json")
DEFAULT_OUT = os.path.join(DEFAULT_OUTPUT, "deepaa_official_light")


class DeepAAOfficialLight(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.BatchNorm2d(16, eps=0.001, momentum=0.99),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.99),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64, eps=0.001, momentum=0.99),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128, eps=0.001, momentum=0.99),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(2048, 411)

    def forward(self, x):
        return self.fc(torch.flatten(self.features(x), 1))


def _keras_conv(layer, kernel, bias):
    layer.weight.data.copy_(torch.from_numpy(np.transpose(kernel, (3, 2, 0, 1))))
    layer.bias.data.copy_(torch.from_numpy(bias))


def _keras_bn(layer, gamma, beta, mean, var):
    layer.weight.data.copy_(torch.from_numpy(gamma))
    layer.bias.data.copy_(torch.from_numpy(beta))
    layer.running_mean.copy_(torch.from_numpy(mean))
    layer.running_var.copy_(torch.from_numpy(var))


def load_official_light(device):
    model = DeepAAOfficialLight()
    with h5py.File(WEIGHT_PATH, "r") as f:
        w = f["model_weights"]

        def arr(*parts):
            node = w
            for part in parts:
                node = node[part]
            return np.array(node)

        convs = [
            (model.features[0], "conv2d_1"),
            (model.features[4], "conv2d_2"),
            (model.features[8], "conv2d_3"),
            (model.features[12], "conv2d_4"),
        ]
        bns = [
            (model.features[1], "batch_normalization_1"),
            (model.features[5], "batch_normalization_2"),
            (model.features[9], "batch_normalization_3"),
            (model.features[13], "batch_normalization_4"),
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
        model.fc.weight.data.copy_(
            torch.from_numpy(arr("predictions", "predictions", "kernel:0").T)
        )
        model.fc.bias.data.copy_(torch.from_numpy(arr("predictions", "predictions", "bias:0")))
    return model.to(device).eval()


def load_charset():
    df = pd.read_csv(CHAR_LIST_PATH, encoding="cp932")
    chars = df.loc[df["frequency"] >= 10, "char"].tolist()
    if len(chars) != 411:
        raise ValueError(f"expected 411 official classes, got {len(chars)}")
    space = next(i for i, ch in enumerate(chars) if ch == " ")
    with open(CHAR_DICT_PATH, "rb") as f:
        char_dict = pickle.load(f, encoding="latin1")
    return chars, space, char_dict


def compare_local_411(official_chars):
    if not os.path.isfile(LOCAL_411):
        return "no local deepaa_411.json"
    with open(LOCAL_411, encoding="utf-8") as f:
        local = json.load(f).get("chars") or []
    same_order = local == official_chars
    same_set = set(local) == set(official_chars)
    return (
        f"local 411 vs official light: same_set={same_set} "
        f"same_order={same_order} local={len(local)}"
    )


def load_gray_official(path, new_width=0):
    img = Image.open(path)
    orig_w, orig_h = img.size
    if new_width <= 0:
        new_width = orig_w
    new_height = int(orig_h * new_width / orig_w)
    img = img.resize((new_width, new_height), Image.LANCZOS)
    arr = np.array(img)
    if arr.ndim == 3:
        arr = arr[:, :, 0]
    return arr.astype(np.uint8), new_width, new_height


def pad_official(gray):
    mergin = (WIN - CELL_H) // 2
    h, w = gray.shape
    canvas = np.ones((h + 2 * mergin + CELL_H, w + 2 * mergin + CELL_H), dtype=np.uint8) * 255
    canvas[mergin : mergin + h, mergin : mergin + w] = gray
    return canvas.astype(np.float32) / 255.0


@torch.no_grad()
def convert_image(model, device, gray, chars, space, char_dict, slide=0):
    img = pad_official(gray)
    img = np.concatenate([np.ones((1 + slide, img.shape[1]), dtype=np.float32), img], axis=0)
    num_line = (img.shape[0] - WIN) // CELL_H
    img_width = img.shape[1]
    predicts = []
    text = []
    for h in range(num_line):
        w = 0
        penalty = 1
        predict_line = []
        text_line = ""
        while w <= img_width - WIN:
            patch = img[h * CELL_H : h * CELL_H + WIN, w : w + WIN]
            x = torch.from_numpy(patch[None, None].astype(np.float32)).to(device)
            logits = model(x)[0]
            if penalty:
                logits = logits.clone()
                logits[space] = -1e9
            idx = int(logits.argmax().item())
            penalty = int(idx == space)
            char = chars[idx]
            predict_line.append(char)
            w += int(char_dict[char].shape[1])
            text_line += char
        predicts.append(predict_line)
        text.append(text_line)
    canvas = np.ones_like(img, dtype=np.uint8) * 255
    for h, line in enumerate(predicts):
        w = 0
        for char in line:
            glyph = char_dict[char]
            char_width = int(glyph.shape[1])
            canvas[h * CELL_H : h * CELL_H + 16, w : w + char_width] = (
                255 - glyph.astype(np.uint8) * 255
            )
            w += char_width
    out_h, out_w = gray.shape
    cropped = Image.fromarray(canvas).crop((0, slide, out_w, out_h + slide))
    return "\n".join(text), cropped


def iter_images(folder):
    for name in sorted(os.listdir(folder)):
        if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".webp")):
            yield os.path.join(folder, name), name


def run_folder(model, device, chars, space, char_dict, inp, out_dir, new_width, slide):
    os.makedirs(out_dir, exist_ok=True)
    arts = {}
    for path, name in iter_images(inp):
        print(f"official-light {name}", flush=True)
        gray, used_w, used_h = load_gray_official(path, new_width)
        text, png = convert_image(model, device, gray, chars, space, char_dict, slide=slide)
        stem = os.path.splitext(name)[0]
        png_path = os.path.join(out_dir, f"{stem}_slide{slide}_w{used_w}.png")
        txt_path = os.path.join(out_dir, f"{stem}_slide{slide}_w{used_w}.txt")
        png.save(png_path)
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        arts[name] = text
        print(f"  {used_w}x{used_h} -> {png_path}", flush=True)
    return arts


def parse_args():
    parser = argparse.ArgumentParser(description="Official DeepAA light inference")
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        help="Image folder. Repeat for multiple folders.",
    )
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--width", type=int, default=0, help="0 = keep original width")
    parser.add_argument("--slide", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    folders = args.input or [
        os.path.join(PROJECT_ROOT, "input", "deepaa_original"),
        os.path.join(PROJECT_ROOT, "input", "eva-line"),
    ]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    chars, space, char_dict = load_charset()
    print(compare_local_411(chars), flush=True)
    model = load_official_light(device)
    print("loaded official light 16-32-64-128 + FC411", flush=True)
    all_arts = {}
    for folder in folders:
        tag = os.path.basename(os.path.normpath(folder))
        out_dir = os.path.join(args.output, tag)
        arts = run_folder(
            model, device, chars, space, char_dict, folder, out_dir, args.width, args.slide
        )
        all_arts.update({f"{tag}/{k}": v for k, v in arts.items()})
        html_path = os.path.join(DEFAULT_OUTPUT, f"all_deepaa_official_light_{tag}.html")
        save_batch_html(arts, html_path, font_family='"MS Gothic", "MS Mincho", monospace')
        print(f"html {html_path}", flush=True)
    if len(folders) > 1:
        html_path = os.path.join(DEFAULT_OUTPUT, "all_deepaa_official_light.html")
        save_batch_html(all_arts, html_path, font_family='"MS Gothic", "MS Mincho", monospace')
        print(f"html {html_path}", flush=True)


if __name__ == "__main__":
    main()
