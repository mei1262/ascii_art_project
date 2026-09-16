"""
AniLines detail 抽线：赛璐璐动画 / 插画 -> 白底黑线。

basic 和 detail 是同一套 U-Net，只差输入通道（3 对 2）和预处理；
算力几乎一样，detail 线更全，所以只保留 detail。

图片文件夹或视频都可以：
  python src/anilines.py --input input
  python src/anilines.py --input video.mp4 --fps 8
再把 output/anilines 喂给 line_art_3 / line_art_4 的 --from-lines。
"""
import argparse
import os
import sys

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps

from line_art import save_line_gallery
from main import DEFAULT_INPUT, DEFAULT_OUTPUT, PROJECT_ROOT

WEIGHT_DIR = os.path.join(PROJECT_ROOT, "models", "anilines")
WEIGHT_PATH = os.path.join(WEIGHT_DIR, "detail.pth")
DEFAULT_OUT = os.path.join(DEFAULT_OUTPUT, "videos", "line")
HF_REPO = "gyrojeff/AniLines"
HF_FILE = "detail.pth"
VIDEO_EXTS = (".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v")
IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".bmp")


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if mid_channels is None:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size(2) - x1.size(2)
        diff_x = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2, diff_y // 2, diff_y - diff_y // 2])
        return self.conv(torch.cat([x2, x1], dim=1))


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class LineExtractor(nn.Module):
    def __init__(self, chan_in, chan_out, bilinear=False):
        super().__init__()
        self.inc = DoubleConv(chan_in, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)
        self.outc = OutConv(64, chan_out)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        return self.outc(x)


def ensure_weights():
    if os.path.isfile(WEIGHT_PATH) and os.path.getsize(WEIGHT_PATH) > 1_000_000:
        return WEIGHT_PATH
    os.makedirs(WEIGHT_DIR, exist_ok=True)
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError("需要 huggingface_hub：pip install huggingface_hub") from exc
    print(f"正在下载 AniLines detail.pth …")
    downloaded = hf_hub_download(HF_REPO, HF_FILE, local_dir=WEIGHT_DIR)
    if os.path.abspath(downloaded) != os.path.abspath(WEIGHT_PATH) and os.path.isfile(downloaded):
        import shutil

        shutil.copy2(downloaded, WEIGHT_PATH)
    if not os.path.isfile(WEIGHT_PATH):
        raise FileNotFoundError(f"未找到权重: {WEIGHT_PATH}")
    return WEIGHT_PATH


def load_model(device):
    ensure_weights()
    model = LineExtractor(2, 1, True)
    ckpt = torch.load(WEIGHT_PATH, map_location=device, weights_only=False)
    model.load_state_dict(ckpt)
    model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return model


def load_rgb(image_path):
    img = Image.open(image_path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[3])
        img = background
    else:
        img = img.convert("RGB")
    return np.array(ImageOps.autocontrast(img, cutoff=0.3), dtype=np.uint8)


def bgr_to_rgb(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return np.array(ImageOps.autocontrast(Image.fromarray(rgb), cutoff=0.3), dtype=np.uint8)


def preprocess(rgb, device):
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    sobelx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    sobel = cv2.magnitude(sobelx, sobely)
    sobel = 255 - cv2.normalize(sobel, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
    t_img = torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float() / 255.0
    t_sobel = torch.from_numpy(sobel).unsqueeze(0).unsqueeze(0).float() / 255.0
    return torch.cat([t_img, t_sobel], dim=1).to(device)


def sketch_to_binary(gray, method="otsu"):
    if method == "raw":
        return gray
    if method == "adaptive":
        return cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


@torch.inference_mode()
def extract_lines(model, rgb, binarize=-1.0):
    device = next(model.parameters()).device
    x_in = preprocess(rgb, device)
    _, _, height, width = x_in.shape
    pad_h = (8 - height % 8) % 8
    pad_w = (8 - width % 8) % 8
    if pad_h or pad_w:
        x_in = F.pad(x_in, (0, pad_w, 0, pad_h), mode="reflect")
    pred = model(x_in)[:, :, :height, :width]
    if binarize >= 0:
        pred = (pred > binarize).float()
    return np.clip(pred[0, 0].cpu().numpy() * 255.0 + 0.5, 0, 255).astype(np.uint8)


def _save_pair(output_folder, raw_dir, base, sketch, binary_img):
    out = os.path.join(output_folder, f"{base}_line.png")
    raw = os.path.join(raw_dir, f"{base}_sketch.png")
    cv2.imwrite(out, binary_img)
    cv2.imwrite(raw, sketch)
    return out


def resize_rgb(rgb, max_side):
    if not max_side:
        return rgb
    height, width = rgb.shape[:2]
    longest = max(height, width)
    if longest <= max_side:
        return rgb
    scale = max_side / float(longest)
    new_size = (max(8, int(round(width * scale))), max(8, int(round(height * scale))))
    return cv2.resize(rgb, new_size, interpolation=cv2.INTER_AREA)


def finalize_mp4(tmp_path, output_path):
    try:
        import subprocess

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
    print(f"线稿视频: {os.path.abspath(output_path)}")
    return output_path


def extract_video(
    model,
    video_path,
    output_folder,
    raw_dir,
    binary="otsu",
    binarize=-1.0,
    fps=0,
    max_side=0,
    save_frames=False,
    video_out=None,
):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"打不开视频: {video_path}")
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = 1
    if fps and fps > 0:
        step = max(1, int(round(src_fps / fps)))
    out_fps = src_fps / step
    stem = os.path.splitext(os.path.basename(video_path))[0]
    if not video_out:
        video_out = os.path.join(output_folder, f"{stem}_line.mp4")
    os.makedirs(output_folder, exist_ok=True)
    tmp_path = video_out + ".tmp.mp4"
    saved = []
    writer = None
    idx = 0
    kept = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if idx % step == 0:
            rgb = resize_rgb(bgr_to_rgb(frame), max_side)
            sketch = extract_lines(model, rgb, binarize=binarize)
            binary_img = sketch_to_binary(sketch, method=binary)
            if writer is None:
                writer = cv2.VideoWriter(
                    tmp_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    max(1.0, float(out_fps)),
                    (binary_img.shape[1], binary_img.shape[0]),
                    True,
                )
                if not writer.isOpened():
                    raise RuntimeError(f"无法写视频: {tmp_path}")
            writer.write(cv2.cvtColor(binary_img, cv2.COLOR_GRAY2BGR))
            if save_frames or kept < 8:
                base = f"{stem}_{kept:05d}"
                out = _save_pair(output_folder, raw_dir, base, sketch, binary_img)
                saved.append((f"{base}.png", out))
            kept += 1
            if kept == 1 or kept % 10 == 0:
                print(f"  video frame {idx + 1}/{total or '?'}  saved {kept}", flush=True)
        idx += 1
    cap.release()
    if writer is not None:
        writer.release()
        finalize_mp4(tmp_path, video_out)
    print(f"视频抽了 {kept} 帧  step={step}  fps={out_fps:.2f}")
    return saved


def batch_extract(
    input_path=DEFAULT_INPUT,
    output_folder=DEFAULT_OUT,
    binary="otsu",
    binarize=-1.0,
    fps=0,
    max_side=0,
    save_frames=False,
    video_out=None,
):
    if not os.path.exists(input_path):
        print(f"找不到输入: {input_path}")
        return []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"AniLines detail  设备: {device}")
    model = load_model(device)
    os.makedirs(output_folder, exist_ok=True)
    raw_dir = os.path.join(output_folder, "raw")
    os.makedirs(raw_dir, exist_ok=True)
    saved = []
    if os.path.isfile(input_path) and input_path.lower().endswith(VIDEO_EXTS):
        saved = extract_video(
            model,
            input_path,
            output_folder,
            raw_dir,
            binary,
            binarize,
            fps,
            max_side,
            save_frames,
            video_out,
        )
    elif os.path.isfile(input_path) and input_path.lower().endswith(IMAGE_EXTS):
        rgb = load_rgb(input_path)
        sketch = extract_lines(model, rgb, binarize=binarize)
        binary_img = sketch_to_binary(sketch, method=binary)
        base, _ = os.path.splitext(os.path.basename(input_path))
        out = _save_pair(output_folder, raw_dir, base, sketch, binary_img)
        saved.append((os.path.basename(input_path), out))
        print(f"[1/1] {os.path.basename(input_path)}")
    else:
        files = [f for f in os.listdir(input_path) if f.lower().endswith(IMAGE_EXTS)]
        videos = [f for f in os.listdir(input_path) if f.lower().endswith(VIDEO_EXTS)]
        print(f"AniLines：{len(files)} 张图  {len(videos)} 个视频")
        for idx, name in enumerate(files, 1):
            src = os.path.join(input_path, name)
            try:
                rgb = load_rgb(src)
                sketch = extract_lines(model, rgb, binarize=binarize)
                binary_img = sketch_to_binary(sketch, method=binary)
                base, _ = os.path.splitext(name)
                out = _save_pair(output_folder, raw_dir, base, sketch, binary_img)
                saved.append((name, out))
                print(f"[{idx}/{len(files)}] {name}  {rgb.shape[1]}x{rgb.shape[0]}")
            except Exception as exc:
                print(f"[{idx}/{len(files)}] 失败 {name}: {exc}")
        for name in videos:
            src = os.path.join(input_path, name)
            try:
                saved.extend(
                    extract_video(
                        model,
                        src,
                        output_folder,
                        raw_dir,
                        binary,
                        binarize,
                        fps,
                        max_side,
                        save_frames,
                    )
                )
            except Exception as exc:
                print(f"失败 {name}: {exc}")
    if saved:
        gallery = os.path.join(DEFAULT_OUTPUT, "all_anilines.html")
        preview = saved[:80]
        save_line_gallery(preview, gallery, title="AniLines detail 线稿")
        print(f"线稿目录: {os.path.abspath(output_folder)}")
        print(f"线稿总览: {os.path.abspath(gallery)}")
    return saved


def parse_args():
    parser = argparse.ArgumentParser(description="AniLines detail 抽线")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument("--binary", default="otsu", choices=["otsu", "adaptive", "raw"])
    parser.add_argument("--binarize", type=float, default=-1.0, help="网络侧阈值，-1 关闭")
    parser.add_argument("--fps", type=float, default=0, help="视频抽帧帧率，0=逐帧")
    parser.add_argument("--max-side", type=int, default=0, help="最长边像素，0=原分辨率")
    parser.add_argument("--save-frames", action="store_true", help="同时保存逐帧 PNG")
    parser.add_argument("--video-out", default="", help="线稿视频路径，默认 output/anilines/<name>_line.mp4")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    batch_extract(
        args.input,
        args.output,
        args.binary,
        args.binarize,
        args.fps,
        args.max_side,
        args.save_frames,
        args.video_out or None,
    )
