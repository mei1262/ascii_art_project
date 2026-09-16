"""把照片收成纯线条勾边图。结构型 ASCII（论文 / RF / 卷积核）都应先走这一步。"""
import argparse
import os
import sys

import cv2
import numpy as np

from PIL import Image, ImageOps

from main import DEFAULT_INPUT, DEFAULT_OUTPUT


def xdog(gray_u8, sigma=1.0, k=1.6, p=20.0, eps=0.02, phi=200.0):
    """
    XDoG：适合把照片收成插画式线稿。
    返回白底黑线 uint8。
    """
    gray = gray_u8.astype(np.float32) / 255.0
    g1 = cv2.GaussianBlur(gray, (0, 0), sigma)
    g2 = cv2.GaussianBlur(gray, (0, 0), sigma * k)
    dog = g1 - g2
    u = (1.0 + p) * g1 - p * g2
    u = np.where(u >= eps, 1.0, 1.0 + np.tanh(phi * (u - eps)))
    u = np.clip(u, 0.0, 1.0)
    return (u * 255.0).astype(np.uint8)


def canny_lines(gray_u8):
    """论文里用的自适应 Canny。返回白底黑线。"""
    blur = cv2.GaussianBlur(gray_u8, (5, 5), 0)
    median = float(np.median(blur))
    lo = int(max(0, 0.66 * median))
    hi = int(min(255, 1.33 * median))
    if hi <= lo:
        lo, hi = 50, 150
    edges = cv2.Canny(blur, lo, hi)
    return 255 - edges


def sketch_lines(gray_u8):
    """双边滤波 + 自适应阈值，接近铅笔勾边。"""
    smooth = cv2.bilateralFilter(gray_u8, 9, 75, 75)
    inv = 255 - smooth
    return cv2.adaptiveThreshold(
        inv, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 7
    )


def load_gray_for_lines(image_path):
    """线稿用灰度：只铺透明底，不做 2 倍对比度（那会把阴影收成杂线）。"""
    img = Image.open(image_path)
    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        background = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        background.paste(img, mask=img.split()[3])
        img = background
    else:
        img = img.convert("RGB")
    img = ImageOps.autocontrast(img, cutoff=0.5)
    return img.convert("L")


def structure_lines(gray_u8):
    """
    只要主轮廓，不要阴影网纹。
    保边平滑之后抓「比周围更暗」的笔划，再骨架化；
    这样粗睫毛/头发是一根中线，而不是 Canny 的双边。
    """
    bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)
    try:
        flat = cv2.edgePreservingFilter(bgr, flags=cv2.RECURS_FILTER, sigma_s=70, sigma_r=0.35)
        flat = cv2.cvtColor(flat, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        flat = cv2.bilateralFilter(gray_u8, 9, 60, 60)
        flat = cv2.bilateralFilter(flat, 9, 60, 60)
    surround = cv2.GaussianBlur(flat, (0, 0), 2.2)
    darker = surround.astype(np.int16) - flat.astype(np.int16)
    ink = np.where(darker > 14, 255, 0).astype(np.uint8)
    ink = np.maximum(ink, np.where(flat < 42, 255, 0).astype(np.uint8))
    dt = cv2.distanceTransform((ink > 0).astype(np.uint8), cv2.DIST_L2, 3)
    fills = ((ink > 0) & (dt > 3.5)).astype(np.uint8) * 255
    strokes = ink.copy()
    strokes[fills > 0] = 0
    if cv2.countNonZero(fills):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fills = cv2.morphologyEx(fills, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(fills, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(strokes, contours, -1, 255, 1)
    return 255 - strokes


def already_line_art(gray_u8):
    """浅底勾边图：Otsu 收成干净黑线。"""
    _, binary = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary


def binarize_ink(line_wb):
    """白底黑线（可含灰）-> 墨水=255 的二值图。"""
    unique = np.unique(line_wb)
    if unique.size <= 3:
        return np.where(line_wb < 128, 255, 0).astype(np.uint8)
    dark = 255 - line_wb
    _, ink = cv2.threshold(dark, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return ink


def thin_ink(ink):
    """Zhang-Suen 细化成 1 像素骨架。ink=255 为笔划。"""
    img = np.pad(ink > 0, 1, mode="constant")
    for _ in range(512):
        changed = False
        for step in (0, 1):
            p2 = img[:-2, 1:-1]
            p3 = img[:-2, 2:]
            p4 = img[1:-1, 2:]
            p5 = img[2:, 2:]
            p6 = img[2:, 1:-1]
            p7 = img[2:, :-2]
            p8 = img[1:-1, :-2]
            p9 = img[:-2, :-2]
            center = img[1:-1, 1:-1]
            n_count = (
                p2.astype(np.uint8) + p3 + p4 + p5 + p6 + p7 + p8 + p9
            )
            trans = (
                (~p2 & p3).astype(np.uint8)
                + (~p3 & p4)
                + (~p4 & p5)
                + (~p5 & p6)
                + (~p6 & p7)
                + (~p7 & p8)
                + (~p8 & p9)
                + (~p9 & p2)
            )
            mark = center & (n_count >= 2) & (n_count <= 6) & (trans == 1)
            if step == 0:
                mark &= ~(p2 & p4 & p6)
                mark &= ~(p4 & p6 & p8)
            else:
                mark &= ~(p2 & p4 & p8)
                mark &= ~(p2 & p6 & p8)
            if mark.any():
                img[1:-1, 1:-1] = center & ~mark
                changed = True
        if not changed:
            break
    return img[1:-1, 1:-1].astype(np.uint8) * 255


def restroke(ink, width):
    if width <= 1:
        return ink
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, width))
    return cv2.dilate(ink, kernel)


def remove_short(ink):
    """去掉短碎线、噪点。"""
    h, w = ink.shape
    min_area = max(28, int(h * w / 90000))
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(
        (ink > 0).astype(np.uint8), connectivity=8
    )
    keep = np.zeros(nlab, dtype=bool)
    keep[0] = False
    for i in range(1, nlab):
        keep[i] = int(stats[i, cv2.CC_STAT_AREA]) >= min_area
    return keep[labels].astype(np.uint8) * 255


def suppress_texture(ink):
    """局部过密的网纹/阴影当成干扰，只留外轮廓。"""
    h, w = ink.shape
    block = max(15, (min(h, w) // 45) | 1)
    density = cv2.blur((ink > 0).astype(np.float32), (block, block))
    dense = density > 0.16
    if float(dense.mean()) < 0.008:
        return ink
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    region = cv2.dilate(dense.astype(np.uint8), kernel)
    mesh = np.where(region > 0, ink, 0).astype(np.uint8)
    closed = cv2.morphologyEx(mesh, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    out = ink.copy()
    out[region > 0] = 0
    cv2.drawContours(out, contours, -1, 255, 1)
    return out


def uniform_ink(ink, stroke_width=1):
    ink = remove_short(ink)
    ink = suppress_texture(ink)
    if cv2.countNonZero(ink):
        ink = thin_ink(ink)
    ink = remove_short(ink)
    return restroke(ink, stroke_width)


def normalize_line_art(line_wb, stroke_width=1):
    """
    粗细不一的勾边 -> 等宽细线（白底黑线）。
    笔划骨架化，色块改描外轮廓，再按固定宽度描回。
    """
    ink = binarize_ink(line_wb)
    if cv2.countNonZero(ink) == 0:
        return np.full_like(line_wb, 255)
    return 255 - uniform_ink(ink, stroke_width)


def resize_line_art(line_wb, width, height, stroke_width=1):
    """缩放到字符网格时保持线条，并再次收成等宽。"""
    ink = binarize_ink(line_wb)
    small = cv2.resize(ink, (width, height), interpolation=cv2.INTER_AREA)
    small = np.where(small > 20, 255, 0).astype(np.uint8)
    if cv2.countNonZero(small):
        small = uniform_ink(small, stroke_width)
    return 255 - small


def gray_from_bgr(frame_bgr):
    """视频帧：BGR -> 和单张图相同的线稿灰度。"""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(rgb)
    img = ImageOps.autocontrast(img, cutoff=0.5)
    return np.array(img.convert("L"), dtype=np.uint8)


def frame_bgr_to_line(frame_bgr, method="auto", stroke_width=1, max_side=0):
    """
    视频帧走 all_line_art 同一套：灰度 + photo_to_line。
    max_side>0 时先缩小长边，加快逐帧细化。
    """
    gray = gray_from_bgr(frame_bgr)
    if max_side and max(gray.shape) > max_side:
        height, width = gray.shape
        scale = max_side / float(max(height, width))
        gray = cv2.resize(
            gray,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return photo_to_line(gray, method=method, stroke_width=stroke_width)


def photo_to_line(gray_u8, method="auto", stroke_width=1):
    """
    照片/线稿 -> 白底黑线。
    method: auto / xdog / canny / sketch
    stroke_width: 0 表示不细化；>=1 细化后描成该像素宽。
    """
    if method == "auto":
        if float(gray_u8.mean()) > 200 and gray_u8.std() > 40:
            raw = already_line_art(gray_u8)
        else:
            raw = structure_lines(gray_u8)
    elif method == "structure":
        raw = structure_lines(gray_u8)
    elif method == "xdog":
        raw = xdog(gray_u8)
    elif method == "canny":
        raw = canny_lines(gray_u8)
    elif method == "sketch":
        raw = sketch_lines(gray_u8)
    else:
        raise ValueError(f"未知线稿方法: {method}")
    if stroke_width and stroke_width > 0:
        return normalize_line_art(raw, stroke_width=stroke_width)
    return 255 - binarize_ink(raw)


def to_white_on_black(line_white_bg):
    """训练字形和匹配都用白线黑底。"""
    return 255 - line_white_bg


def save_line_gallery(image_paths, output_html, title="线稿预览"):
    cards = []
    for rel, abs_path in image_paths:
        src = os.path.relpath(abs_path, os.path.dirname(output_html)).replace("\\", "/")
        cards.append(
            f'<figure><img src="{src}" alt="{rel}"><figcaption>{rel}</figcaption></figure>'
        )
    html = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ background:#111; color:#eee; font-family:system-ui,sans-serif; margin:0; padding:24px; }}
        h1 {{ font-size:20px; }}
        .grid {{ display:grid; grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); gap:16px; }}
        figure {{ margin:0; background:#000; padding:10px; border-radius:8px; }}
        img {{ width:100%; height:auto; background:#fff; }}
        figcaption {{ margin-top:8px; font-size:13px; word-break:break-all; }}
    </style>
</head>
<body>
    <h1>{title}（{len(image_paths)}）</h1>
    <div class="grid">{''.join(cards)}</div>
</body>
</html>"""
    with open(output_html, "w", encoding="utf-8") as f:
        f.write(html)


def batch_extract_lines(
    input_folder=DEFAULT_INPUT,
    output_folder=None,
    method="auto",
    stroke_width=1,
    from_lines=False,
    cols=80,
):
    if output_folder is None:
        output_folder = os.path.join(DEFAULT_OUTPUT, "line_art")
    if not os.path.exists(input_folder):
        print(f"找不到输入文件夹: {input_folder}")
        return []

    os.makedirs(output_folder, exist_ok=True)
    valid = (".jpg", ".jpeg", ".png")
    files = [f for f in os.listdir(input_folder) if f.lower().endswith(valid)]
    saved = []
    print(f"线稿提取：{len(files)} 张，方法 {method}，线宽 {stroke_width}  from_lines={from_lines}")
    ascii_fn = None
    if from_lines:
        from glyphs import save_ascii_png
        from line_art_2 import line_to_thin_ascii

        ascii_fn = (save_ascii_png, line_to_thin_ascii)
    for idx, name in enumerate(files, 1):
        src = os.path.join(input_folder, name)
        try:
            if from_lines:
                gray = np.array(Image.open(src).convert("L"), dtype=np.uint8)
                lines = gray
            else:
                gray = np.array(load_gray_for_lines(src), dtype=np.uint8)
                lines = photo_to_line(gray, method=method, stroke_width=stroke_width)
            base, _ = os.path.splitext(name)
            out = os.path.join(output_folder, f"{base}_line.png")
            cv2.imwrite(out, lines)
            saved.append((name, out))
            if ascii_fn:
                save_ascii_png, line_to_thin_ascii = ascii_fn
                save_ascii_png(
                    line_to_thin_ascii(lines, cols=cols),
                    os.path.join(output_folder, f"{base}_ascii.png"),
                    cell_w=16,
                    cell_h=16,
                )
            print(f"[{idx}/{len(files)}] {name}")
        except Exception as exc:
            print(f"[{idx}/{len(files)}] 失败 {name}: {exc}")

    if saved:
        gallery = os.path.join(DEFAULT_OUTPUT, "all_line_art.html")
        save_line_gallery(saved, gallery)
        print(f"线稿总览: {os.path.abspath(gallery)}")
    return saved


def parse_args():
    parser = argparse.ArgumentParser(description="照片转纯线条勾边图")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=os.path.join(DEFAULT_OUTPUT, "line_art"))
    parser.add_argument(
        "--method",
        default="auto",
        choices=["auto", "structure", "xdog", "canny", "sketch"],
        help="auto=浅底当线稿否则主轮廓；structure=保边平滑+Canny；xdog/sketch 更易出杂线",
    )
    parser.add_argument(
        "--stroke-width",
        type=int,
        default=1,
        help="细化后的统一线宽，像素；0 表示不细化",
    )
    parser.add_argument("--from-lines", action="store_true", help="输入已是线稿，不再抽线")
    parser.add_argument("--width", type=int, default=80)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    batch_extract_lines(
        args.input,
        args.output,
        args.method,
        args.stroke_width,
        args.from_lines,
        args.width,
    )
