"""
处理已有勾边图：修骨架，不负责从照片抽线。

去掉短刺、结上的小环，让一格里更接近一根线。
"""
import argparse
import os
import sys

import cv2
import numpy as np

from line_art import binarize_ink, remove_short, restroke, save_line_gallery, thin_ink
from main import DEFAULT_OUTPUT

DEFAULT_INPUT = os.path.join(DEFAULT_OUTPUT, "line_art")
DEFAULT_CLEAN = os.path.join(DEFAULT_OUTPUT, "line_art_clean")
_N8 = ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1))


def _degree(binary):
    kernel = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
    return cv2.filter2D((binary > 0).astype(np.uint8), -1, kernel, borderType=cv2.BORDER_CONSTANT)


def fill_beads(ink, max_area=40):
    """骨架交叉处常留下空心小环，填掉再细化。"""
    holes = np.where(ink > 0, 0, 255).astype(np.uint8)
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(holes, connectivity=8)
    if nlab <= 2:
        return ink
    bg = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    out = ink.copy()
    for i in range(1, nlab):
        if i != bg and int(stats[i, cv2.CC_STAT_AREA]) <= max_area:
            out[labels == i] = 255
    return out


def prune_spurs(ink, max_len=10):
    """从端点走到交叉口，短枝删掉；长发梢保留。"""
    remain = ink > 0
    h, w = remain.shape

    def neighbors(y, x):
        found = []
        for dy, dx in _N8:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and remain[ny, nx]:
                found.append((ny, nx))
        return found

    deg = _degree(remain)
    ys, xs = np.where(remain & (deg == 1))
    delete = np.zeros_like(remain)
    for y, x in zip(ys.tolist(), xs.tolist()):
        if delete[y, x] or not remain[y, x]:
            continue
        path = [(y, x)]
        seen = {(y, x)}
        cy, cx = y, x
        hit_junction = False
        for _ in range(max_len + 1):
            cand = [p for p in neighbors(cy, cx) if p not in seen]
            if len(cand) != 1:
                hit_junction = len(cand) >= 2
                break
            cy, cx = cand[0]
            seen.add((cy, cx))
            path.append((cy, cx))
            if len(neighbors(cy, cx)) >= 3:
                hit_junction = True
                break
        else:
            continue
        if len(path) > max_len:
            continue
        cut = path[:-1] if hit_junction else path
        for py, px in cut:
            delete[py, px] = True
    return (remain & ~delete).astype(np.uint8) * 255


def clean_line_image(line_wb, stroke_width=1):
    """白底黑线 -> 去短刺和小环后的白底黑线。"""
    ink = binarize_ink(line_wb)
    if cv2.countNonZero(ink) == 0:
        return np.full_like(line_wb, 255)
    ink = fill_beads(ink)
    close_k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, close_k)
    if cv2.countNonZero(ink):
        ink = thin_ink(ink)
    ink = prune_spurs(ink)
    ink = fill_beads(ink)
    if cv2.countNonZero(ink):
        ink = thin_ink(ink)
    ink = remove_short(ink)
    ink = restroke(ink, stroke_width)
    return 255 - ink


def batch_clean(
    input_folder=DEFAULT_INPUT,
    output_folder=DEFAULT_CLEAN,
    stroke_width=1,
):
    if not os.path.exists(input_folder):
        print(f"找不到线稿文件夹: {input_folder}")
        return []
    os.makedirs(output_folder, exist_ok=True)
    files = [f for f in os.listdir(input_folder) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    saved = []
    print(f"线稿清理：{len(files)} 张")
    for idx, name in enumerate(files, 1):
        src = os.path.join(input_folder, name)
        try:
            gray = cv2.imread(src, cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise ValueError("读图失败")
            cleaned = clean_line_image(gray, stroke_width=stroke_width)
            out = os.path.join(output_folder, name)
            cv2.imwrite(out, cleaned)
            saved.append((name, out))
            print(f"[{idx}/{len(files)}] {name}")
        except Exception as exc:
            print(f"[{idx}/{len(files)}] 失败 {name}: {exc}")
    if saved:
        gallery = os.path.join(DEFAULT_OUTPUT, "all_line_art_clean.html")
        save_line_gallery(saved, gallery, title="清理后线稿预览")
        print(f"清理总览: {os.path.abspath(gallery)}")
    return saved


def parse_args():
    parser = argparse.ArgumentParser(description="清理线稿骨架：去短刺、去结上小环")
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_CLEAN)
    parser.add_argument("--stroke-width", type=int, default=1)
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    batch_clean(args.input, args.output, args.stroke_width)
