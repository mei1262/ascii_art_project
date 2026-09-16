# ascii_art_project

线稿提取与 ASCII 转换。 / Line-art extraction and ASCII conversion.

## 引用 / Reference

Taizan Yonetsuji. *DeepAA*. 2017. <https://github.com/taizan/DeepAA>

Official DeepAA converts line drawings to proportional Japanese ASCII (411 classes, 64×64 windows, variable glyph width). This repo keeps that decode for models 9 and 10, and also includes a separate self-made converter (model 5).

## 模型对比 / Models

| | 模型 5 / Model 5 | 模型 9 / Model 9 | 模型 10 / Model 10 | 官方 DeepAA / Official DeepAA |
|---|---|---|---|---|
| 来源 Origin | **自研**，无网络权重 Self-made, 0 NN weights | 官方卷积冻住 + 自训 FC512 Frozen official conv + trained FC512 | 同 9，空白窗跳过 Same as 9, skip empty windows | 完整官方网 Full official net |
| 字符 Characters | 仅方向笔画 `/ \ - \| ~` 等 Direction strokes only | 官方 411 日文变宽字 Official 411, variable width | 同 9 Same as 9 | 官方 411 Official 411 |
| 排版 Layout | 固定 18×30 等宽格 Fixed grid | `char_dict` 变宽、行距 18px | 同 9 | 同 9（本仓库 slide=0） |
| 参数 Params | 0 | ~10.6M | ~10.6M | ~87.3M |
| 入口 Script | `src/line_art_demo.py` | `src/line_art_9.py` | `src/line_art_10.py` | `src/line_art_8_puredeepaa.py` |

Model 9/10 trade a small accuracy drop (subset-trained head) for a much smaller net than official FC4096 DeepAA. They do **not** change `--width` or frame rate. Model 10 only skips 64×64 windows whose ink fraction is below `--ink-skip` (default `0.0008`, quality first).

On the 8 EVA stills below, model 10 matched model 9 **character-for-character**, skipped **13.9%** of windows (range 0.8%–41.2% per frame), so DeepAA forwards are about **86%** of model 9. Wall-clock speedup is a bit less than 1.16× because the serial decode loop still walks every window; empty frames (e.g. f253) gain more, busy frames (e.g. f706) almost none.

## EVA 8 帧对照 / EVA 8-frame comparison

Source line art: AniLines detail from *明日香来日*. Model 5 uses a fixed glyph grid (different scale). Models 9 and 10 use official pixel width (`--width 0`).

Each row: **line** · **5** · **9** · **10**

### 01 · f72

<img src="samples/eva-compare/01_line.png" width="200"> <img src="samples/eva-compare/01_m5.png" width="200"> <img src="samples/eva-compare/01_m9.png" width="200"> <img src="samples/eva-compare/01_m10.png" width="200">

### 02 · f163

<img src="samples/eva-compare/02_line.png" width="200"> <img src="samples/eva-compare/02_m5.png" width="200"> <img src="samples/eva-compare/02_m9.png" width="200"> <img src="samples/eva-compare/02_m10.png" width="200">

### 03 · f253

<img src="samples/eva-compare/03_line.png" width="200"> <img src="samples/eva-compare/03_m5.png" width="200"> <img src="samples/eva-compare/03_m9.png" width="200"> <img src="samples/eva-compare/03_m10.png" width="200">

### 04 · f344

<img src="samples/eva-compare/04_line.png" width="200"> <img src="samples/eva-compare/04_m5.png" width="200"> <img src="samples/eva-compare/04_m9.png" width="200"> <img src="samples/eva-compare/04_m10.png" width="200">

### 05 · f434

<img src="samples/eva-compare/05_line.png" width="200"> <img src="samples/eva-compare/05_m5.png" width="200"> <img src="samples/eva-compare/05_m9.png" width="200"> <img src="samples/eva-compare/05_m10.png" width="200">

### 06 · f525

<img src="samples/eva-compare/06_line.png" width="200"> <img src="samples/eva-compare/06_m5.png" width="200"> <img src="samples/eva-compare/06_m9.png" width="200"> <img src="samples/eva-compare/06_m10.png" width="200">

### 07 · f616

<img src="samples/eva-compare/07_line.png" width="200"> <img src="samples/eva-compare/07_m5.png" width="200"> <img src="samples/eva-compare/07_m9.png" width="200"> <img src="samples/eva-compare/07_m10.png" width="200">

### 08 · f706

<img src="samples/eva-compare/08_line.png" width="200"> <img src="samples/eva-compare/08_m5.png" width="200"> <img src="samples/eva-compare/08_m9.png" width="200"> <img src="samples/eva-compare/08_m10.png" width="200">

## 目录 / Layout

```text
src/                     converters
models/deepaa_official/  official DeepAA charset, glyphs, small-head
models/anilines/         AniLines (downloaded on first run)
models/own/              self-trained CNNs (not official DeepAA)
samples/eva-compare/     EVA stills: line / model 5 / 9 / 10
output/videos/line|ascii|source   local videos (gitignored)
```

## 运行 / Run

```text
pip install -r requirements.txt
```

Model 5 (direction ASCII):

```text
python src/line_art_demo.py --from-lines --input input/eva-line --output output/line_art_demo --width 160
```

Model 9 (lightweight DeepAA):

```text
python src/line_art_9.py --from-lines --width 0 --input input/eva-line --output output/line-art-9-deepaa-light
```

Model 10 (same as 9, skip empty windows):

```text
python src/line_art_10.py --from-lines --width 0 --input input/eva-line --output output/line-art-10-deepaa-light-empty-skip
```

Raise `--ink-skip` only after checking that strokes are not dropped (try `0.002`, then `0.005`). `0` equals model 9.

AniLines line video:

```text
python src/anilines.py --input path/to/clip.mp4 --output output/videos/line --fps 0 --max-side 960
```

Official full DeepAA (`weight_full.pt`, local only; convert from `data/weight.hdf5`):

```text
python src/line_art_8_puredeepaa.py --from-lines --width 0 --input input/eva-line
```
