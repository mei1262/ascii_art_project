# ascii_art_project

突然对ASCII艺术很感兴趣，做了非结构ASCII转化模型，觉得不过瘾就研究一下结构性ASCII艺术；自己做的line_art_demo局限性比较大，也没有标注标签训练的条件，于是借用DeepAA项目的经验，数据和权重做了一些优化。

I suddenly got interested in ASCII art, built a non-structural ASCII converter, then went further into structural ASCII. line_art_demo is quite limited, and I had no labeled data to train with, so I borrowed experience, data, and weights from the DeepAA project and made some optimizations.

## 引用 / Reference

Taizan Yonetsuji. *DeepAA*. 2017. <https://github.com/taizan/DeepAA>

## 模型对比 / Models

| | line_art_demo（模型 5） | 模型 9 | 模型 10 | 官方 DeepAA |
|---|---|---|---|---|
| Origin | **Self-made**, 0 NN weights | Frozen official conv + FC512 | Same as 9, skip empty windows | Full official net |
| Characters | Direction strokes `/ \ - \| ~` only | Official 411, variable width | Same as 9 | Official 411 |
| Layout | Fixed 18×30 grid; **glyphs look smaller** in the comparison | `char_dict` variable width, 18px rows | Same as 9 | Same as 9 (slide=0 here) |
| Params | 0 | ~10.6M | ~10.6M | ~87.3M |
| Script | `src/line_art_demo.py` | `src/line_art_9.py` | `src/line_art_10.py` | `src/line_art_8_puredeepaa.py` |

Model 9/10 keep official DeepAA decode but use a smaller head. Model 10 only skips nearly empty 64×64 windows (`--ink-skip 0.0008`). On the 8 EVA stills, 10 matched 9 exactly and skipped 13.9% of windows (~1.16× fewer forwards).

## EVA 8 帧对照 / EVA 8-frame comparison

AniLines detail from *明日香来日*. Each row: **line** · **line_art_demo (5)** · **9** · **10** · **official DeepAA**.

line_art_demo uses a fixed grid, so **its characters are smaller** than DeepAA / 9 / 10 in these thumbnails. 9 and 10 use `--width 0`. Official DeepAA here is `line-art-8-puredeepaa` (full FC4096 weights).

### 01 · f72

<img src="samples/eva-compare/01_line.png" width="160"> <img src="samples/eva-compare/01_m5.png" width="160"> <img src="samples/eva-compare/01_m9.png" width="160"> <img src="samples/eva-compare/01_m10.png" width="160"> <img src="samples/eva-compare/01_deepaa.png" width="160">

### 02 · f163

<img src="samples/eva-compare/02_line.png" width="160"> <img src="samples/eva-compare/02_m5.png" width="160"> <img src="samples/eva-compare/02_m9.png" width="160"> <img src="samples/eva-compare/02_m10.png" width="160"> <img src="samples/eva-compare/02_deepaa.png" width="160">

### 03 · f253

<img src="samples/eva-compare/03_line.png" width="160"> <img src="samples/eva-compare/03_m5.png" width="160"> <img src="samples/eva-compare/03_m9.png" width="160"> <img src="samples/eva-compare/03_m10.png" width="160"> <img src="samples/eva-compare/03_deepaa.png" width="160">

### 04 · f344

<img src="samples/eva-compare/04_line.png" width="160"> <img src="samples/eva-compare/04_m5.png" width="160"> <img src="samples/eva-compare/04_m9.png" width="160"> <img src="samples/eva-compare/04_m10.png" width="160"> <img src="samples/eva-compare/04_deepaa.png" width="160">

### 05 · f434

<img src="samples/eva-compare/05_line.png" width="160"> <img src="samples/eva-compare/05_m5.png" width="160"> <img src="samples/eva-compare/05_m9.png" width="160"> <img src="samples/eva-compare/05_m10.png" width="160"> <img src="samples/eva-compare/05_deepaa.png" width="160">

### 06 · f525

<img src="samples/eva-compare/06_line.png" width="160"> <img src="samples/eva-compare/06_m5.png" width="160"> <img src="samples/eva-compare/06_m9.png" width="160"> <img src="samples/eva-compare/06_m10.png" width="160"> <img src="samples/eva-compare/06_deepaa.png" width="160">

### 07 · f616

<img src="samples/eva-compare/07_line.png" width="160"> <img src="samples/eva-compare/07_m5.png" width="160"> <img src="samples/eva-compare/07_m9.png" width="160"> <img src="samples/eva-compare/07_m10.png" width="160"> <img src="samples/eva-compare/07_deepaa.png" width="160">

### 08 · f706

<img src="samples/eva-compare/08_line.png" width="160"> <img src="samples/eva-compare/08_m5.png" width="160"> <img src="samples/eva-compare/08_m9.png" width="160"> <img src="samples/eva-compare/08_m10.png" width="160"> <img src="samples/eva-compare/08_deepaa.png" width="160">

## 目录 / Layout

```text
src/                     converters
models/deepaa_official/  official DeepAA charset, glyphs, small-head
models/anilines/         AniLines (downloaded on first run)
models/own/              self-trained CNNs (not official DeepAA)
samples/eva-compare/     EVA stills: line / line_art_demo / 9 / 10 / official DeepAA
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
