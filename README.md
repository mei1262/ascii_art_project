# ascii_art_project

Line-art extraction and ASCII conversion. Model 9 is official DeepAA convolution with a small FC512 head and official variable-width decode.

## Layout

```text
src/                     converters and training
models/
  deepaa_official/       official DeepAA (charset, glyphs, small-head weights)
  anilines/              AniLines detail weights (downloaded on first run)
  own/                   models trained in this repo
samples/line-art-9-eva/  8 EVA frames from model 9 (line + ASCII)
output/videos/line/      local line videos (gitignored)
output/videos/ascii/     local ASCII videos (gitignored)
output/videos/source/    local source clips (gitignored)
```

Showcase stills: `samples/line-art-9-eva/` (明日香来日, model 9, `--width 0`).

## Setup

```text
pip install -r requirements.txt
```

Place official full weights locally if you need 8pure / converting hdf5:

- `data/weight.hdf5` or `models/deepaa_official/weight_full.pt`

AniLines `detail.pth` is pulled automatically. Model 9 needs `models/deepaa_official/weight_smallhead.pt` (in this repo).

## Commands

Line video (AniLines detail):

```text
python src/anilines.py --input path/to/clip.mp4 --output output/videos/line --fps 0 --max-side 960
```

Model 9 (line already extracted):

```text
python src/line_art_9.py --from-lines --width 0 --input output/videos/line/clip_line.mp4
```

ASCII mp4 defaults to `output/videos/ascii/`. Stop with Ctrl+C; run the same command to resume.

Official full DeepAA decode (8pure):

```text
python src/line_art_8_puredeepaa.py --from-lines --width 0 --input path/to/line.png
```
