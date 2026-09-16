# Weights

## `deepaa_official/`

Official DeepAA (Takezaki / Keras `weight.hdf5` lineage).

| file | what |
|---|---|
| `char_list.csv` | 411-class order (`frequency >= 10`) |
| `char_dict.pkl` | variable-width glyphs for layout |
| `weight_smallhead.pt` | frozen official conv + FC512 head (model 9) |
| `weight_light.hdf5` | official light net (unused by 9/10) |
| `weight_full.pt` | full FC4096 net (~333MB, **local only**, gitignored) |

Convert official `data/weight.hdf5` to `weight_full.pt` with:

```text
python src/deepaa_official_full.py
```

## `anilines/`

Third-party AniLines `detail.pth`. Not in git; first run of `src/anilines.py` downloads it from Hugging Face (`gyrojeff/AniLines`).

## `own/`

Models trained in this project (not official DeepAA weights).

| file | used by |
|---|---|
| `deepaa_cnn.pt` | `src/deepaa_cnn.py` (training helper for small-head data) |
| `deepaa_struct.pt` | structure-char variant of the same CNN |
