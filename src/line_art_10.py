"""
line-art-10-deepaa-light-empty-skip

Same conversion as model 9. 64x64 windows with almost no ink emit an
ideographic space and skip the network.

Default --ink-skip is very low so strokes are not dropped. Raise it only
after checking that valuable cells are not turned into spaces.
"""
import argparse
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import line_art_9 as m9
from line_art_4 import VIDEO_EXTS
from main import DEFAULT_INPUT, DEFAULT_OUTPUT

STRATEGY_NAME = "line-art-10-deepaa-light-empty-skip"
DEFAULT_OUT = os.path.join(DEFAULT_OUTPUT, STRATEGY_NAME)
INK_SKIP_DEFAULT = 0.0008


def parse_args():
    parser = argparse.ArgumentParser(description=STRATEGY_NAME)
    parser.add_argument("--input", default=DEFAULT_INPUT)
    parser.add_argument("--output", default=DEFAULT_OUT)
    parser.add_argument(
        "--width",
        type=int,
        default=0,
        help="Official pixel width. 0 = original. Not column count.",
    )
    parser.add_argument(
        "--method", default="auto", choices=["auto", "structure", "xdog", "canny", "sketch"]
    )
    parser.add_argument("--from-lines", action="store_true")
    parser.add_argument("--video-out", default="")
    parser.add_argument("--slide", type=int, default=0)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument(
        "--ink-skip",
        type=float,
        default=INK_SKIP_DEFAULT,
        help=(
            "Skip the net when the 64x64 ink fraction is below this. "
            "Default 0.0008 (~3 dark pixels). Raise slowly if empty skip looks safe. "
            "0 = same as model 9."
        ),
    )
    return parser.parse_args()


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass
    args = parse_args()
    m9.STRATEGY_NAME = STRATEGY_NAME
    m9.DEFAULT_OUT = args.output
    print(
        f"{STRATEGY_NAME}: ink_skip={args.ink_skip}  "
        "(low = fewer skips, quality first)",
        flush=True,
    )
    if os.path.isfile(args.input) and args.input.lower().endswith(VIDEO_EXTS):
        m9.convert_line_video(
            args.input,
            args.output,
            args.width,
            args.video_out or None,
            slide=args.slide,
            chunk=args.chunk,
            restart=args.restart,
            ink_skip=args.ink_skip,
        )
    else:
        m9.batch_extract(
            args.input,
            args.output,
            args.width,
            args.method,
            args.from_lines,
            slide=args.slide,
            chunk=args.chunk,
            ink_skip=args.ink_skip,
        )
