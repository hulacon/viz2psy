#!/usr/bin/env python3
"""Score arbitrary images with any viz2psy model.

Examples
--------
    # Single image
    python scripts/score.py resmem photo.jpg

    # Multiple images
    python scripts/score.py emonet img1.png img2.png img3.png

    # Glob a directory and save to CSV
    python scripts/score.py resmem /path/to/images/*.png -o scores.csv
"""

import argparse
from pathlib import Path

from viz2psy.models.resmem import ResMemModel
from viz2psy.models.emonet import EmoNetModel
from viz2psy.models.clip import CLIPModel
from viz2psy.models.gist import GISTModel
from viz2psy.models.llstat import LLStatModel
from viz2psy.models.saliency import SaliencyModel
from viz2psy.pipeline import score_images

MODELS = {
    "resmem": ResMemModel,
    "emonet": EmoNetModel,
    "clip": CLIPModel,
    "gist": GISTModel,
    "llstat": LLStatModel,
    "saliency": SaliencyModel,
}


def main():
    parser = argparse.ArgumentParser(
        description="Score images with a viz2psy model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python scripts/score.py resmem photo.jpg",
    )
    parser.add_argument("model", choices=MODELS, help="Model to run.")
    parser.add_argument("images", nargs="+", type=Path, help="Image file paths.")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Save results to this CSV (prints to stdout if omitted).")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    model = MODELS[args.model]()
    df = score_images(model, args.images, batch_size=args.batch_size)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} rows to {args.output}")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
