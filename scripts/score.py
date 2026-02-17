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
import importlib
from pathlib import Path

from viz2psy.pipeline import score_images

# Lazy model registry: maps name -> (module_path, class_name).
_MODEL_REGISTRY = {
    "resmem": ("viz2psy.models.resmem", "ResMemModel"),
    "emonet": ("viz2psy.models.emonet", "EmoNetModel"),
    "clip": ("viz2psy.models.clip", "CLIPModel"),
    "gist": ("viz2psy.models.gist", "GISTModel"),
    "llstat": ("viz2psy.models.llstat", "LLStatModel"),
    "saliency": ("viz2psy.models.saliency", "SaliencyModel"),
    "dinov2": ("viz2psy.models.dinov2", "DINOv2Model"),
    "aesthetics": ("viz2psy.models.aesthetics", "AestheticsModel"),
    "places": ("viz2psy.models.places", "PlacesModel"),
    "yolo": ("viz2psy.models.yolo", "YOLOModel"),
}


def _load_model_class(name: str):
    module_path, class_name = _MODEL_REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def main():
    parser = argparse.ArgumentParser(
        description="Score images with a viz2psy model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Example: python scripts/score.py resmem photo.jpg",
    )
    parser.add_argument("model", choices=_MODEL_REGISTRY, help="Model to run.")
    parser.add_argument("images", nargs="+", type=Path, help="Image file paths.")
    parser.add_argument("-o", "--output", type=Path, default=None,
                        help="Save results to this CSV (prints to stdout if omitted).")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    model_cls = _load_model_class(args.model)
    model = model_cls()
    df = score_images(model, args.images, batch_size=args.batch_size)

    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} rows to {args.output}")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
