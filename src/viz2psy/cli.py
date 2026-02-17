#!/usr/bin/env python3
"""CLI for scoring images with viz2psy models.

Examples
--------
    # List available models
    viz2psy --list-models

    # Single image
    viz2psy resmem photo.jpg

    # Multiple images
    viz2psy emonet img1.png img2.png img3.png

    # Glob a directory and save to CSV
    viz2psy resmem /path/to/images/*.png -o scores.csv

    # Use CPU explicitly
    viz2psy clip images/*.jpg --device cpu
"""

import argparse
import importlib
import sys
from pathlib import Path

# Lazy model registry: maps name -> (module_path, class_name, description).
MODEL_REGISTRY = {
    "resmem": ("viz2psy.models.resmem", "ResMemModel", "Image memorability (0-1)"),
    "emonet": ("viz2psy.models.emonet", "EmoNetModel", "20 emotion category probabilities"),
    "clip": ("viz2psy.models.clip", "CLIPModel", "512-dim CLIP ViT-B-32 embeddings"),
    "gist": ("viz2psy.models.gist", "GISTModel", "512-dim Gabor spatial envelope features"),
    "llstat": ("viz2psy.models.llstat", "LLStatModel", "17 low-level image statistics"),
    "saliency": ("viz2psy.models.saliency", "SaliencyModel", "576-dim fixation density grid"),
    "dinov2": ("viz2psy.models.dinov2", "DINOv2Model", "768-dim DINOv2 ViT-B/14 embeddings"),
    "aesthetics": ("viz2psy.models.aesthetics", "AestheticsModel", "Aesthetic quality score (1-10)"),
    "places": ("viz2psy.models.places", "PlacesModel", "365 scene + 102 attribute probabilities"),
    "yolo": ("viz2psy.models.yolo", "YOLOModel", "80 object counts + 5 summary statistics"),
}


def _load_model_class(name: str):
    """Dynamically import and return a model class."""
    module_path, class_name, _ = MODEL_REGISTRY[name]
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def list_models():
    """Print available models and their descriptions."""
    print("Available models:\n")
    for name, (_, _, desc) in MODEL_REGISTRY.items():
        print(f"  {name:12s}  {desc}")
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Extract psychological/perceptual features from images.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  viz2psy resmem photo.jpg\n"
               "  viz2psy clip images/*.png -o embeddings.csv\n"
               "  viz2psy --list-models",
    )
    parser.add_argument(
        "model",
        nargs="?",
        choices=list(MODEL_REGISTRY.keys()),
        help="Model to run.",
    )
    parser.add_argument(
        "images",
        nargs="*",
        type=Path,
        help="Image file paths.",
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=None,
        help="Save results to this CSV (prints to stdout if omitted).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Number of images per forward pass (default: 32).",
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default=None,
        help="Device for inference (default: auto-detect).",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit.",
    )

    args = parser.parse_args()

    if args.list_models:
        list_models()
        sys.exit(0)

    if args.model is None:
        parser.print_help()
        sys.exit(1)

    if not args.images:
        print("Error: No images provided.", file=sys.stderr)
        sys.exit(1)

    # Import here to avoid slow startup when just listing models.
    from viz2psy.pipeline import score_images

    model_cls = _load_model_class(args.model)

    # Pass device if specified.
    if args.device:
        model = model_cls(device=args.device)
    else:
        model = model_cls()

    df = score_images(model, args.images, batch_size=args.batch_size, quiet=args.quiet)

    if args.output:
        df.to_csv(args.output, index=False)
        if not args.quiet:
            print(f"Saved {len(df)} rows to {args.output}")
    else:
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
