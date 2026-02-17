#!/usr/bin/env python3
"""Score shared1000 images with the EmoNet emotion classification model."""

import argparse
from pathlib import Path

from viz2psych.models.emonet import EmoNetModel
from viz2psych.pipeline import run_model, DEFAULT_IMAGE_DIR, DEFAULT_STIM_INFO, DEFAULT_OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="Run EmoNet emotion scoring.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--stim-info", type=Path, default=DEFAULT_STIM_INFO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--weights-path", type=Path, default=None,
                        help="Path to cached EmoNet weights (downloads if absent).")
    args = parser.parse_args()

    model = EmoNetModel(weights_path=args.weights_path)
    run_model(
        model,
        image_dir=args.image_dir,
        stim_info_csv=args.stim_info,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
