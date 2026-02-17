#!/usr/bin/env python3
"""Score shared1000 images with the ResMem memorability model."""

import argparse
from pathlib import Path

from viz2psych.models.resmem import ResMemModel
from viz2psych.pipeline import run_model, DEFAULT_IMAGE_DIR, DEFAULT_STIM_INFO, DEFAULT_OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="Run ResMem memorability scoring.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--stim-info", type=Path, default=DEFAULT_STIM_INFO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    model = ResMemModel()
    run_model(
        model,
        image_dir=args.image_dir,
        stim_info_csv=args.stim_info,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
