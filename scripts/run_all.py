#!/usr/bin/env python3
"""Run all models over the shared1000 images."""

import argparse
from pathlib import Path

from viz2psy.models.resmem import ResMemModel
from viz2psy.models.emonet import EmoNetModel
from viz2psy.models.clip import CLIPModel
from viz2psy.models.gist import GISTModel
from viz2psy.models.llstat import LLStatModel
from viz2psy.models.saliency import SaliencyModel
from viz2psy.pipeline import run_model, DEFAULT_IMAGE_DIR, DEFAULT_STIM_INFO, DEFAULT_OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser(description="Run all viz2psy models.")
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--stim-info", type=Path, default=DEFAULT_STIM_INFO)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    models = [
        ResMemModel(),
        EmoNetModel(),
        CLIPModel(),
        GISTModel(),
        LLStatModel(),
        SaliencyModel(),
    ]

    for model in models:
        print(f"\n{'='*60}")
        print(f"Running {model.name}")
        print(f"{'='*60}")
        run_model(
            model,
            image_dir=args.image_dir,
            stim_info_csv=args.stim_info,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )

    print("\nAll models complete.")


if __name__ == "__main__":
    main()
