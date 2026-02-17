#!/usr/bin/env python3
"""CLI for scoring images with viz2psy models.

Examples
--------
    # List available models
    viz2psy --list-models

    # Single model on images
    viz2psy resmem photo.jpg

    # Multiple models on images
    viz2psy resmem clip emonet images/*.png -o scores.csv

    # All models on images
    viz2psy --all images/*.png -o scores.csv

    # Score video frames (default: every 0.5s)
    viz2psy resmem movie.mp4 -o scores.csv

    # Custom frame interval
    viz2psy resmem movie.mp4 --frame-interval 1.0 -o scores.csv

    # Save extracted frames to disk (for large videos)
    viz2psy resmem movie.mp4 --save-frames ./frames -o scores.csv
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


def _parse_models_and_inputs(args: list[str]) -> tuple[list[str], list[Path]]:
    """Separate model names from input paths in positional arguments."""
    models = []
    inputs = []

    for arg in args:
        if arg in MODEL_REGISTRY and not inputs:
            # It's a model name (and we haven't started seeing inputs yet)
            models.append(arg)
        else:
            # It's an input path (image or video)
            inputs.append(Path(arg))

    return models, inputs


def _format_bytes(n_bytes: int) -> str:
    """Format bytes as human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if n_bytes < 1024:
            return f"{n_bytes:.1f} {unit}"
        n_bytes /= 1024
    return f"{n_bytes:.1f} TB"


def _process_video(
    video_path: Path,
    models: list[str],
    frame_interval: float,
    save_frames: Path | None,
    batch_size: int,
    device: str | None,
    quiet: bool,
):
    """Process a video file and return a DataFrame with time-based scores."""
    from viz2psy.pipeline import score_images
    from viz2psy.video import (
        extract_frames,
        extract_frames_to_temp,
        estimate_memory_usage,
        get_available_memory,
        get_video_info,
    )

    video_info = get_video_info(video_path)
    if not quiet:
        print(f"Video: {video_path.name}")
        print(f"  Duration: {video_info['duration']:.1f}s, "
              f"Resolution: {video_info['width']}x{video_info['height']}, "
              f"FPS: {video_info['fps']:.1f}")

    # Check memory usage
    estimated_mem = estimate_memory_usage(video_info, frame_interval)
    available_mem = get_available_memory()

    n_frames = int(video_info["duration"] / frame_interval) + 1
    if not quiet:
        print(f"  Frames to extract: {n_frames} (every {frame_interval}s)")
        print(f"  Estimated memory: {_format_bytes(estimated_mem)}")

    # Warn if memory might be an issue
    use_temp_dir = False
    temp_dir = None
    if save_frames:
        # User explicitly requested saving frames
        if not quiet:
            print(f"  Saving frames to: {save_frames}")
        frames = extract_frames(
            video_path,
            frame_interval=frame_interval,
            save_dir=save_frames,
            quiet=quiet,
        )
    elif estimated_mem > available_mem * 0.8:
        # Memory might be tight, use temp directory
        if not quiet:
            print(f"  Warning: Estimated memory ({_format_bytes(estimated_mem)}) "
                  f"exceeds 80% of available ({_format_bytes(available_mem)})")
            print("  Using temporary directory for frames...")
        frames, temp_dir = extract_frames_to_temp(
            video_path,
            frame_interval=frame_interval,
            quiet=quiet,
        )
        use_temp_dir = True
    else:
        # Load frames into memory
        frames = extract_frames(
            video_path,
            frame_interval=frame_interval,
            save_dir=None,
            quiet=quiet,
        )

    try:
        # Extract times and images/paths
        times = [t for t, _ in frames]
        images_or_paths = [img for _, img in frames]

        # If we have paths, load them as PIL images for scoring
        if save_frames or use_temp_dir:
            from viz2psy.utils import load_image
            images = [load_image(p) for p in images_or_paths]
        else:
            images = images_or_paths

        # Run each model
        import pandas as pd
        result_df = pd.DataFrame({"time": times})

        for model_name in models:
            model_cls = _load_model_class(model_name)
            model = model_cls(device=device) if device else model_cls()

            if not quiet:
                print(f"Loading {model.name} model on {model.device} ...")
            model.load()

            # Score images in batches
            from tqdm import tqdm
            all_scores = []
            iterator = range(0, len(images), batch_size)
            if not quiet:
                iterator = tqdm(iterator, desc=model.name)

            for batch_start in iterator:
                batch = images[batch_start : batch_start + batch_size]
                scores_list = model.predict_batch(batch)
                all_scores.extend(scores_list)

            # Add scores to result DataFrame
            scores_df = pd.DataFrame(all_scores)
            for col in scores_df.columns:
                result_df[col] = scores_df[col]

        return result_df

    finally:
        # Clean up temp directory if used
        if temp_dir is not None:
            temp_dir.cleanup()


def _process_images(
    image_paths: list[Path],
    models: list[str],
    batch_size: int,
    device: str | None,
    quiet: bool,
):
    """Process image files and return a DataFrame."""
    from viz2psy.pipeline import score_images

    result_df = None
    for model_name in models:
        model_cls = _load_model_class(model_name)
        model = model_cls(device=device) if device else model_cls()

        df = score_images(model, image_paths, batch_size=batch_size, quiet=quiet)

        if result_df is None:
            result_df = df
        else:
            # Merge on filename/filepath, adding new score columns
            score_cols = [c for c in df.columns if c not in ("filename", "filepath")]
            result_df = result_df.merge(
                df[["filename"] + score_cols],
                on="filename",
                how="outer",
            )

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract psychological/perceptual features from images or videos.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  viz2psy resmem photo.jpg\n"
               "  viz2psy resmem clip images/*.png -o scores.csv\n"
               "  viz2psy --all images/*.png -o all_scores.csv\n"
               "  viz2psy resmem movie.mp4 --frame-interval 0.5 -o scores.csv\n"
               "  viz2psy --list-models",
    )
    parser.add_argument(
        "args",
        nargs="*",
        help="Model name(s) followed by image/video paths.",
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
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Device for inference (default: auto-detect). Use 'mps' for Apple Silicon GPU.",
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress progress output.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available models.",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit.",
    )
    # Video-specific options
    parser.add_argument(
        "--frame-interval",
        type=float,
        default=0.5,
        help="Time between extracted frames in seconds (default: 0.5).",
    )
    parser.add_argument(
        "--save-frames",
        type=Path,
        default=None,
        help="Save extracted video frames to this directory.",
    )

    args = parser.parse_args()

    if args.list_models:
        list_models()
        sys.exit(0)

    # Parse positional arguments into models and inputs
    models, inputs = _parse_models_and_inputs(args.args)

    # Handle --all flag
    if args.all:
        if models:
            print("Error: Cannot specify both --all and model names.", file=sys.stderr)
            sys.exit(1)
        models = list(MODEL_REGISTRY.keys())

    if not models:
        parser.print_help()
        sys.exit(1)

    if not inputs:
        print("Error: No input files provided.", file=sys.stderr)
        sys.exit(1)

    # Validate model names
    invalid = [m for m in models if m not in MODEL_REGISTRY]
    if invalid:
        print(f"Error: Unknown model(s): {', '.join(invalid)}", file=sys.stderr)
        print(f"Available models: {', '.join(MODEL_REGISTRY.keys())}", file=sys.stderr)
        sys.exit(1)

    # Detect if input is a video
    from viz2psy.video import is_video_file

    if len(inputs) == 1 and is_video_file(inputs[0]):
        # Video processing
        result_df = _process_video(
            video_path=inputs[0],
            models=models,
            frame_interval=args.frame_interval,
            save_frames=args.save_frames,
            batch_size=args.batch_size,
            device=args.device,
            quiet=args.quiet,
        )
    else:
        # Image processing
        if args.save_frames:
            print("Warning: --save-frames is only used with video input.", file=sys.stderr)
        result_df = _process_images(
            image_paths=inputs,
            models=models,
            batch_size=args.batch_size,
            device=args.device,
            quiet=args.quiet,
        )

    if args.output:
        result_df.to_csv(args.output, index=False)
        if not args.quiet:
            print(f"Saved {len(result_df)} rows to {args.output}")
    else:
        print(result_df.to_string(index=False))


if __name__ == "__main__":
    main()
