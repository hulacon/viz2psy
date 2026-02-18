#!/usr/bin/env python3
"""CLI for scoring images, videos, and HDF5 bricks with viz2psy models.

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

    # Score HDF5 image brick (e.g., NSD dataset)
    viz2psy resmem data.hdf5 -o scores.csv

    # HDF5 with slice and dataset selection
    viz2psy resmem data.hdf5 --dataset imgBrick --start 0 --end 1000 -o scores.csv

    # List datasets in HDF5 file
    viz2psy --list-datasets data.hdf5
"""

import argparse
import importlib
import sys
from pathlib import Path

from viz2psy.exceptions import (
    DeviceError,
    ImageLoadError,
    InferenceError,
    ModelLoadError,
    VideoError,
    Viz2PsyError,
)

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


def is_hdf5_file(path: Path) -> bool:
    """Check if a file is an HDF5 file."""
    return path.suffix.lower() in (".h5", ".hdf5")


def list_datasets(hdf5_path: Path) -> None:
    """Print datasets in an HDF5 file."""
    import h5py

    print(f"Datasets in {hdf5_path}:\n")
    with h5py.File(hdf5_path, "r") as f:
        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                shape_str = " x ".join(str(d) for d in obj.shape)
                print(f"  {name}: [{shape_str}] {obj.dtype}")
        f.visititems(visitor)
    print()


def _find_image_dataset(hdf5_path: Path, dataset_name: str | None) -> str:
    """Find the image dataset in an HDF5 file."""
    import h5py
    import numpy as np

    with h5py.File(hdf5_path, "r") as f:
        if dataset_name:
            if dataset_name not in f:
                print(f"Error: Dataset '{dataset_name}' not found in {hdf5_path}", file=sys.stderr)
                list_datasets(hdf5_path)
                sys.exit(1)
            return dataset_name

        # Auto-detect: look for common names or 4D uint8 arrays
        common_names = ["imgBrick", "images", "stimuli", "data"]
        for name in common_names:
            if name in f:
                return name

        # Find first 4D uint8 dataset
        for name in f.keys():
            ds = f[name]
            if isinstance(ds, h5py.Dataset) and len(ds.shape) == 4 and ds.dtype == np.uint8:
                return name

        print("Error: Could not auto-detect image dataset. Use --dataset to specify.", file=sys.stderr)
        list_datasets(hdf5_path)
        sys.exit(1)


def _clear_gpu_memory():
    """Clear GPU memory if available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except ImportError:
        pass


def _cuda_available() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def _mps_available() -> bool:
    try:
        import torch
        return torch.backends.mps.is_available()
    except ImportError:
        return False


def _process_hdf5(
    hdf5_path: Path,
    dataset_name: str,
    models: list[str],
    output_path: Path,
    batch_size: int = 32,
    start_idx: int = 0,
    end_idx: int | None = None,
    device: str | None = None,
    quiet: bool = False,
    metadata=None,
):
    """Process an HDF5 image brick with multiple models."""
    import time

    import h5py
    import numpy as np
    import pandas as pd
    from PIL import Image
    from tqdm import tqdm

    with h5py.File(hdf5_path, "r") as f:
        dataset = f[dataset_name]
        n_total = dataset.shape[0]
        img_shape = dataset.shape[1:3]

        if end_idx is None:
            end_idx = n_total
        end_idx = min(end_idx, n_total)
        n_to_process = end_idx - start_idx

        # Set metadata input info
        if metadata:
            metadata.set_input_hdf5(hdf5_path, dataset_name, start_idx, end_idx)

        if not quiet:
            print(f"HDF5 file: {hdf5_path}")
            print(f"Dataset: {dataset_name}")
            print(f"Total images: {n_total:,} @ {img_shape[0]}x{img_shape[1]}")
            print(f"Processing: indices {start_idx:,} to {end_idx:,} ({n_to_process:,} images)")
            print(f"Output: {output_path}")
            print()

        all_results = {"image_idx": list(range(start_idx, end_idx))}

        for model_name in models:
            # Load model
            model_cls = _load_model_class(model_name)
            model = model_cls(device=device) if device else model_cls()

            if not quiet:
                print(f"Loading {model.name} on {model.device}...")
            try:
                model.load()
            except Exception as e:
                raise ModelLoadError(model_name, str(e)) from e

            # Track device for metadata
            if metadata and metadata.device is None:
                metadata.set_device(str(model.device))

            # Get feature dimensions by running on one image
            test_img = Image.fromarray(dataset[start_idx], mode="RGB")
            test_result = model.predict(test_img)
            feature_names = list(test_result.keys())
            n_features = len(feature_names)

            if not quiet:
                print(f"  Features: {n_features}")

            # Process in batches
            start_time = time.time()
            model_scores = []

            iterator = range(start_idx, end_idx, batch_size)
            if not quiet:
                n_batches = (end_idx - start_idx + batch_size - 1) // batch_size
                iterator = tqdm(iterator, desc=model_name, total=n_batches, unit="batch")

            for batch_start in iterator:
                batch_end = min(batch_start + batch_size, end_idx)

                # Load batch of images
                batch_arrays = dataset[batch_start:batch_end]
                batch_images = [Image.fromarray(arr, mode="RGB") for arr in batch_arrays]

                # Score batch
                try:
                    results = model.predict_batch(batch_images)
                except Exception as e:
                    raise InferenceError(model_name, str(e)) from e
                model_scores.extend(results)

            # Add to results dict
            for feat_name in feature_names:
                all_results[feat_name] = [r[feat_name] for r in model_scores]

            elapsed = time.time() - start_time
            if not quiet and n_to_process > 0:
                rate = n_to_process / elapsed
                print(f"  Completed: {n_to_process:,} images in {elapsed:.1f}s ({rate:.1f} img/s)")
                print()

            # Record model metadata
            if metadata:
                metadata.add_model(model_name, feature_names, elapsed)

            # Clean up GPU memory
            del model
            _clear_gpu_memory()

    return pd.DataFrame(all_results)


def _process_video(
    video_path: Path,
    models: list[str],
    frame_interval: float,
    save_frames: Path | None,
    batch_size: int,
    device: str | None,
    quiet: bool,
    metadata=None,
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

    # Set metadata input info
    if metadata:
        metadata.set_input_video(video_path, frame_interval, n_frames, save_frames)

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
        import time
        result_df = pd.DataFrame({"time": times})

        for model_name in models:
            model_cls = _load_model_class(model_name)
            model = model_cls(device=device) if device else model_cls()

            if not quiet:
                print(f"Loading {model.name} model on {model.device} ...")
            try:
                model.load()
            except Exception as e:
                raise ModelLoadError(model_name, str(e)) from e

            # Track device for metadata
            if metadata and metadata.device is None:
                metadata.set_device(str(model.device))

            # Score images in batches
            from tqdm import tqdm
            start_time = time.time()
            all_scores = []
            iterator = range(0, len(images), batch_size)
            if not quiet:
                iterator = tqdm(iterator, desc=model.name)

            for batch_start in iterator:
                batch = images[batch_start : batch_start + batch_size]
                try:
                    scores_list = model.predict_batch(batch)
                except Exception as e:
                    raise InferenceError(model_name, str(e)) from e
                all_scores.extend(scores_list)

            elapsed = time.time() - start_time

            # Add scores to result DataFrame
            scores_df = pd.DataFrame(all_scores)
            feature_names = list(scores_df.columns)
            for col in scores_df.columns:
                result_df[col] = scores_df[col]

            # Record model metadata
            if metadata:
                metadata.add_model(model_name, feature_names, elapsed)

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
    metadata=None,
):
    """Process image files and return a DataFrame."""
    import time
    from viz2psy.pipeline import score_images

    # Set metadata input info
    if metadata:
        metadata.set_input_images(image_paths)

    result_df = None
    for model_name in models:
        model_cls = _load_model_class(model_name)
        model = model_cls(device=device) if device else model_cls()

        start_time = time.time()
        df = score_images(model, image_paths, batch_size=batch_size, quiet=quiet)
        elapsed = time.time() - start_time

        # Track device for metadata (after model.load() inside score_images)
        if metadata and metadata.device is None:
            metadata.set_device(str(model.device))

        # Extract feature names (all columns except filename)
        feature_names = [c for c in df.columns if c != "filename"]

        # Record model metadata
        if metadata:
            metadata.add_model(model_name, feature_names, elapsed)

        if result_df is None:
            result_df = df
        else:
            # Merge on filename, adding new score columns
            score_cols = [c for c in df.columns if c != "filename"]
            result_df = result_df.merge(
                df[["filename"] + score_cols],
                on="filename",
                how="outer",
            )

    return result_df


def main():
    parser = argparse.ArgumentParser(
        description="Extract psychological/perceptual features from images, videos, or HDF5 bricks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  viz2psy resmem photo.jpg\n"
               "  viz2psy resmem clip images/*.png -o scores.csv\n"
               "  viz2psy --all images/*.png -o all_scores.csv\n"
               "  viz2psy resmem movie.mp4 --frame-interval 0.5 -o scores.csv\n"
               "  viz2psy resmem data.hdf5 --start 0 --end 1000 -o scores.csv\n"
               "  viz2psy --list-datasets data.hdf5\n"
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
    # HDF5-specific options
    parser.add_argument(
        "--dataset", "-d",
        type=str,
        default=None,
        help="HDF5 dataset name (auto-detected if not specified).",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index for HDF5 processing (default: 0).",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index for HDF5 processing (default: all images).",
    )
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List datasets in HDF5 file and exit.",
    )

    args = parser.parse_args()

    if args.list_models:
        list_models()
        sys.exit(0)

    # Parse positional arguments into models and inputs
    models, inputs = _parse_models_and_inputs(args.args)

    # Handle --list-datasets (needs at least one input file)
    if args.list_datasets:
        if not inputs:
            print("Error: No HDF5 file provided.", file=sys.stderr)
            sys.exit(1)
        if not is_hdf5_file(inputs[0]):
            print(f"Error: {inputs[0]} is not an HDF5 file.", file=sys.stderr)
            sys.exit(1)
        list_datasets(inputs[0])
        sys.exit(0)

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

    # Detect input type and route accordingly
    from viz2psy.video import is_video_file
    from viz2psy.metadata import MetadataBuilder

    # Initialize metadata builder (only save if output is specified)
    metadata = MetadataBuilder() if args.output else None

    try:
        if len(inputs) == 1 and is_hdf5_file(inputs[0]):
            # HDF5 processing
            if args.save_frames:
                print("Warning: --save-frames is only used with video input.", file=sys.stderr)
            if args.frame_interval != 0.5:
                print("Warning: --frame-interval is only used with video input.", file=sys.stderr)

            hdf5_path = inputs[0]
            if not hdf5_path.exists():
                print(f"Error: File not found: {hdf5_path}", file=sys.stderr)
                sys.exit(1)

            dataset_name = _find_image_dataset(hdf5_path, args.dataset)

            # Default output path for HDF5
            output_path = args.output
            if output_path is None:
                output_path = hdf5_path.with_name(hdf5_path.stem + "_scores.csv")

            result_df = _process_hdf5(
                hdf5_path=hdf5_path,
                dataset_name=dataset_name,
                models=models,
                output_path=output_path,
                batch_size=args.batch_size,
                start_idx=args.start,
                end_idx=args.end,
                device=args.device,
                quiet=args.quiet,
                metadata=metadata,
            )
            # Override output path for HDF5 (it has default)
            args.output = output_path

        elif len(inputs) == 1 and is_video_file(inputs[0]):
            # Video processing
            if args.dataset:
                print("Warning: --dataset is only used with HDF5 input.", file=sys.stderr)
            if args.start != 0 or args.end is not None:
                print("Warning: --start/--end are only used with HDF5 input.", file=sys.stderr)

            result_df = _process_video(
                video_path=inputs[0],
                models=models,
                frame_interval=args.frame_interval,
                save_frames=args.save_frames,
                batch_size=args.batch_size,
                device=args.device,
                quiet=args.quiet,
                metadata=metadata,
            )
        else:
            # Image processing
            if args.save_frames:
                print("Warning: --save-frames is only used with video input.", file=sys.stderr)
            if args.frame_interval != 0.5:
                print("Warning: --frame-interval is only used with video input.", file=sys.stderr)
            if args.dataset:
                print("Warning: --dataset is only used with HDF5 input.", file=sys.stderr)
            if args.start != 0 or args.end is not None:
                print("Warning: --start/--end are only used with HDF5 input.", file=sys.stderr)

            result_df = _process_images(
                image_paths=inputs,
                models=models,
                batch_size=args.batch_size,
                device=args.device,
                quiet=args.quiet,
                metadata=metadata,
            )

        if args.output:
            result_df.to_csv(args.output, index=False)

            # Save metadata sidecar
            metadata.set_output(args.output, len(result_df), len(result_df.columns))
            meta_path = metadata.save(args.output)

            if not args.quiet:
                print(f"Saved {len(result_df)} rows to {args.output}")
                print(f"Metadata saved to {meta_path}")
        else:
            print(result_df.to_string(index=False))

    except DeviceError as e:
        print(f"Error: {e}", file=sys.stderr)
        print(f"Available devices: cpu"
              + (", cuda" if _cuda_available() else "")
              + (", mps" if _mps_available() else ""),
              file=sys.stderr)
        sys.exit(1)
    except ImageLoadError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except VideoError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except ModelLoadError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Check your internet connection and available disk space.", file=sys.stderr)
        sys.exit(1)
    except InferenceError as e:
        print(f"Error: {e}", file=sys.stderr)
        print("Try reducing --batch-size or using --device cpu.", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)
    except OSError as e:
        print(f"Error writing output: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
