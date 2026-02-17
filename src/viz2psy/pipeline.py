"""Batch inference pipeline: load images, run model, save CSV."""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .models.base import BaseModel
from .utils import build_image_manifest, load_image

# Default paths for the shared1000 dataset.
DEFAULT_IMAGE_DIR = Path("/projects/hulacon/shared/mmmdata/stimuli/shared1000/images")
DEFAULT_STIM_INFO = Path("/projects/hulacon/shared/mmmdata/stimuli/shared1000/nsd_stim_info.csv")
DEFAULT_OUTPUT_DIR = Path("/projects/hulacon/shared/mmmdata/stimuli/shared1000")


def score_images(
    model: BaseModel,
    image_paths: list[str | Path],
    batch_size: int = 32,
) -> pd.DataFrame:
    """Score arbitrary images with a model.

    Parameters
    ----------
    model : BaseModel
        An instantiated (but not yet loaded) model wrapper.
    image_paths : list of str or Path
        Paths to image files (any format PIL can read).
    batch_size : int
        Number of images per forward pass.

    Returns
    -------
    pd.DataFrame
        One row per image with columns: filename, filepath, and model scores.
    """
    image_paths = [Path(p) for p in image_paths]

    print(f"Loading {model.name} model on {model.device} ...")
    model.load()

    all_rows: list[dict] = []
    for batch_start in tqdm(range(0, len(image_paths), batch_size), desc=model.name):
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        images = [load_image(p) for p in batch_paths]
        scores_list = model.predict_batch(images)

        for path, scores in zip(batch_paths, scores_list):
            row = {"filename": path.name, "filepath": str(path)}
            row.update(scores)
            all_rows.append(row)

    return pd.DataFrame(all_rows)


def run_model(
    model: BaseModel,
    image_dir: Path = DEFAULT_IMAGE_DIR,
    stim_info_csv: Path = DEFAULT_STIM_INFO,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    batch_size: int = 32,
    output_csv: Path | None = None,
) -> pd.DataFrame:
    """Run a model over all shared1000 images and save results to CSV.

    Parameters
    ----------
    model : BaseModel
        An instantiated (but not yet loaded) model wrapper.
    image_dir : Path
        Directory containing the shared1000 images.
    stim_info_csv : Path
        Path to nsd_stim_info.csv for cocoId lookup.
    output_dir : Path
        Directory for the output CSV.
    batch_size : int
        Number of images per forward pass.
    output_csv : Path, optional
        Override the default output path ({output_dir}/{model.name}_scores.csv).

    Returns
    -------
    pd.DataFrame
        The scores DataFrame that was written to CSV.
    """
    # Build manifest of images + IDs.
    manifest = build_image_manifest(image_dir, stim_info_csv)
    print(f"Found {len(manifest)} images.")

    # Load model.
    print(f"Loading {model.name} model on {model.device} ...")
    model.load()

    # Check for existing partial results (resumability).
    if output_csv is None:
        output_csv = output_dir / f"{model.name}_scores.csv"

    existing_ids: set[int] = set()
    all_rows: list[dict] = []
    if output_csv.exists():
        existing = pd.read_csv(output_csv)
        existing_ids = set(existing["nsdId"].tolist())
        all_rows = existing.to_dict("records")
        print(f"Resuming: {len(existing_ids)} images already processed.")

    # Filter to unprocessed images.
    todo = manifest[~manifest["nsdId"].isin(existing_ids)]
    if todo.empty:
        print("All images already processed.")
        return pd.read_csv(output_csv)

    # Process in batches.
    filepaths = todo["filepath"].tolist()
    meta_rows = todo.to_dict("records")

    for batch_start in tqdm(range(0, len(filepaths), batch_size), desc=model.name):
        batch_end = min(batch_start + batch_size, len(filepaths))
        batch_paths = filepaths[batch_start:batch_end]
        batch_meta = meta_rows[batch_start:batch_end]

        images = [load_image(Path(p)) for p in batch_paths]
        scores_list = model.predict_batch(images)

        for meta, scores in zip(batch_meta, scores_list):
            row = {
                "nsdId": meta["nsdId"],
                "cocoId": meta.get("cocoId"),
                "filename": meta["filename"],
            }
            row.update(scores)
            all_rows.append(row)

        # Flush after each batch for resumability.
        _save_csv(all_rows, output_csv, model.name)

    result = pd.read_csv(output_csv)
    print(f"Saved {len(result)} rows to {output_csv}")
    return result


def _save_csv(rows: list[dict], path: Path, model_name: str) -> None:
    """Write rows to CSV, keeping ID columns first."""
    df = pd.DataFrame(rows)
    # Ensure ID columns come first.
    id_cols = ["nsdId", "cocoId", "filename"]
    score_cols = [c for c in df.columns if c not in id_cols]
    df = df[id_cols + score_cols]
    df.to_csv(path, index=False)
