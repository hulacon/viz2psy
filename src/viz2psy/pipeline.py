"""Batch inference pipeline: load images, run model, return scores."""

from pathlib import Path

import pandas as pd
from tqdm import tqdm

from .models.base import BaseModel
from .utils import load_image


def score_images(
    model: BaseModel,
    image_paths: list[str | Path],
    batch_size: int = 32,
    quiet: bool = False,
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
    quiet : bool
        If True, suppress progress output.

    Returns
    -------
    pd.DataFrame
        One row per image with columns: filename and model scores.
    """
    image_paths = [Path(p) for p in image_paths]

    if not quiet:
        print(f"Loading {model.name} model on {model.device} ...")
    model.load()

    all_rows: list[dict] = []
    iterator = range(0, len(image_paths), batch_size)
    if not quiet:
        iterator = tqdm(iterator, desc=model.name)

    for batch_start in iterator:
        batch_paths = image_paths[batch_start : batch_start + batch_size]
        images = [load_image(p) for p in batch_paths]
        scores_list = model.predict_batch(images)

        for path, scores in zip(batch_paths, scores_list):
            row = {"filename": path.name}
            row.update(scores)
            all_rows.append(row)

    return pd.DataFrame(all_rows)
