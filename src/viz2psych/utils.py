"""Shared utilities: image loading, filename parsing, ID mapping."""

import re
from pathlib import Path

import pandas as pd
from PIL import Image

# Regex for shared1000 filenames: shared{NNNN}_nsd{NNNNN}.png
_FILENAME_RE = re.compile(r"^shared(\d+)_nsd(\d+)\.png$")


def parse_filename(filename: str) -> tuple[int, int]:
    """Extract shared index and 0-based nsdId from an image filename.

    The filename nsd ID is 1-based; this returns the 0-based version
    consistent with the metadata CSVs.
    """
    m = _FILENAME_RE.match(filename)
    if m is None:
        raise ValueError(f"Filename does not match expected pattern: {filename}")
    shared_idx = int(m.group(1))
    nsd_id_0based = int(m.group(2)) - 1  # convert 1-based -> 0-based
    return shared_idx, nsd_id_0based


def load_image(path: Path) -> Image.Image:
    """Load an image as RGB PIL Image."""
    return Image.open(path).convert("RGB")


def build_image_manifest(
    image_dir: Path,
    stim_info_csv: Path | None = None,
) -> pd.DataFrame:
    """Build a DataFrame of all images with their IDs.

    Returns a DataFrame with columns:
        filename, shared_idx, nsdId (0-based), cocoId (if stim_info provided)
    sorted by shared_idx.
    """
    image_dir = Path(image_dir)
    rows = []
    for p in sorted(image_dir.glob("shared*_nsd*.png")):
        shared_idx, nsd_id = parse_filename(p.name)
        rows.append({
            "filename": p.name,
            "filepath": str(p),
            "shared_idx": shared_idx,
            "nsdId": nsd_id,
        })

    df = pd.DataFrame(rows)

    if stim_info_csv is not None:
        stim = pd.read_csv(stim_info_csv)
        # nsd_stim_info.csv uses 'nsdId' as 0-based index
        coco_map = stim.set_index("nsdId")["cocoId"].to_dict()
        df["cocoId"] = df["nsdId"].map(coco_map)

    return df.sort_values("shared_idx").reset_index(drop=True)
