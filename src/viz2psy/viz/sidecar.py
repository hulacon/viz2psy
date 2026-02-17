"""Sidecar metadata utilities for visualization.

Loads and parses .meta.json sidecar files to provide:
- Semantic labels for features (e.g., "Adoration" instead of column name)
- Index column detection (time, filename, image_idx)
- Model information for tooltips
- Image path/data resolution for all input types
"""

import json
import re
from pathlib import Path
from typing import Any, Union
import io

import numpy as np
import pandas as pd
from PIL import Image


class SidecarMetadata:
    """Parsed sidecar metadata with helper methods."""

    def __init__(self, data: dict[str, Any], path: Path | None = None):
        self.data = data
        self.path = path

    @property
    def index_column(self) -> str | None:
        """Get the index column name (time, filename, image_idx)."""
        return self.data.get("index_column")

    @property
    def input_type(self) -> str | None:
        """Get input type (video, image_folder, hdf5_brick)."""
        return self.data.get("input", {}).get("type")

    @property
    def models(self) -> dict[str, Any]:
        """Get model info dict."""
        return self.data.get("models", {})

    @property
    def feature_definitions(self) -> dict[str, Any]:
        """Get feature definitions dict."""
        return self.data.get("feature_definitions", {})

    def get_model_for_column(self, column: str) -> str | None:
        """Determine which model produced a column."""
        for model_name, model_info in self.models.items():
            features = model_info.get("features", {})

            # Check pattern-based columns (clip_000, gist_001, etc.)
            pattern = features.get("pattern")
            if pattern:
                # Convert pattern like "clip_{NNN}" to regex "clip_\d+"
                regex = pattern.replace("{NNN}", r"\d+").replace("{XX}", r"\d+").replace("{YY}", r"\d+")
                if re.match(f"^{regex}$", column):
                    return model_name

            # Check named columns
            columns = features.get("columns", [])
            if column in columns:
                return model_name

        return None

    def get_semantic_label(self, column: str) -> str:
        """Get semantic label for a column, or return column name if not found."""
        model_name = self.get_model_for_column(column)
        if not model_name:
            return column

        # For named columns, the column name IS the semantic label
        features = self.models[model_name].get("features", {})
        if "columns" in features:
            return column  # Already semantic (e.g., "Adoration", "memorability")

        # For pattern-based, extract index and lookup if definitions exist
        pattern = features.get("pattern", "")

        # Handle places scenes and sun attributes specially
        if column.startswith("places_"):
            defs = self.feature_definitions.get("places", {})
            if isinstance(defs, dict):
                scenes = defs.get("scenes", [])
                idx = self._extract_index(column, "places_")
                if idx is not None and idx < len(scenes):
                    return scenes[idx]

        if column.startswith("sunattr_"):
            defs = self.feature_definitions.get("places", {})
            if isinstance(defs, dict):
                attrs = defs.get("attributes", [])
                idx = self._extract_index(column, "sunattr_")
                if idx is not None and idx < len(attrs):
                    return attrs[idx]

        if column.startswith("yolo_"):
            defs = self.feature_definitions.get("yolo", {})
            if isinstance(defs, dict):
                objects = defs.get("objects", [])
                obj_name = column.replace("yolo_", "")
                if obj_name in objects:
                    return obj_name

        # For clip_, gist_, dinov2_, saliency_ - no semantic labels available
        return column

    def _extract_index(self, column: str, prefix: str) -> int | None:
        """Extract numeric index from column name."""
        try:
            suffix = column.replace(prefix, "")
            return int(suffix)
        except ValueError:
            return None

    def get_feature_labels(self, columns: list[str]) -> dict[str, str]:
        """Get semantic labels for a list of columns."""
        return {col: self.get_semantic_label(col) for col in columns}

    def get_model_summary(self) -> str:
        """Get a summary of models used."""
        parts = []
        for name, info in self.models.items():
            count = info.get("features", {}).get("count", "?")
            parts.append(f"{name}({count})")
        return ", ".join(parts)


def load_sidecar(csv_path: str | Path) -> SidecarMetadata | None:
    """Load sidecar metadata for a CSV file.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file. Will look for .meta.json sidecar.

    Returns
    -------
    SidecarMetadata or None
        Parsed metadata, or None if sidecar doesn't exist.
    """
    csv_path = Path(csv_path)

    # Sidecar is CSV path with .csv replaced by .meta.json
    sidecar_path = csv_path.with_suffix(".meta.json")

    if not sidecar_path.exists():
        return None

    try:
        with open(sidecar_path) as f:
            data = json.load(f)
        return SidecarMetadata(data, sidecar_path)
    except (json.JSONDecodeError, OSError):
        return None


def auto_detect_index_column(
    df,
    sidecar: SidecarMetadata | None = None,
) -> str | None:
    """Auto-detect the index/time column from sidecar or DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The data frame.
    sidecar : SidecarMetadata, optional
        Sidecar metadata if available.

    Returns
    -------
    str or None
        Name of detected index column.
    """
    # Prefer sidecar info
    if sidecar and sidecar.index_column:
        if sidecar.index_column in df.columns:
            return sidecar.index_column

    # Fall back to heuristics
    candidates = ["time", "filename", "image_idx", "frame", "index"]
    for col in candidates:
        if col in df.columns:
            return col

    return None


class ImagePathResolver:
    """Resolves image paths from CSV rows using multiple strategies.

    Resolution order:
    1. If row value is absolute path and exists → use it
    2. image_root + filename (if image_root provided)
    3. sidecar input.path + filename (if sidecar exists)
    4. CSV directory + filename
    5. Current working directory + filename

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file (used for relative resolution).
    sidecar : SidecarMetadata, optional
        Sidecar metadata with input.path info.
    image_root : Path, optional
        Explicit base directory for images (highest priority).
    filename_column : str
        Column name containing filenames. Default: "filename".
    """

    def __init__(
        self,
        csv_path: Path | str,
        sidecar: SidecarMetadata | None = None,
        image_root: Path | str | None = None,
        filename_column: str = "filename",
    ):
        self.csv_path = Path(csv_path)
        self.csv_dir = self.csv_path.parent
        self.sidecar = sidecar
        self.image_root = Path(image_root) if image_root else None
        self.filename_column = filename_column

        # Extract input path from sidecar
        self.sidecar_input_path = None
        if sidecar:
            input_info = sidecar.data.get("input", {})
            input_path = input_info.get("path")
            if input_path:
                self.sidecar_input_path = Path(input_path)

        # Cache for resolved paths
        self._cache: dict[str, Path | None] = {}

    def resolve(self, filename_or_row: str | pd.Series) -> Path | None:
        """Resolve a filename or CSV row to an image path.

        Parameters
        ----------
        filename_or_row : str or pd.Series
            Either a filename string or a DataFrame row.

        Returns
        -------
        Path or None
            Resolved path if found, None otherwise.
        """
        if isinstance(filename_or_row, pd.Series):
            if self.filename_column not in filename_or_row.index:
                return None
            filename = filename_or_row[self.filename_column]
        else:
            filename = filename_or_row

        if pd.isna(filename) or not filename:
            return None

        filename = str(filename)

        # Check cache
        if filename in self._cache:
            return self._cache[filename]

        resolved = self._resolve_impl(filename)
        self._cache[filename] = resolved
        return resolved

    def _resolve_impl(self, filename: str) -> Path | None:
        """Internal resolution logic."""
        # Strategy 1: Already absolute and exists
        path = Path(filename)
        if path.is_absolute() and path.exists():
            return path

        basename = Path(filename).name

        # Strategy 2: image_root + filename (explicit override)
        if self.image_root:
            candidate = self.image_root / filename
            if candidate.exists():
                return candidate
            # Also try just the basename in case filename has subdirs
            candidate = self.image_root / basename
            if candidate.exists():
                return candidate

        # Strategy 3: sidecar input.path + filename
        if self.sidecar_input_path:
            candidate = self.sidecar_input_path / filename
            if candidate.exists():
                return candidate
            candidate = self.sidecar_input_path / basename
            if candidate.exists():
                return candidate

        # Strategy 4: CSV directory + filename
        candidate = self.csv_dir / filename
        if candidate.exists():
            return candidate
        candidate = self.csv_dir / basename
        if candidate.exists():
            return candidate

        # Strategy 5: Infer directory from CSV name pattern
        # e.g., "image_folder_scores.csv" -> look in "image_folder/"
        csv_stem = self.csv_path.stem
        for suffix in ["_scores", "_features", "_output", ""]:
            if csv_stem.endswith(suffix):
                inferred_dir = csv_stem[: -len(suffix)] if suffix else csv_stem
                candidate = self.csv_dir / inferred_dir / filename
                if candidate.exists():
                    return candidate
                candidate = self.csv_dir / inferred_dir / basename
                if candidate.exists():
                    return candidate

        # Strategy 6: Current working directory
        candidate = Path.cwd() / filename
        if candidate.exists():
            return candidate

        return None

    def resolve_row_idx(self, df: pd.DataFrame, row_idx: int) -> Path | None:
        """Resolve image path for a specific row index.

        Parameters
        ----------
        df : pd.DataFrame
            The scores DataFrame.
        row_idx : int
            Row index.

        Returns
        -------
        Path or None
            Resolved path if found.
        """
        if row_idx < 0 or row_idx >= len(df):
            return None
        return self.resolve(df.iloc[row_idx])

    def get_search_paths(self) -> list[Path]:
        """Get the list of directories that will be searched (in order).

        Useful for debugging path resolution issues.
        """
        paths = []
        if self.image_root:
            paths.append(self.image_root)
        if self.sidecar_input_path:
            paths.append(self.sidecar_input_path)
        paths.append(self.csv_dir)

        # Inferred directories from CSV name
        csv_stem = self.csv_path.stem
        for suffix in ["_scores", "_features", "_output", ""]:
            if csv_stem.endswith(suffix):
                inferred_dir = csv_stem[: -len(suffix)] if suffix else csv_stem
                inferred_path = self.csv_dir / inferred_dir
                if inferred_path.exists() and inferred_path.is_dir():
                    paths.append(inferred_path)

        paths.append(Path.cwd())
        return paths


def create_image_resolver(
    csv_path: str | Path,
    image_root: str | Path | None = None,
) -> ImagePathResolver:
    """Create an ImagePathResolver with automatic sidecar loading.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.
    image_root : str or Path, optional
        Explicit image root directory (overrides sidecar).

    Returns
    -------
    ImagePathResolver
    """
    csv_path = Path(csv_path)
    sidecar = load_sidecar(csv_path)
    return ImagePathResolver(
        csv_path=csv_path,
        sidecar=sidecar,
        image_root=image_root,
    )


class UnifiedImageResolver:
    """Resolves images from any input type: image folder, video, or HDF5.

    This resolver handles all three input types and returns either:
    - A Path to an image file (for image folders)
    - A PIL Image (for video frames or HDF5 images)

    Parameters
    ----------
    csv_path : Path
        Path to the CSV file.
    sidecar : SidecarMetadata, optional
        Sidecar metadata with input info.
    image_root : Path, optional
        Override for image directory (image_folder only).
    video_path : Path, optional
        Override for video file path.
    hdf5_path : Path, optional
        Override for HDF5 file path.
    hdf5_dataset : str, optional
        Override for HDF5 dataset name.
    """

    def __init__(
        self,
        csv_path: Path | str,
        sidecar: SidecarMetadata | None = None,
        image_root: Path | str | None = None,
        video_path: Path | str | None = None,
        hdf5_path: Path | str | None = None,
        hdf5_dataset: str | None = None,
    ):
        self.csv_path = Path(csv_path)
        self.sidecar = sidecar
        self.image_root = Path(image_root) if image_root else None
        self.video_path = Path(video_path) if video_path else None
        self.hdf5_path = Path(hdf5_path) if hdf5_path else None
        self.hdf5_dataset = hdf5_dataset

        # Detect input type from sidecar or CLI overrides
        self.input_type = None
        if sidecar:
            self.input_type = sidecar.input_type

        # Override input type based on CLI arguments
        if video_path:
            self.input_type = "video"
        elif hdf5_path:
            self.input_type = "hdf5_brick"

        # Extract paths from sidecar if available
        self._sidecar_paths = None  # For image_folder
        self._sidecar_video_path = None
        self._sidecar_hdf5_path = None
        self._sidecar_hdf5_dataset = None
        self._sidecar_hdf5_indices = None
        self._frame_interval = None

        if sidecar:
            input_info = sidecar.data.get("input", {})

            if self.input_type == "image_folder":
                self._sidecar_paths = input_info.get("paths", [])

            elif self.input_type == "video":
                path = input_info.get("path")
                if path:
                    self._sidecar_video_path = Path(path)
                self._frame_interval = input_info.get("frame_interval_sec")

            elif self.input_type == "hdf5_brick":
                path = input_info.get("path")
                if path:
                    self._sidecar_hdf5_path = Path(path)
                self._sidecar_hdf5_dataset = input_info.get("dataset", "stimuli")
                self._sidecar_hdf5_indices = input_info.get("indices")

        # Create fallback file resolver for image_folder
        self._file_resolver = ImagePathResolver(
            csv_path=self.csv_path,
            sidecar=sidecar,
            image_root=image_root,
        )

        # Cache for video/HDF5 handles
        self._video_capture = None
        self._hdf5_file = None
        self._hdf5_data = None

    @property
    def detected_input_type(self) -> str | None:
        """Return the detected input type."""
        return self.input_type

    def resolve(
        self,
        df: pd.DataFrame,
        row_idx: int,
    ) -> Path | Image.Image | None:
        """Resolve image for a given row.

        Parameters
        ----------
        df : pd.DataFrame
            The scores DataFrame.
        row_idx : int
            Row index.

        Returns
        -------
        Path or PIL.Image or None
            For image_folder: returns Path
            For video/HDF5: returns PIL.Image
            Returns None if resolution fails.
        """
        if row_idx < 0 or row_idx >= len(df):
            return None

        row = df.iloc[row_idx]

        # Route to appropriate resolver
        if self.input_type == "image_folder" or self.input_type is None:
            return self._resolve_image_folder(row, row_idx)
        elif self.input_type == "video":
            return self._resolve_video(row, row_idx)
        elif self.input_type == "hdf5_brick":
            return self._resolve_hdf5(row, row_idx)
        else:
            # Unknown type, try file resolution
            return self._resolve_image_folder(row, row_idx)

    def _resolve_image_folder(
        self,
        row: pd.Series,
        row_idx: int,
    ) -> Path | None:
        """Resolve image from folder."""
        # Try sidecar paths first (direct mapping by index)
        if self._sidecar_paths and row_idx < len(self._sidecar_paths):
            path = Path(self._sidecar_paths[row_idx])
            if path.exists():
                return path

        # Fall back to filename-based resolution
        return self._file_resolver.resolve(row)

    def _resolve_video(
        self,
        row: pd.Series,
        row_idx: int,
    ) -> Image.Image | None:
        """Extract frame from video at given timestamp."""
        # Get video path
        video_path = self.video_path or self._sidecar_video_path
        if not video_path or not video_path.exists():
            return None

        # Get timestamp from row
        time_col = "time"
        if time_col not in row.index:
            return None

        timestamp = row[time_col]

        try:
            import cv2
        except ImportError:
            print("Warning: opencv-python required for video frame extraction")
            return None

        # Open video (lazy, cached)
        if self._video_capture is None:
            self._video_capture = cv2.VideoCapture(str(video_path))

        cap = self._video_capture
        if not cap.isOpened():
            return None

        # Seek to timestamp (in milliseconds)
        cap.set(cv2.CAP_PROP_POS_MSEC, timestamp * 1000)

        # Read frame
        ret, frame = cap.read()
        if not ret:
            return None

        # Convert BGR to RGB and return as PIL Image
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return Image.fromarray(frame_rgb)

    def _resolve_hdf5(
        self,
        row: pd.Series,
        row_idx: int,
    ) -> Image.Image | None:
        """Extract image from HDF5 dataset."""
        # Get HDF5 path
        hdf5_path = self.hdf5_path or self._sidecar_hdf5_path
        if not hdf5_path or not hdf5_path.exists():
            return None

        # Get dataset name
        dataset_name = self.hdf5_dataset or self._sidecar_hdf5_dataset or "stimuli"

        # Get image index from row
        idx_col = "image_idx"
        if idx_col in row.index:
            image_idx = int(row[idx_col])
        else:
            # Use row_idx as fallback
            image_idx = row_idx

        # Adjust for sidecar indices if present
        if self._sidecar_hdf5_indices:
            start_idx = self._sidecar_hdf5_indices[0]
            # image_idx in CSV is relative to start_idx
            actual_idx = image_idx
        else:
            actual_idx = image_idx

        try:
            import h5py
        except ImportError:
            print("Warning: h5py required for HDF5 image extraction")
            return None

        # Open HDF5 file (lazy, cached)
        if self._hdf5_file is None:
            self._hdf5_file = h5py.File(str(hdf5_path), "r")
            self._hdf5_data = self._hdf5_file[dataset_name]

        # Extract image
        try:
            img_data = self._hdf5_data[actual_idx]

            # Handle different formats
            if img_data.ndim == 2:
                # Grayscale
                return Image.fromarray(img_data, mode="L")
            elif img_data.ndim == 3:
                if img_data.shape[2] == 3:
                    return Image.fromarray(img_data, mode="RGB")
                elif img_data.shape[2] == 4:
                    return Image.fromarray(img_data, mode="RGBA")

            return None
        except (IndexError, KeyError):
            return None

    def close(self):
        """Close any open file handles."""
        if self._video_capture is not None:
            self._video_capture.release()
            self._video_capture = None

        if self._hdf5_file is not None:
            self._hdf5_file.close()
            self._hdf5_file = None
            self._hdf5_data = None

    def __del__(self):
        self.close()


def create_unified_resolver(
    csv_path: str | Path,
    image_root: str | Path | None = None,
    video_path: str | Path | None = None,
    hdf5_path: str | Path | None = None,
    hdf5_dataset: str | None = None,
) -> UnifiedImageResolver:
    """Create a UnifiedImageResolver with automatic sidecar loading.

    Parameters
    ----------
    csv_path : str or Path
        Path to the CSV file.
    image_root : str or Path, optional
        Override for image directory (image_folder only).
    video_path : str or Path, optional
        Override for video file path.
    hdf5_path : str or Path, optional
        Override for HDF5 file path.
    hdf5_dataset : str, optional
        Override for HDF5 dataset name.

    Returns
    -------
    UnifiedImageResolver
    """
    csv_path = Path(csv_path)
    sidecar = load_sidecar(csv_path)
    return UnifiedImageResolver(
        csv_path=csv_path,
        sidecar=sidecar,
        image_root=image_root,
        video_path=video_path,
        hdf5_path=hdf5_path,
        hdf5_dataset=hdf5_dataset,
    )
