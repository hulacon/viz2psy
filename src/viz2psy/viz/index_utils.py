"""Index column detection and axis formatting utilities.

Handles detection of index columns (time, filename, image_idx) and
provides formatting utilities for x-axis labels across visualizations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .sidecar import SidecarMetadata


def detect_index_column(
    df: pd.DataFrame,
    sidecar: "SidecarMetadata | None" = None,
) -> tuple[str | None, str]:
    """Detect the index column and its type from DataFrame and sidecar.

    Parameters
    ----------
    df : pd.DataFrame
        The data frame to analyze.
    sidecar : SidecarMetadata, optional
        Sidecar metadata for authoritative index info.

    Returns
    -------
    column_name : str or None
        Name of the detected index column, or None if not found.
    index_type : str
        Type of index: "time", "integer", or "ordinal".
        - "time": Continuous time values (seconds)
        - "integer": Integer indices (image_idx, frame)
        - "ordinal": Categorical/filename strings

    Examples
    --------
    >>> col, idx_type = detect_index_column(df, sidecar)
    >>> print(f"Using {col} as {idx_type} index")
    Using time as time index
    """
    # Prefer sidecar info
    if sidecar and sidecar.index_column and sidecar.index_column in df.columns:
        col = sidecar.index_column
        idx_type = _classify_index_type(df, col, sidecar)
        return col, idx_type

    # Fall back to heuristics
    candidates = [
        ("time", "time"),
        ("filename", "ordinal"),
        ("image_idx", "integer"),
        ("frame", "integer"),
        ("index", "integer"),
    ]

    for col, default_type in candidates:
        if col in df.columns:
            idx_type = _classify_index_type(df, col, sidecar, default_type)
            return col, idx_type

    return None, "integer"


def _classify_index_type(
    df: pd.DataFrame,
    col: str,
    sidecar: "SidecarMetadata | None" = None,
    default: str = "integer",
) -> str:
    """Classify the type of an index column."""
    if col == "time":
        return "time"
    elif col == "filename":
        return "ordinal"
    elif col in ("image_idx", "frame", "index"):
        return "integer"

    # Check if sidecar indicates input type
    if sidecar:
        input_type = sidecar.input_type
        if input_type == "video":
            return "time"
        elif input_type == "hdf5_brick":
            return "integer"
        elif input_type == "image_folder":
            return "ordinal"

    # Infer from data type
    dtype = df[col].dtype
    if pd.api.types.is_float_dtype(dtype):
        return "time"
    elif pd.api.types.is_integer_dtype(dtype):
        return "integer"
    else:
        return "ordinal"


def prepare_index_values(
    df: pd.DataFrame,
    index_col: str | None,
    index_type: str,
) -> tuple[np.ndarray, dict]:
    """Prepare x-axis values and formatting info.

    Parameters
    ----------
    df : pd.DataFrame
        The data frame.
    index_col : str or None
        The index column name.
    index_type : str
        The index type ("time", "integer", or "ordinal").

    Returns
    -------
    x_values : np.ndarray
        Values for x-axis.
    format_info : dict
        Formatting information with keys:
        - "xlabel": X-axis label string
        - "tickmode": Plotly tickmode ("linear", "array", etc.)
        - "tickvals": Tick positions (for ordinal)
        - "ticktext": Tick labels (for ordinal)
        - "tickformat": Format string for ticks
        - "hoverformat": Format string for hover

    Examples
    --------
    >>> x_vals, fmt = prepare_index_values(df, "time", "time")
    >>> ax.set_xlabel(fmt["xlabel"])
    """
    format_info = {}

    if index_col is None:
        # Use row numbers
        x_values = np.arange(len(df))
        format_info["xlabel"] = "Index"
        format_info["tickmode"] = "linear"
        format_info["hoverformat"] = "d"
        return x_values, format_info

    x_values = df[index_col].values

    if index_type == "time":
        format_info["xlabel"] = "Time (s)"
        format_info["tickmode"] = "linear"
        format_info["tickformat"] = ".1f"
        format_info["hoverformat"] = ".2f"

    elif index_type == "integer":
        format_info["xlabel"] = index_col.replace("_", " ").title()
        format_info["tickmode"] = "linear"
        format_info["hoverformat"] = "d"

    elif index_type == "ordinal":
        # For ordinal data (filenames), use numeric x-axis with text labels
        x_values = np.arange(len(df))
        labels = df[index_col].astype(str).values

        # Truncate long filenames for display
        max_label_len = 25
        truncated = [
            (s[:max_label_len] + "...") if len(s) > max_label_len else s
            for s in labels
        ]

        format_info["xlabel"] = index_col.replace("_", " ").title()
        format_info["tickmode"] = "array"
        format_info["tickvals"] = x_values
        format_info["ticktext"] = truncated
        format_info["original_labels"] = labels  # Full labels for hover
        format_info["hoverformat"] = ""

        # For many points, subsample tick labels
        if len(labels) > 20:
            step = max(1, len(labels) // 10)
            format_info["tickvals"] = x_values[::step]
            format_info["ticktext"] = [truncated[i] for i in range(0, len(truncated), step)]

    return x_values, format_info


def is_video_data(
    df: pd.DataFrame,
    sidecar: "SidecarMetadata | None" = None,
) -> bool:
    """Check if the data appears to be from video input.

    Parameters
    ----------
    df : pd.DataFrame
        The data frame.
    sidecar : SidecarMetadata, optional
        Sidecar metadata.

    Returns
    -------
    bool
        True if data is from video input.
    """
    # Check sidecar first
    if sidecar:
        return sidecar.input_type == "video"

    # Heuristic: has "time" column with float values
    if "time" in df.columns:
        if pd.api.types.is_float_dtype(df["time"].dtype):
            return True

    return False
