"""Time series visualization for video scores."""

import fnmatch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def get_feature_columns(
    df: pd.DataFrame,
    patterns: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[str]:
    """Get feature columns matching patterns.

    Parameters
    ----------
    df : pd.DataFrame
    patterns : list of str, optional
        Glob patterns to match (e.g., ["memorability", "clip_*"]).
        If None, returns all numeric columns except common ID columns.
    exclude : list of str, optional
        Columns to exclude.

    Returns
    -------
    list of str
    """
    exclude = exclude or ["time", "index", "filename", "filepath"]

    if patterns is None:
        # All numeric columns except excluded
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in exclude]

    # Match patterns
    all_cols = df.columns.tolist()
    matched = set()
    for pattern in patterns:
        for col in all_cols:
            if fnmatch.fnmatch(col, pattern):
                matched.add(col)

    return [c for c in matched if c not in exclude]


def plot_timeseries(
    df: pd.DataFrame,
    features: list[str] | None = None,
    time_col: str = "time",
    figsize: tuple[int, int] | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot feature values over time.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a time column and feature columns.
    features : list of str, optional
        Features to plot. Supports glob patterns (e.g., "clip_*").
        If None, plots all scalar features (not high-dim embeddings).
    time_col : str
        Name of the time column (default: "time").
    figsize : tuple, optional
        Figure size (width, height) in inches.
    title : str, optional
        Plot title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame.")

    # Get feature columns
    feature_cols = get_feature_columns(df, features)

    # For default (no patterns), exclude high-dimensional embeddings
    if features is None:
        # Exclude columns that look like embeddings (clip_*, gist_*, dinov2_*, saliency_*, etc.)
        embedding_prefixes = ["clip_", "gist_", "dinov2_", "saliency_", "places_", "sunattr_"]
        feature_cols = [
            c for c in feature_cols
            if not any(c.startswith(p) for p in embedding_prefixes)
        ]

    if not feature_cols:
        raise ValueError("No features found to plot.")

    # Determine layout
    n_features = len(feature_cols)
    if n_features == 1:
        nrows, ncols = 1, 1
    elif n_features <= 3:
        nrows, ncols = n_features, 1
    elif n_features <= 6:
        nrows, ncols = (n_features + 1) // 2, 2
    else:
        nrows, ncols = (n_features + 2) // 3, 3

    if figsize is None:
        figsize = (4 * ncols, 2.5 * nrows)

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, squeeze=False)
    axes = axes.flatten()

    time = df[time_col].values

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        values = df[col].values
        ax.plot(time, values, linewidth=1)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(col)
        ax.set_title(col)
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for i in range(n_features, len(axes)):
        axes[i].set_visible(False)

    if title:
        fig.suptitle(title, fontsize=12)

    fig.tight_layout()
    return fig
