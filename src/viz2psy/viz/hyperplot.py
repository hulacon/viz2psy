"""HyperTools visualization wrapper.

Provides 3D scatter plots, trajectory animations, and clustering
using the HyperTools library as an optional dependency.
"""

from __future__ import annotations

import fnmatch
import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .sidecar import SidecarMetadata


def get_feature_columns(
    df: pd.DataFrame,
    patterns: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[str]:
    """Get feature columns matching patterns."""
    exclude = exclude or ["time", "index", "filename", "filepath", "image_idx"]

    if patterns is None:
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        return [c for c in numeric_cols if c not in exclude]

    all_cols = df.columns.tolist()
    matched = set()
    for pattern in patterns:
        for col in all_cols:
            if fnmatch.fnmatch(col, pattern):
                matched.add(col)

    return sorted([c for c in matched if c not in exclude])


def plot_hypertools(
    df: pd.DataFrame,
    features: list[str] | None = None,
    ndims: int = 3,
    reduce: str = "pca",
    n_clusters: int | None = None,
    group: str | None = None,
    animate: bool = False,
    title: str | None = None,
    legend: bool = True,
    sidecar: "SidecarMetadata | None" = None,
) -> Any:
    """Create HyperTools visualization.

    Wraps hyp.plot() with viz2psy feature selection and sidecar integration.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature columns.
    features : list of str, optional
        Feature columns or glob patterns to include.
        If None, uses all numeric columns.
    ndims : int
        Number of dimensions for visualization (2 or 3).
    reduce : str
        Reduction method: "pca", "umap", or "ppca" (probabilistic PCA).
    n_clusters : int, optional
        If provided, auto-cluster data into N groups and color accordingly.
    group : str, optional
        Column name for categorical coloring (overrides n_clusters).
    animate : bool
        If True, create trajectory animation. Points are connected
        in order and animated over time. Best for sequential data.
    title : str, optional
        Plot title.
    legend : bool
        Whether to show legend (default: True).
    sidecar : SidecarMetadata, optional
        Sidecar metadata for model info.

    Returns
    -------
    fig : hypertools figure
        The HyperTools figure object.

    Raises
    ------
    ImportError
        If hypertools is not installed.
    ValueError
        If n_clusters > n_samples or invalid parameters.

    Examples
    --------
    >>> # 3D scatter with clustering
    >>> fig = plot_hypertools(df, features=["clip_*"], n_clusters=5)

    >>> # Trajectory animation for video data
    >>> fig = plot_hypertools(df, animate=True)

    >>> # Color by categorical column
    >>> fig = plot_hypertools(df, group="condition")
    """
    try:
        import hypertools as hyp
    except ImportError:
        raise ImportError(
            "hypertools is required for this visualization. "
            "Install with: pip install viz2psy[hypertools]"
        )

    # Select feature columns
    cols = get_feature_columns(df, features)

    if len(cols) < 2:
        raise ValueError(f"Need at least 2 features, found {len(cols)}.")

    X = df[cols].values
    n_samples = len(X)

    # Handle NaN values
    if np.any(np.isnan(X)):
        warnings.warn("NaN values found, filling with column means.")
        col_means = np.nanmean(X, axis=0)
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            X[mask, i] = col_means[i]

    # Validate n_clusters
    if n_clusters is not None:
        if n_clusters > n_samples:
            warnings.warn(
                f"n_clusters ({n_clusters}) > n_samples ({n_samples}), "
                f"reducing to {n_samples}."
            )
            n_clusters = n_samples
        if n_clusters < 2:
            warnings.warn("n_clusters must be >= 2, disabling clustering.")
            n_clusters = None

    # Validate animation
    if animate:
        # Check if data appears sequential
        index_col = None
        for candidate in ["time", "image_idx", "frame", "index"]:
            if candidate in df.columns:
                index_col = candidate
                break

        if index_col is None:
            warnings.warn(
                "animate=True but no sequential index found. "
                "Animation may not be meaningful. "
                "Data will be connected in row order."
            )

    # Build hyp.plot kwargs
    kwargs: dict[str, Any] = {
        "ndims": ndims,
        "reduce": reduce,
    }

    # Handle grouping/coloring
    if group and group in df.columns:
        kwargs["group"] = df[group].tolist()
    elif n_clusters:
        kwargs["n_clusters"] = n_clusters

    if animate:
        kwargs["animate"] = True

    if not legend:
        kwargs["legend"] = False

    # Generate title
    if title is None:
        parts = [f"{ndims}D {reduce.upper()}"]
        if n_clusters:
            parts.append(f"{n_clusters} clusters")
        if animate:
            parts.append("animated")
        parts.append(f"({len(cols)} features)")
        if sidecar:
            model_summary = sidecar.get_model_summary()
            if model_summary:
                parts.append(f"- {model_summary}")
        title = " ".join(parts)

    kwargs["title"] = title

    # Create plot
    return hyp.plot(X, **kwargs)


def save_hypertools_figure(
    fig: Any,
    output_path: str,
) -> None:
    """Save a HyperTools figure to file.

    Parameters
    ----------
    fig : hypertools figure
        The figure returned by plot_hypertools.
    output_path : str
        Output file path. For static images, use .png/.pdf.
        For interactive/animated, use .html.
    """
    try:
        import hypertools as hyp
    except ImportError:
        raise ImportError(
            "hypertools is required. "
            "Install with: pip install viz2psy[hypertools]"
        )

    # HyperTools figures are matplotlib figures for static,
    # or can be saved via the figure's save method
    if hasattr(fig, "savefig"):
        fig.savefig(output_path, bbox_inches="tight", dpi=150)
    elif hasattr(fig, "save"):
        fig.save(output_path)
    else:
        # Try to get the matplotlib figure
        import matplotlib.pyplot as plt
        plt.savefig(output_path, bbox_inches="tight", dpi=150)
