"""Interactive scatter plot with dimensionality reduction using Plotly.

Provides interactive 2D scatter plots with:
- Multiple DR methods (PCA, UMAP, t-SNE)
- Point selection and highlighting
- Hover tooltips with feature values
- Automatic sampling for large datasets
- Sidecar metadata integration for semantic labels
"""

from __future__ import annotations

import fnmatch
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .base import configure_theme

if TYPE_CHECKING:
    from ..sidecar import SidecarMetadata


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


def _compute_projection(
    X: np.ndarray,
    method: str = "pca",
) -> tuple[np.ndarray, dict]:
    """Compute 2D projection of feature matrix.

    Returns
    -------
    X_2d : np.ndarray
        2D coordinates (n_samples, 2).
    info : dict
        Additional info (axis labels, etc.).
    """
    # Handle NaN values
    if np.any(np.isnan(X)):
        col_means = np.nanmean(X, axis=0)
        for i in range(X.shape[1]):
            mask = np.isnan(X[:, i])
            X[mask, i] = col_means[i]

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if method == "pca":
        projector = PCA(n_components=2, random_state=42)
        X_2d = projector.fit_transform(X_scaled)
        var_explained = projector.explained_variance_ratio_
        info = {
            "xlabel": f"PC1 ({var_explained[0]:.1%} var)",
            "ylabel": f"PC2 ({var_explained[1]:.1%} var)",
        }
    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError(
                "umap-learn is required for UMAP projection. "
                "Install with: pip install umap-learn"
            )
        projector = umap.UMAP(n_components=2, random_state=42)
        X_2d = projector.fit_transform(X_scaled)
        info = {"xlabel": "UMAP1", "ylabel": "UMAP2"}
    elif method == "tsne":
        from sklearn.manifold import TSNE

        perplexity = min(30, len(X) - 1)
        projector = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_2d = projector.fit_transform(X_scaled)
        info = {"xlabel": "t-SNE1", "ylabel": "t-SNE2"}
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca', 'umap', or 'tsne'.")

    return X_2d, info


def plot_scatter_interactive(
    df: pd.DataFrame,
    features: list[str] | None = None,
    method: str = "pca",
    color_by: str | None = None,
    hover_data: list[str] | None = None,
    max_points: int = 5000,
    point_size: int = 8,
    title: str | None = None,
    width: int = 700,
    height: int = 500,
    sidecar: SidecarMetadata | None = None,
) -> go.Figure:
    """Create interactive scatter plot with dimensionality reduction.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature columns.
    features : list of str, optional
        Feature columns or glob patterns to project.
    method : str
        Projection method: "pca", "umap", or "tsne".
    color_by : str, optional
        Column to use for point coloring.
    hover_data : list of str, optional
        Additional columns to show in hover tooltip.
    max_points : int
        Maximum points to display. Samples if exceeded.
    point_size : int
        Size of scatter points.
    title : str, optional
        Chart title.
    width : int
        Chart width in pixels.
    height : int
        Chart height in pixels.
    sidecar : SidecarMetadata, optional
        Sidecar metadata for semantic labels and model info.

    Returns
    -------
    go.Figure
        Interactive Plotly figure.
    """
    configure_theme()

    # Sample if too large
    original_len = len(df)
    if len(df) > max_points:
        warnings.warn(
            f"Dataset has {len(df)} points, sampling to {max_points}. "
            f"Set max_points higher to include all points."
        )
        df = df.sample(n=max_points, random_state=42).reset_index(drop=True)

    # Get feature columns
    feature_cols = get_feature_columns(df, features)
    if len(feature_cols) < 2:
        raise ValueError(
            f"Need at least 2 features for projection, found {len(feature_cols)}."
        )

    # Compute projection
    X = df[feature_cols].values
    X_2d, info = _compute_projection(X, method=method)

    # Build plot DataFrame
    plot_df = pd.DataFrame({
        "_x": X_2d[:, 0],
        "_y": X_2d[:, 1],
    })

    # Detect index column from sidecar or fallback
    index_col = None
    if sidecar and sidecar.index_column and sidecar.index_column in df.columns:
        index_col = sidecar.index_column
    else:
        for candidate in ["filename", "time", "image_idx", "index"]:
            if candidate in df.columns:
                index_col = candidate
                break

    if index_col:
        plot_df[index_col] = df[index_col].values

    # Add color column with semantic label
    color_label = None
    if color_by and color_by in df.columns:
        plot_df[color_by] = df[color_by].values
        color_label = sidecar.get_semantic_label(color_by) if sidecar else color_by

    # Add hover data columns
    hover_cols = []
    if hover_data:
        for col in hover_data:
            if col in df.columns and col not in plot_df.columns:
                plot_df[col] = df[col].values
                hover_cols.append(col)

    # Build hover template
    hover_template = []
    if index_col:
        if index_col == "time":
            hover_template.append(f"Time: %{{customdata[0]:.2f}}s")
        elif index_col == "filename":
            hover_template.append("File: %{customdata[0]}")
        else:
            hover_template.append(f"{index_col}: %{{customdata[0]}}")

    if color_by and color_by in plot_df.columns:
        hover_template.append(f"{color_label}: %{{marker.color:.3f}}")

    hover_template.append(f"{info['xlabel']}: %{{x:.2f}}")
    hover_template.append(f"{info['ylabel']}: %{{y:.2f}}")

    # Add extra hover data
    for i, col in enumerate(hover_cols):
        label = sidecar.get_semantic_label(col) if sidecar else col
        hover_template.append(f"{label}: %{{customdata[{i + 1 if index_col else i}]:.3f}}")

    hover_template.append("<extra></extra>")

    # Build customdata array
    customdata_cols = []
    if index_col:
        customdata_cols.append(index_col)
    customdata_cols.extend(hover_cols)

    customdata = plot_df[customdata_cols].values if customdata_cols else None

    # Create figure
    if color_by and color_by in plot_df.columns:
        fig = go.Figure(data=go.Scatter(
            x=plot_df["_x"],
            y=plot_df["_y"],
            mode="markers",
            marker=dict(
                size=point_size,
                color=plot_df[color_by],
                colorscale="Viridis",
                colorbar=dict(title=color_label),
                opacity=0.7,
            ),
            customdata=customdata,
            hovertemplate="<br>".join(hover_template),
        ))
    else:
        fig = go.Figure(data=go.Scatter(
            x=plot_df["_x"],
            y=plot_df["_y"],
            mode="markers",
            marker=dict(
                size=point_size,
                color="steelblue",
                opacity=0.7,
            ),
            customdata=customdata,
            hovertemplate="<br>".join(hover_template),
        ))

    # Generate title
    if title is None:
        title = f"{method.upper()} Projection ({len(feature_cols)} features)"
        if sidecar:
            model_summary = sidecar.get_model_summary()
            if model_summary:
                title = f"{method.upper()} Projection - {model_summary}"
        if original_len > max_points:
            title += f" [sampled {max_points}/{original_len}]"

    fig.update_layout(
        title=title,
        xaxis_title=info["xlabel"],
        yaxis_title=info["ylabel"],
        width=width,
        height=height,
    )

    return fig


def _find_image_column(
    df: pd.DataFrame,
    image_col: str | None = None,
) -> str | None:
    """Find column containing image paths."""
    if image_col and image_col in df.columns:
        return image_col

    candidates = ["filename", "filepath", "image_path", "path", "file"]
    for col in candidates:
        if col in df.columns:
            return col

    return None
