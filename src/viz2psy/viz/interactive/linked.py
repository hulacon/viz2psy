"""Linked scatter and timeseries dashboard using Plotly.

Provides coordinated views where:
- Scatter plot shows DR projection
- Timeseries shows feature evolution
- Shared hover highlighting across views
- Sidecar metadata integration for semantic labels
"""

from __future__ import annotations

import fnmatch
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .base import configure_theme
from ..projection import compute_projection, PROJECTION_METHODS

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


def create_linked_explorer(
    df: pd.DataFrame,
    scatter_features: list[str],
    timeseries_features: list[str] | None = None,
    method: str = "pca",
    time_col: str = "time",
    color_by: str | None = None,
    max_points: int = 5000,
    title: str | None = None,
    width: int = 1000,
    height: int = 700,
    sidecar: SidecarMetadata | None = None,
) -> go.Figure:
    """Create linked scatter + timeseries explorer.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature columns and a time column.
    scatter_features : list of str
        Features for scatter projection (supports glob patterns).
    timeseries_features : list of str, optional
        Features to show in timeseries. If None, uses color_by or
        first scalar features found.
    method : str
        Projection method: "pca", "umap", "tsne", "mds", or "mds_nonmetric".
    time_col : str
        Time column name for timeseries.
    color_by : str, optional
        Column to color scatter points by.
    max_points : int
        Maximum points before sampling.
    title : str, optional
        Dashboard title.
    width : int
        Total width in pixels.
    height : int
        Total height in pixels.
    sidecar : SidecarMetadata, optional
        Sidecar metadata for semantic labels.

    Returns
    -------
    go.Figure
        Linked Plotly figure with scatter and timeseries.
    """
    configure_theme()

    # Check for time column
    has_time = time_col in df.columns

    # Sample if too large
    original_len = len(df)
    if len(df) > max_points:
        warnings.warn(f"Dataset has {len(df)} points, sampling to {max_points}.")
        if has_time:
            # Sort by time before sampling to preserve temporal structure
            df = df.sort_values(time_col).reset_index(drop=True)
            # Sample evenly across time
            indices = np.linspace(0, len(df) - 1, max_points, dtype=int)
            df = df.iloc[indices].reset_index(drop=True)
        else:
            df = df.sample(n=max_points, random_state=42).reset_index(drop=True)

    # Compute scatter projection using unified module
    scatter_cols = get_feature_columns(df, scatter_features)
    if len(scatter_cols) < 2:
        raise ValueError(f"Need at least 2 scatter features, found {len(scatter_cols)}.")

    X = df[scatter_cols].values.copy()
    X_2d, proj_info = compute_projection(X, method=method, n_components=2)

    # Determine timeseries features
    if timeseries_features is None:
        if color_by and color_by in df.columns:
            ts_cols = [color_by]
        else:
            embedding_prefixes = [
                "clip_", "gist_", "dinov2_", "saliency_", "places_", "sunattr_"
            ]
            ts_cols = [
                c for c in get_feature_columns(df)
                if not any(c.startswith(p) for p in embedding_prefixes)
            ][:3]
    else:
        ts_cols = get_feature_columns(df, timeseries_features)

    # Build label mapping for timeseries features
    ts_label_map = {}
    if sidecar:
        ts_label_map = sidecar.get_feature_labels(ts_cols)
    else:
        ts_label_map = {col: col for col in ts_cols}

    # Get color label
    color_label = None
    if color_by and color_by in df.columns:
        color_label = sidecar.get_semantic_label(color_by) if sidecar else color_by

    # Create subplots: scatter on left, timeseries on right
    if has_time and ts_cols:
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.45, 0.55],
            subplot_titles=[
                f"{method.upper()} Projection ({len(scatter_cols)} features)",
                "Feature Time Series",
            ],
            horizontal_spacing=0.1,
        )
    else:
        # No time column, just scatter
        fig = go.Figure()

    # Build scatter hover template
    scatter_hover = []
    if has_time:
        scatter_hover.append(f"Time: %{{customdata[0]:.2f}}s")
    if color_by and color_by in df.columns:
        scatter_hover.append(f"{color_label}: %{{marker.color:.3f}}")
    scatter_hover.append(f"{proj_info['xlabel']}: %{{x:.2f}}")
    scatter_hover.append(f"{proj_info['ylabel']}: %{{y:.2f}}")
    scatter_hover.append("<extra></extra>")

    # Build customdata for scatter
    customdata = None
    if has_time:
        customdata = df[[time_col]].values

    # Add scatter trace
    if color_by and color_by in df.columns:
        scatter_trace = go.Scatter(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            mode="markers",
            marker=dict(
                size=8,
                color=df[color_by].values,
                colorscale="Viridis",
                colorbar=dict(title=color_label, x=0.42) if has_time else dict(title=color_label),
                opacity=0.7,
            ),
            customdata=customdata,
            hovertemplate="<br>".join(scatter_hover),
            name="Points",
        )
    else:
        scatter_trace = go.Scatter(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            mode="markers",
            marker=dict(
                size=8,
                color="steelblue",
                opacity=0.7,
            ),
            customdata=customdata,
            hovertemplate="<br>".join(scatter_hover),
            name="Points",
        )

    if has_time and ts_cols:
        fig.add_trace(scatter_trace, row=1, col=1)

        # Add timeseries traces
        time_values = df[time_col].values
        for col in ts_cols:
            label = ts_label_map.get(col, col)
            fig.add_trace(
                go.Scatter(
                    x=time_values,
                    y=df[col].values,
                    mode="lines",
                    name=label,
                    hovertemplate=f"{label}<br>Time: %{{x:.2f}}s<br>Value: %{{y:.4f}}<extra></extra>",
                ),
                row=1, col=2,
            )

        # Update axes
        fig.update_xaxes(title_text=proj_info["xlabel"], row=1, col=1)
        fig.update_yaxes(title_text=proj_info["ylabel"], row=1, col=1)
        fig.update_xaxes(title_text="Time (s)", row=1, col=2)
        fig.update_yaxes(title_text="Feature Value", row=1, col=2)
    else:
        fig.add_trace(scatter_trace)
        fig.update_xaxes(title_text=proj_info["xlabel"])
        fig.update_yaxes(title_text=proj_info["ylabel"])

    # Update layout
    dashboard_title = title
    if dashboard_title is None:
        dashboard_title = "Feature Explorer"
        if sidecar:
            model_summary = sidecar.get_model_summary()
            if model_summary:
                dashboard_title = f"Feature Explorer - {model_summary}"
        if original_len > max_points:
            dashboard_title += f" [sampled {len(df)}/{original_len}]"

    fig.update_layout(
        title=dashboard_title,
        width=width,
        height=height,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="right",
            x=0.99,
        ),
        hovermode="closest",
    )

    return fig
