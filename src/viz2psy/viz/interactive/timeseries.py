"""Interactive timeseries visualization using Plotly.

Provides interactive time series plots with:
- Multi-feature overlay with legend toggle
- Zoom/pan on time axis
- Hover tooltips with values
- Range slider for navigation
- Sidecar metadata integration for semantic labels
"""

from __future__ import annotations

import fnmatch
import warnings
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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


def plot_timeseries_interactive(
    df: pd.DataFrame,
    features: list[str] | None = None,
    time_col: str = "time",
    normalize: bool = False,
    show_range_slider: bool = True,
    title: str | None = None,
    width: int = 800,
    height: int = 400,
    sidecar: SidecarMetadata | None = None,
) -> go.Figure:
    """Create interactive timeseries plot.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time and feature columns.
    features : list of str, optional
        Features to plot. Supports glob patterns.
        If None, plots all scalar features (excludes embeddings).
    time_col : str
        Name of the time column.
    normalize : bool
        If True, normalize features to [0, 1] for comparison.
    show_range_slider : bool
        If True, show range slider for time navigation.
    title : str, optional
        Chart title.
    width : int
        Chart width in pixels.
    height : int
        Chart height in pixels.
    sidecar : SidecarMetadata, optional
        Sidecar metadata for semantic labels.

    Returns
    -------
    go.Figure
        Interactive Plotly figure.
    """
    configure_theme()

    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame.")

    # Get feature columns
    feature_cols = get_feature_columns(df, features)

    # For default (no patterns), exclude high-dimensional embeddings
    if features is None:
        embedding_prefixes = [
            "clip_", "gist_", "dinov2_", "saliency_", "places_", "sunattr_"
        ]
        feature_cols = [
            c for c in feature_cols
            if not any(c.startswith(p) for p in embedding_prefixes)
        ]

    if not feature_cols:
        raise ValueError("No features found to plot.")

    if len(feature_cols) > 15:
        warnings.warn(
            f"Plotting {len(feature_cols)} features may be cluttered. "
            f"Consider specifying fewer features."
        )

    # Build semantic label mapping from sidecar
    label_map = {}
    if sidecar:
        label_map = sidecar.get_feature_labels(feature_cols)
    else:
        label_map = {col: col for col in feature_cols}

    # Prepare data
    time_values = df[time_col].values

    # Normalize if requested
    feature_data = {}
    for col in feature_cols:
        values = df[col].values.copy()
        if normalize:
            min_val, max_val = np.nanmin(values), np.nanmax(values)
            if max_val > min_val:
                values = (values - min_val) / (max_val - min_val)
        feature_data[col] = values

    # Create figure
    fig = go.Figure()

    for col in feature_cols:
        label = label_map.get(col, col)
        fig.add_trace(go.Scatter(
            x=time_values,
            y=feature_data[col],
            mode="lines",
            name=label,
            hovertemplate=f"{label}<br>Time: %{{x:.2f}}s<br>Value: %{{y:.4f}}<extra></extra>",
        ))

    # Update layout
    fig.update_layout(
        title=title or f"Feature Time Series ({len(feature_cols)} features)",
        xaxis_title="Time (s)",
        yaxis_title="Value" if normalize else "Feature Value",
        width=width,
        height=height,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
        ),
        hovermode="x unified",
    )

    if show_range_slider:
        fig.update_xaxes(rangeslider_visible=True)

    return fig


def plot_timeseries_subplots(
    df: pd.DataFrame,
    features: list[str] | None = None,
    time_col: str = "time",
    shared_xaxis: bool = True,
    title: str | None = None,
    width: int = 800,
    height_per_row: int = 150,
    sidecar: SidecarMetadata | None = None,
) -> go.Figure:
    """Create faceted timeseries with one row per feature.

    Useful when features have different scales.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with time and feature columns.
    features : list of str, optional
        Features to plot. If None, uses all scalar features.
    time_col : str
        Name of the time column.
    shared_xaxis : bool
        If True, all subplots share the same x-axis (linked zoom).
    title : str, optional
        Chart title.
    width : int
        Chart width in pixels.
    height_per_row : int
        Height per feature subplot.
    sidecar : SidecarMetadata, optional
        Sidecar metadata for semantic labels.

    Returns
    -------
    go.Figure
        Faceted Plotly figure.
    """
    configure_theme()

    if time_col not in df.columns:
        raise ValueError(f"Time column '{time_col}' not found in DataFrame.")

    # Get feature columns
    feature_cols = get_feature_columns(df, features)

    if features is None:
        embedding_prefixes = [
            "clip_", "gist_", "dinov2_", "saliency_", "places_", "sunattr_"
        ]
        feature_cols = [
            c for c in feature_cols
            if not any(c.startswith(p) for p in embedding_prefixes)
        ]

    if not feature_cols:
        raise ValueError("No features found to plot.")

    # Limit to reasonable number
    if len(feature_cols) > 10:
        warnings.warn(f"Limiting to first 10 features (of {len(feature_cols)}).")
        feature_cols = feature_cols[:10]

    # Build label mapping
    label_map = {}
    if sidecar:
        label_map = sidecar.get_feature_labels(feature_cols)
    else:
        label_map = {col: col for col in feature_cols}

    # Create subplots
    n_features = len(feature_cols)
    fig = make_subplots(
        rows=n_features,
        cols=1,
        shared_xaxes=shared_xaxis,
        vertical_spacing=0.02,
        subplot_titles=[label_map.get(col, col) for col in feature_cols],
    )

    time_values = df[time_col].values

    for i, col in enumerate(feature_cols):
        label = label_map.get(col, col)
        fig.add_trace(
            go.Scatter(
                x=time_values,
                y=df[col].values,
                mode="lines",
                name=label,
                showlegend=False,
                hovertemplate=f"Time: %{{x:.2f}}s<br>{label}: %{{y:.4f}}<extra></extra>",
            ),
            row=i + 1,
            col=1,
        )

    # Update layout
    total_height = height_per_row * n_features + 100  # Extra for title/margins
    fig.update_layout(
        title=title or "Feature Time Series",
        width=width,
        height=total_height,
        hovermode="x unified",
    )

    # Only show x-axis label on bottom subplot
    fig.update_xaxes(title_text="Time (s)", row=n_features, col=1)

    return fig
