"""Interactive timeseries visualization using Plotly.

Provides interactive time series plots with:
- Multi-feature overlay with legend toggle
- Zoom/pan on time axis
- Hover tooltips with values
- Range slider for navigation
- Auto-detect index type (time, filename, image_idx)
- Diff and rolling average overlays
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
from ..index_utils import detect_index_column, prepare_index_values, is_video_data

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
    time_col: str | None = None,
    normalize: bool = False,
    show_range_slider: bool = True,
    show_diff: bool = False,
    rolling_window: int | None = None,
    auto_video_mode: bool = True,
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
    time_col : str, optional
        Name of the time/index column. If None, auto-detects.
    normalize : bool
        If True, normalize features to [0, 1] for comparison.
    show_range_slider : bool
        If True, show range slider for time navigation.
    show_diff : bool
        If True, plot first-order differences between consecutive points.
        Auto-enabled for video data if auto_video_mode=True.
    rolling_window : int, optional
        If provided, overlay rolling average with this window size.
        Auto-enabled (window=5) for video data if auto_video_mode=True.
    auto_video_mode : bool
        If True and data appears to be from video, auto-enable show_diff
        and rolling_window=5 unless explicitly disabled.
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

    # Auto-detect index column
    detected_col, index_type = detect_index_column(df, sidecar)
    if time_col is None:
        time_col = detected_col

    if time_col is None:
        # Fall back to row numbers
        time_col = "_row_index"
        df = df.copy()
        df[time_col] = np.arange(len(df))
        index_type = "integer"

    if time_col not in df.columns and time_col != "_row_index":
        raise ValueError(f"Index column '{time_col}' not found in DataFrame.")

    # Auto-enable video mode features
    if auto_video_mode and is_video_data(df, sidecar):
        if not show_diff:
            show_diff = True
        if rolling_window is None:
            rolling_window = 5

    # Validate rolling window
    if rolling_window is not None:
        if rolling_window >= len(df):
            new_window = max(1, len(df) // 3)
            warnings.warn(
                f"rolling_window ({rolling_window}) >= data length ({len(df)}), "
                f"reducing to {new_window}."
            )
            rolling_window = new_window

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

    # Prepare index values and formatting
    x_values, format_info = prepare_index_values(df, time_col, index_type)

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

    # Build hover format based on index type
    if index_type == "time":
        x_hover_format = "Time: %{x:.2f}s"
    elif index_type == "ordinal":
        x_hover_format = "%{text}"  # Use text for ordinal labels
    else:
        x_hover_format = "%{x:d}"

    for col in feature_cols:
        label = label_map.get(col, col)

        # Build hover template
        if index_type == "ordinal":
            hover = f"{label}<br>%{{text}}<br>Value: %{{y:.4f}}<extra></extra>"
            text = format_info.get("original_labels", df[time_col].astype(str).values)
        else:
            hover = f"{label}<br>{x_hover_format}<br>Value: %{{y:.4f}}<extra></extra>"
            text = None

        # Main trace
        trace_kwargs = dict(
            x=x_values,
            y=feature_data[col],
            mode="lines",
            name=label,
            hovertemplate=hover,
        )
        if text is not None:
            trace_kwargs["text"] = text

        fig.add_trace(go.Scatter(**trace_kwargs))

        # Add diff trace if requested
        if show_diff:
            diff_values = np.diff(feature_data[col])
            diff_x = x_values[1:]
            diff_hover = f"{label} (diff)<br>{x_hover_format}<br>Value: %{{y:.4f}}<extra></extra>"
            diff_kwargs = dict(
                x=diff_x,
                y=diff_values,
                mode="lines",
                name=f"{label} (diff)",
                line=dict(dash="dash"),
                hovertemplate=diff_hover,
                visible="legendonly",  # Hidden by default
            )
            if text is not None:
                diff_kwargs["text"] = text[1:]
            fig.add_trace(go.Scatter(**diff_kwargs))

        # Add rolling average if requested
        if rolling_window is not None:
            rolling_avg = pd.Series(feature_data[col]).rolling(
                window=rolling_window, center=True
            ).mean().values
            roll_hover = f"{label} (rolling {rolling_window})<br>{x_hover_format}<br>Value: %{{y:.4f}}<extra></extra>"
            roll_kwargs = dict(
                x=x_values,
                y=rolling_avg,
                mode="lines",
                name=f"{label} (rolling {rolling_window})",
                line=dict(width=2),
                hovertemplate=roll_hover,
            )
            if text is not None:
                roll_kwargs["text"] = text
            fig.add_trace(go.Scatter(**roll_kwargs))

    # Update layout
    fig.update_layout(
        title=title or f"Feature Time Series ({len(feature_cols)} features)",
        xaxis_title=format_info["xlabel"],
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

    # Handle ordinal x-axis
    if index_type == "ordinal" and "tickvals" in format_info:
        fig.update_xaxes(
            tickmode="array",
            tickvals=format_info["tickvals"],
            ticktext=format_info["ticktext"],
            tickangle=45,
        )

    if show_range_slider:
        fig.update_xaxes(rangeslider_visible=True)

    return fig


def plot_timeseries_subplots(
    df: pd.DataFrame,
    features: list[str] | None = None,
    time_col: str | None = None,
    shared_xaxis: bool = True,
    show_diff: bool = False,
    rolling_window: int | None = None,
    auto_video_mode: bool = True,
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
    time_col : str, optional
        Name of the time/index column. If None, auto-detects.
    shared_xaxis : bool
        If True, all subplots share the same x-axis (linked zoom).
    show_diff : bool
        If True, overlay differences in each subplot.
    rolling_window : int, optional
        If provided, overlay rolling average.
    auto_video_mode : bool
        If True and video data, auto-enable diff/rolling.
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

    # Auto-detect index column
    detected_col, index_type = detect_index_column(df, sidecar)
    if time_col is None:
        time_col = detected_col

    if time_col is None:
        time_col = "_row_index"
        df = df.copy()
        df[time_col] = np.arange(len(df))
        index_type = "integer"

    if time_col not in df.columns and time_col != "_row_index":
        raise ValueError(f"Index column '{time_col}' not found in DataFrame.")

    # Auto-enable video mode features
    if auto_video_mode and is_video_data(df, sidecar):
        if not show_diff:
            show_diff = True
        if rolling_window is None:
            rolling_window = 5

    # Validate rolling window
    if rolling_window is not None:
        if rolling_window >= len(df):
            rolling_window = max(1, len(df) // 3)

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

    # Prepare index values
    x_values, format_info = prepare_index_values(df, time_col, index_type)

    # Create subplots
    n_features = len(feature_cols)
    fig = make_subplots(
        rows=n_features,
        cols=1,
        shared_xaxes=shared_xaxis,
        vertical_spacing=0.02,
        subplot_titles=[label_map.get(col, col) for col in feature_cols],
    )

    for i, col in enumerate(feature_cols):
        label = label_map.get(col, col)
        values = df[col].values

        # Main trace
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=values,
                mode="lines",
                name=label,
                showlegend=False,
                hovertemplate=f"%{{x}}<br>{label}: %{{y:.4f}}<extra></extra>",
            ),
            row=i + 1,
            col=1,
        )

        # Diff trace
        if show_diff:
            diff_values = np.diff(values)
            diff_x = x_values[1:]
            fig.add_trace(
                go.Scatter(
                    x=diff_x,
                    y=diff_values,
                    mode="lines",
                    name=f"{label} (diff)",
                    showlegend=False,
                    line=dict(dash="dash", width=1),
                    opacity=0.6,
                ),
                row=i + 1,
                col=1,
            )

        # Rolling average
        if rolling_window is not None:
            rolling_avg = pd.Series(values).rolling(
                window=rolling_window, center=True
            ).mean().values
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=rolling_avg,
                    mode="lines",
                    name=f"{label} (rolling)",
                    showlegend=False,
                    line=dict(color="red", width=2),
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
    fig.update_xaxes(title_text=format_info["xlabel"], row=n_features, col=1)

    # Handle ordinal x-axis
    if index_type == "ordinal" and "tickvals" in format_info:
        fig.update_xaxes(
            tickmode="array",
            tickvals=format_info["tickvals"],
            ticktext=format_info["ticktext"],
            tickangle=45,
            row=n_features,
            col=1,
        )

    return fig
