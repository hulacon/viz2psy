"""Interactive model-visualization dashboard.

Creates a single HTML file with controls to select:
- Model output (emonet, clip, etc.)
- Visualization type (timeseries, clustering, trajectory)

Clicking on data points shows single-image detail view.
Invalid combinations show a warning message.
"""

from __future__ import annotations

import base64
import io
import json
import warnings
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .feature_config import FEATURE_CONFIGS, detect_models_in_dataframe
from .projection import compute_projection
from .index_utils import detect_index_column, prepare_index_values, is_video_data
from .interactive.single_image import create_single_image_viewer

if TYPE_CHECKING:
    from PIL import Image
    from .sidecar import SidecarMetadata, UnifiedImageResolver


# Emotion feature names (EmoNet)
EMOTION_FEATURES = [
    "Adoration", "Aesthetic Appreciation", "Amusement", "Anxiety", "Awe",
    "Boredom", "Confusion", "Craving", "Disgust", "Empathic Pain",
    "Entrancement", "Excitement", "Fear", "Horror", "Interest",
    "Joy", "Romance", "Sadness", "Sexual Desire", "Surprise"
]

# Scalar features with known ranges for normalization
SCALAR_FEATURES = {
    "memorability": (0, 1),
    "aesthetic_score": (0, 10),
}


def _image_to_base64(img: "Image.Image | Path | None", max_size: int = 200) -> str | None:
    """Convert image to base64 data URI for embedding in HTML."""
    if img is None:
        return None

    try:
        from PIL import Image as PILImage

        if isinstance(img, Path):
            if not img.exists():
                return None
            img = PILImage.open(img)

        # Resize for thumbnail
        img.thumbnail((max_size, max_size), PILImage.Resampling.LANCZOS)

        # Convert to RGB if necessary
        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")

        # Encode as JPEG
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        b64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/jpeg;base64,{b64}"
    except Exception:
        return None


def _extract_row_details(
    df: pd.DataFrame,
    row_idx: int,
    index_col: str | None,
    index_type: str,
) -> dict:
    """Extract feature details for a single row (for preview panel)."""
    row = df.iloc[row_idx]
    details = {"row_idx": row_idx}

    # Add index value (time, filename, etc.)
    if index_col and index_col in df.columns:
        val = row[index_col]
        if index_type == "time":
            details["label"] = f"Time: {val:.2f}s"
        elif index_type == "ordinal":
            details["label"] = str(val)[:30]  # Truncate long filenames
        else:
            details["label"] = f"Index: {val}"
    else:
        details["label"] = f"Row {row_idx}"

    # Extract emotion values if present
    emotions = {}
    for em in EMOTION_FEATURES:
        if em in row.index:
            emotions[em] = float(row[em]) if pd.notna(row[em]) else 0
    if emotions:
        details["emotions"] = emotions

    # Extract scalar features
    scalars = {}
    for feat, (vmin, vmax) in SCALAR_FEATURES.items():
        if feat in row.index and pd.notna(row[feat]):
            raw = float(row[feat])
            normalized = (raw - vmin) / (vmax - vmin) if vmax > vmin else 0
            scalars[feat] = {"raw": raw, "normalized": min(1, max(0, normalized))}
    if scalars:
        details["scalars"] = scalars

    return details


def _generate_single_image_viewer_html(
    df: pd.DataFrame,
    row_idx: int,
    image_data: str | None,
    index_col: str | None,
    index_type: str,
) -> str:
    """Generate a standalone single-image viewer HTML for one row.

    This creates a simplified self-contained HTML viewer that can be
    opened in a new tab.
    """
    row = df.iloc[row_idx]

    # Get title/label
    if index_col and index_col in df.columns:
        val = row[index_col]
        if index_type == "time":
            title = f"Frame at {val:.2f}s"
        elif index_type == "ordinal":
            title = str(val)[:50]
        else:
            title = f"Row {row_idx}"
    else:
        title = f"Row {row_idx}"

    # Extract emotions
    emotions_data = []
    for em in EMOTION_FEATURES:
        if em in row.index and pd.notna(row[em]):
            emotions_data.append({"name": em, "value": float(row[em])})
    emotions_data.sort(key=lambda x: x["value"], reverse=True)

    # Extract scalars
    scalars_data = []
    for feat, (vmin, vmax) in SCALAR_FEATURES.items():
        if feat in row.index and pd.notna(row[feat]):
            scalars_data.append({
                "name": feat,
                "value": float(row[feat]),
                "min": vmin,
                "max": vmax,
            })

    # Build the HTML
    emotions_json = json.dumps(emotions_data)
    scalars_json = json.dumps(scalars_data)

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>{title} - viz2psy Viewer</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{ box-sizing: border-box; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }}
        body {{ margin: 0; padding: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        h1 {{ color: #333; margin-bottom: 20px; }}
        .content {{ display: flex; gap: 20px; flex-wrap: wrap; }}
        .image-panel {{
            background: white; border-radius: 8px; padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; min-width: 300px;
        }}
        .image-panel img {{ max-width: 100%; height: auto; border-radius: 4px; }}
        .no-image {{
            height: 300px; display: flex; align-items: center; justify-content: center;
            background: #eee; border-radius: 4px; color: #999; font-size: 48px;
        }}
        .features-panel {{
            background: white; border-radius: 8px; padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1); flex: 1; min-width: 400px;
        }}
        .plot {{ width: 100%; height: 350px; }}
        .section-title {{ font-weight: 600; color: #333; margin: 20px 0 10px 0; }}
        .section-title:first-child {{ margin-top: 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>
        <div class="content">
            <div class="image-panel">
                {"<img src='" + image_data + "' alt='Image' />" if image_data else "<div class='no-image'>🖼️</div>"}
            </div>
            <div class="features-panel">
                <div class="section-title">Emotions</div>
                <div id="emotions-plot" class="plot"></div>
                <div class="section-title">Scalar Features</div>
                <div id="scalars-plot" class="plot"></div>
            </div>
        </div>
    </div>
    <script>
        const emotions = {emotions_json};
        const scalars = {scalars_json};

        // Emotions bar chart
        if (emotions.length > 0) {{
            const trace = {{
                type: 'bar',
                orientation: 'h',
                y: emotions.map(e => e.name),
                x: emotions.map(e => e.value * 100),
                marker: {{ color: 'steelblue' }},
                hovertemplate: '%{{y}}: %{{x:.1f}}%<extra></extra>'
            }};
            const layout = {{
                margin: {{ l: 120, r: 20, t: 10, b: 40 }},
                xaxis: {{ title: 'Probability (%)', range: [0, 100] }},
                yaxis: {{ autorange: 'reversed' }},
                height: 350
            }};
            Plotly.newPlot('emotions-plot', [trace], layout);
        }}

        // Scalars bar chart
        if (scalars.length > 0) {{
            const trace = {{
                type: 'bar',
                x: scalars.map(s => s.name),
                y: scalars.map(s => s.value),
                marker: {{ color: '#4CAF50' }},
                hovertemplate: '%{{x}}: %{{y:.3f}}<extra></extra>'
            }};
            const layout = {{
                margin: {{ l: 50, r: 20, t: 10, b: 60 }},
                yaxis: {{ title: 'Value' }},
                height: 350
            }};
            Plotly.newPlot('scalars-plot', [trace], layout);
        }}
    </script>
</body>
</html>'''

    return html


def _get_model_columns(df: pd.DataFrame, model_name: str) -> list[str]:
    """Get columns belonging to a specific model."""
    import fnmatch

    config = FEATURE_CONFIGS.get(model_name)
    if not config:
        return []

    cols = []
    for pattern in config.column_patterns:
        for col in df.columns:
            if fnmatch.fnmatch(col, pattern) and col not in cols:
                cols.append(col)
    return cols


def _create_timeseries_trace(
    df: pd.DataFrame,
    model_name: str,
    x_values: np.ndarray,
    x_label: str,
) -> list[go.Scatter]:
    """Create timeseries traces for a model."""
    config = FEATURE_CONFIGS.get(model_name)
    if not config or not config.timeseries:
        return []

    cols = _get_model_columns(df, model_name)
    if not cols:
        return []

    # Limit columns for high-dim models
    if config.timeseries_mode == "top_k":
        # For distributions, show top-k by variance
        variances = [(col, df[col].var()) for col in cols]
        variances.sort(key=lambda x: x[1], reverse=True)
        cols = [c for c, _ in variances[:config.top_k]]
    elif config.timeseries_mode == "none":
        return []

    traces = []
    for col in cols:
        traces.append(go.Scatter(
            x=x_values,
            y=df[col].values,
            mode="lines",
            name=col,
            hovertemplate=f"{col}<br>{x_label}: %{{x}}<br>Value: %{{y:.4f}}<extra></extra>",
        ))

    return traces


def _create_cluster_trace(
    df: pd.DataFrame,
    model_name: str,
    color_values: np.ndarray | None = None,
    color_label: str = "time",
    n_components: int = 2,
) -> tuple[go.Scatter | go.Scatter3d | None, dict]:
    """Create MDS/clustering scatter trace for a model (2D or 3D)."""
    config = FEATURE_CONFIGS.get(model_name)
    if not config or not config.mds_clustering:
        return None, {}

    cols = _get_model_columns(df, model_name)
    if len(cols) < 2:
        return None, {}

    # Compute MDS projection
    X = df[cols].values.copy()
    try:
        X_proj, info = compute_projection(X, method="mds", n_components=n_components)
    except Exception as e:
        warnings.warn(f"MDS failed for {model_name}: {e}")
        return None, {}

    n_points = len(X_proj)
    # customdata stores row indices for click handling
    customdata = np.arange(n_points).reshape(-1, 1)

    # Create scatter trace (2D or 3D)
    if n_components == 3:
        if color_values is not None:
            trace = go.Scatter3d(
                x=X_proj[:, 0],
                y=X_proj[:, 1],
                z=X_proj[:, 2],
                mode="markers",
                marker=dict(
                    size=6,
                    color=color_values,
                    colorscale="Viridis",
                    colorbar=dict(title=color_label),
                    opacity=0.8,
                ),
                customdata=customdata,
                hovertemplate=f"Point %{{customdata[0]}}<br>{color_label}: %{{marker.color:.2f}}<br>MDS1: %{{x:.2f}}<br>MDS2: %{{y:.2f}}<br>MDS3: %{{z:.2f}}<extra></extra>",
                name=model_name,
            )
        else:
            trace = go.Scatter3d(
                x=X_proj[:, 0],
                y=X_proj[:, 1],
                z=X_proj[:, 2],
                mode="markers",
                marker=dict(
                    size=6,
                    color="steelblue",
                    opacity=0.8,
                ),
                customdata=customdata,
                hovertemplate=f"Point %{{customdata[0]}}<br>MDS1: %{{x:.2f}}<br>MDS2: %{{y:.2f}}<br>MDS3: %{{z:.2f}}<extra></extra>",
                name=model_name,
            )
    else:
        if color_values is not None:
            trace = go.Scatter(
                x=X_proj[:, 0],
                y=X_proj[:, 1],
                mode="markers",
                marker=dict(
                    size=10,
                    color=color_values,
                    colorscale="Viridis",
                    colorbar=dict(title=color_label),
                    opacity=0.8,
                ),
                customdata=customdata,
                hovertemplate=f"Point %{{customdata[0]}}<br>{color_label}: %{{marker.color:.2f}}<br>MDS1: %{{x:.2f}}<br>MDS2: %{{y:.2f}}<extra></extra>",
                name=model_name,
            )
        else:
            trace = go.Scatter(
                x=X_proj[:, 0],
                y=X_proj[:, 1],
                mode="markers",
                marker=dict(
                    size=10,
                    color="steelblue",
                    opacity=0.8,
                ),
                customdata=customdata,
                hovertemplate=f"Point %{{customdata[0]}}<br>MDS1: %{{x:.2f}}<br>MDS2: %{{y:.2f}}<extra></extra>",
                name=model_name,
            )

    return trace, info


def _create_trajectory_trace(
    df: pd.DataFrame,
    model_name: str,
    color_values: np.ndarray | None = None,
    color_label: str = "time",
) -> tuple[list[go.Scatter], dict]:
    """Create trajectory traces for a model (points + connecting lines)."""
    config = FEATURE_CONFIGS.get(model_name)
    if not config or not config.trajectories:
        return [], {}

    cols = _get_model_columns(df, model_name)
    if len(cols) < 2:
        return [], {}

    # Compute PCA projection (better for trajectories than MDS)
    X = df[cols].values.copy()
    try:
        X_2d, info = compute_projection(X, method="pca", n_components=2)
    except Exception as e:
        warnings.warn(f"PCA failed for {model_name}: {e}")
        return [], {}

    n_points = len(X_2d)
    customdata = np.arange(n_points).reshape(-1, 1)

    traces = []

    # Line trace connecting points in order
    traces.append(go.Scatter(
        x=X_2d[:, 0],
        y=X_2d[:, 1],
        mode="lines",
        line=dict(color="rgba(100,100,100,0.5)", width=1),
        hoverinfo="skip",
        showlegend=False,
        name="trajectory",
    ))

    # Points with color (clickable)
    if color_values is not None:
        traces.append(go.Scatter(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            mode="markers",
            marker=dict(
                size=12,
                color=color_values,
                colorscale="Viridis",
                colorbar=dict(title=color_label),
                opacity=0.9,
            ),
            customdata=customdata,
            hovertemplate=f"Point %{{customdata[0]}}<br>{color_label}: %{{marker.color:.2f}}<br>{info['xlabel']}: %{{x:.2f}}<br>{info['ylabel']}: %{{y:.2f}}<extra></extra>",
            name=model_name,
        ))
    else:
        traces.append(go.Scatter(
            x=X_2d[:, 0],
            y=X_2d[:, 1],
            mode="markers",
            marker=dict(
                size=12,
                color="steelblue",
                opacity=0.9,
            ),
            customdata=customdata,
            hovertemplate=f"Point %{{customdata[0]}}<br>{info['xlabel']}: %{{x:.2f}}<br>{info['ylabel']}: %{{y:.2f}}<extra></extra>",
            name=model_name,
        ))

    # Add start/end markers (not clickable for detail)
    traces.append(go.Scatter(
        x=[X_2d[0, 0]],
        y=[X_2d[0, 1]],
        mode="markers",
        marker=dict(size=16, color="green", symbol="circle"),
        name="Start",
        hovertemplate="Start<extra></extra>",
    ))
    traces.append(go.Scatter(
        x=[X_2d[-1, 0]],
        y=[X_2d[-1, 1]],
        mode="markers",
        marker=dict(size=16, color="red", symbol="square"),
        name="End",
        hovertemplate="End<extra></extra>",
    ))

    return traces, info


def _create_trajectory_animated(
    df: pd.DataFrame,
    model_name: str,
    color_values: np.ndarray | None = None,
    color_label: str = "time",
) -> tuple[list, list, dict]:
    """Create animated trajectory using Plotly animation frames.

    Returns
    -------
    traces : list
        Initial traces for the animation.
    frames : list
        Animation frames showing trajectory evolution.
    info : dict
        Projection metadata.
    """
    config = FEATURE_CONFIGS.get(model_name)
    if not config or not config.trajectories:
        return [], [], {}

    cols = _get_model_columns(df, model_name)
    if len(cols) < 2:
        return [], [], {}

    # Compute PCA projection
    X = df[cols].values.copy()
    try:
        X_2d, info = compute_projection(X, method="pca", n_components=2)
    except Exception as e:
        warnings.warn(f"PCA failed for {model_name}: {e}")
        return [], [], {}

    n_points = len(X_2d)

    # Determine axis ranges with padding
    x_min, x_max = X_2d[:, 0].min(), X_2d[:, 0].max()
    y_min, y_max = X_2d[:, 1].min(), X_2d[:, 1].max()
    x_pad = (x_max - x_min) * 0.1
    y_pad = (y_max - y_min) * 0.1
    info["xaxis_range"] = [x_min - x_pad, x_max + x_pad]
    info["yaxis_range"] = [y_min - y_pad, y_max + y_pad]

    # Initial traces (show first point)
    traces = [
        # Trail line (initially empty)
        go.Scatter(
            x=[X_2d[0, 0]],
            y=[X_2d[0, 1]],
            mode="lines",
            line=dict(color="rgba(100,100,100,0.6)", width=2),
            hoverinfo="skip",
            showlegend=False,
            name="trail",
        ),
        # Current point marker
        go.Scatter(
            x=[X_2d[0, 0]],
            y=[X_2d[0, 1]],
            mode="markers",
            marker=dict(
                size=14,
                color="red",
                symbol="circle",
            ),
            hovertemplate=f"Frame: 0<br>{info['xlabel']}: %{{x:.2f}}<br>{info['ylabel']}: %{{y:.2f}}<extra></extra>",
            name="current",
        ),
        # Start marker (stays fixed)
        go.Scatter(
            x=[X_2d[0, 0]],
            y=[X_2d[0, 1]],
            mode="markers",
            marker=dict(size=12, color="green", symbol="circle-open", line=dict(width=2)),
            name="Start",
            hovertemplate="Start<extra></extra>",
        ),
        # End marker (stays fixed)
        go.Scatter(
            x=[X_2d[-1, 0]],
            y=[X_2d[-1, 1]],
            mode="markers",
            marker=dict(size=12, color="blue", symbol="square-open", line=dict(width=2)),
            name="End",
            hovertemplate="End<extra></extra>",
        ),
    ]

    # Create animation frames
    frames = []
    for i in range(n_points):
        frame_data = [
            # Trail up to current point
            go.Scatter(
                x=X_2d[:i+1, 0],
                y=X_2d[:i+1, 1],
                mode="lines",
                line=dict(color="rgba(100,100,100,0.6)", width=2),
                hoverinfo="skip",
                showlegend=False,
            ),
            # Current point
            go.Scatter(
                x=[X_2d[i, 0]],
                y=[X_2d[i, 1]],
                mode="markers",
                marker=dict(size=14, color="red", symbol="circle"),
                hovertemplate=f"Frame: {i}<br>{info['xlabel']}: %{{x:.2f}}<br>{info['ylabel']}: %{{y:.2f}}<extra></extra>",
            ),
            # Start marker
            go.Scatter(
                x=[X_2d[0, 0]],
                y=[X_2d[0, 1]],
                mode="markers",
                marker=dict(size=12, color="green", symbol="circle-open", line=dict(width=2)),
            ),
            # End marker
            go.Scatter(
                x=[X_2d[-1, 0]],
                y=[X_2d[-1, 1]],
                mode="markers",
                marker=dict(size=12, color="blue", symbol="square-open", line=dict(width=2)),
            ),
        ]

        frame_label = f"{color_values[i]:.2f}" if color_values is not None else str(i)
        frames.append(dict(
            data=frame_data,
            name=frame_label,
        ))

    return traces, frames, info


def create_dashboard(
    df: pd.DataFrame,
    sidecar: "SidecarMetadata | None" = None,
    image_resolver: "UnifiedImageResolver | None" = None,
    width: int = 900,
    height: int = 600,
    max_thumbnails: int = 100,
) -> str:
    """Create an interactive dashboard HTML.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature columns.
    sidecar : SidecarMetadata, optional
        Sidecar metadata.
    image_resolver : UnifiedImageResolver, optional
        Resolver for loading images from rows. If provided, clicking
        points will show image thumbnails in detail panel.
    width : int
        Figure width in pixels.
    height : int
        Figure height in pixels.
    max_thumbnails : int
        Maximum number of thumbnail images to embed (default: 100).
        Set to 0 to disable image embedding.

    Returns
    -------
    str
        Complete HTML string for the dashboard.
    """
    # Detect models and prepare data
    detected_models = detect_models_in_dataframe(df.columns.tolist())

    # Detect index for x-axis
    index_col, index_type = detect_index_column(df, sidecar)
    if index_col and index_col in df.columns:
        x_values, format_info = prepare_index_values(df, index_col, index_type)
        color_values = x_values if index_type == "time" else np.arange(len(df))
        color_label = "Time (s)" if index_type == "time" else "Index"
        x_label = format_info["xlabel"]
    else:
        x_values = np.arange(len(df))
        color_values = x_values
        color_label = "Index"
        x_label = "Index"

    # Pre-compute all visualizations
    viz_data = {}

    for model in detected_models:
        config = FEATURE_CONFIGS.get(model)
        if not config:
            continue

        viz_data[model] = {
            "timeseries": {"available": config.timeseries, "traces": [], "layout": {}},
            "clustering_2d": {"available": config.mds_clustering, "traces": [], "layout": {}},
            "clustering_3d": {"available": config.mds_clustering, "traces": [], "layout": {}},
            "trajectory_static": {"available": config.trajectories, "traces": [], "layout": {}},
            "trajectory_animated": {"available": config.trajectories, "traces": [], "frames": [], "layout": {}},
        }

        # Timeseries
        if config.timeseries:
            traces = _create_timeseries_trace(df, model, x_values, x_label)
            if traces:
                viz_data[model]["timeseries"]["traces"] = traces
                viz_data[model]["timeseries"]["layout"] = {
                    "xaxis_title": x_label,
                    "yaxis_title": "Value",
                    "title": f"{model} - Time Series",
                }

        # Clustering 2D
        if config.mds_clustering:
            trace, info = _create_cluster_trace(df, model, color_values, color_label, n_components=2)
            if trace:
                viz_data[model]["clustering_2d"]["traces"] = [trace]
                viz_data[model]["clustering_2d"]["layout"] = {
                    "xaxis_title": info.get("xlabel", "MDS1"),
                    "yaxis_title": info.get("ylabel", "MDS2"),
                    "title": f"{model} - MDS Clustering (2D)",
                }

        # Clustering 3D
        if config.mds_clustering:
            trace, info = _create_cluster_trace(df, model, color_values, color_label, n_components=3)
            if trace:
                viz_data[model]["clustering_3d"]["traces"] = [trace]
                viz_data[model]["clustering_3d"]["layout"] = {
                    "scene": {
                        "xaxis_title": info.get("xlabel", "MDS1"),
                        "yaxis_title": info.get("ylabel", "MDS2"),
                        "zaxis_title": info.get("zlabel", "MDS3"),
                    },
                    "title": f"{model} - MDS Clustering (3D)",
                }

        # Trajectory Static
        if config.trajectories:
            traces, info = _create_trajectory_trace(df, model, color_values, color_label)
            if traces:
                viz_data[model]["trajectory_static"]["traces"] = traces
                viz_data[model]["trajectory_static"]["layout"] = {
                    "xaxis_title": info.get("xlabel", "PC1"),
                    "yaxis_title": info.get("ylabel", "PC2"),
                    "title": f"{model} - State-Space Trajectory",
                }

        # Trajectory Animated
        if config.trajectories:
            traces, frames, info = _create_trajectory_animated(df, model, color_values, color_label)
            if traces and frames:
                viz_data[model]["trajectory_animated"]["traces"] = traces
                viz_data[model]["trajectory_animated"]["frames"] = frames
                viz_data[model]["trajectory_animated"]["layout"] = {
                    "xaxis_title": info.get("xlabel", "PC1"),
                    "yaxis_title": info.get("ylabel", "PC2"),
                    "xaxis_range": info.get("xaxis_range"),
                    "yaxis_range": info.get("yaxis_range"),
                    "title": f"{model} - Animated Trajectory",
                }

    # Pre-compute single-image viewer HTMLs for click-to-open feature
    viewer_htmls = []
    n_rows = len(df)
    has_resolver = image_resolver is not None

    if has_resolver and max_thumbnails > 0:
        print(f"Pre-rendering {min(n_rows, max_thumbnails)} single-image viewers...")
        for i in range(n_rows):
            if i < max_thumbnails:
                try:
                    # Get image for this row
                    img = image_resolver.resolve(df, i)

                    # Generate row label for title
                    row = df.iloc[i]
                    if index_col and index_col in df.columns:
                        val = row[index_col]
                        if index_type == "time":
                            img_title = f"Frame at {val:.2f}s"
                        elif index_type == "ordinal":
                            img_title = str(val)[:50]
                        else:
                            img_title = f"Row {i}"
                    else:
                        img_title = f"Row {i}"

                    # Create full single-image viewer figure
                    fig = create_single_image_viewer(
                        image=img,
                        scores_df=df,
                        row_idx=i,
                        panels=None,  # Auto-detect
                        width=1100,
                        height=650,
                        sidecar=sidecar,
                        normalize_scalars=True,
                        image_title=img_title,
                    )

                    # Convert to standalone HTML
                    viewer_html = fig.to_html(
                        full_html=True,
                        include_plotlyjs='cdn',
                        config={'displayModeBar': True, 'responsive': True},
                    )
                    viewer_htmls.append(viewer_html)
                except Exception as e:
                    # Fallback to simple viewer on error
                    viewer_html = _generate_single_image_viewer_html(
                        df, i, None, index_col, index_type
                    )
                    viewer_htmls.append(viewer_html)
            else:
                # Beyond max_thumbnails - use simple fallback
                viewer_html = _generate_single_image_viewer_html(
                    df, i, None, index_col, index_type
                )
                viewer_htmls.append(viewer_html)
    else:
        # No image resolver - generate simple viewers for all rows
        for i in range(n_rows):
            viewer_html = _generate_single_image_viewer_html(
                df, i, None, index_col, index_type
            )
            viewer_htmls.append(viewer_html)

    # Build the HTML
    return _build_dashboard_html(viz_data, detected_models, viewer_htmls, width, height)


def _numpy_to_python(obj):
    """Recursively convert numpy types to Python types for JSON serialization."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, dict):
        return {k: _numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_numpy_to_python(item) for item in obj]
    return obj


def _build_dashboard_html(
    viz_data: dict,
    models: list[str],
    viewer_htmls: list[str],
    width: int,
    height: int,
) -> str:
    """Build the complete dashboard HTML with controls."""

    # Serialize trace data to JSON
    traces_json = {}
    layouts_json = {}
    frames_json = {}
    availability_json = {}

    for model, vizs in viz_data.items():
        traces_json[model] = {}
        layouts_json[model] = {}
        frames_json[model] = {}
        availability_json[model] = {}

        for viz_type, data in vizs.items():
            availability_json[model][viz_type] = data["available"] and len(data.get("traces", [])) > 0

            if data.get("traces"):
                # Convert traces to JSON-serializable format
                traces_json[model][viz_type] = [
                    _numpy_to_python(trace.to_plotly_json()) for trace in data["traces"]
                ]
                layouts_json[model][viz_type] = data["layout"]

                # Handle animation frames
                if data.get("frames"):
                    frames_json[model][viz_type] = [
                        {
                            "data": [_numpy_to_python(t.to_plotly_json()) for t in frame["data"]],
                            "name": frame["name"],
                        }
                        for frame in data["frames"]
                    ]
                else:
                    frames_json[model][viz_type] = []
            else:
                traces_json[model][viz_type] = []
                layouts_json[model][viz_type] = {}
                frames_json[model][viz_type] = []

    # Model display names and descriptions
    model_info = {}
    for model in models:
        config = FEATURE_CONFIGS.get(model)
        if config:
            model_info[model] = {
                "name": model,
                "description": config.description,
                "n_dims": config.n_dims,
            }

    html = f'''<!DOCTYPE html>
<html>
<head>
    <title>viz2psy Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        * {{
            box-sizing: border-box;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
        }}
        body {{
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: {width + 100}px;
            margin: 0 auto;
        }}
        h1 {{
            color: #333;
            margin-bottom: 5px;
        }}
        .subtitle {{
            color: #666;
            margin-bottom: 20px;
        }}
        .controls {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .control-group {{
            background: white;
            padding: 15px 20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        .control-group label {{
            display: block;
            font-weight: 600;
            margin-bottom: 8px;
            color: #333;
        }}
        select {{
            padding: 8px 12px;
            font-size: 14px;
            border: 1px solid #ddd;
            border-radius: 4px;
            min-width: 200px;
        }}
        .viz-buttons {{
            display: flex;
            gap: 8px;
        }}
        .viz-btn {{
            padding: 8px 16px;
            font-size: 14px;
            border: 2px solid #ddd;
            border-radius: 4px;
            background: white;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .viz-btn:hover {{
            border-color: #007bff;
        }}
        .viz-btn.active {{
            background: #007bff;
            color: white;
            border-color: #007bff;
        }}
        .viz-btn.disabled {{
            opacity: 0.4;
            cursor: not-allowed;
        }}
        .sub-options {{
            display: none;
            margin-top: 10px;
            padding-top: 10px;
            border-top: 1px solid #eee;
        }}
        .sub-options.visible {{
            display: block;
        }}
        .sub-toggle {{
            display: inline-flex;
            background: #f0f0f0;
            border-radius: 4px;
            overflow: hidden;
        }}
        .sub-toggle button {{
            padding: 6px 12px;
            font-size: 12px;
            border: none;
            background: transparent;
            cursor: pointer;
            transition: all 0.2s;
        }}
        .sub-toggle button.active {{
            background: #007bff;
            color: white;
        }}
        .sub-toggle button:hover:not(.active) {{
            background: #e0e0e0;
        }}
        .plot-container {{
            background: white;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            padding: 20px;
        }}
        .warning {{
            display: none;
            text-align: center;
            padding: 60px 20px;
            color: #666;
        }}
        .warning .emoji {{
            font-size: 64px;
            margin-bottom: 20px;
        }}
        .warning .message {{
            font-size: 18px;
            margin-bottom: 10px;
        }}
        .warning .detail {{
            font-size: 14px;
            color: #999;
        }}
        .model-info {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>viz2psy Feature Dashboard</h1>
        <p class="subtitle">Explore model outputs with different visualization types. Click a point to see details.</p>

        <div class="controls">
            <div class="control-group">
                <label>Model Output</label>
                <select id="model-select" onchange="updatePlot()">
                    {_generate_model_options(models, model_info)}
                </select>
                <div id="model-info" class="model-info"></div>
            </div>

            <div class="control-group">
                <label>Visualization</label>
                <div class="viz-buttons">
                    <button class="viz-btn active" data-viz="timeseries" onclick="selectViz('timeseries')">
                        📈 Time Series
                    </button>
                    <button class="viz-btn" data-viz="clustering" onclick="selectViz('clustering')">
                        🔵 Clustering
                    </button>
                    <button class="viz-btn" data-viz="trajectory" onclick="selectViz('trajectory')">
                        🚀 Trajectory
                    </button>
                </div>

                <div id="sub-clustering" class="sub-options">
                    <div class="sub-toggle">
                        <button class="active" data-sub="2d" onclick="selectSubOption('clustering', '2d')">2D</button>
                        <button data-sub="3d" onclick="selectSubOption('clustering', '3d')">3D</button>
                    </div>
                </div>

                <div id="sub-trajectory" class="sub-options">
                    <div class="sub-toggle">
                        <button class="active" data-sub="static" onclick="selectSubOption('trajectory', 'static')">Static</button>
                        <button data-sub="animated" onclick="selectSubOption('trajectory', 'animated')">▶ Animated</button>
                    </div>
                </div>
            </div>
        </div>

        <div class="plot-container">
            <div id="plot" style="width: {width}px; height: {height}px;"></div>
            <div id="warning" class="warning">
                <div class="emoji">😕</div>
                <div class="message" id="warning-message">Not available</div>
                <div class="detail" id="warning-detail"></div>
            </div>
        </div>
    </div>

    <script>
        // Data
        const traces = {json.dumps(traces_json)};
        const layouts = {json.dumps(layouts_json)};
        const frames = {json.dumps(frames_json)};
        const availability = {json.dumps(availability_json)};
        const modelInfo = {json.dumps(model_info)};
        // Viewer HTMLs are base64 encoded to avoid escaping issues
        const viewerHtmlsB64 = {json.dumps([base64.b64encode(h.encode()).decode() for h in viewer_htmls])};

        // State
        let currentModel = '{models[0] if models else ""}';
        let currentViz = 'timeseries';
        let subOptions = {{
            clustering: '2d',
            trajectory: 'static'
        }};

        // Warning messages for unavailable combinations
        const warnings = {{
            'timeseries': {{
                'message': 'Time series not available for this model',
                'detail': 'High-dimensional embeddings are not interpretable as individual time series.'
            }},
            'clustering': {{
                'message': 'Clustering not available for this model',
                'detail': 'Single-value outputs cannot be projected to 2D/3D.'
            }},
            'trajectory': {{
                'message': 'Trajectory not available for this model',
                'detail': 'Single-value outputs cannot show state-space evolution.'
            }}
        }};

        // Open single-image viewer in new tab
        function openViewer(rowIdx) {{
            if (rowIdx < 0 || rowIdx >= viewerHtmlsB64.length) return;

            // Decode base64
            const viewerHtml = atob(viewerHtmlsB64[rowIdx]);
            const newWindow = window.open('', '_blank');
            if (newWindow) {{
                newWindow.document.write(viewerHtml);
                newWindow.document.close();
            }} else {{
                alert('Popup blocked. Please allow popups for this site.');
            }}
        }}

        // Handle plot clicks - directly open viewer
        function setupClickHandler() {{
            const plotDiv = document.getElementById('plot');
            plotDiv.on('plotly_click', function(data) {{
                if (data.points && data.points.length > 0) {{
                    const point = data.points[0];
                    // Get row index from customdata or pointIndex
                    let rowIdx;
                    if (point.customdata && point.customdata.length > 0) {{
                        rowIdx = point.customdata[0];
                    }} else {{
                        rowIdx = point.pointIndex;
                    }}
                    if (typeof rowIdx === 'number') {{
                        openViewer(rowIdx);
                    }}
                }}
            }});
        }}

        function selectViz(viz) {{
            currentViz = viz;

            // Update button states
            document.querySelectorAll('.viz-btn').forEach(btn => {{
                btn.classList.remove('active');
                if (btn.dataset.viz === viz) {{
                    btn.classList.add('active');
                }}
            }});

            // Show/hide sub-options
            document.getElementById('sub-clustering').classList.toggle('visible', viz === 'clustering');
            document.getElementById('sub-trajectory').classList.toggle('visible', viz === 'trajectory');

            updatePlot();
        }}

        function selectSubOption(viz, sub) {{
            subOptions[viz] = sub;

            // Update toggle button states
            const container = document.getElementById('sub-' + viz);
            container.querySelectorAll('button').forEach(btn => {{
                btn.classList.toggle('active', btn.dataset.sub === sub);
            }});

            updatePlot();
        }}

        function getVizKey() {{
            if (currentViz === 'timeseries') {{
                return 'timeseries';
            }} else if (currentViz === 'clustering') {{
                return 'clustering_' + subOptions.clustering;
            }} else if (currentViz === 'trajectory') {{
                return 'trajectory_' + subOptions.trajectory;
            }}
            return currentViz;
        }}

        function updatePlot() {{
            currentModel = document.getElementById('model-select').value;
            const vizKey = getVizKey();

            // Update model info
            const info = modelInfo[currentModel];
            if (info) {{
                document.getElementById('model-info').textContent =
                    `${{info.n_dims}} dimensions - ${{info.description}}`;
            }}

            // Update main viz button disabled states
            document.querySelectorAll('.viz-btn').forEach(btn => {{
                const viz = btn.dataset.viz;
                let isAvailable = false;
                if (viz === 'timeseries') {{
                    isAvailable = availability[currentModel] && availability[currentModel]['timeseries'];
                }} else if (viz === 'clustering') {{
                    isAvailable = availability[currentModel] &&
                                 (availability[currentModel]['clustering_2d'] || availability[currentModel]['clustering_3d']);
                }} else if (viz === 'trajectory') {{
                    isAvailable = availability[currentModel] &&
                                 (availability[currentModel]['trajectory_static'] || availability[currentModel]['trajectory_animated']);
                }}
                btn.classList.toggle('disabled', !isAvailable);
            }});

            // Check availability for current selection
            const isAvailable = availability[currentModel] &&
                               availability[currentModel][vizKey] &&
                               traces[currentModel] &&
                               traces[currentModel][vizKey] &&
                               traces[currentModel][vizKey].length > 0;

            const plotDiv = document.getElementById('plot');
            const warningDiv = document.getElementById('warning');

            if (!isAvailable) {{
                // Show warning
                plotDiv.style.display = 'none';
                warningDiv.style.display = 'block';

                const warn = warnings[currentViz] || {{message: 'Not available', detail: ''}};
                document.getElementById('warning-message').textContent = warn.message;
                document.getElementById('warning-detail').textContent = warn.detail;
                return;
            }}

            // Show plot
            plotDiv.style.display = 'block';
            warningDiv.style.display = 'none';

            const traceData = traces[currentModel][vizKey];
            const layoutData = layouts[currentModel][vizKey];
            const frameData = frames[currentModel] && frames[currentModel][vizKey];

            // Build layout
            let layout = {{
                ...layoutData,
                width: {width},
                height: {height},
                margin: {{ t: 50, r: 50, b: 50, l: 60 }},
                hovermode: 'closest',
            }};

            // Handle 3D layout
            if (vizKey === 'clustering_3d') {{
                layout.scene = layoutData.scene || {{}};
            }}

            // Handle animation
            if (vizKey === 'trajectory_animated' && frameData && frameData.length > 0) {{
                // Fix axis ranges for animation
                if (layoutData.xaxis_range) {{
                    layout.xaxis = layout.xaxis || {{}};
                    layout.xaxis.range = layoutData.xaxis_range;
                    layout.xaxis.autorange = false;
                }}
                if (layoutData.yaxis_range) {{
                    layout.yaxis = layout.yaxis || {{}};
                    layout.yaxis.range = layoutData.yaxis_range;
                    layout.yaxis.autorange = false;
                }}

                // Add animation controls
                layout.updatemenus = [{{
                    type: 'buttons',
                    showactive: false,
                    y: 0,
                    x: 0.1,
                    xanchor: 'right',
                    yanchor: 'top',
                    pad: {{t: 60, r: 10}},
                    buttons: [
                        {{
                            label: '▶ Play',
                            method: 'animate',
                            args: [null, {{
                                fromcurrent: true,
                                frame: {{duration: 100, redraw: true}},
                                transition: {{duration: 50}}
                            }}]
                        }},
                        {{
                            label: '⏸ Pause',
                            method: 'animate',
                            args: [[null], {{
                                mode: 'immediate',
                                frame: {{duration: 0, redraw: false}},
                                transition: {{duration: 0}}
                            }}]
                        }}
                    ]
                }}];

                // Add slider
                layout.sliders = [{{
                    active: 0,
                    steps: frameData.map((f, i) => ({{
                        label: f.name,
                        method: 'animate',
                        args: [[f.name], {{
                            mode: 'immediate',
                            frame: {{duration: 0, redraw: true}},
                            transition: {{duration: 0}}
                        }}]
                    }})),
                    x: 0.1,
                    len: 0.8,
                    xanchor: 'left',
                    y: 0,
                    yanchor: 'top',
                    pad: {{t: 40, b: 10}},
                    currentvalue: {{
                        visible: true,
                        prefix: 'Frame: ',
                        xanchor: 'right',
                        font: {{size: 12}}
                    }},
                    transition: {{duration: 0}}
                }}];

                // Make room for controls
                layout.margin.b = 100;

                Plotly.react('plot', traceData, layout).then(() => {{
                    Plotly.addFrames('plot', frameData);
                    setupClickHandler();
                }});
            }} else {{
                Plotly.react('plot', traceData, layout).then(() => {{
                    setupClickHandler();
                }});
            }}
        }}

        // Initialize
        document.addEventListener('DOMContentLoaded', function() {{
            updatePlot();
        }});
    </script>
</body>
</html>'''

    return html


def _generate_model_options(models: list[str], model_info: dict) -> str:
    """Generate HTML options for model select."""
    options = []
    for model in models:
        info = model_info.get(model, {})
        desc = info.get("description", "")
        dims = info.get("n_dims", "?")
        options.append(f'<option value="{model}">{model} ({dims}d)</option>')
    return "\n                    ".join(options)
