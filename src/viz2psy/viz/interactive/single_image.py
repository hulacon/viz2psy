"""Single-image viewer with feature visualizations using Plotly.

Provides a two-panel layout:
- Left: The image itself
- Right: Selectable feature visualizations

Feature visualization options:
- Caption text display (caption model)
- Emotion bar chart (emonet)
- Emotion spider/radar plot (emonet)
- Saliency heatmap
- Scalar features bar chart (memorability, aesthetics, etc.)
- CLIP word cloud (requires wordcloud package)
"""

from __future__ import annotations

import fnmatch
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from PIL import Image as PILImage

from .base import configure_theme, encode_image_base64

# For wordcloud rendering
import io
import base64

if TYPE_CHECKING:
    from ..sidecar import SidecarMetadata


# Known feature groups
EMOTION_FEATURES = [
    "Adoration", "Aesthetic Appreciation", "Amusement", "Anxiety", "Awe",
    "Boredom", "Confusion", "Craving", "Disgust", "Empathic Pain",
    "Entrancement", "Excitement", "Fear", "Horror", "Interest",
    "Joy", "Romance", "Sadness", "Sexual Desire", "Surprise"
]

SCALAR_FEATURES = ["memorability", "aesthetics"]


def _wrap_caption(text: str, max_chars: int = 50) -> str:
    """Wrap caption text with <br> tags for display in panels.

    Parameters
    ----------
    text : str
        The caption text to wrap.
    max_chars : int
        Maximum characters per line before wrapping.

    Returns
    -------
    str
        Caption with <br> tags inserted at word boundaries.
    """
    if len(text) <= max_chars:
        return text

    words = text.split()
    lines = []
    current_line = []
    current_length = 0

    for word in words:
        word_len = len(word)
        # +1 for space between words
        if current_length + word_len + (1 if current_line else 0) <= max_chars:
            current_line.append(word)
            current_length += word_len + (1 if len(current_line) > 1 else 0)
        else:
            if current_line:
                lines.append(" ".join(current_line))
            current_line = [word]
            current_length = word_len

    if current_line:
        lines.append(" ".join(current_line))

    return "<br>".join(lines)


def _render_saliency_to_image(
    grid: np.ndarray,
    target_width: int = 400,
    target_height: int = 300,
    colormap: str = "hot",
) -> np.ndarray:
    """Render saliency grid as a colored image array matching target dimensions.

    Parameters
    ----------
    grid : np.ndarray
        2D saliency grid (e.g., 24x24).
    target_width : int
        Target image width in pixels.
    target_height : int
        Target image height in pixels.
    colormap : str
        Matplotlib colormap name.

    Returns
    -------
    np.ndarray
        RGB image array of shape (target_height, target_width, 3).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    # Normalize grid to [0, 1]
    norm = Normalize(vmin=grid.min(), vmax=grid.max())
    normalized = norm(grid)

    # Apply colormap
    cmap = plt.get_cmap(colormap)
    colored = cmap(normalized)[:, :, :3]  # Drop alpha channel

    # Convert to uint8
    colored_uint8 = (colored * 255).astype(np.uint8)

    # Resize to target dimensions using PIL
    img = PILImage.fromarray(colored_uint8)
    img_resized = img.resize((target_width, target_height), PILImage.Resampling.NEAREST)

    return np.array(img_resized)


def _render_wordcloud_to_base64(
    row: pd.Series,
    top_n: int = 50,
    width: int = 400,
    height: int = 300,
) -> str | None:
    """Render CLIP word cloud and return as base64 PNG.

    Returns None if CLIP columns not found or wordcloud fails.
    """
    try:
        from ..wordcloud import (
            get_clip_columns,
            load_clip_text_encoder,
            compute_text_embeddings,
            compute_word_similarities,
            DEFAULT_VOCABULARY,
        )
        from wordcloud import WordCloud
        import torch
    except ImportError:
        return None

    # Get CLIP embedding
    clip_cols = [c for c in row.index if c.startswith("clip_")]
    if not clip_cols:
        return None

    image_embedding = row[clip_cols].values.astype(np.float32)

    # Load CLIP text encoder
    try:
        model, tokenizer = load_clip_text_encoder()
    except Exception:
        return None

    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Compute text embeddings and similarities
    text_embeddings = compute_text_embeddings(DEFAULT_VOCABULARY, model, tokenizer, device)
    word_scores = compute_word_similarities(image_embedding, text_embeddings, DEFAULT_VOCABULARY)

    # Keep top N
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    word_scores = dict(sorted_words)

    if not word_scores:
        return None

    # Generate word cloud
    wc = WordCloud(
        width=width,
        height=height,
        background_color="white",
        colormap="viridis",
        max_words=top_n,
    ).generate_from_frequencies(word_scores)

    # Render to PNG bytes
    img_array = wc.to_array()
    from PIL import Image
    img = Image.fromarray(img_array)

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)

    return base64.b64encode(buf.read()).decode("utf-8")


# Known ranges for normalization (min, max)
FEATURE_RANGES = {
    # Model outputs (typically 0-1)
    "memorability": (0, 1),
    "aesthetic_score": (0, 10),  # LAION aesthetics
    "aesthetics": (0, 1),

    # Low-level stats
    "luminance_mean": (0, 255),
    "luminance_std": (0, 128),
    "rms_contrast": (0, 1),
    "r_mean": (0, 255),
    "r_std": (0, 128),
    "g_mean": (0, 255),
    "g_std": (0, 128),
    "b_mean": (0, 255),
    "b_std": (0, 128),
    "lab_l_mean": (0, 100),
    "lab_a_mean": (-128, 127),
    "lab_b_mean": (-128, 127),
    "saturation_mean": (0, 1),
    "hf_energy": (0, 1),
    "lf_energy": (0, 1),
    "edge_density": (0, 1),
    "colorfulness": (0, 200),

    # YOLO stats
    "object_count": (0, 50),
    "category_count": (0, 20),
    "object_coverage": (0, 1),
    "largest_object_ratio": (0, 1),
    "mean_confidence": (0, 1),
}


def _get_available_features(row: pd.Series) -> dict[str, list[str]]:
    """Detect which feature groups are available in the data."""
    available = {}

    # Check for emotion features
    emotions = [f for f in EMOTION_FEATURES if f in row.index]
    if emotions:
        available["emotions"] = emotions

    # Check for scalar features
    scalars = [f for f in SCALAR_FEATURES if f in row.index]
    if scalars:
        available["scalars"] = scalars

    # Check for saliency grid
    saliency_cols = [c for c in row.index if c.startswith("saliency_")]
    if saliency_cols:
        available["saliency"] = saliency_cols

    # Check for CLIP embeddings
    clip_cols = [c for c in row.index if c.startswith("clip_")]
    if clip_cols:
        available["clip"] = clip_cols

    # Check for YOLO detections
    yolo_cols = [c for c in row.index if c.startswith("yolo_")]
    if yolo_cols:
        available["yolo"] = yolo_cols

    # Check for Places predictions
    places_cols = [c for c in row.index if c.startswith("places_")]
    if places_cols:
        available["places"] = places_cols

    # Check for caption
    if "caption" in row.index and pd.notna(row["caption"]):
        available["caption"] = ["caption"]

    return available


def plot_emotions_bar(
    row: pd.Series,
    features: list[str] | None = None,
    title: str = "Emotion Predictions",
    color: str = "#636EFA",
) -> go.Figure:
    """Create horizontal bar chart of emotion predictions.

    Parameters
    ----------
    row : pd.Series
        Single row from scores DataFrame.
    features : list of str, optional
        Emotion features to include. If None, uses all available.
    title : str
        Chart title.
    color : str
        Bar color.

    Returns
    -------
    go.Figure
    """
    if features is None:
        features = [f for f in EMOTION_FEATURES if f in row.index]

    values = [row[f] for f in features]

    # Sort by value
    sorted_pairs = sorted(zip(features, values), key=lambda x: x[1], reverse=True)
    features, values = zip(*sorted_pairs)

    fig = go.Figure(go.Bar(
        x=values,
        y=features,
        orientation="h",
        marker_color=color,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Probability",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),  # Highest at top
        height=max(400, len(features) * 25),
    )

    return fig


def plot_emotions_spider(
    row: pd.Series,
    features: list[str] | None = None,
    title: str = "Emotion Profile",
    fill_color: str = "rgba(99, 110, 250, 0.3)",
    line_color: str = "#636EFA",
) -> go.Figure:
    """Create spider/radar plot of emotion predictions.

    Parameters
    ----------
    row : pd.Series
        Single row from scores DataFrame.
    features : list of str, optional
        Emotion features to include. If None, uses all available.
    title : str
        Chart title.
    fill_color : str
        Fill color (with alpha).
    line_color : str
        Line color.

    Returns
    -------
    go.Figure
    """
    if features is None:
        features = [f for f in EMOTION_FEATURES if f in row.index]

    values = [row[f] for f in features]
    # Close the polygon
    values = values + [values[0]]
    features = list(features) + [features[0]]

    fig = go.Figure(go.Scatterpolar(
        r=values,
        theta=features,
        fill="toself",
        fillcolor=fill_color,
        line=dict(color=line_color),
        hovertemplate="%{theta}: %{r:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        polar=dict(
            radialaxis=dict(visible=True, range=[0, max(values) * 1.1]),
        ),
        height=500,
    )

    return fig


def plot_scalars_bar(
    row: pd.Series,
    features: list[str] | None = None,
    title: str = "Image Scores",
    color: str = "#00CC96",
) -> go.Figure:
    """Create bar chart of scalar features.

    Parameters
    ----------
    row : pd.Series
        Single row from scores DataFrame.
    features : list of str, optional
        Scalar features to include.
    title : str
        Chart title.
    color : str
        Bar color.

    Returns
    -------
    go.Figure
    """
    if features is None:
        features = [f for f in SCALAR_FEATURES if f in row.index]

    if not features:
        # Try to find any scalar-like features
        exclude = ["time", "index", "filename", "filepath", "image_idx"]
        embedding_prefixes = ["clip_", "gist_", "dinov2_", "saliency_", "places_", "yolo_", "sunattr_"]
        features = [
            c for c in row.index
            if c not in exclude
            and c not in EMOTION_FEATURES
            and not any(c.startswith(p) for p in embedding_prefixes)
            and isinstance(row[c], (int, float))
            and not pd.isna(row[c])
        ]

    values = [row[f] for f in features]

    fig = go.Figure(go.Bar(
        x=features,
        y=values,
        marker_color=color,
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="",
        yaxis_title="Score",
        height=400,
    )

    return fig


def plot_saliency_heatmap(
    row: pd.Series,
    grid_size: int = 24,
    title: str = "Saliency Map",
    colorscale: str = "Hot",
) -> go.Figure:
    """Create heatmap of saliency predictions.

    Parameters
    ----------
    row : pd.Series
        Single row from scores DataFrame.
    grid_size : int
        Size of saliency grid (assumes square).
    title : str
        Chart title.
    colorscale : str
        Plotly colorscale name.

    Returns
    -------
    go.Figure
    """
    # Extract saliency values and reshape to grid
    saliency_cols = sorted([c for c in row.index if c.startswith("saliency_")])

    if not saliency_cols:
        raise ValueError("No saliency features found in data.")

    # Try to infer grid size from column names (e.g., saliency_23_23)
    try:
        last_col = saliency_cols[-1]
        parts = last_col.replace("saliency_", "").split("_")
        if len(parts) == 2:
            grid_size = int(parts[0]) + 1
    except (ValueError, IndexError):
        pass

    values = np.array([row[c] for c in saliency_cols])

    # Reshape to 2D grid
    try:
        grid = values.reshape(grid_size, grid_size)
    except ValueError:
        # If reshape fails, try square root
        side = int(np.sqrt(len(values)))
        grid = values.reshape(side, side)

    fig = go.Figure(go.Heatmap(
        z=grid,
        colorscale=colorscale,
        hovertemplate="Row: %{y}<br>Col: %{x}<br>Saliency: %{z:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis=dict(scaleanchor="y", constrain="domain"),
        yaxis=dict(autorange="reversed"),  # Origin at top-left like image
        height=450,
        width=450,
    )

    return fig


def plot_top_detections(
    row: pd.Series,
    prefix: str = "yolo_",
    top_n: int = 10,
    title: str = "Top Detections",
    color: str = "#EF553B",
) -> go.Figure:
    """Create bar chart of top object/scene detections.

    Works for YOLO, Places, or any prefix_name format.

    Parameters
    ----------
    row : pd.Series
        Single row from scores DataFrame.
    prefix : str
        Column prefix (e.g., "yolo_", "places_").
    top_n : int
        Number of top detections to show.
    title : str
        Chart title.
    color : str
        Bar color.

    Returns
    -------
    go.Figure
    """
    cols = [c for c in row.index if c.startswith(prefix)]

    if not cols:
        raise ValueError(f"No {prefix}* features found in data.")

    # Get values and labels
    data = [(c.replace(prefix, ""), row[c]) for c in cols if not pd.isna(row[c])]

    # Sort by value and take top N
    data = sorted(data, key=lambda x: x[1], reverse=True)[:top_n]

    if not data:
        raise ValueError(f"No non-zero {prefix}* values found.")

    labels, values = zip(*data)

    fig = go.Figure(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=color,
        hovertemplate="%{y}: %{x:.3f}<extra></extra>",
    ))

    fig.update_layout(
        title=title,
        xaxis_title="Score/Count",
        yaxis_title="",
        yaxis=dict(autorange="reversed"),
        height=max(300, len(labels) * 30),
    )

    return fig


def create_single_image_viewer(
    image: str | Path | PILImage.Image,
    scores_df: pd.DataFrame,
    row_idx: int = 0,
    panels: list[str] | None = None,
    width: int = 1200,
    height: int = 600,
    sidecar: SidecarMetadata | None = None,
    normalize_scalars: bool = True,
    image_title: str | None = None,
) -> go.Figure:
    """Create two-panel viewer: image + feature visualization with dropdown.

    Parameters
    ----------
    image : str, Path, or PIL.Image
        Path to the image file, or a PIL Image object (for video frames/HDF5).
    scores_df : pd.DataFrame
        DataFrame with feature scores.
    row_idx : int
        Row index in scores_df for this image.
    panels : list of str, optional
        Which feature panels to include. Options:
        - "caption": Text display of image caption
        - "emotions_bar": Horizontal bar chart of emotions
        - "emotions_spider": Radar plot of emotions
        - "scalars": Bar chart of memorability, aesthetics, etc.
        - "saliency": Saliency heatmap
        - "yolo": Top YOLO detections
        - "places": Top Places predictions
        - "wordcloud": CLIP word cloud (requires wordcloud package)
        If None, auto-detects available features.
    width : int
        Total figure width.
    height : int
        Total figure height.
    sidecar : SidecarMetadata, optional
        Sidecar metadata for semantic labels.
    normalize_scalars : bool
        If True, normalize scalar features to [0, 1] using known ranges.
        Hover text shows original values. Default True.
    image_title : str, optional
        Title for the image (used if image is PIL Image without filename).

    Returns
    -------
    go.Figure
        Two-panel figure with image and feature viz, dropdown to switch panels.
    """
    configure_theme()

    # Handle different image input types
    if image is None:
        pil_image = None
        image_path = None
        title_name = image_title or f"Row {row_idx}"
    elif isinstance(image, PILImage.Image):
        pil_image = image
        image_path = None
        title_name = image_title or f"Row {row_idx}"
    else:
        image_path = Path(image)
        pil_image = None
        title_name = image_title or image_path.name

    row = scores_df.iloc[row_idx]

    # Detect available features
    available = _get_available_features(row)

    # Determine which panels to show
    if panels is None:
        panels = []
        if "caption" in available:
            panels.append("caption")
        if "emotions" in available:
            panels.append("emotions_bar")
        if "scalars" in available:
            panels.append("scalars")
        if "saliency" in available:
            panels.append("saliency")
        if "yolo" in available:
            panels.append("yolo")
        if "places" in available:
            panels.append("places")
        # Note: wordcloud disabled by default (slow CLIP text encoding)
        # Can be explicitly requested via panels=["wordcloud", ...]

    if not panels:
        panels = ["scalars"]  # Fallback

    # Separate polar (spider) from cartesian panels
    polar_panels = [p for p in panels if p == "emotions_spider"]
    cartesian_panels = [p for p in panels if p != "emotions_spider"]

    # Use polar subplot if spider is the only panel, otherwise cartesian
    has_polar = len(polar_panels) > 0 and len(cartesian_panels) == 0

    if has_polar:
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.45, 0.55],
            specs=[[{"type": "xy"}, {"type": "polar"}]],
            horizontal_spacing=0.05,
        )
    else:
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.45, 0.55],
            specs=[[{"type": "xy"}, {"type": "xy"}]],
            horizontal_spacing=0.05,
        )

    # Add image
    img_b64 = None

    if pil_image is not None:
        # Encode PIL Image to base64
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    elif image_path is not None and image_path.exists():
        img_b64 = encode_image_base64(image_path)

    if img_b64:
        # Use paper coordinates to constrain image to left panel
        # Left panel domain is approximately [0, 0.45] with 0.05 spacing
        left_panel_width = 0.45

        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{img_b64}",
                xref="paper",
                yref="paper",
                x=0,
                y=1,
                sizex=left_panel_width - 0.02,  # Leave small margin
                sizey=1,
                xanchor="left",
                yanchor="top",
                layer="below",
                sizing="contain",
            ),
        )

    # Hide axes for image panel
    fig.update_xaxes(visible=False, row=1, col=1)
    fig.update_yaxes(visible=False, row=1, col=1)

    # Track trace indices and axis configs for each panel
    panel_traces = {}
    panel_axis_configs = {}

    # Add all panel traces (initially hidden except first)
    active_panels = polar_panels if has_polar else cartesian_panels
    for i, panel_type in enumerate(active_panels):
        is_first = (i == 0)
        trace_count_before = len(fig.data)

        # Get trace data and axis config for this panel
        axis_config = _add_feature_panel_with_config(
            fig, row, panel_type, available, subplot_row=1, subplot_col=2,
            sidecar=sidecar, normalize_scalars=normalize_scalars,
        )

        trace_count_after = len(fig.data)

        # Record which traces belong to this panel
        panel_traces[panel_type] = list(range(trace_count_before, trace_count_after))
        panel_axis_configs[panel_type] = axis_config

        # Hide traces for non-first panels
        if not is_first:
            for t_idx in panel_traces[panel_type]:
                fig.data[t_idx].visible = False

    # Apply first panel's axis config
    if active_panels:
        first_config = panel_axis_configs[active_panels[0]]
        fig.update_xaxes(**first_config.get("xaxis", {}), row=1, col=2)
        fig.update_yaxes(**first_config.get("yaxis", {}), row=1, col=2)

    # Get image filename for title
    title = f"Image: {title_name}"

    fig.update_layout(
        title=title,
        width=width,
        height=height,
        showlegend=False,
    )

    # Add dropdown to switch panels if multiple available
    if len(active_panels) > 1:
        buttons = []
        total_traces = len(fig.data)

        for panel in active_panels:
            label = _panel_label(panel)

            # Create visibility array: True only for this panel's traces
            visibility = [False] * total_traces
            for t_idx in panel_traces[panel]:
                visibility[t_idx] = True

            # Get axis config for this panel
            axis_config = panel_axis_configs[panel]

            # Build layout update for axes (xaxis2 and yaxis2 for second subplot)
            layout_update = {}
            if "xaxis" in axis_config:
                layout_update["xaxis2"] = axis_config["xaxis"]
            if "yaxis" in axis_config:
                layout_update["yaxis2"] = axis_config["yaxis"]

            buttons.append(dict(
                label=label,
                method="update",
                args=[
                    {"visible": visibility},
                    layout_update,
                ],
            ))

        fig.update_layout(
            updatemenus=[dict(
                type="dropdown",
                direction="down",
                x=0.55,
                y=1.15,
                showactive=True,
                active=0,
                buttons=buttons,
                bgcolor="white",
                bordercolor="#ccc",
                font=dict(size=12),
            )]
        )

    return fig


def _panel_label(panel_type: str) -> str:
    """Get human-readable label for panel type."""
    labels = {
        "emotions_bar": "Emotions (Bar)",
        "emotions_spider": "Emotions (Radar)",
        "scalars": "Scores",
        "saliency": "Saliency",
        "yolo": "Objects (YOLO)",
        "places": "Scenes (Places)",
        "wordcloud": "Word Cloud (CLIP)",
        "caption": "Caption",
    }
    return labels.get(panel_type, panel_type)


def _normalize_scalar(value: float, feature: str) -> tuple[float, str]:
    """Normalize a scalar value to [0, 1] using known ranges.

    Returns
    -------
    tuple
        (normalized_value, tooltip_text)
    """
    if feature in FEATURE_RANGES:
        min_val, max_val = FEATURE_RANGES[feature]
        # Handle features that can be negative (like lab_a_mean)
        normalized = (value - min_val) / (max_val - min_val)
        normalized = max(0, min(1, normalized))  # Clamp to [0, 1]
        tooltip = f"{feature}: {value:.3f} (range: {min_val}-{max_val})"
        return normalized, tooltip
    else:
        # Unknown feature - assume 0-1 if in range, otherwise just use raw
        if 0 <= value <= 1:
            return value, f"{feature}: {value:.3f}"
        else:
            # Can't normalize, return raw but capped for display
            return min(1, max(0, value)), f"{feature}: {value:.3f} (raw)"


def _add_feature_panel_with_config(
    fig: go.Figure,
    data_row: pd.Series,
    panel_type: str,
    available: dict,
    subplot_row: int,
    subplot_col: int,
    sidecar=None,
    normalize_scalars: bool = True,
) -> dict:
    """Add feature visualization to subplot and return axis config.

    Returns
    -------
    dict
        Axis configuration with 'xaxis' and 'yaxis' keys.
    """
    # Base axis config - ensures axes stay in right panel domain
    axis_config = {
        "xaxis": {
            "anchor": "y2",
            "domain": [0.55, 1.0],
        },
        "yaxis": {
            "anchor": "x2",
            "domain": [0.0, 1.0],
        }
    }

    if panel_type == "emotions_bar" and "emotions" in available:
        values = [data_row[f] for f in available["emotions"]]
        sorted_pairs = sorted(zip(available["emotions"], values), key=lambda x: x[1], reverse=True)
        features, values = zip(*sorted_pairs)

        fig.add_trace(go.Bar(
            x=values,
            y=features,
            orientation="h",
            marker_color="#636EFA",
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        ), row=subplot_row, col=subplot_col)

        axis_config["xaxis"].update({
            "title": {"text": "Probability"},
            "range": [0, max(values) * 1.1],
            "autorange": False,
            "type": "linear",
            "visible": True,
            "scaleanchor": None,
        })
        axis_config["yaxis"].update({
            "title": {"text": ""},
            "autorange": "reversed",
            "range": None,
            "type": "category",
            "categoryorder": "array",
            "categoryarray": list(features),
            "visible": True,
        })

    elif panel_type == "scalars":
        # Find scalar features
        exclude = ["time", "index", "filename", "filepath", "image_idx"]
        embedding_prefixes = ["clip_", "gist_", "dinov2_", "saliency_", "places_", "yolo_", "sunattr_"]
        features = [
            c for c in data_row.index
            if c not in exclude
            and c not in EMOTION_FEATURES
            and not any(c.startswith(p) for p in embedding_prefixes)
            and isinstance(data_row[c], (int, float, np.floating))
            and not pd.isna(data_row[c])
        ]

        if features:
            raw_values = [data_row[f] for f in features]

            # Get semantic labels from sidecar if available
            if sidecar:
                labels = [sidecar.get_semantic_label(f) for f in features]
            else:
                labels = list(features)

            # Normalize values and create hover text
            if normalize_scalars:
                normalized = []
                hover_texts = []
                for f, v in zip(features, raw_values):
                    norm_v, tooltip = _normalize_scalar(v, f)
                    normalized.append(norm_v)
                    hover_texts.append(tooltip)
                values = normalized
                y_title = "Normalized Score"
            else:
                values = raw_values
                hover_texts = [f"{f}: {v:.3f}" for f, v in zip(features, raw_values)]
                y_title = "Score"

            fig.add_trace(go.Bar(
                x=labels,
                y=values,
                marker_color="#00CC96",
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover_texts,
            ), row=subplot_row, col=subplot_col)

            axis_config["xaxis"].update({
                "title": {"text": ""},
                "type": "category",
                "categoryorder": "array",
                "categoryarray": labels,
                "visible": True,
                "autorange": True,
                "range": None,
                "scaleanchor": None,
            })
            axis_config["yaxis"].update({
                "title": {"text": y_title},
                "range": [0, max(values) * 1.1] if values else [0, 1],
                "autorange": False,
                "type": "linear",
                "visible": True,
                "categoryorder": None,
                "categoryarray": None,
            })

    elif panel_type == "saliency" and "saliency" in available:
        saliency_cols = sorted(available["saliency"])
        values = np.array([data_row[c] for c in saliency_cols])
        side = int(np.sqrt(len(values)))
        grid = values.reshape(side, side)
        # Saliency columns are named saliency_X_Y (col_row), but alphabetical sort
        # gives us grid[x][y]. Transpose to get grid[y][x] for correct display.
        grid = grid.T

        fig.add_trace(go.Heatmap(
            z=grid,
            colorscale="Hot",
            hovertemplate="Row: %{y}, Col: %{x}<br>Saliency: %{z:.3f}<extra></extra>",
        ), row=subplot_row, col=subplot_col)

        axis_config["xaxis"].update({
            "title": {"text": ""},
            "range": [-0.5, side - 0.5],
            "autorange": False,
            "visible": True,
            "type": "linear",
            "categoryorder": None,
            "categoryarray": None,
            "scaleanchor": "y2",
            "scaleratio": 1,
            "constrain": "domain",
        })
        axis_config["yaxis"].update({
            "title": {"text": ""},
            "range": [side - 0.5, -0.5],  # Reversed for image-like orientation
            "autorange": False,
            "visible": True,
            "type": "linear",
            "categoryorder": None,
            "categoryarray": None,
            "constrain": "domain",
        })

    elif panel_type == "yolo" and "yolo" in available:
        cols = available["yolo"]
        data = [(c.replace("yolo_", ""), data_row[c]) for c in cols if data_row[c] > 0]
        data = sorted(data, key=lambda x: x[1], reverse=True)[:10]

        if data:
            labels, values = zip(*data)
            fig.add_trace(go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color="#EF553B",
                hovertemplate="%{y}: %{x:.0f}<extra></extra>",
            ), row=subplot_row, col=subplot_col)

            axis_config["xaxis"].update({
                "title": {"text": "Count"},
                "range": [0, max(values) * 1.1],
                "autorange": False,
                "type": "linear",
                "visible": True,
                "scaleanchor": None,
            })
            axis_config["yaxis"].update({
                "title": {"text": ""},
                "autorange": "reversed",
                "range": None,
                "type": "category",
                "categoryorder": "array",
                "categoryarray": list(labels),
                "visible": True,
            })

    elif panel_type == "places" and "places" in available:
        cols = available["places"]
        data = [(c.replace("places_", ""), data_row[c]) for c in cols if data_row[c] > 0.01]
        data = sorted(data, key=lambda x: x[1], reverse=True)[:10]

        if data:
            labels, values = zip(*data)
            fig.add_trace(go.Bar(
                x=values,
                y=labels,
                orientation="h",
                marker_color="#AB63FA",
                hovertemplate="%{y}: %{x:.3f}<extra></extra>",
            ), row=subplot_row, col=subplot_col)

            axis_config["xaxis"].update({
                "title": {"text": "Probability"},
                "range": [0, max(values) * 1.1],
                "autorange": False,
                "type": "linear",
                "visible": True,
                "scaleanchor": None,
            })
            axis_config["yaxis"].update({
                "title": {"text": ""},
                "autorange": "reversed",
                "range": None,
                "type": "category",
                "categoryorder": "array",
                "categoryarray": list(labels),
                "visible": True,
            })

    elif panel_type == "wordcloud" and "clip" in available:
        # Render wordcloud and embed as Image trace
        wc_b64 = _render_wordcloud_to_base64(data_row, top_n=50, width=500, height=400)

        if wc_b64:
            # Use go.Image to display the wordcloud (can toggle visibility)
            from PIL import Image as PILImage

            # Decode and get dimensions
            img_bytes = base64.b64decode(wc_b64)
            img = PILImage.open(io.BytesIO(img_bytes))
            img_array = np.array(img)

            fig.add_trace(go.Image(
                z=img_array,
                hoverinfo="skip",
            ), row=subplot_row, col=subplot_col)

            axis_config["xaxis"].update({
                "title": {"text": ""},
                "visible": False,
                "autorange": True,
                "range": None,
                "scaleanchor": "y2",
                "scaleratio": 1,
                "constrain": "domain",
            })
            axis_config["yaxis"].update({
                "title": {"text": ""},
                "visible": False,
                "autorange": "reversed",
                "range": None,
                "constrain": "domain",
            })
        else:
            # Fallback: show message that wordcloud couldn't be generated
            fig.add_trace(go.Scatter(
                x=[0.5],
                y=[0.5],
                mode="text",
                text=["Word cloud requires CLIP model"],
                textfont=dict(size=14, color="gray"),
                hoverinfo="skip",
                showlegend=False,
            ), row=subplot_row, col=subplot_col)

            axis_config["xaxis"].update({
                "title": {"text": ""},
                "range": [0, 1],
                "autorange": False,
                "visible": False,
                "scaleanchor": None,
            })
            axis_config["yaxis"].update({
                "title": {"text": ""},
                "range": [0, 1],
                "autorange": False,
                "visible": False,
            })

    elif panel_type == "caption" and "caption" in available:
        caption_text = str(data_row.get("caption", ""))
        wrapped_caption = _wrap_caption(caption_text, max_chars=45)

        # Display caption as centered text
        fig.add_trace(go.Scatter(
            x=[0.5],
            y=[0.5],
            mode="text",
            text=[f"<b>Caption:</b><br><br><i>{wrapped_caption}</i>"],
            textfont=dict(size=14, color="#333"),
            textposition="middle center",
            hoverinfo="skip",
            showlegend=False,
        ), row=subplot_row, col=subplot_col)

        axis_config["xaxis"].update({
            "title": {"text": ""},
            "range": [0, 1],
            "autorange": False,
            "visible": False,
            "scaleanchor": None,
        })
        axis_config["yaxis"].update({
            "title": {"text": ""},
            "range": [0, 1],
            "autorange": False,
            "visible": False,
        })

    return axis_config


def _add_feature_panel(
    fig: go.Figure,
    data_row: pd.Series,
    panel_type: str,
    available: dict,
    subplot_row: int,
    subplot_col: int,
    sidecar=None,
) -> None:
    """Add feature visualization to subplot (legacy, no config return)."""
    _add_feature_panel_with_config(fig, data_row, panel_type, available, subplot_row, subplot_col, sidecar=sidecar)


# Standalone visualization functions for use in notebooks

def view_image_emotions(
    image_path: str | Path,
    scores_df: pd.DataFrame,
    row_idx: int = 0,
    style: str = "bar",
    width: int = 1000,
    height: int = 500,
) -> go.Figure:
    """View image alongside emotion predictions.

    Parameters
    ----------
    image_path : str or Path
        Path to the image.
    scores_df : pd.DataFrame
        Feature scores DataFrame.
    row_idx : int
        Row index for this image.
    style : str
        "bar" for horizontal bars, "spider" for radar plot.
    width, height : int
        Figure dimensions.

    Returns
    -------
    go.Figure
    """
    configure_theme()

    image_path = Path(image_path)
    row = scores_df.iloc[row_idx]

    emotions = [f for f in EMOTION_FEATURES if f in row.index]
    if not emotions:
        raise ValueError("No emotion features found in data.")

    # Create side-by-side layout
    if style == "spider":
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.5, 0.5],
            specs=[[{"type": "xy"}, {"type": "polar"}]],
        )
    else:
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.45, 0.55],
        )

    # Add image
    if image_path.exists():
        img_b64 = encode_image_base64(image_path)
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{img_b64}",
                xref="x domain",
                yref="y domain",
                x=0, y=1,
                sizex=1, sizey=1,
                xanchor="left", yanchor="top",
                sizing="contain",
            ),
            row=1, col=1,
        )
        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_yaxes(visible=False, row=1, col=1)

    # Add emotion viz
    values = [row[f] for f in emotions]

    if style == "spider":
        # Radar plot
        values_closed = values + [values[0]]
        emotions_closed = emotions + [emotions[0]]
        fig.add_trace(go.Scatterpolar(
            r=values_closed,
            theta=emotions_closed,
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.3)",
            line=dict(color="#636EFA"),
        ), row=1, col=2)
    else:
        # Bar chart
        sorted_pairs = sorted(zip(emotions, values), key=lambda x: x[1], reverse=True)
        emotions_sorted, values_sorted = zip(*sorted_pairs)
        fig.add_trace(go.Bar(
            x=values_sorted,
            y=emotions_sorted,
            orientation="h",
            marker_color="#636EFA",
        ), row=1, col=2)
        fig.update_yaxes(autorange="reversed", row=1, col=2)

    fig.update_layout(
        title=f"Emotions: {image_path.name}",
        width=width,
        height=height,
        showlegend=False,
    )

    return fig


def view_image_saliency(
    image_path: str | Path,
    scores_df: pd.DataFrame,
    row_idx: int = 0,
    width: int = 900,
    height: int = 400,
) -> go.Figure:
    """View image alongside saliency heatmap.

    Parameters
    ----------
    image_path : str or Path
        Path to the image.
    scores_df : pd.DataFrame
        Feature scores DataFrame.
    row_idx : int
        Row index for this image.
    width, height : int
        Figure dimensions.

    Returns
    -------
    go.Figure
    """
    configure_theme()

    image_path = Path(image_path)
    row = scores_df.iloc[row_idx]

    saliency_cols = sorted([c for c in row.index if c.startswith("saliency_")])
    if not saliency_cols:
        raise ValueError("No saliency features found in data.")

    fig = make_subplots(
        rows=1, cols=2,
        column_widths=[0.5, 0.5],
        subplot_titles=["Original Image", "Saliency Map"],
    )

    # Add image
    if image_path.exists():
        img_b64 = encode_image_base64(image_path)
        fig.add_layout_image(
            dict(
                source=f"data:image/png;base64,{img_b64}",
                xref="x domain",
                yref="y domain",
                x=0, y=1,
                sizex=1, sizey=1,
                xanchor="left", yanchor="top",
                sizing="contain",
            ),
            row=1, col=1,
        )
        fig.update_xaxes(visible=False, row=1, col=1)
        fig.update_yaxes(visible=False, row=1, col=1)

    # Add saliency
    values = np.array([row[c] for c in saliency_cols])
    side = int(np.sqrt(len(values)))
    grid = values.reshape(side, side)

    fig.add_trace(go.Heatmap(
        z=grid,
        colorscale="Hot",
        showscale=True,
    ), row=1, col=2)

    fig.update_yaxes(autorange="reversed", scaleanchor="x2", row=1, col=2)
    fig.update_xaxes(visible=False, row=1, col=2)
    fig.update_yaxes(visible=False, row=1, col=2)

    fig.update_layout(
        title=f"Saliency: {image_path.name}",
        width=width,
        height=height,
    )

    return fig


def create_browsable_viewer(
    scores_df: pd.DataFrame,
    image_resolver,
    panels: list[str] | None = None,
    width: int = 1200,
    height: int = 700,
    max_rows: int = 1000,
    sidecar: SidecarMetadata | None = None,
    normalize_scalars: bool = True,
    embed_images: bool = False,
) -> go.Figure:
    """Create a browsable viewer with slider to navigate through rows.

    Parameters
    ----------
    scores_df : pd.DataFrame
        DataFrame with feature scores.
    image_resolver : UnifiedImageResolver
        Resolver for getting images from rows.
    panels : list of str, optional
        Which feature panels to include. Options:
        emotions_bar, emotions_spider, scalars, saliency, yolo, places
        If None, auto-detects available features.
    width : int
        Figure width.
    height : int
        Figure height.
    max_rows : int
        Maximum number of rows to include (default: 500).
    sidecar : SidecarMetadata, optional
        Sidecar metadata for semantic labels.
    normalize_scalars : bool
        Normalize scalar features to [0, 1].
    embed_images : bool
        If True, embed images as base64 (portable but large files).
        If False (default), use file:// URLs (fast, requires images to stay in place).

    Returns
    -------
    go.Figure
        Figure with slider for browsing rows and dropdown for panel selection.
    """
    configure_theme()

    n_rows = min(len(scores_df), max_rows)
    if n_rows <= 1:
        # Single row - just use regular viewer
        image = image_resolver.resolve(scores_df, 0)
        return create_single_image_viewer(
            image=image,
            scores_df=scores_df,
            row_idx=0,
            panels=panels,
            width=width,
            height=height,
            sidecar=sidecar,
            normalize_scalars=normalize_scalars,
        )

    # Detect available features from first row
    available = _get_available_features(scores_df.iloc[0])

    # Determine which panels to show
    if panels is None:
        panels = []
        if "caption" in available:
            panels.append("caption")
        if "emotions" in available:
            panels.append("emotions_bar")
        if "scalars" in available:
            panels.append("scalars")
        if "saliency" in available:
            panels.append("saliency")
        if "yolo" in available:
            panels.append("yolo")
        if "places" in available:
            panels.append("places")
        # Note: wordcloud disabled by default (slow CLIP text encoding)
        # Can be explicitly requested via panels=["wordcloud", ...]

    if not panels:
        panels = ["scalars"]  # Fallback

    # Pre-render all images and feature data for ALL panel types
    mode_desc = "embedded" if embed_images else "file:// URLs"
    print(f"Pre-rendering {n_rows} frames for {len(panels)} panel types ({mode_desc})...")
    frames_data = []

    for idx in range(n_rows):
        if idx % 100 == 0:
            print(f"  Processing frame {idx}/{n_rows}...")
        row = scores_df.iloc[idx]

        # Get image
        image = image_resolver.resolve(scores_df, idx) if image_resolver else None
        img_src = None  # Will be either base64 data URI or file:// URL
        img_aspect = 1.0  # Default aspect ratio (width/height)

        if image is None:
            pass
        elif isinstance(image, PILImage.Image):
            # PIL Image (from video frame or HDF5) - must embed
            img_aspect = image.width / image.height if image.height > 0 else 1.0
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)
            img_src = f"data:image/png;base64,{base64.b64encode(buf.read()).decode('utf-8')}"
        else:
            # Path to image file
            image_path = Path(image) if not isinstance(image, Path) else image
            try:
                with PILImage.open(image_path) as img:
                    img_aspect = img.width / img.height if img.height > 0 else 1.0
            except Exception:
                pass

            if embed_images:
                # Embed as base64
                img_src = f"data:image/png;base64,{encode_image_base64(image_path)}"
            else:
                # Use file:// URL (much faster, smaller HTML)
                abs_path = image_path.resolve()
                img_src = f"file://{abs_path}"

        # Get feature data for ALL panel types
        panel_data = {}
        for panel_type in panels:
            panel_data[panel_type] = _extract_panel_data(
                row, panel_type, available, normalize_scalars, img_aspect=img_aspect
            )

        # Generate label for slider
        input_type = image_resolver.detected_input_type if image_resolver else None
        label = _get_row_label(row, idx, input_type)

        frames_data.append({
            "idx": idx,
            "img_src": img_src,
            "img_aspect": img_aspect,
            "panel_data": panel_data,
            "label": label,
        })

    # Create base figure
    fig = go.Figure()

    # Add traces for each panel type (first panel visible, rest hidden)
    panel_trace_indices = {}  # Maps panel_type -> trace index
    first_frame = frames_data[0]

    for i, panel_type in enumerate(panels):
        is_first = (i == 0)
        feature_data = first_frame["panel_data"][panel_type]
        trace_idx = len(fig.data)
        panel_trace_indices[panel_type] = trace_idx

        trace = _create_panel_trace(panel_type, feature_data, visible=is_first)
        fig.add_trace(trace)

    # Create frames - each frame updates ALL panel traces
    frames = []
    for frame_data in frames_data:
        trace_updates = []
        for panel_type in panels:
            feature = frame_data["panel_data"][panel_type]
            trace = _create_panel_trace(panel_type, feature, visible=None)  # Keep current visibility
            trace_updates.append(trace)

        # Build layout update with image or placeholder
        layout_update = {}
        if frame_data["img_src"]:
            layout_update["images"] = [dict(
                source=frame_data["img_src"],
                xref="paper",
                yref="paper",
                x=0,
                y=1,
                sizex=0.43,
                sizey=0.85,
                xanchor="left",
                yanchor="top",
                layer="below",
                sizing="contain",
            )]
            layout_update["annotations"] = []  # Clear placeholder
        else:
            layout_update["images"] = []  # Clear any previous image
            layout_update["annotations"] = [dict(
                text="Image not available<br><br><i>Use --save-frames during<br>inference to enable viewing</i>",
                xref="paper",
                yref="paper",
                x=0.215,  # Center of left panel (0.43/2)
                y=0.575,  # Center vertically (1 - 0.85/2)
                showarrow=False,
                font=dict(size=14, color="gray"),
                align="center",
            )]
        layout_update["title"] = {"text": frame_data["label"]}

        frames.append(go.Frame(
            data=trace_updates,
            layout=layout_update,
            name=str(frame_data["idx"]),
        ))

    fig.frames = frames

    # Add initial image or placeholder
    if first_frame["img_src"]:
        fig.add_layout_image(dict(
            source=first_frame["img_src"],
            xref="paper",
            yref="paper",
            x=0,
            y=1,
            sizex=0.43,
            sizey=0.85,
            xanchor="left",
            yanchor="top",
            layer="below",
            sizing="contain",
        ))
    else:
        fig.add_annotation(dict(
            text="Image not available<br><br><i>Use --save-frames during<br>inference to enable viewing</i>",
            xref="paper",
            yref="paper",
            x=0.215,  # Center of left panel
            y=0.575,  # Center vertically
            showarrow=False,
            font=dict(size=14, color="gray"),
            align="center",
        ))

    # Print file size warning for embedded images
    if embed_images and n_rows > 50:
        print(f"  Note: HTML file may be large with {n_rows} embedded images")

    # Create slider
    sliders = [dict(
        active=0,
        currentvalue={"prefix": "Row: ", "visible": True, "xanchor": "center"},
        pad={"b": 10, "t": 50},
        len=0.9,
        x=0.05,
        y=0,
        steps=[
            dict(
                args=[
                    [str(fd["idx"])],
                    {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}
                ],
                label=fd["label"][:20],  # Truncate for slider
                method="animate",
            )
            for fd in frames_data
        ],
    )]

    # Configure base layout with constrained right panel domain
    fig.update_layout(
        title=first_frame["label"],
        width=width,
        height=height,
        sliders=sliders,
        xaxis=dict(
            domain=[0.52, 0.98],  # Right panel with margins
            anchor="y",
        ),
        yaxis=dict(
            domain=[0.15, 0.92],
            anchor="x",
        ),
        showlegend=False,
    )

    # Set initial axis config based on first panel
    _apply_panel_axis_config(fig, panels[0], first_frame.get("img_aspect", 1.0))

    # Add dropdown to switch panels if multiple available
    if len(panels) > 1:
        buttons = []
        # Use first frame's aspect ratio for saliency
        representative_aspect = first_frame.get("img_aspect", 1.0)

        for panel_type in panels:
            label = _panel_label(panel_type)

            # Create visibility array
            visibility = [False] * len(panels)
            visibility[panels.index(panel_type)] = True

            # Get axis config for this panel
            axis_updates = _get_panel_axis_updates(panel_type, representative_aspect)

            buttons.append(dict(
                label=label,
                method="update",
                args=[
                    {"visible": visibility},
                    axis_updates,
                ],
            ))

        fig.update_layout(
            updatemenus=[dict(
                type="dropdown",
                direction="down",
                x=0.55,
                y=1.12,
                showactive=True,
                active=0,
                buttons=buttons,
                bgcolor="white",
                bordercolor="#ccc",
                font=dict(size=12),
            )]
        )

    return fig


def _create_panel_trace(panel_type: str, feature_data: dict, visible: bool | None = True) -> go.Bar | go.Heatmap | go.Scatterpolar | go.Image:
    """Create a trace for a panel type with given data."""
    if panel_type in ("emotions_bar", "yolo", "places"):
        trace = go.Bar(
            x=feature_data.get("values", []),
            y=feature_data.get("labels", []),
            orientation="h",
            marker_color=feature_data.get("color", "#636EFA"),
            hovertemplate="%{y}: %{x:.3f}<extra></extra>",
        )
    elif panel_type == "scalars":
        trace = go.Bar(
            x=feature_data.get("labels", []),
            y=feature_data.get("values", []),
            marker_color="#00CC96",
            hovertemplate="%{customdata}<extra></extra>",
            customdata=feature_data.get("hover_texts", []),
        )
    elif panel_type == "saliency":
        grid = feature_data.get("grid")
        if grid is not None:
            trace = go.Heatmap(
                z=grid,
                colorscale="Hot",
                hovertemplate="Row: %{y}, Col: %{x}<br>Saliency: %{z:.3f}<extra></extra>",
                showscale=False,
            )
        else:
            # Fallback to empty
            trace = go.Scatter(
                x=[0.5], y=[0.5],
                mode="text",
                text=["Saliency unavailable"],
                textfont=dict(size=14, color="gray"),
                hoverinfo="skip",
            )
    elif panel_type == "emotions_spider":
        trace = go.Scatterpolar(
            r=feature_data.get("values", []),
            theta=feature_data.get("labels", []),
            fill="toself",
            fillcolor="rgba(99, 110, 250, 0.3)",
            line=dict(color="#636EFA"),
        )
    elif panel_type == "wordcloud":
        img_array = feature_data.get("image_array")
        if img_array is not None:
            trace = go.Image(
                z=img_array,
                hoverinfo="skip",
            )
        else:
            # Fallback: empty scatter with message
            trace = go.Scatter(
                x=[0.5], y=[0.5],
                mode="text",
                text=["Word cloud unavailable"],
                textfont=dict(size=14, color="gray"),
                hoverinfo="skip",
            )
    elif panel_type == "caption":
        caption_text = feature_data.get("caption", "")
        wrapped_caption = _wrap_caption(caption_text, max_chars=45)
        trace = go.Scatter(
            x=[0.5],
            y=[0.5],
            mode="text",
            text=[f"<b>Caption:</b><br><br><i>{wrapped_caption}</i>"],
            textfont=dict(size=14, color="#333"),
            textposition="middle center",
            hoverinfo="skip",
            showlegend=False,
        )
    else:
        trace = go.Bar(x=[], y=[])

    if visible is not None:
        trace.visible = visible

    return trace


def _apply_panel_axis_config(fig: go.Figure, panel_type: str, img_aspect: float = 1.0) -> None:
    """Apply axis configuration for a panel type."""
    if panel_type in ("emotions_bar", "yolo", "places"):
        fig.update_yaxes(autorange="reversed", type="category", visible=True, scaleanchor=None)
        fig.update_xaxes(type="linear", visible=True, scaleanchor=None, constrain=None)
    elif panel_type == "scalars":
        fig.update_yaxes(autorange=True, type="linear", visible=True, scaleanchor=None)
        fig.update_xaxes(type="category", visible=True, scaleanchor=None, constrain=None)
    elif panel_type == "saliency":
        # Saliency is rendered as go.Heatmap
        fig.update_yaxes(autorange="reversed", visible=False, scaleanchor="x", scaleratio=1)
        fig.update_xaxes(visible=False, constrain="domain")
    elif panel_type == "wordcloud":
        fig.update_yaxes(autorange="reversed", visible=False, scaleanchor="x", scaleratio=1.0)
        fig.update_xaxes(visible=False, constrain="domain")
    elif panel_type == "caption":
        fig.update_yaxes(range=[0, 1], visible=False, scaleanchor=None)
        fig.update_xaxes(range=[0, 1], visible=False, scaleanchor=None, constrain=None)
    elif panel_type == "emotions_spider":
        pass  # Polar axes handled differently


def _get_panel_axis_updates(panel_type: str, img_aspect: float = 1.0) -> dict:
    """Get axis update dict for dropdown button."""
    # Base domain to keep content in right panel
    base_x_domain = [0.52, 0.98]
    base_y_domain = [0.15, 0.92]

    if panel_type in ("emotions_bar", "yolo", "places"):
        return {
            "yaxis": {
                "autorange": "reversed",
                "type": "category",
                "visible": True,
                "scaleanchor": None,
                "domain": base_y_domain,
            },
            "xaxis": {
                "type": "linear",
                "visible": True,
                "scaleanchor": None,
                "constrain": None,
                "domain": base_x_domain,
            },
        }
    elif panel_type == "scalars":
        return {
            "yaxis": {
                "autorange": True,
                "type": "linear",
                "visible": True,
                "scaleanchor": None,
                "domain": base_y_domain,
            },
            "xaxis": {
                "type": "category",
                "visible": True,
                "scaleanchor": None,
                "constrain": None,
                "domain": base_x_domain,
            },
        }
    elif panel_type == "saliency":
        # Saliency is rendered as go.Heatmap
        return {
            "yaxis": {
                "autorange": "reversed",  # Flip so origin is top-left
                "visible": False,
                "scaleanchor": "x",  # Square cells
                "scaleratio": 1.0,
                "domain": base_y_domain,
            },
            "xaxis": {
                "autorange": True,
                "visible": False,
                "constrain": "domain",
                "domain": base_x_domain,
            },
        }
    elif panel_type == "wordcloud":
        return {
            "yaxis": {
                "autorange": "reversed",
                "visible": False,
                "scaleanchor": "x",
                "scaleratio": 1.0,
                "domain": base_y_domain,
            },
            "xaxis": {
                "visible": False,
                "constrain": "domain",
                "domain": base_x_domain,
            },
        }
    elif panel_type == "caption":
        return {
            "yaxis": {
                "range": [0, 1],
                "autorange": False,
                "visible": False,
                "scaleanchor": None,
                "domain": base_y_domain,
            },
            "xaxis": {
                "range": [0, 1],
                "autorange": False,
                "visible": False,
                "scaleanchor": None,
                "constrain": None,
                "domain": base_x_domain,
            },
        }
    else:
        return {}


def _extract_panel_data(
    row: pd.Series,
    panel_type: str,
    available: dict,
    normalize_scalars: bool = True,
    img_aspect: float = 1.0,
) -> dict:
    """Extract data for a specific panel type from a row.

    Parameters
    ----------
    img_aspect : float
        Image aspect ratio (width/height) for scaling saliency display.
    """
    data = {}

    if panel_type == "emotions_bar" and "emotions" in available:
        values = [row[f] for f in available["emotions"]]
        sorted_pairs = sorted(zip(available["emotions"], values), key=lambda x: x[1], reverse=True)
        labels, values = zip(*sorted_pairs) if sorted_pairs else ([], [])
        data = {"labels": list(labels), "values": list(values), "color": "#636EFA"}

    elif panel_type == "scalars":
        exclude = ["time", "index", "filename", "filepath", "image_idx"]
        embedding_prefixes = ["clip_", "gist_", "dinov2_", "saliency_", "places_", "yolo_", "sunattr_"]
        features = [
            c for c in row.index
            if c not in exclude
            and c not in EMOTION_FEATURES
            and not any(c.startswith(p) for p in embedding_prefixes)
            and isinstance(row[c], (int, float, np.floating))
            and not pd.isna(row[c])
        ]

        if features:
            raw_values = [row[f] for f in features]
            if normalize_scalars:
                normalized = []
                hover_texts = []
                for f, v in zip(features, raw_values):
                    norm_v, tooltip = _normalize_scalar(v, f)
                    normalized.append(norm_v)
                    hover_texts.append(tooltip)
                values = normalized
            else:
                values = raw_values
                hover_texts = [f"{f}: {v:.3f}" for f, v in zip(features, raw_values)]

            data = {"labels": features, "values": values, "hover_texts": hover_texts}

    elif panel_type == "saliency" and "saliency" in available:
        saliency_cols = sorted(available["saliency"])
        values = np.array([row[c] for c in saliency_cols])
        side = int(np.sqrt(len(values)))
        grid = values.reshape(side, side)
        # Saliency columns are named saliency_X_Y (col_row), but alphabetical sort
        # gives us grid[x][y]. Transpose to get grid[y][x] for correct display.
        grid = grid.T

        # Store raw grid for go.Heatmap (much faster than matplotlib rendering)
        data = {
            "grid": grid.tolist(),
            "grid_size": side,
        }

    elif panel_type == "yolo" and "yolo" in available:
        cols = available["yolo"]
        pairs = [(c.replace("yolo_", ""), row[c]) for c in cols if row[c] > 0]
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:10]
        if pairs:
            labels, values = zip(*pairs)
            data = {"labels": list(labels), "values": list(values), "color": "#EF553B"}

    elif panel_type == "places" and "places" in available:
        cols = available["places"]
        pairs = [(c.replace("places_", ""), row[c]) for c in cols if row[c] > 0.01]
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)[:10]
        if pairs:
            labels, values = zip(*pairs)
            data = {"labels": list(labels), "values": list(values), "color": "#AB63FA"}

    elif panel_type == "emotions_spider" and "emotions" in available:
        values = [row[f] for f in available["emotions"]]
        labels = list(available["emotions"])
        # Close the polygon
        values = values + [values[0]]
        labels = labels + [labels[0]]
        data = {"labels": labels, "values": values}

    elif panel_type == "wordcloud" and "clip" in available:
        # Render wordcloud to base64 image
        wc_b64 = _render_wordcloud_to_base64(row, top_n=50, width=500, height=400)
        if wc_b64:
            # Decode to get image array for go.Image
            img_bytes = base64.b64decode(wc_b64)
            img = PILImage.open(io.BytesIO(img_bytes))
            img_array = np.array(img)
            data = {"image_array": img_array.tolist()}
        else:
            data = {"image_array": None}

    elif panel_type == "caption" and "caption" in available:
        caption_text = str(row.get("caption", ""))
        data = {"caption": caption_text}

    return data


def _get_row_label(row: pd.Series, idx: int, input_type: str | None) -> str:
    """Generate a label for a row based on input type."""
    if input_type == "video" and "time" in row.index:
        time_val = row["time"]
        return f"Frame {idx}: {time_val:.2f}s"
    elif input_type == "hdf5_brick" and "image_idx" in row.index:
        img_idx = int(row["image_idx"])
        return f"Image {img_idx}"
    elif "filename" in row.index:
        return str(row["filename"])
    else:
        return f"Row {idx}"
