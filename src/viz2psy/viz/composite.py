"""Composite visualization: image + feature panels."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image


# Emotion categories from EmoNet
EMOTION_CATEGORIES = [
    "Adoration", "Aesthetic Appreciation", "Amusement", "Anxiety", "Awe",
    "Boredom", "Confusion", "Craving", "Disgust", "Empathic Pain",
    "Entrancement", "Excitement", "Fear", "Horror", "Interest",
    "Joy", "Romance", "Sadness", "Sexual Desire", "Surprise",
]


def _get_saliency_columns(df: pd.DataFrame) -> list[str]:
    """Get saliency grid columns in order."""
    cols = [c for c in df.columns if c.startswith("saliency_")]
    # Sort by row, then column (saliency_00_00, saliency_00_01, ...)
    return sorted(cols)


def _get_emotion_columns(df: pd.DataFrame) -> list[str]:
    """Get emotion category columns."""
    return [c for c in EMOTION_CATEGORIES if c in df.columns]


def _plot_saliency_panel(ax, df: pd.DataFrame, image_idx: int):
    """Plot 24x24 saliency heatmap."""
    saliency_cols = _get_saliency_columns(df)

    if not saliency_cols:
        ax.text(0.5, 0.5, "No saliency data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Saliency")
        return

    # Extract saliency values and reshape to 24x24
    values = df.loc[image_idx, saliency_cols].values.astype(float)

    # Determine grid size from column count
    grid_size = int(np.sqrt(len(values)))
    if grid_size * grid_size != len(values):
        ax.text(0.5, 0.5, f"Invalid saliency grid ({len(values)} values)",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Saliency")
        return

    saliency_map = values.reshape(grid_size, grid_size)

    im = ax.imshow(saliency_map, cmap="hot", interpolation="bilinear")
    ax.set_title("Saliency Map")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_emotions_panel(ax, df: pd.DataFrame, image_idx: int):
    """Plot emotion category bar chart."""
    emotion_cols = _get_emotion_columns(df)

    if not emotion_cols:
        ax.text(0.5, 0.5, "No emotion data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Emotions")
        return

    values = df.loc[image_idx, emotion_cols].values.astype(float)

    # Sort by value for better visualization
    sorted_idx = np.argsort(values)[::-1]
    sorted_emotions = [emotion_cols[i] for i in sorted_idx]
    sorted_values = values[sorted_idx]

    # Show top 10 emotions
    n_show = min(10, len(sorted_emotions))
    y_pos = np.arange(n_show)

    colors = plt.cm.viridis(sorted_values[:n_show] / max(sorted_values[:n_show].max(), 1e-6))

    ax.barh(y_pos, sorted_values[:n_show], color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_emotions[:n_show], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Probability")
    ax.set_title("Top Emotions")
    ax.set_xlim(0, min(1.0, sorted_values[:n_show].max() * 1.1))


def _plot_bars_panel(ax, df: pd.DataFrame, image_idx: int):
    """Plot bar chart of scalar features (memorability, aesthetics, etc.)."""
    scalar_features = []

    # Check for common scalar features
    candidates = [
        ("memorability", "Memorability"),
        ("aesthetic_score", "Aesthetics"),
    ]

    for col, label in candidates:
        if col in df.columns:
            scalar_features.append((label, df.loc[image_idx, col]))

    # Add YOLO summary stats if present
    yolo_stats = ["yolo_total_objects", "yolo_unique_classes", "yolo_total_area"]
    for col in yolo_stats:
        if col in df.columns:
            label = col.replace("yolo_", "").replace("_", " ").title()
            scalar_features.append((label, df.loc[image_idx, col]))

    if not scalar_features:
        ax.text(0.5, 0.5, "No scalar features", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Features")
        return

    labels, values = zip(*scalar_features)
    values = np.array(values, dtype=float)
    y_pos = np.arange(len(labels))

    # Normalize for color mapping
    norm_values = (values - values.min()) / (values.max() - values.min() + 1e-6)
    colors = plt.cm.viridis(norm_values)

    ax.barh(y_pos, values, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_title("Features")

    # Add value labels
    for i, v in enumerate(values):
        ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=8)


def _plot_wordcloud_panel(ax, df: pd.DataFrame, image_idx: int):
    """Plot CLIP word cloud in a panel."""
    from .wordcloud import (
        get_clip_columns,
        load_clip_text_encoder,
        compute_text_embeddings,
        compute_word_similarities,
        DEFAULT_VOCABULARY,
    )
    from wordcloud import WordCloud
    import torch

    clip_cols = get_clip_columns(df)

    if not clip_cols:
        ax.text(0.5, 0.5, "No CLIP data", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Word Cloud")
        return

    image_embedding = df.loc[image_idx, clip_cols].values.astype(np.float32)

    # Load CLIP and compute text embeddings
    model, tokenizer = load_clip_text_encoder()

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    text_embeddings = compute_text_embeddings(DEFAULT_VOCABULARY, model, tokenizer, device)
    word_scores = compute_word_similarities(image_embedding, text_embeddings, DEFAULT_VOCABULARY)

    # Keep top 50 words
    sorted_words = sorted(word_scores.items(), key=lambda x: x[1], reverse=True)[:50]
    word_scores = dict(sorted_words)

    if not word_scores:
        ax.text(0.5, 0.5, "No matching words", ha="center", va="center", transform=ax.transAxes)
        ax.set_title("Word Cloud")
        return

    wc = WordCloud(
        width=400,
        height=300,
        background_color="white",
        colormap="viridis",
        max_words=50,
    ).generate_from_frequencies(word_scores)

    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    ax.set_title("CLIP Word Cloud")


# Panel registry
PANEL_FUNCTIONS = {
    "saliency": _plot_saliency_panel,
    "emotions": _plot_emotions_panel,
    "bars": _plot_bars_panel,
    "wordcloud": _plot_wordcloud_panel,
}


def plot_composite(
    image_path: Path | str,
    scores_df: pd.DataFrame,
    image_idx: int = 0,
    panels: list[str] | None = None,
    figsize: tuple[int, int] | None = None,
    title: str | None = None,
) -> plt.Figure:
    """Plot image alongside feature visualization panels.

    Parameters
    ----------
    image_path : Path or str
        Path to the image file.
    scores_df : pd.DataFrame
        DataFrame with feature scores.
    image_idx : int
        Row index in scores_df corresponding to this image.
    panels : list of str, optional
        Panel types to include: "saliency", "emotions", "bars", "wordcloud".
        Default: ["saliency", "emotions", "bars"]
    figsize : tuple, optional
        Figure size (width, height) in inches.
    title : str, optional
        Figure title.

    Returns
    -------
    matplotlib.figure.Figure
    """
    if panels is None:
        panels = ["saliency", "emotions", "bars"]

    # Validate panels
    invalid = [p for p in panels if p not in PANEL_FUNCTIONS]
    if invalid:
        raise ValueError(f"Unknown panel type(s): {invalid}. Available: {list(PANEL_FUNCTIONS.keys())}")

    n_panels = len(panels)

    # Layout: image on left, panels on right in a grid
    if n_panels == 1:
        fig, axes = plt.subplots(1, 2, figsize=figsize or (12, 5))
        panel_axes = [axes[1]]
    elif n_panels == 2:
        fig, axes = plt.subplots(1, 3, figsize=figsize or (15, 5))
        panel_axes = [axes[1], axes[2]]
    elif n_panels == 3:
        fig = plt.figure(figsize=figsize or (14, 8))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1])
        ax_img = fig.add_subplot(gs[:, 0])
        panel_axes = [
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 1:]),
        ]
        axes = [ax_img] + panel_axes
    else:  # 4 panels
        fig = plt.figure(figsize=figsize or (16, 8))
        gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1])
        ax_img = fig.add_subplot(gs[:, 0])
        panel_axes = [
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2]),
            fig.add_subplot(gs[1, 1]),
            fig.add_subplot(gs[1, 2]),
        ]
        axes = [ax_img] + panel_axes

    # Plot image
    ax_img = axes[0]
    img = Image.open(image_path)
    ax_img.imshow(img)
    ax_img.axis("off")
    ax_img.set_title(Path(image_path).name)

    # Plot panels
    for ax, panel_name in zip(panel_axes, panels):
        panel_func = PANEL_FUNCTIONS[panel_name]
        panel_func(ax, scores_df, image_idx)

    if title:
        fig.suptitle(title, fontsize=14)

    fig.tight_layout()
    return fig
