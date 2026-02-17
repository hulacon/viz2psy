"""2D scatter projection visualization."""

import fnmatch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler


def get_feature_columns(
    df: pd.DataFrame,
    patterns: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[str]:
    """Get feature columns matching patterns."""
    exclude = exclude or ["time", "index", "filename", "filepath"]

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


def plot_scatter(
    df: pd.DataFrame,
    features: list[str] | None = None,
    method: str = "pca",
    color_by: str | None = None,
    figsize: tuple[int, int] = (8, 6),
    title: str | None = None,
    point_size: int = 20,
    alpha: float = 0.7,
) -> plt.Figure:
    """Plot 2D scatter projection of features.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with feature columns.
    features : list of str, optional
        Features to project. Supports glob patterns (e.g., "clip_*").
        If None, uses all numeric features.
    method : str
        Projection method: "pca", "umap", or "tsne".
    color_by : str, optional
        Column to use for coloring points.
    figsize : tuple
        Figure size (width, height) in inches.
    title : str, optional
        Plot title.
    point_size : int
        Size of scatter points.
    alpha : float
        Transparency of points.

    Returns
    -------
    matplotlib.figure.Figure
    """
    feature_cols = get_feature_columns(df, features)

    if len(feature_cols) < 2:
        raise ValueError("Need at least 2 features for projection.")

    # Extract feature matrix
    X = df[feature_cols].values

    # Handle NaN values
    if np.any(np.isnan(X)):
        print("Warning: NaN values found, filling with column means.")
        X = np.nan_to_num(X, nan=np.nanmean(X, axis=0))

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Project to 2D
    if method == "pca":
        projector = PCA(n_components=2, random_state=42)
        X_2d = projector.fit_transform(X_scaled)
        var_explained = projector.explained_variance_ratio_
        xlabel = f"PC1 ({var_explained[0]:.1%} var)"
        ylabel = f"PC2 ({var_explained[1]:.1%} var)"
    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("umap-learn is required for UMAP projection. Install with: pip install umap-learn")
        projector = umap.UMAP(n_components=2, random_state=42)
        X_2d = projector.fit_transform(X_scaled)
        xlabel = "UMAP1"
        ylabel = "UMAP2"
    elif method == "tsne":
        # Adjust perplexity for small datasets
        perplexity = min(30, len(X) - 1)
        projector = TSNE(n_components=2, random_state=42, perplexity=perplexity)
        X_2d = projector.fit_transform(X_scaled)
        xlabel = "t-SNE1"
        ylabel = "t-SNE2"
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca', 'umap', or 'tsne'.")

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)

    # Color mapping
    if color_by and color_by in df.columns:
        colors = df[color_by].values
        scatter = ax.scatter(
            X_2d[:, 0], X_2d[:, 1],
            c=colors,
            s=point_size,
            alpha=alpha,
            cmap="viridis",
        )
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label(color_by)
    else:
        ax.scatter(
            X_2d[:, 0], X_2d[:, 1],
            s=point_size,
            alpha=alpha,
            c="steelblue",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title or f"{method.upper()} Projection ({len(feature_cols)} features)")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    return fig
