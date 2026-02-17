"""Interactive visualizations using Plotly.

This module provides interactive versions of viz2psy visualizations
that work in Jupyter notebooks and export to standalone HTML files.

Examples
--------
    >>> from viz2psy.viz.interactive import plot_scatter_interactive
    >>> import pandas as pd
    >>> df = pd.read_csv("scores.csv")
    >>> fig = plot_scatter_interactive(df, features=["clip_*"], method="pca")
    >>> fig.write_html("scatter.html")
"""

__all__ = [
    "plot_scatter_interactive",
    "plot_timeseries_interactive",
    "plot_timeseries_subplots",
    "create_linked_explorer",
    "configure_theme",
    "save_figure",
    "encode_image_base64",
    "encode_image_data_url",
    # Single-image viewers
    "create_single_image_viewer",
    "create_browsable_viewer",
    "plot_emotions_bar",
    "plot_emotions_spider",
    "plot_scalars_bar",
    "plot_saliency_heatmap",
    "plot_top_detections",
    "view_image_emotions",
    "view_image_saliency",
]


def __getattr__(name):
    """Lazy imports to avoid loading Plotly until needed."""
    if name == "plot_scatter_interactive":
        from .scatter import plot_scatter_interactive
        return plot_scatter_interactive
    elif name == "plot_timeseries_interactive":
        from .timeseries import plot_timeseries_interactive
        return plot_timeseries_interactive
    elif name == "plot_timeseries_subplots":
        from .timeseries import plot_timeseries_subplots
        return plot_timeseries_subplots
    elif name == "create_linked_explorer":
        from .linked import create_linked_explorer
        return create_linked_explorer
    elif name == "configure_theme":
        from .base import configure_theme
        return configure_theme
    elif name == "save_figure":
        from .base import save_figure
        return save_figure
    elif name == "encode_image_base64":
        from .base import encode_image_base64
        return encode_image_base64
    elif name == "encode_image_data_url":
        from .base import encode_image_data_url
        return encode_image_data_url
    # Single-image viewers
    elif name == "create_single_image_viewer":
        from .single_image import create_single_image_viewer
        return create_single_image_viewer
    elif name == "create_browsable_viewer":
        from .single_image import create_browsable_viewer
        return create_browsable_viewer
    elif name == "plot_emotions_bar":
        from .single_image import plot_emotions_bar
        return plot_emotions_bar
    elif name == "plot_emotions_spider":
        from .single_image import plot_emotions_spider
        return plot_emotions_spider
    elif name == "plot_scalars_bar":
        from .single_image import plot_scalars_bar
        return plot_scalars_bar
    elif name == "plot_saliency_heatmap":
        from .single_image import plot_saliency_heatmap
        return plot_saliency_heatmap
    elif name == "plot_top_detections":
        from .single_image import plot_top_detections
        return plot_top_detections
    elif name == "view_image_emotions":
        from .single_image import view_image_emotions
        return view_image_emotions
    elif name == "view_image_saliency":
        from .single_image import view_image_saliency
        return view_image_saliency
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
