"""Visualization tools for viz2psy feature outputs."""

# Lazy imports to avoid dependency errors at CLI startup
__all__ = [
    # Static matplotlib visualizations
    "make_wordcloud",
    "plot_timeseries",
    "plot_heatmap",
    "plot_scatter",
    "plot_composite",
    # Interactive Altair visualizations
    "plot_scatter_interactive",
    "plot_timeseries_interactive",
    "create_linked_explorer",
]


def __getattr__(name):
    # Static visualizations
    if name == "make_wordcloud":
        from .wordcloud import make_wordcloud
        return make_wordcloud
    elif name == "plot_timeseries":
        from .timeseries import plot_timeseries
        return plot_timeseries
    elif name == "plot_heatmap":
        from .heatmap import plot_heatmap
        return plot_heatmap
    elif name == "plot_scatter":
        from .scatter import plot_scatter
        return plot_scatter
    elif name == "plot_composite":
        from .composite import plot_composite
        return plot_composite
    # Interactive visualizations
    elif name == "plot_scatter_interactive":
        from .interactive.scatter import plot_scatter_interactive
        return plot_scatter_interactive
    elif name == "plot_timeseries_interactive":
        from .interactive.timeseries import plot_timeseries_interactive
        return plot_timeseries_interactive
    elif name == "create_linked_explorer":
        from .interactive.linked import create_linked_explorer
        return create_linked_explorer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
