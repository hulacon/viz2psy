"""Visualization tools for viz2psy feature outputs."""

# Lazy imports to avoid dependency errors at CLI startup
__all__ = [
    "make_wordcloud",
    "plot_timeseries",
    "plot_heatmap",
    "plot_scatter",
    "plot_composite",
]


def __getattr__(name):
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
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
