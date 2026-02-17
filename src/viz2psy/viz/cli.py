#!/usr/bin/env python3
"""CLI for visualizing viz2psy feature outputs.

Examples
--------
    # Word cloud from CLIP embeddings
    viz2psy-viz wordcloud scores.csv -o cloud.png

    # Time series of video scores
    viz2psy-viz timeseries scores.csv --features memorability -o plot.png

    # Correlation heatmap
    viz2psy-viz heatmap scores.csv -o heatmap.png

    # 2D scatter projection of embeddings
    viz2psy-viz scatter scores.csv --features "clip_*" -o scatter.png

    # Composite: image + feature visualizations
    viz2psy-viz composite image.jpg scores.csv --panels saliency,emotions -o composite.png
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def cmd_wordcloud(args):
    """Generate word cloud from CLIP embeddings."""
    from .wordcloud import make_wordcloud

    df = pd.read_csv(args.input)
    fig = make_wordcloud(
        df,
        image_idx=args.image,
        top_n=args.top,
    )

    if args.output:
        fig.savefig(args.output, bbox_inches="tight", dpi=150)
        print(f"Saved to {args.output}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


def cmd_timeseries(args):
    """Plot feature time series."""
    from .timeseries import plot_timeseries

    df = pd.read_csv(args.input)
    fig = plot_timeseries(
        df,
        features=args.features,
        time_col=args.time_col,
    )

    if args.output:
        fig.savefig(args.output, bbox_inches="tight", dpi=150)
        print(f"Saved to {args.output}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


def cmd_heatmap(args):
    """Plot correlation heatmap."""
    from .heatmap import plot_heatmap

    df = pd.read_csv(args.input)
    fig = plot_heatmap(
        df,
        features=args.features,
        method=args.method,
    )

    if args.output:
        fig.savefig(args.output, bbox_inches="tight", dpi=150)
        print(f"Saved to {args.output}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


def cmd_scatter(args):
    """Plot 2D scatter projection."""
    from .scatter import plot_scatter

    df = pd.read_csv(args.input)
    fig = plot_scatter(
        df,
        features=args.features,
        method=args.method,
        color_by=args.color_by,
    )

    if args.output:
        fig.savefig(args.output, bbox_inches="tight", dpi=150)
        print(f"Saved to {args.output}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


def cmd_composite(args):
    """Plot image with feature visualizations."""
    from .composite import plot_composite

    df = pd.read_csv(args.scores)
    fig = plot_composite(
        image_path=args.image,
        scores_df=df,
        image_idx=args.image_idx,
        panels=args.panels.split(",") if args.panels else None,
    )

    if args.output:
        fig.savefig(args.output, bbox_inches="tight", dpi=150)
        print(f"Saved to {args.output}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Visualize viz2psy feature outputs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Visualization type")

    # wordcloud
    p_wc = subparsers.add_parser("wordcloud", help="Word cloud from CLIP embeddings")
    p_wc.add_argument("input", type=Path, help="CSV file with CLIP embeddings")
    p_wc.add_argument("-o", "--output", type=Path, help="Output image path")
    p_wc.add_argument("--image", type=int, default=0, help="Image index/row (default: 0)")
    p_wc.add_argument("--top", type=int, default=100, help="Top N words (default: 100)")
    p_wc.set_defaults(func=cmd_wordcloud)

    # timeseries
    p_ts = subparsers.add_parser("timeseries", help="Plot feature time series")
    p_ts.add_argument("input", type=Path, help="CSV file with scores")
    p_ts.add_argument("-o", "--output", type=Path, help="Output image path")
    p_ts.add_argument("--features", nargs="+", help="Features to plot (default: all numeric)")
    p_ts.add_argument("--time-col", default="time", help="Time column name (default: time)")
    p_ts.set_defaults(func=cmd_timeseries)

    # heatmap
    p_hm = subparsers.add_parser("heatmap", help="Correlation heatmap")
    p_hm.add_argument("input", type=Path, help="CSV file with scores")
    p_hm.add_argument("-o", "--output", type=Path, help="Output image path")
    p_hm.add_argument("--features", nargs="+", help="Features to include (default: all numeric)")
    p_hm.add_argument("--method", default="pearson", choices=["pearson", "spearman"],
                      help="Correlation method (default: pearson)")
    p_hm.set_defaults(func=cmd_heatmap)

    # scatter
    p_sc = subparsers.add_parser("scatter", help="2D scatter projection")
    p_sc.add_argument("input", type=Path, help="CSV file with scores")
    p_sc.add_argument("-o", "--output", type=Path, help="Output image path")
    p_sc.add_argument("--features", nargs="+", help="Features to project (supports glob patterns)")
    p_sc.add_argument("--method", default="pca", choices=["pca", "umap", "tsne"],
                      help="Projection method (default: pca)")
    p_sc.add_argument("--color-by", help="Column to use for coloring points")
    p_sc.set_defaults(func=cmd_scatter)

    # composite
    p_cp = subparsers.add_parser("composite", help="Image + feature visualizations")
    p_cp.add_argument("image", type=Path, help="Image file path")
    p_cp.add_argument("scores", type=Path, help="CSV file with scores")
    p_cp.add_argument("-o", "--output", type=Path, help="Output image path")
    p_cp.add_argument("--image-idx", type=int, default=0, help="Row index in scores CSV (default: 0)")
    p_cp.add_argument("--panels", type=str, default="saliency,emotions,bars",
                      help="Comma-separated panel types: saliency,emotions,bars,wordcloud (default: saliency,emotions,bars)")
    p_cp.set_defaults(func=cmd_composite)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
