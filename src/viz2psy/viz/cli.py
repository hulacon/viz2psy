#!/usr/bin/env python3
"""CLI for visualizing viz2psy feature outputs.

Examples
--------
    # Word cloud from CLIP embeddings
    viz2psy-viz wordcloud scores.csv -o cloud.png

    # Time series of video scores
    viz2psy-viz timeseries scores.csv --features memorability -o plot.png

    # Interactive timeseries with zoom/pan
    viz2psy-viz timeseries scores.csv --features memorability -i -o plot.html

    # Timeseries with diff and rolling average (auto for video)
    viz2psy-viz timeseries scores.csv --show-diff --rolling-window 10 -o plot.png

    # Correlation heatmap
    viz2psy-viz heatmap scores.csv -o heatmap.png

    # 2D scatter projection of embeddings
    viz2psy-viz scatter scores.csv --features "clip_*" -o scatter.png

    # MDS scatter projection
    viz2psy-viz scatter scores.csv --features "clip_*" --method mds -o mds.png

    # Interactive scatter with image thumbnails
    viz2psy-viz scatter scores.csv --features "clip_*" -i --show-images -o scatter.html

    # Composite: image + feature visualizations
    viz2psy-viz composite image.jpg scores.csv --panels saliency,emotions -o composite.png

    # Linked scatter + timeseries explorer
    viz2psy-viz explorer scores.csv --scatter-features "clip_*" --timeseries-features memorability -o explorer.html

    # Single-image viewer with feature panels (explicit image path)
    viz2psy-viz image scores.csv photo.jpg --panels emotions_bar,scalars -o viewer.html

    # Single-image viewer with auto-resolved image path (uses filename column)
    viz2psy-viz image scores.csv --row-idx 5 -o viewer.html

    # Single-image viewer with custom image root directory
    viz2psy-viz image scores.csv --image-root /data/stimuli --row-idx 5 -o viewer.html

    # HyperTools 3D visualization
    viz2psy-viz hyperplot scores.csv --dims 3 --n-clusters 5 -o hyper.html

    # HyperTools trajectory animation
    viz2psy-viz hyperplot scores.csv --animate --features "clip_*"
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

from .projection import PROJECTION_METHODS


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
    df = pd.read_csv(args.input)

    if args.interactive:
        from .interactive.timeseries import plot_timeseries_interactive
        from .interactive.base import save_figure
        from .sidecar import load_sidecar

        sidecar = load_sidecar(args.input)

        fig = plot_timeseries_interactive(
            df,
            features=args.features,
            time_col=args.index_col,
            normalize=args.normalize,
            show_diff=args.show_diff,
            rolling_window=args.rolling_window,
            auto_video_mode=not args.no_auto_video,
            sidecar=sidecar,
        )

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_path = _default_output_path(args.input, "timeseries")
            print(f"(Tip: use -o to specify output path)")

        save_figure(fig, output_path)
        print(f"Saved to {output_path}")
    else:
        from .timeseries import plot_timeseries
        from .sidecar import load_sidecar

        sidecar = load_sidecar(args.input)

        fig = plot_timeseries(
            df,
            features=args.features,
            index_col=args.index_col,
            show_diff=args.show_diff,
            rolling_window=args.rolling_window,
            auto_video_mode=not args.no_auto_video,
            sidecar=sidecar,
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
    df = pd.read_csv(args.input)

    if args.interactive:
        from .interactive.scatter import plot_scatter_interactive
        from .interactive.base import save_figure
        from .sidecar import load_sidecar

        sidecar = load_sidecar(args.input)

        fig = plot_scatter_interactive(
            df,
            features=args.features,
            method=args.method,
            color_by=args.color_by,
            max_points=args.max_points,
            sidecar=sidecar,
        )

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_path = _default_output_path(args.input, f"scatter_{args.method}")
            print(f"(Tip: use -o to specify output path)")

        save_figure(fig, output_path)
        print(f"Saved to {output_path}")
    else:
        from .scatter import plot_scatter

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


def cmd_explorer(args):
    """Create linked scatter + timeseries explorer."""
    from .interactive.linked import create_linked_explorer
    from .interactive.base import save_figure
    from .sidecar import load_sidecar

    df = pd.read_csv(args.input)
    sidecar = load_sidecar(args.input)

    fig = create_linked_explorer(
        df,
        scatter_features=args.scatter_features,
        timeseries_features=args.timeseries_features,
        method=args.method,
        time_col=args.time_col,
        color_by=args.color_by,
        max_points=args.max_points,
        sidecar=sidecar,
    )

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = _default_output_path(args.input, "explorer")
        print(f"(Tip: use -o to specify output path)")

    save_figure(fig, output_path)
    print(f"Saved to {output_path}")


def _default_output_path(csv_path: Path, suffix: str, ext: str = ".html") -> Path:
    """Generate default output path alongside CSV."""
    return csv_path.parent / f"{csv_path.stem}_{suffix}{ext}"


def cmd_image(args):
    """View single image with feature visualizations."""
    from .interactive.single_image import create_single_image_viewer, create_browsable_viewer
    from .interactive.base import save_figure
    from .sidecar import load_sidecar, create_unified_resolver

    df = pd.read_csv(args.scores)
    sidecar = load_sidecar(args.scores)

    # Create unified resolver for all input types
    resolver = create_unified_resolver(
        args.scores,
        image_root=args.image_root,
        video_path=args.video_path,
        frames_dir=args.frames_dir,
        hdf5_path=args.hdf5_path,
        hdf5_dataset=args.hdf5_dataset,
    )

    # Check if browse mode requested or should auto-detect
    use_browse_mode = args.browse
    if use_browse_mode is None:
        # Auto-detect: use browse mode if multiple rows and no explicit image
        use_browse_mode = len(df) > 1 and args.image is None

    if use_browse_mode and len(df) > 1:
        # Browse mode: create viewer with slider
        panels = args.panels.split(",") if args.panels else None

        print(f"Creating browsable viewer for {min(len(df), args.max_rows)} rows...")

        fig = create_browsable_viewer(
            scores_df=df,
            image_resolver=resolver,
            panels=panels,
            width=args.width,
            height=args.height + 100,  # Extra height for slider
            max_rows=args.max_rows,
            sidecar=sidecar,
            normalize_scalars=not args.no_normalize,
            embed_images=args.embed_images,
        )

        # Determine output path
        if args.output:
            output_path = args.output
        else:
            output_path = _default_output_path(args.scores, "browse")
            print(f"(Tip: use -o to specify output path)")

        save_figure(fig, output_path)
        print(f"Saved to {output_path}")
        return

    # Single image mode
    image = None
    image_title = None

    if args.image:
        # Explicit image path provided
        image = args.image
        image_title = args.image.name
    else:
        # Auto-resolve using unified resolver
        input_type = resolver.detected_input_type or "image_folder"
        image = resolver.resolve(df, args.row_idx)

        if image is None:
            print(f"Could not resolve image for row {args.row_idx}")
            print(f"Detected input type: {input_type}")

            if input_type == "image_folder":
                search_paths = resolver._file_resolver.get_search_paths()
                print(f"Searched in: {[str(p) for p in search_paths]}")
                if "filename" in df.columns:
                    filename = df.iloc[args.row_idx].get("filename", "<no filename>")
                    print(f"Looking for: {filename}")
            elif input_type == "video":
                print("Ensure video file exists and opencv-python is installed")
            elif input_type == "hdf5_brick":
                print("Ensure HDF5 file exists and h5py is installed")

            sys.exit(1)

        # Generate title based on input type
        if input_type == "image_folder":
            image_title = image.name if hasattr(image, 'name') else f"Row {args.row_idx}"
            print(f"Resolved image: {image}")
        elif input_type == "video":
            time_val = df.iloc[args.row_idx].get("time", args.row_idx)
            image_title = f"Frame at {time_val:.2f}s"
            print(f"Extracted frame at {time_val:.2f}s")
        elif input_type == "hdf5_brick":
            idx_val = df.iloc[args.row_idx].get("image_idx", args.row_idx)
            image_title = f"Image {idx_val}"
            print(f"Extracted image {idx_val} from HDF5")

    panels = args.panels.split(",") if args.panels else None

    fig = create_single_image_viewer(
        image=image,
        scores_df=df,
        row_idx=args.row_idx,
        panels=panels,
        width=args.width,
        height=args.height,
        sidecar=sidecar,
        normalize_scalars=not args.no_normalize,
        image_title=image_title,
    )

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = _default_output_path(args.scores, f"image_{args.row_idx}")
        print(f"(Tip: use -o to specify output path)")

    save_figure(fig, output_path)
    print(f"Saved to {output_path}")


def cmd_hyperplot(args):
    """Create HyperTools visualization."""
    from .hyperplot import plot_hypertools, save_hypertools_figure
    from .sidecar import load_sidecar

    df = pd.read_csv(args.input)
    sidecar = load_sidecar(args.input)

    fig = plot_hypertools(
        df,
        features=args.features,
        ndims=args.dims,
        reduce=args.method,
        n_clusters=args.n_clusters,
        group=args.group,
        animate=args.animate,
        sidecar=sidecar,
    )

    if args.output:
        save_hypertools_figure(fig, str(args.output))
        print(f"Saved to {args.output}")
    else:
        import matplotlib.pyplot as plt
        plt.show()


def cmd_dashboard(args):
    """Create interactive model-visualization dashboard."""
    from .dashboard import create_dashboard
    from .sidecar import load_sidecar, UnifiedImageResolver

    df = pd.read_csv(args.input)
    sidecar = load_sidecar(args.input)

    print(f"Creating dashboard for {args.input}...")
    print(f"Rows: {len(df):,}, Columns: {len(df.columns)}")

    # Create image resolver if images are available
    image_resolver = None
    if not args.no_images:
        try:
            image_resolver = UnifiedImageResolver(
                csv_path=args.input,
                sidecar=sidecar,
                image_root=Path(args.image_root) if args.image_root else None,
                video_path=Path(args.video_path) if args.video_path else None,
                frames_dir=Path(args.frames_dir) if args.frames_dir else None,
                hdf5_path=Path(args.hdf5_path) if args.hdf5_path else None,
            )
            print(f"Image resolver: {image_resolver.input_type or 'auto-detect'}")
        except Exception as e:
            print(f"Note: Could not set up image resolver: {e}")

    html = create_dashboard(
        df,
        sidecar=sidecar,
        image_resolver=image_resolver,
        width=args.width,
        height=args.height,
        max_thumbnails=args.max_thumbnails,
        embed_images=args.embed_images,
    )

    # Determine output path
    if args.output:
        output_path = args.output
    else:
        output_path = _default_output_path(args.input, "dashboard")
        print(f"(Tip: use -o to specify output path)")

    with open(output_path, "w") as f:
        f.write(html)

    print(f"Saved to {output_path}")

    if args.open:
        import webbrowser
        webbrowser.open(f"file://{output_path.absolute()}")


def cmd_recommend(args):
    """Analyze CSV and recommend visualizations."""
    from .feature_config import (
        get_visualization_recommendations,
        VISUALIZATION_MATRIX,
        FEATURE_CONFIGS,
    )

    df = pd.read_csv(args.input)
    columns = df.columns.tolist()

    recommendations = get_visualization_recommendations(columns)

    print(f"Analyzing: {args.input}")
    print(f"Rows: {len(df):,}, Columns: {len(columns)}")
    print()

    # Detected models
    models = recommendations["detected_models"]
    if models:
        print(f"Detected models: {', '.join(models)}")
    else:
        print("No viz2psy models detected in columns.")
    print()

    # Timeseries recommendations
    ts = recommendations["timeseries"]
    print("=" * 60)
    print("TIMESERIES")
    print("=" * 60)
    if ts["available"]:
        print(f"Recommended features ({len(ts['features'])}):")
        # Group by model for cleaner output
        for model in models:
            config = FEATURE_CONFIGS.get(model)
            if config and config.timeseries:
                model_feats = [f for f in ts["features"] if any(
                    __import__("fnmatch").fnmatch(f, p) for p in config.column_patterns
                )]
                if model_feats:
                    mode = config.timeseries_mode
                    if mode == "top_k":
                        print(f"  {model} (top-{config.top_k}): {', '.join(model_feats[:5])}...")
                    elif mode == "aggregate":
                        print(f"  {model} (aggregate): spatial attention patterns")
                    else:
                        print(f"  {model}: {', '.join(model_feats[:5])}" +
                              ("..." if len(model_feats) > 5 else ""))
        print()
        print("Example command:")
        example_feats = ts["features"][:3]
        print(f"  viz2psy-viz timeseries {args.input} --features {' '.join(example_feats)} -i")
    else:
        print("No features suitable for timeseries visualization.")
    print()

    # MDS/Clustering recommendations
    mds = recommendations["mds_clustering"]
    print("=" * 60)
    print("MDS / CLUSTERING")
    print("=" * 60)
    if mds["available"]:
        print("Available feature groups:")
        for model, cols in mds["groups"].items():
            config = FEATURE_CONFIGS.get(model)
            desc = config.description if config else ""
            print(f"  {model}: {len(cols)} dims - {desc}")
        print()
        print("Example commands:")
        for model in list(mds["groups"].keys())[:2]:
            # Use correct pattern based on model type
            if model in ["clip", "gist", "dinov2", "saliency", "places"]:
                pattern = f'"{model}_*"'
            elif model == "emonet":
                pattern = "Adoration Amusement Excitement Fear Joy"
            elif model == "llstat":
                pattern = '"luminance_*" rms_contrast colorfulness edge_density'
            elif model == "yolo":
                pattern = '"yolo_*"'
            else:
                pattern = f'"{model}_*"'
            print(f"  viz2psy-viz scatter {args.input} --features {pattern} --method mds -i")
    else:
        print("No features suitable for MDS/clustering visualization.")
    print()

    # Trajectory recommendations
    traj = recommendations["trajectories"]
    print("=" * 60)
    print("TRAJECTORIES (HyperTools)")
    print("=" * 60)
    if traj["available"]:
        # Check if data is sequential
        has_time = "time" in columns
        has_index = any(c in columns for c in ["image_idx", "frame", "index"])
        is_sequential = has_time or has_index

        if is_sequential:
            print("Data appears sequential (has time/index column).")
        else:
            print("Warning: No time/index column detected. Trajectories connect points in row order.")
        print()
        print("Available feature groups:")
        for model, cols in traj["groups"].items():
            config = FEATURE_CONFIGS.get(model)
            desc = config.description if config else ""
            print(f"  {model}: {len(cols)} dims - {desc}")
        print()
        print("Example commands:")
        for model in list(traj["groups"].keys())[:2]:
            # Use correct pattern based on model type
            if model in ["clip", "gist", "dinov2", "saliency", "places"]:
                pattern = f'"{model}_*"'
            elif model == "emonet":
                pattern = "Adoration Amusement Excitement Fear Joy Sadness"
            elif model == "llstat":
                pattern = '"luminance_*" rms_contrast colorfulness'
            elif model == "yolo":
                pattern = '"yolo_*"'
            else:
                pattern = f'"{model}_*"'
            print(f"  viz2psy-viz hyperplot {args.input} --features {pattern} --animate --dims 3")
    else:
        print("No features suitable for trajectory visualization.")
    print()

    # Show matrix if requested
    if args.matrix:
        print("=" * 60)
        print("VISUALIZATION MATRIX (all models)")
        print("=" * 60)
        print(VISUALIZATION_MATRIX)


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
    p_ts.add_argument("--index-col", default=None,
                      help="Index column name (auto-detects: time, filename, image_idx)")
    p_ts.add_argument("-i", "--interactive", action="store_true",
                      help="Generate interactive Plotly chart (HTML output)")
    p_ts.add_argument("--normalize", action="store_true",
                      help="Normalize features to [0,1] for comparison (interactive only)")
    p_ts.add_argument("--show-diff", action="store_true",
                      help="Plot first-order differences (auto for video)")
    p_ts.add_argument("--rolling-window", type=int, default=None,
                      help="Overlay rolling average with window size (auto=5 for video)")
    p_ts.add_argument("--no-auto-video", action="store_true",
                      help="Disable auto diff/rolling for video data")
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
    p_sc.add_argument("--method", default="pca", choices=PROJECTION_METHODS,
                      help="Projection method (default: pca)")
    p_sc.add_argument("--color-by", help="Column to use for coloring points")
    p_sc.add_argument("-i", "--interactive", action="store_true",
                      help="Generate interactive Plotly chart (HTML output)")
    p_sc.add_argument("--max-points", type=int, default=5000,
                      help="Maximum points before sampling (interactive only, default: 5000)")
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

    # explorer (linked scatter + timeseries)
    p_ex = subparsers.add_parser("explorer", help="Linked scatter + timeseries explorer (interactive)")
    p_ex.add_argument("input", type=Path, help="CSV file with scores")
    p_ex.add_argument("-o", "--output", type=Path, help="Output HTML path")
    p_ex.add_argument("--scatter-features", nargs="+", required=True,
                      help="Features for scatter projection (supports glob patterns)")
    p_ex.add_argument("--timeseries-features", nargs="+",
                      help="Features for timeseries (default: color-by or first scalar features)")
    p_ex.add_argument("--method", default="pca", choices=PROJECTION_METHODS,
                      help="Projection method (default: pca)")
    p_ex.add_argument("--time-col", default="time", help="Time column name (default: time)")
    p_ex.add_argument("--color-by", help="Column to use for coloring scatter points")
    p_ex.add_argument("--max-points", type=int, default=5000,
                      help="Maximum points before sampling (default: 5000)")
    p_ex.set_defaults(func=cmd_explorer)

    # image (single-image viewer)
    p_img = subparsers.add_parser("image", help="Single-image viewer with feature panels (interactive)")
    p_img.add_argument("scores", type=Path, help="CSV file with scores")
    p_img.add_argument("image", type=Path, nargs="?", default=None,
                      help="Image file path (optional if CSV has filename column)")
    p_img.add_argument("-o", "--output", type=Path, help="Output HTML path")
    p_img.add_argument("--row-idx", type=int, default=0, help="Row index in scores CSV (default: 0)")
    p_img.add_argument("--image-root", type=Path,
                      help="Base directory for resolving image paths (image_folder input)")
    p_img.add_argument("--video-path", type=Path,
                      help="Path to video file (overrides sidecar, for video input)")
    p_img.add_argument("--frames-dir", type=Path,
                      help="Path to saved video frames directory (avoids re-extracting from video)")
    p_img.add_argument("--hdf5-path", type=Path,
                      help="Path to HDF5 file (overrides sidecar, for hdf5_brick input)")
    p_img.add_argument("--hdf5-dataset", type=str, default="stimuli",
                      help="HDF5 dataset name (default: stimuli)")
    p_img.add_argument("--panels", type=str, default=None,
                      help="Comma-separated panel types: emotions_bar,emotions_spider,scalars,saliency,yolo,places,wordcloud (default: auto-detect)")
    p_img.add_argument("--width", type=int, default=1200, help="Figure width in pixels (default: 1200)")
    p_img.add_argument("--height", type=int, default=600, help="Figure height in pixels (default: 600)")
    p_img.add_argument("--no-normalize", action="store_true",
                      help="Show raw scalar values instead of normalized [0-1]")
    p_img.add_argument("--browse", action="store_true", default=None,
                      help="Enable browse mode with slider for multiple rows (auto-detected if multiple rows)")
    p_img.add_argument("--no-browse", action="store_false", dest="browse",
                      help="Disable browse mode (single image only)")
    p_img.add_argument("--max-rows", type=int, default=1000,
                      help="Maximum rows to include in browse mode (default: 1000)")
    p_img.add_argument("--embed-images", action="store_true",
                      help="Embed images as base64 (portable but larger file). Default uses file:// URLs.")
    p_img.set_defaults(func=cmd_image)

    # hyperplot (HyperTools visualization)
    p_hyp = subparsers.add_parser("hyperplot", help="HyperTools 3D/animation visualization")
    p_hyp.add_argument("input", type=Path, help="CSV file with scores")
    p_hyp.add_argument("-o", "--output", type=Path, help="Output file path")
    p_hyp.add_argument("--features", nargs="+",
                      help="Features to include (supports glob patterns)")
    p_hyp.add_argument("--dims", type=int, default=3, choices=[2, 3],
                      help="Number of dimensions (default: 3)")
    p_hyp.add_argument("--method", default="pca", choices=["pca", "umap", "ppca"],
                      help="Reduction method (default: pca)")
    p_hyp.add_argument("--n-clusters", type=int,
                      help="Auto-cluster into N groups and color by cluster")
    p_hyp.add_argument("--group", type=str,
                      help="Column name for categorical coloring (overrides n-clusters)")
    p_hyp.add_argument("--animate", action="store_true",
                      help="Create trajectory animation (connects points in row order)")
    p_hyp.set_defaults(func=cmd_hyperplot)

    # dashboard (interactive model-visualization explorer)
    p_dash = subparsers.add_parser("dashboard", help="Interactive model-visualization dashboard")
    p_dash.add_argument("input", type=Path, help="CSV file with scores")
    p_dash.add_argument("-o", "--output", type=Path, help="Output HTML path")
    p_dash.add_argument("--width", type=int, default=900, help="Plot width in pixels (default: 900)")
    p_dash.add_argument("--height", type=int, default=600, help="Plot height in pixels (default: 600)")
    p_dash.add_argument("--image-root", type=str, help="Base directory for image lookup")
    p_dash.add_argument("--video-path", type=str, help="Video file for frame extraction")
    p_dash.add_argument("--frames-dir", type=str, help="Saved video frames directory (avoids re-extracting)")
    p_dash.add_argument("--hdf5-path", type=str, help="HDF5 file for image extraction")
    p_dash.add_argument("--max-thumbnails", type=int, default=1000,
                       help="Max rows to include in viewer (default: 1000)")
    p_dash.add_argument("--no-images", action="store_true", help="Disable image thumbnails in detail panel")
    p_dash.add_argument("--embed-images", action="store_true",
                       help="Embed images as base64 (portable but larger file). Default uses file:// URLs.")
    p_dash.add_argument("--open", action="store_true", help="Open dashboard in browser after creation")
    p_dash.set_defaults(func=cmd_dashboard)

    # recommend (analyze CSV and suggest visualizations)
    p_rec = subparsers.add_parser("recommend", help="Analyze CSV and recommend visualizations")
    p_rec.add_argument("input", type=Path, help="CSV file with scores")
    p_rec.add_argument("--matrix", action="store_true",
                      help="Show full visualization matrix for all models")
    p_rec.set_defaults(func=cmd_recommend)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
