# Visualization

viz2psy includes `viz2psy-viz`, a CLI tool for visualizing feature outputs with both static (matplotlib) and interactive (Plotly) charts.

## Quick Start

```bash
# Word cloud from CLIP embeddings
viz2psy-viz wordcloud scores.csv -o cloud.png

# Interactive timeseries
viz2psy-viz timeseries scores.csv --features memorability -i -o plot.html

# Scatter projection of embeddings
viz2psy-viz scatter scores.csv --features "clip_*" -o scatter.png

# Single image viewer with feature panels
viz2psy-viz image scores.csv photo.jpg -o viewer.html

# Browse multiple images with slider
viz2psy-viz image scores.csv --browse --image-root ./images -o browser.html

# Interactive dashboard with model selection
viz2psy-viz dashboard scores.csv --image-root ./images -o dashboard.html
```

---

## Commands

### wordcloud

Generate word cloud from CLIP embeddings showing semantically similar words.

```bash
viz2psy-viz wordcloud scores.csv -o cloud.png
viz2psy-viz wordcloud scores.csv --image 5 --top 50 -o cloud.png
```

Options:
- `--image`: Row index (default: 0)
- `--top`: Number of words (default: 100)

### timeseries

Plot feature values over time (for video frames or sequential data).

```bash
# Static plot
viz2psy-viz timeseries scores.csv --features memorability aesthetics -o plot.png

# Interactive with zoom/pan
viz2psy-viz timeseries scores.csv --features memorability -i -o plot.html

# Normalize for comparison
viz2psy-viz timeseries scores.csv --features memorability aesthetics -i --normalize -o plot.html
```

Options:
- `--features`: Features to plot (default: all numeric)
- `--time-col`: Time column name (default: `time`)
- `-i, --interactive`: Generate interactive HTML
- `--normalize`: Normalize features to [0,1] (interactive only)

### heatmap

Correlation heatmap between features.

```bash
viz2psy-viz heatmap scores.csv -o heatmap.png
viz2psy-viz heatmap scores.csv --features memorability aesthetics Awe Joy --method spearman -o heatmap.png
```

Options:
- `--features`: Features to include (default: all numeric)
- `--method`: `pearson` or `spearman` (default: pearson)

### scatter

2D/3D scatter projection of high-dimensional embeddings.

```bash
# Static PCA
viz2psy-viz scatter scores.csv --features "clip_*" -o scatter.png

# Interactive UMAP colored by memorability
viz2psy-viz scatter scores.csv --features "clip_*" --method umap --color-by memorability -i -o scatter.html

# MDS projection
viz2psy-viz scatter scores.csv --features "emonet_*" --method mds -i -o mds.html

# Probabilistic PCA (handles missing data)
viz2psy-viz scatter scores.csv --features "clip_*" --method ppca -i -o ppca.html
```

Options:
- `--features`: Features to project (supports glob patterns)
- `--method`: Projection method (default: pca)
  - `pca`: Principal Component Analysis
  - `ppca`: Probabilistic PCA (handles missing data via EM algorithm)
  - `umap`: Uniform Manifold Approximation
  - `tsne`: t-Distributed Stochastic Neighbor Embedding
  - `mds`: Metric Multi-Dimensional Scaling
  - `mds_nonmetric`: Non-metric MDS
- `--color-by`: Column for point coloring
- `-i, --interactive`: Generate interactive HTML
- `--max-points`: Sampling limit for large datasets (default: 5000)

### composite

Image with multiple feature visualization panels.

```bash
viz2psy-viz composite image.jpg scores.csv -o composite.png
viz2psy-viz composite image.jpg scores.csv --panels saliency,emotions,bars -o composite.png
```

Options:
- `--image-idx`: Row index in CSV (default: 0)
- `--panels`: Comma-separated panel types (default: saliency,emotions,bars)

### explorer

Linked scatter + timeseries dashboard for interactive exploration.

```bash
viz2psy-viz explorer scores.csv \
    --scatter-features "clip_*" \
    --timeseries-features memorability Awe \
    -o explorer.html
```

Options:
- `--scatter-features`: Features for scatter projection (required)
- `--timeseries-features`: Features for timeseries plot
- `--method`: Projection method (default: pca)
- `--time-col`: Time column (default: time)
- `--color-by`: Scatter point coloring
- `--max-points`: Sampling limit (default: 5000)

### dashboard

Interactive dashboard for exploring multi-image datasets with model selection and visualization controls.

```bash
# Basic dashboard
viz2psy-viz dashboard scores.csv -o dashboard.html

# With image thumbnails (auto-detected from sidecar)
viz2psy-viz dashboard scores.csv --image-root ./images -o dashboard.html

# For video data
viz2psy-viz dashboard scores.csv --video-path movie.mp4 -o dashboard.html

# Limit embedded thumbnails for large datasets
viz2psy-viz dashboard scores.csv --max-thumbnails 50 -o dashboard.html
```

Features:
- **Model selection**: Dropdown to switch between detected models (emonet, clip, etc.)
- **Visualization types**: Time series, MDS clustering (2D/3D), state-space trajectories (static/animated)
- **Click-to-view**: Click any data point to open the full single-image viewer in a new tab
- **Auto-detection**: Automatically detects available models and appropriate visualizations

Options:
- `--image-root`: Base directory for image resolution
- `--video-path`: Path to source video file
- `--hdf5-path`: Path to HDF5 file
- `--max-thumbnails`: Maximum images to embed (default: 100, 0 to disable)
- `--no-images`: Disable image embedding entirely
- `--width`, `--height`: Dashboard dimensions

### image

Single-image viewer with selectable feature panels.

```bash
# Explicit image path
viz2psy-viz image scores.csv photo.jpg -o viewer.html

# Auto-resolve from filename column
viz2psy-viz image scores.csv --row-idx 5 --image-root ./images -o viewer.html

# Video frame extraction
viz2psy-viz image scores.csv --video-path movie.mp4 --row-idx 10 -o viewer.html

# HDF5 image brick
viz2psy-viz image scores.csv --hdf5-path data.h5 --hdf5-dataset imgBrick --row-idx 0 -o viewer.html

# Browse mode with slider
viz2psy-viz image scores.csv --browse --image-root ./images -o browser.html
```

Options:
- `--row-idx`: Row index in CSV (default: 0)
- `--image-root`: Base directory for image resolution
- `--video-path`: Path to video file
- `--hdf5-path`: Path to HDF5 file
- `--hdf5-dataset`: HDF5 dataset name (default: stimuli)
- `--panels`: Panel types to show (default: auto-detect)
- `--width`, `--height`: Figure dimensions
- `--no-normalize`: Show raw scalar values
- `--browse`: Enable slider for multiple rows
- `--no-browse`: Force single-image mode
- `--max-rows`: Maximum rows in browse mode (default: 100)

---

## Feature Panels

The `image` command supports multiple visualization panels accessible via dropdown:

| Panel | Description | Requires |
|-------|-------------|----------|
| `emotions_bar` | Horizontal bar chart of emotion probabilities | emonet |
| `emotions_spider` | Radar plot of emotion profile | emonet |
| `scalars` | Bar chart of scalar features (memorability, etc.) | Any scalar |
| `saliency` | Fixation density heatmap | saliency |
| `yolo` | Top detected objects | yolo |
| `places` | Top scene predictions | places |
| `wordcloud` | CLIP-based semantic word cloud | clip |

Panels are auto-detected based on available features in the CSV.

---

## Output Formats

- **Static images**: `.png`, `.jpg`, `.svg`, `.pdf` (matplotlib)
- **Interactive HTML**: `.html` (Plotly, self-contained)

Interactive HTML files are standalone and can be shared/viewed in any browser.

---

## Metadata Sidecar

When a CSV has an accompanying `.meta.json` sidecar file (created by `viz2psy`), visualizations use it for:

- **Semantic labels**: Display "Adoration" instead of column names
- **Input type detection**: Auto-detect video/HDF5/image folder
- **Path resolution**: Find images relative to original input path

---

## Python API

Interactive visualizations can also be used programmatically:

```python
from viz2psy.viz.interactive import (
    plot_scatter_interactive,
    plot_timeseries_interactive,
    create_linked_explorer,
    create_single_image_viewer,
)
from viz2psy.viz.dashboard import create_dashboard
from viz2psy.viz.projection import compute_projection
import pandas as pd

df = pd.read_csv("scores.csv")

# Scatter plot with different projection methods
fig = plot_scatter_interactive(df, features=["clip_*"], method="pca")
fig.write_html("scatter.html")

# Direct projection access (PCA, PPCA, UMAP, t-SNE, MDS)
X = df.filter(like="clip_").values
X_2d, info = compute_projection(X, method="ppca", n_components=2)
print(f"Variance explained: {info['variance_explained']}")

# Single image viewer
fig = create_single_image_viewer(
    image="photo.jpg",
    scores_df=df,
    row_idx=0,
    panels=["emotions_bar", "scalars", "saliency"],
)
fig.write_html("viewer.html")

# Interactive dashboard
from viz2psy.viz.sidecar import UnifiedImageResolver
resolver = UnifiedImageResolver(csv_path="scores.csv", image_root="./images")
html = create_dashboard(df, image_resolver=resolver)
with open("dashboard.html", "w") as f:
    f.write(html)
```

See [api.md](api.md) for full API documentation.
