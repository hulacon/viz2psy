# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Dashboard** (`viz2psy-viz dashboard`)
  - Interactive model-visualization dashboard for multi-image datasets
  - Model selection dropdown (auto-detects available models)
  - Visualization type buttons: time series, MDS clustering, state-space trajectories
  - 2D/3D toggle for clustering visualizations
  - Static/animated toggle for trajectory visualizations
  - Click-to-view: clicking any data point opens full single-image viewer in new tab
  - Automatic feature-to-visualization compatibility detection

- **Projection methods**
  - `ppca`: Probabilistic PCA via EM algorithm (handles missing data natively)
  - `mds`: Metric Multi-Dimensional Scaling
  - `mds_nonmetric`: Non-metric MDS

- Interactive visualizations (Plotly-based)
  - Single-image viewer with dropdown panel selection
  - Browsable viewer with slider for multiple images/frames
  - Panel types: emotions, scalars, saliency, objects, scenes, wordcloud
  - Linked scatter + timeseries explorer
- Path resolution system (`UnifiedImageResolver`)
  - Sidecar-based path inference
  - Video frame extraction (OpenCV)
  - HDF5 image extraction (h5py)
- Documentation restructure
  - `docs/cli.md` - CLI reference
  - `docs/models.md` - Model documentation
  - `docs/visualization.md` - Visualization guide
  - `docs/api.md` - Python API reference
- `KNOWN_ISSUES.md` for tracking limitations
- `CHANGELOG.md` for version history
- `__version__` attribute in package

## [0.1.0] - 2024-02-17

### Added

- **Models**: 10 pre-integrated models
  - `resmem` - Image memorability (ResMem)
  - `emonet` - 20 emotion categories (EmoNet)
  - `clip` - 512-dim vision-language embeddings (CLIP ViT-B/32)
  - `dinov2` - 768-dim self-supervised features (DINOv2 ViT-B/14)
  - `gist` - 512-dim spatial envelope (GIST)
  - `places` - 365 scene categories + 102 SUN attributes (Places365)
  - `llstat` - 17 low-level image statistics
  - `saliency` - 576-dim fixation density grid (DeepGaze IIE)
  - `aesthetics` - Aesthetic quality score (LAION Aesthetics)
  - `yolo` - 80 object counts + summary stats (YOLOv8)

- **CLI** (`viz2psy`)
  - Unified interface for images, video, and HDF5 input
  - Multi-model batch processing
  - Automatic device detection (CUDA, MPS, CPU)
  - Metadata sidecar generation (`.meta.json`)

- **Visualization CLI** (`viz2psy-viz`)
  - `wordcloud` - CLIP-based semantic word clouds
  - `timeseries` - Feature plots over time (static + interactive)
  - `heatmap` - Correlation heatmaps
  - `scatter` - 2D projections (PCA, UMAP, t-SNE)
  - `composite` - Image + feature panel layouts
  - `explorer` - Linked scatter + timeseries dashboard
  - `image` - Single-image viewer with feature panels
    - Dropdown panel selection (emotions, scalars, saliency, objects, scenes, wordcloud)
    - Browse mode with slider for multiple images/frames
    - Support for image folders, video frames, HDF5 bricks

- **Documentation**
  - `docs/cli.md` - CLI reference
  - `docs/models.md` - Model documentation
  - `docs/visualization.md` - Visualization guide
  - `docs/api.md` - Python API reference

### Known Issues

- Saliency maps display smaller than source images in browse mode
- Saliency model uses CPU on Apple Silicon (MPS doesn't support float64)
