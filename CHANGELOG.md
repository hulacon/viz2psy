# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.7.0] - 2026-08-17

### Added

- `ebind` model: 1024-d L2-normalized image embeddings from EBind's
  Perception Encoder vision arm (checkpoint `encord-team/ebind-full`,
  revision-pinned). Shares one cross-modal space with word2psy
  `ebind_text` and aud2psy `ebind_audio`. First >999-d space in the
  family: columns are fixed-width 4-digit (`ebind_0000..ebind_1023`)
  per the amended contracts §4.1.

## [0.6.0] - 2026-08-10

### Changed (BREAKING — output column names)

Conform to the constellation Contract B extractor-output convention
(mmmdata-agents `docs/constellation-contracts.md` §4.1): every feature
column now carries its model's registry-name prefix. **Consumers must do:**
re-extract, or apply `viz2psy.columns.apply_legacy_renames()` when loading
pre-0.6.0 CSVs (the viz CLI does this automatically).

- emonet: bare capitalized names (`Adoration`, `Aesthetic Appreciation`, …)
  → `emonet_adoration`, `emonet_aesthetic_appreciation`, … (snake_case)
- llstat: 17 bare scalars (`luminance_mean`, …) → `llstat_*`
- yolo aggregates: `object_count`, `category_count`, `object_coverage`,
  `largest_object_ratio`, `mean_confidence` → `yolo_*`
- resmem: `memorability` → `resmem_memorability`
- aesthetics: `aesthetic_score` → `aesthetics_score`
- caption: `caption` → `caption_text`
- places `sunattr_*` columns unchanged — declared as an extra prefix of the
  places model (distinct feature space from `places_*` scenes)

### Added

- Sidecar (`.meta.json`): `schema_version` ("1.0"), `extractor`,
  `extractor_version`, and per-model `package_version` + **`checkpoint`**
  (exact architecture+weights identifier, e.g. `ViT-B-32/laion2b_s34b_b79k`)
  — checkpoint identity backs the cross-modal space guarantees in psytwill.
  Legacy `viz2psy_version` and per-model `version` keys retained for one
  deprecation cycle.
- `viz2psy.columns` module: `LEGACY_RENAMES` map +
  `apply_legacy_renames(df)` for loading pre-0.6.0 scores CSVs (the viz
  CLI applies it automatically at load).
- `BaseModel.checkpoint` / `BaseModel.extra_prefixes` class attributes.
- **`stimulus_id` column** (first column of every scores CSV): filename
  stem per image, input-file stem for video/HDF5 (`time` / `image_idx`
  disambiguate rows), `--stimulus-id` to override with a constant.
- Output parent directories are created automatically.


## Initial release feature summary

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

## v0.5.0 (2026-02-19)

### Feat

- **viz**: add Browse All Images button to dashboard
- **viz**: add axis labels, caption wrapping, and example screenshots
- **viz**: add smoothing toggle and human-readable labels to dashboard

## v0.4.0 (2026-02-19)

### Feat

- **cli**: add --parallel flag for concurrent model execution
- add BLIP captioning model and auto-generate HTML visualizations

### Perf

- **viz**: use file:// URLs for images instead of base64 embedding

## v0.3.0 (2026-02-18)

### Feat

- **viz**: optimize viewer performance and UX
- **viz**: reuse saved video frames in visualization

### Fix

- add missing dependencies and structured error handling

## v0.2.0 (2026-02-17)

### Feat

- **viz**: add interactive dashboard with click-to-view functionality
