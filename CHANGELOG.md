# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0] - 2026-09-25

### Fixed

- **Interlaced video was scored as one blank frame.** OpenCV's FFmpeg backend
  cannot convert interlaced frames ("Cannot convert interlaced to progressive
  frames") and returns `ok=True` with a uniform frame, so every model scored
  the same image at every timestamp: valid output, silently void. Found on
  71 interlaced (top-field-first) Friends episodes, where `faces` detected no
  face in any frame and every CLIP row was identical. Interlaced sources
  (`video.is_interlaced`, the leading frames' interlaced flag) are now
  decoded with PyAV through FFmpeg's `yadif` in `send_frame` mode
  (`video.iter_native_frames`), at the same native frame indices OpenCV
  addresses. `motion` reads its frame pairs the same way. **Progressive
  sources take the OpenCV path unchanged**, so their features do not move.
  Features extracted from interlaced video with ≤ 0.10.0 should be
  re-extracted.

### Added

- `extract_frames` refuses a clip in which more than half the decoded frames
  are a single uniform colour (`VideoError`), so a decoder failure of this
  kind fails loudly instead of emitting; and a clip of ≥ 10 sampled frames
  in which more than 90 % are bit-identical to the previous sample (a
  decoder returning one stale picture; the interlaced MPEG-2 synthetic fails
  this way under OpenCV).
- `av>=12` (PyAV) as a dependency.
- `tests/test_video_interlaced.py`: an interlaced synthetic that reproduces
  the OpenCV failure, the deinterlaced path's content, OpenCV-path identity
  on progressive input, the blank-frame refusal, and `motion` on interlaced
  input.

## [0.10.0] - 2026-09-23

### Added

- **Contract B schema 1.1 (`nulls`):** every model entry in the sidecar
  carries `nulls`, mapping each column the model can set to NaN to
  `{"means", "when"}` (constellation-contracts §4.1). Declared on the model
  class (`BaseModel.nulls`, `{}` = never null), stamped in
  `MetadataBuilder.add_model` filtered to the emitted columns, and never
  swallowed on a lookup error (a silent `{}` would be a false "never null").
  `faces`: `center_dist` (no face) and `mutual_dist` (< 2 faces), `undefined`.
  `motion`: six flow statistics `undefinable` at a timestamp with no next
  native frame (the video's end), `motion_coherence` `undefined` (no motion
  has no direction). `schema_version` is now `"1.1"`.
- `viz2psy sidecar refresh PATH... [--dry-run]`: rewrites existing sidecars to
  1.1 in place (JSON only, never the CSV), checking each model's columns in the
  recorded output table and refusing a sidecar with NaN in an undeclared column.
- `tests/test_nulls.py`: the producer-duty contract test (no face, one face,
  a static video pair, a timestamp past the video's end; refresh round-trip).

- `viz2psy-viz dashboard` accepts multiple feature CSVs, or a directory of
  per-model CSVs, and merges them into one all-models dashboard
  (`viz2psy.viz.merge`). Static tables merge on `stimulus_id`; time-indexed
  tables merge exactly on a shared grid or to the nearest spine row within
  half the sampling step (`--merge-tolerance` overrides). The same model
  run on two inputs (e.g. model vs. human captions) gets `__<source>`
  column suffixes; directory-collected tables that cannot merge 1:1 are
  skipped with a note.

## [0.9.2] - 2026-09-08

### Fixed

- `saliency` no longer runs out of GPU memory on high-resolution video.
  DeepGaze IIE's activation footprint scales with pixels x batch, so a
  64-frame batch that fits for 720x360 video overflowed a 40 GB device at
  1920x800. `predict_batch` now splits the caller's batch under a fixed
  pixel budget (`_PIXEL_BUDGET`, 12 M pixels per forward pass). Values are
  unchanged; only the forward-pass grouping differs.

## [0.9.1] - 2026-09-07

### Added

- `--id-pattern` for HDF5 bricks: a format string over the brick index
  (`{idx}` 0-based, `{idx1}` 1-based) that stamps a per-row `stimulus_id`,
  e.g. `--id-pattern 'ext-nsd-{idx1:06d}'`. Until now an HDF5 run could only
  carry one constant id for every row, so a brick of 73,000 stimuli could
  not be scored into Contract B's per-stimulus identity without a post-hoc
  rewrite. Overrides `--stimulus-id` for HDF5 input; ignored otherwise.

## [0.9.0] - 2026-08-25

### Added

- **`depth` model:** six scene-layout statistics from Depth Anything V2
  Small relative depth — `depth_fg_fraction`, `_openness`, `_iqr`,
  `_skew`, `_edge_density` (spatial clutter), `_center_near`
  (composition). Relative depth is normalized per image, so every
  statistic is deliberately invariant to that normalization
  (robust 2nd/98th-percentile scaling); cross-stimulus "mean depth" is
  never emitted because it would be meaningless. Complements gist's
  spatial envelope and places' coarse SUN layout attributes with a
  continuous depth arm. Checkpoint
  `depth-anything/Depth-Anything-V2-Small-hf` (Apache-2.0 — the Small
  variant specifically; larger V2 variants are CC-BY-NC and not used).
  Standard per-image model (stills and video frames). Depth backbones
  are trained on real scenes: animated/stylized clips are out-of-domain,
  a covariate rather than ground truth.

## [0.8.0] - 2026-08-25

### Added

- **`motion` model — viz2psy's first temporal model.** Seven optical-flow
  statistics per video timestamp: `motion_energy`, `_energy_p95`,
  `_coherence` (global vs incoherent motion; NaN when there is no motion),
  `_horizontal`/`_vertical` (signed pan/tilt), `_radial` (positive =
  expansion/looming), `_frame_diff` (classic visual change; spikes at hard
  cuts). Flow is dense Farnebäck between the native frame at each grid
  timestamp and the NEXT native frame (one 1/fps step — flow across the
  0.5 s grid spacing would measure nothing), grayscale, downscaled to
  240 px height; flow-derived values are in frame-heights/second, so they
  are resolution- and frame-rate-independent. Analytic (`checkpoint`
  null). Rows align exactly with the per-frame models' rows; timestamps
  with no next frame score NaN. Flow across a hard cut is left in as a
  real visual transient (`motion_frame_diff` flags it) — cut-aware
  handling belongs downstream.
- **Video-only model class**: `motion` refuses still-image and HDF5
  inputs with an error naming the fix, and runs off the video file inside
  the video pipeline (sequential and parallel modes both).
- **First test suite** (`tests/`): synthetic-video ground-truth tests for
  the motion model (known translation direction, static-video zeros,
  cut detection via frame_diff, image-input refusal).

### Added (0.8.0, continued)

- **`faces` model:** five face count/size/configuration statistics per
  image — `faces_count`, `_total_area`, `_max_area` (shot scale),
  `_center_dist` (framing), `_mutual_dist` (face clustering — the
  two-shot/conversation configuration; NaN below two faces). Detection is
  OpenCV's FaceDetectorYN (YuNet, `opencv_zoo/face_detection_yunet_2023mar`,
  ~230 KB ONNX downloaded on first use to `~/.cache/viz2psy/` and
  SHA-256-verified) — deliberately not mediapipe, whose dependency tree is
  disproportionate to five scalars. A frame with no faces scores explicit
  zeros, not NaN. Standard per-image model: runs on stills and video
  frames alike.

## [0.7.1] - 2026-08-18

Housekeeping release for public use — documentation, packaging metadata, and
citations; no feature-output changes.

### Added

- `caption` and `ebind` optional-dependency extras (`transformers` and the
  GitHub-only `ebind` package were previously undeclared, so those two models
  failed at load time on a clean install).
- `ebind` documented in README and `docs/models.md`, with EBind and
  Perception Encoder references.
- `CITATION.cff` for citing viz2psy itself.
- README "Related packages" section pointing at aud2psy, word2psy, and
  psytwill, and describing the shared CLIP and EBind cross-modal spaces.
- Full references with DOIs/arXiv links for every model.

### Fixed

- `pyproject.toml` project URLs pointed at the pre-transfer
  `github.com/bhutch/viz2psy`; now `hulacon/viz2psy`.
- LICENSE copyright line named the wrong person ("Brice Hutcheson", 2024);
  now Ben Hutchinson, 2026.
- `viz2psy.models` lazy registry was missing `CaptionModel`.
- README install instructions claimed `pip install viz2psy` (not on PyPI);
  now the `git+` form.
- docs: `clip` mis-attributed to OpenAI weights (it uses OpenCLIP
  `laion2b_s34b_b79k`); `LowLevelStatModel` corrected to `LLStatModel` in
  two code snippets; DeepGaze IIE reference corrected to Linardos et al.
  (2021, ICCV); removed unused `setuptools-scm` build requirement.

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
