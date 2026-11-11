# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.21.1] - 2026-11-10

### Changed
- `faces`: the `faces_count` Description and the module docstring now state
  the detector's domain. YuNet is trained on photographs of human faces and
  misses most faces in animated material (drawn, CG, stylised or animal
  characters), so a zero there does not mean "no face"; detections it does
  make are reliable in both domains. Wording only: values, weights and
  preprocessing are unchanged, and nothing needs re-extracting.

## [0.21.0] - 2026-11-10

### Changed
- `faces` detects on a canonical input size: every picture is resized
  (bilinear, aspect kept) to a short side of 540 px, and the YuNet score
  threshold is 0.8 (was: native pixels, 0.9). At native pixels the smallest
  detectable face is a fixed pixel count, so `faces_count` and the
  frame-relative area columns carried a floor that moved with each corpus's
  resolution; it is now a fixed fraction of the frame. The size and threshold
  come from a calibration against human face boxes on live-action video. At
  the old setting YuNet's scores cluster just under 0.9, so many visible
  faces were recorded as `faces_count` 0. **Values change**: `faces` outputs
  from earlier versions are not comparable and should be re-extracted. The
  sidecar's `preprocessing` now records the resize and threshold. Weights and
  `checkpoint` are unchanged.

## [0.20.1] - 2026-11-10

### Fixed
- `yolo` no longer drops `yolov8n.pt` into the working directory. Given a
  bare weights name it cannot find, ultralytics downloads it into the cwd
  (its own `weights_dir` setting defaults to the relative `weights`), so every
  directory a job ran from collected a copy. A bare name now resolves under
  `<torch hub dir>/ultralytics/` (follows `TORCH_HOME`, like the other
  torch-based models). If the file is missing it is downloaded there once,
  through a private temporary directory and an atomic rename, so concurrent
  array tasks sharing the cache never read a partly written file. An explicit
  path is used as given, and a bare name already present in the cwd is still
  honoured. Features and the recorded `checkpoint` (`yolov8n.pt`) are
  unchanged.

## [0.20.0] - 2026-11-10

### Added
- Several videos in one call: `viz2psy <model> a.mp4 b.mp4 ... -o out.csv`.
  Each model is loaded once and run over every video in turn, so a corpus of
  short clips no longer pays a model load per clip. Every video goes through
  the same stage 0 as a single-video call (frames decoded, cropped to the
  framing table's box, pixels squared), and the features are identical to one
  call per video, `motion` included. Rows are keyed (`stimulus_id`, `time`);
  `--id-pattern` takes `{stem}` to name each video's rows (default: the
  stem). The sidecar's `input.type` is `"video_set"`: the shared framing rule
  is recorded once as `input.framing` (so `framing_table_sha256` sits where it
  does for one video), and each file's path, `stimulus_id`, `n_frames`,
  `frame_cache` and per-file stage-0 values (box, sizes, `box_source`) are in
  `input.videos`. Additive, so the schema version stays 1.3. A set that mixes
  videos and images, two videos with one `stimulus_id`, `--stimulus-id`,
  `--save-frames` and `--parallel` are refused; no dashboard is built. Every
  video is looked up in the framing table before any model loads.
- `viz2psy.framing.RULE_RECORD_KEYS`: the stage-0 record keys every video
  under one rule shares.

## [0.19.1] - 2026-11-10

### Changed
- `gist` convolves its 32 Gabor kernels in the Fourier domain, all at once,
  instead of one direct `scipy.ndimage.convolve` per kernel. The largest
  kernel (frequency 0.05) is 69 x 69, so the direct path cost ~2.8 G
  multiply-adds per 256 px image, ~4 s on one core: on video, `gist` was
  slower than every GPU model combined. Now ~75 ms per frame end to end
  (~55x). The boundary rule is unchanged (`reflect`, reproduced by symmetric
  padding), and the descriptor equals the old one to floating-point rounding
  (relative difference ~1e-16; `tests/test_gist.py` checks it against the
  direct path). Columns, labels and preprocessing are unchanged.

## [0.19.0] - 2026-11-09

### Added
- Optional sidecar key `device_name`: the GPU model behind a `cuda` device
  (e.g. `NVIDIA H100 80GB HBM3`), next to `device`. Written when it can be
  read, left out otherwise (CPU, MPS, an unreadable name), and never a reason
  for a run to fail. Additive and optional, so the schema version stays 1.3.
  `caption_vlm`'s greedy bf16 captions reword some frames between GPU types,
  so its outputs should be compared or pooled within one `device_name`.

## [0.18.0] - 2026-11-09

### Added
- `caption_vlm`: one-sentence frame captions (`caption_vlm_text`, string) from
  `Qwen/Qwen3-VL-8B-Instruct` at a pinned Hub revision (`checkpoint` =
  `<repo>@<revision>`), with a frozen, versioned prompt (`caption-prompt v1`),
  greedy decoding (<= 64 new tokens, bfloat16) and inputs shrunk to <= 768 px
  on the long side, aspect kept. The sidecar's `preprocessing` record carries
  the prompt, its version and the decoding. `caption` (BLIP) is unchanged.
  Needs transformers >= 4.57; an older version is a load error naming the fix.

## [0.17.0] - 2026-11-08

### Added
- `--framing-table`: the framing table's reader. Every video frame goes
  through *stage 0* before any model sees it: crop to the file's reviewed bar
  box, then resize to square pixels (cropped width x sample aspect ratio,
  bicubic). A source missing from the table, changed since detection, or with
  a different sample aspect ratio is an error. Without a table, pixels are
  still squared and nothing is cropped. `motion` applies stage 0 to the
  frames it reads itself, so its frame-height units and radial centre are the
  picture's.
- `input.framing` (Contract B 1.3, additive) records stage 0: `stage0`,
  `stage0_version`, `stored_wh`, `pixel_aspect`, `crop`, `active_wh`,
  `framing_table`, `framing_table_sha256`, `detect_rule_version`,
  `box_source`. Still-image runs record `frame_source: "still image"`.
- Every model declares `describe_preprocessing()`; the sidecar records it
  under `models.<name>.preprocessing`.

### Changed
- **Square-input networks see the whole picture.** `clip`, `aesthetics`,
  `ebind`, `dinov2` and `places` squash the input to a short-side square
  before their own transform, instead of centre-cropping it. Square inputs
  are unchanged; a non-square image or frame changes.
- **`llstat` runs at the canonical pixel area** (425 x 425 at the picture's
  aspect ratio, bicubic), like `saliency`: its spectral and edge columns
  depend on absolute pixel scale. 425 x 425 inputs are unchanged.
- **Video pixels are squared for every model** (stage 0). Before, only
  `saliency` corrected a non-square sample aspect ratio; it now runs at
  `pixel_aspect` 1 on video, since its frames are already square-pixelled.

## [0.16.0] - 2026-11-07

### Added
- `viz2psy framing {detect,finalize,sheet}`: one letterbox/pillarbox crop box
  per video file (the *framing table*), detected once per file rather than
  on the fly, because a dark scene or a fade looks like a bar in any single
  frame. `detect` samples at 1 frame/s and writes a strict box (lines never
  lit in any non-black frame; never crops a pixel that was ever shown) plus a
  robust box (lines sparse in at least half the frames) used only to flag
  files where subtitles or credits cross a bar. Files are also flagged for a
  large crop (> 30 %), lopsided bars, or a cropped band that is not black
  (`dim_band`: a dark picture edge that never exceeds the darkness threshold
  is cropped like a bar, but still carries detail; each band's level is
  recorded as `band_<edge>`). `sheet` draws review contact sheets
  (per-pixel max over all frames + 11 frames, box drawn). `finalize` merges a
  hand-edited overrides file (`confirm`/`override`, each with a reason) and
  refuses while a flagged file is unreviewed. Tables carry a sha256-pinned
  parameter record; a table edited by hand, or a source missing from it or
  changed since detection, is a loud error. The table's reader for
  extraction (`--framing-table`) is in 0.17.0.

### Changed
- The video decode loop is now `viz2psy.video.iter_frames`, shared by
  `extract_frames` (scoring) and `framing` (bar detection), so a framing box
  is in the pixel coordinates of the frames the models score. Frames are
  unchanged.

## [0.15.0] - 2026-11-07

### Fixed

- **Video features are always scored on the decoded frames.** Before this
  release, the CLI scored JPEG (quality 85) copies of the frames whenever they
  went to disk: on `--save-frames` (the default unless `--no-save-frames`), on
  `--parallel` runs with several models, and, silently, when the estimated
  frame memory exceeded 80 % of available. JPEG moves some models' features a
  long way (clip, ebind and aesthetics correlate about .95 with the same model
  on decoded frames), so the same video and model could emit different
  features depending on flags and node memory, and the sidecar did not say
  which. Saved frames are now viewing copies only; frames cached on disk for
  scoring are lossless PNG. Tables written with `--no-save-frames`, without
  `--parallel` and without the memory fallback were already scored on decoded
  frames and are unchanged.

### Changed

- **Contract B schema 1.3:** video sidecars carry `input.framing` =
  `{"frame_source": "raw decode", "frame_cache": "memory" | "png"}`. Absent
  means unrecorded (written before 0.15.0). `viz2psy sidecar refresh` cannot
  recover it, so it refreshes a video sidecar without it to 1.2 at most;
  image and HDF5 sidecars refresh to 1.3.
- `viz2psy.video.extract_frames` gains `view_dir` / `view_format` (viewing
  copies that are never returned). `extract_frames_to_temp` always writes
  lossless PNG and drops `frame_format` in favour of `view_dir` /
  `view_format`. PNG frames are written at compress level 1 (lossless,
  faster to write, larger on disk).

## [0.14.0] - 2026-11-05

Feature values are unchanged; nothing needs re-extracting. Existing sidecars
gain labels with `viz2psy sidecar refresh` (JSON only).

### Added

- **Contract B schema 1.2 (per-column `labels`):** every model entry in the
  sidecar carries `labels`, saying what each feature column is and how its
  value is computed: `{"default": {...}, "columns": {<column>: {...}}}`, where
  a `columns` entry replaces the default's keys for that column (key-wise
  merge). Each column resolves to `interpretability` (`nameable` / `opaque`)
  and `derivation` (`formula`, `formula-on-model`, `trained-on-labels`,
  `probe`, `zero-shot`, `embedding`, `bookkeeping`), plus `derived_from` (the
  parent's exact checkpoint) for `probe` / `formula-on-model` / `zero-shot`,
  and BIDS `LongName` / `Description` / `Units` where they are obvious.
  Declared on the model class (`BaseModel.labels`), written by
  `MetadataBuilder.add_model` with `columns` filtered to the emitted columns.
  `schema_version` is now `"1.2"`. Vocabularies, the merge and the rule checks
  live in `viz2psy.labels`. All 16 models are labelled:
  - `clip`, `dinov2`, `ebind`: opaque embeddings. `vgg19`: opaque embedding,
    except the 1,000 `fc8` units, which are ImageNet class logits
    (nameable, trained-on-labels, class name in `Description`).
  - `gist`, `llstat`, `motion`: nameable formulas (each `gist_NNN` column's
    frequency, orientation and grid cell is in its `Description`).
  - `saliency` and `emonet`: nameable probes. DeepGaze IIE trains only its
    readout networks over four frozen ImageNet backbones; EmoNet retrains
    only AlexNet's last fully connected layer. `derived_from` names the
    backbones.
  - `aesthetics`: nameable probe on CLIP `ViT-L-14/openai`.
  - `places`: scene probabilities trained-on-labels; the 102 `sunattr_*`
    columns are probes with `derived_from` = places' own checkpoint.
  - `yolo`, `faces`, `depth`, `resmem`, `caption`: nameable,
    trained-on-labels; `yolo_mean_confidence` is bookkeeping.
- `tests/test_labels.py`: the producer-duty contract test (every inventory
  column resolves to valid labels, `columns` keys are real columns,
  `derived_from` present exactly where required; sidecar and refresh
  round-trip).

### Changed

- `viz2psy sidecar refresh` writes `labels` as well as `nulls` and brings
  sidecars to 1.2. It still rewrites JSON only and never a CSV.
- `places` hard-codes the 102 SUN attribute column names (`SUN_ATTRIBUTES`)
  that its labels are keyed on, and `load()` refuses a downloaded attribute
  list that differs. The module docstring now says what the code does: the
  attribute scores are a linear readout of the 512-d pooled features, not of
  the scene probabilities.
- The EBind checkpoint `encord-team/ebind-full` loads at a pinned hub revision (`482831d`, `DEFAULT_REVISION`) instead of whatever `main` is. That snapshot has been `main` since 2025-11-19, so features are unchanged. viz2psy, word2psy and aud2psy pin the same revision, so the shared space cannot split when the hub moves. A non-default checkpoint still loads from `main`. The `ebind` extra pins the `ebind` package to `7909701`, and the DeepGaze dependency pins `7698f83`: the builds the family has been running on (psytwill's `setup_env.sh` already pinned DeepGaze to it).

## [0.13.0] - 2026-10-15

Outputs change. Re-extract anything listed under **Changed outputs** before
comparing with ≤ 0.12.0 results.

### Changed outputs

- **`places` scene columns: 314 -> 365.** Category names kept only the last
  path segment, so the 51 qualified Places365 classes (`church/indoor`,
  `church/outdoor`, ...) collapsed onto `places_indoor`, `places_outdoor`,
  ... and each later class overwrote the earlier one: those columns held one
  arbitrary class's probability under a generic name. They are now
  `places_church_indoor`, `places_church_outdoor`, etc. Single-segment names
  (`places_abbey`) are unchanged. Re-extract `places`.
- **Video and `motion` frame timing.** Sample times accumulated
  `t += interval` and frames were picked by truncating `t * fps`, so float
  error scored the frame *before* the intended one at many samples (0.1 s at
  30 fps: `t = 0.9999999999999999` read frame 29, not 30). Times are now
  `i * interval` and the frame is `floor(t * fps)` with an epsilon. The `time`
  column holds clean values, and the scored frame changes at roughly half the
  samples on common fps/interval pairs. Re-extract video outputs from every
  model.
- **Multi-model image runs with duplicate basenames** (`sub-01/x.png`,
  `sub-02/x.png`) no longer multiply rows. The sequential path merged models
  on the basename, a cross join: N such inputs gave N x N rows. Rows are now
  one per input, aligned by position (as the `--parallel` path already was).
  `filename` / `stimulus_id` are still the basename/stem, so such inputs
  remain ambiguous by name.
- **dinov2 sidecar pattern** is `dinov2_{NNN}` (was `dinov_{NNN}`, which
  named no real column). CSV columns are unchanged; regenerate sidecars or
  run `viz2psy sidecar` to refresh them.

### Changed

- **`plotly` is a core dependency** (was the `interactive` extra, which now
  holds only `kaleido`). The dashboard is generated by default on every `-o`
  run, so a bare install ended in a traceback after writing its outputs.
- **Timed merges (`viz2psy-viz` over several CSVs) raise** when an input has
  stimuli that the spine (the longest input) lacks. Before, those stimuli were
  dropped silently. A tolerance of 0 (the default when the spine has one row
  per stimulus) now means an exact time match; it used to mean unbounded, so
  samples at any distance were attached.
- Python floor raised to 3.11 and `scikit-learn>=1.8` (sklearn 1.8 dropped
  3.10, and MDS changed its keyword arguments).

### Fixed

- HDF5 input without `-o` crashed after scoring, leaving a CSV with no
  sidecar.
- Dashboard projection panels no longer turn a broken call into a warning and
  an empty panel; only too-few-rows/features (`ValueError`) is skipped.
- `compute_projection` no longer writes its NaN imputation into the caller's
  array.
- A corrupt `.meta.json` now warns instead of being treated as absent.

## [0.12.0] - 2026-09-29

### Changed

- **`saliency` resizes every input to a canonical pixel area before
  DeepGaze IIE**: NSD's 425 x 425, at the input's display aspect ratio
  (`saliency.canonical_size`, bicubic). DeepGaze's output depends on absolute
  pixel scale. Native 1920 x 800 film frames gave saliency grids ~5x the
  dimensionality of NSD's, and outside NSD's subspace. At NSD's pixel area,
  with the aspect kept, they fall back to it. The grid stays in frame
  coordinates. **425 x 425 inputs (NSD) are unchanged, bit for bit**; any
  other size scores differently from ≤ 0.11.0 and should be re-extracted if
  it is compared with NSD. `SaliencyModel(canonical_area=None)` keeps the
  stored pixel count (the old behaviour for square-pixel input).
- For video, `saliency` also takes the stream's **sample aspect ratio**, so an
  anamorphic source (e.g. 720 x 480 at 8:9, displayed 4:3) is scored at its
  display shape. **The other models still score stored pixels.** A framing
  rule shared by every model is future work.

### Fixed

- **`llstat_rms_contrast` exploded on near-black frames.** It is
  `luminance std / mean`, unbounded as the mean goes to 0: fades and cuts to
  black reached ~600 on Friends and ~230 on movie10, against a maximum of 4.1
  over the 73,000 NSD images. Those rows dominated any projection that
  whitens llstat. It is now **NaN (declared `undefined`) below a mean
  luminance of 0.005** (`llstat.LUMINANCE_FLOOR`), the smallest floor that
  bounds it on those corpora and drops no NSD image. The old `0.0` for an
  exactly black image, a created value, is gone too.
- `llstat_hf_energy` / `llstat_lf_energy` on an all-black image were a created
  `0.5 / 0.5`. They are now NaN, declared `undefined`.
- llstat features extracted with ≤ 0.11.0 on video with dark frames should be
  re-extracted. NSD images are unchanged.

### Added

- `video.get_pixel_aspect` and `pixel_aspect` in `video.get_video_info`
  (via PyAV; 1.0 when undeclared).
- A `preprocessing` entry in the sidecar for a model that declares an input
  transform (`describe_preprocessing`; now `saliency`): canonical area,
  pixel aspect, interpolation.
- `vgg19` (torchvision VGG19, per-channel spatial means; added after 0.11.0,
  see its commit).

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
