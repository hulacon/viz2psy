# Command Line Interface

The `viz2psy` CLI extracts features from images, videos, and HDF5 image bricks.

## Basic Usage

```bash
viz2psy [models...] [inputs...] [options]
```

Models and inputs are positional arguments. Models are recognized by name, everything else is treated as input.

```bash
# Single model, single image
viz2psy resmem photo.jpg

# Multiple models, multiple images
viz2psy resmem clip emonet images/*.jpg

# All models
viz2psy --all images/*.png -o scores.csv
```

---

## Input Types

### Images

Standard image files (.jpg, .png, .gif, .bmp, .webp, etc.)

```bash
viz2psy resmem photo.jpg
viz2psy clip images/*.png -o embeddings.csv
```

### Video

Video files are automatically detected by extension (.mp4, .mov, .avi, .mkv, etc.) and processed frame-by-frame.

```bash
# Default: extract frame every 0.5 seconds
viz2psy resmem movie.mp4 -o scores.csv

# Custom frame interval (1 second)
viz2psy resmem movie.mp4 --frame-interval 1.0 -o scores.csv

# Save extracted frames to disk (for large videos)
viz2psy resmem movie.mp4 --save-frames ./frames -o scores.csv
```

Output includes a `time` column with frame timestamps in seconds.

### HDF5 Image Bricks

HDF5 files containing image arrays (e.g., NSD dataset).

```bash
# Auto-detect dataset
viz2psy resmem data.hdf5 -o scores.csv

# Specify dataset name
viz2psy resmem data.hdf5 --dataset imgBrick -o scores.csv

# Process slice of images
viz2psy resmem data.hdf5 --dataset imgBrick --start 0 --end 1000 -o scores.csv

# List available datasets
viz2psy --list-datasets data.hdf5
```

Output includes an `image_idx` column with indices into the HDF5 dataset.

---

## Options

### Model Selection

| Option | Description |
|--------|-------------|
| `--list-models` | List available models and exit |
| `--all` | Run all available models |

### Output

| Option | Description |
|--------|-------------|
| `-o, --output PATH` | Save results to CSV file |
| `-q, --quiet` | Suppress progress output |

When `-o` is specified, a metadata sidecar file (`.meta.json`) is automatically created alongside the CSV.

### Device

| Option | Description |
|--------|-------------|
| `--device DEVICE` | Force device: `cpu`, `cuda`, `cuda:0`, `mps` |

Default: auto-detect (CUDA > MPS > CPU).

### Video Options

| Option | Description |
|--------|-------------|
| `--frame-interval SEC` | Seconds between extracted frames (default: 0.5) |
| `--save-frames DIR` | Save viewing copies of the frames to directory (default: beside the output) |
| `--no-save-frames` | Do not save viewing copies |
| `--frame-format {jpg,png}` | Format of the viewing copies (default: jpg, quality 85) |
| `--framing-table CSV` | A finalized framing table (see [Framing Tables](#framing-tables-viz2psy-framing)): crop each frame to its file's bar box before any model sees it |

Models always score the decoded frames, never the saved copies. When frames
must go to disk (`--parallel` with several models, or when the frames would
exceed 80 % of available memory), the scoring copies are lossless PNG in a
temporary directory. The sidecar records this as `input.framing`
(`frame_source: "raw decode"`, `frame_cache: "memory" | "png"`).

**Stage 0 (every video, every model).** Each decoded frame is cropped to the
file's bar box from `--framing-table`, then resized to square pixels (the
cropped width times the stream's sample aspect ratio, bicubic). Without a
table nothing is cropped and only the pixels are squared. A video missing
from a given table, changed since detection, or with a different sample
aspect ratio is an error. `motion` applies the same stage 0 to the frames it
reads itself. The sidecar's `input.framing` records it: `stage0`,
`stage0_version`, `stored_wh`, `pixel_aspect`, `crop`, `active_wh`,
`framing_table`, `framing_table_sha256`, `detect_rule_version`, `box_source`.
Still images record `frame_source: "still image"` (stage 0 is a no-op).

**Stage 1 (per model, every input).** Each model's input transform is
recorded in the sidecar under `models.<name>.preprocessing`. Square-input
networks see the whole picture: `clip`, `aesthetics`, `ebind`, `dinov2` and
`places` squash it to a short-side square before their own transform, as
`caption`, `emonet`, `vgg19`, `gist` and `resmem` already did. `saliency`
and `llstat` work at a canonical pixel area (425 x 425) at the picture's
aspect ratio. Square stills (e.g. NSD's 425 x 425) are unchanged by all of
this.

### HDF5 Options

| Option | Description |
|--------|-------------|
| `--list-datasets` | List datasets in HDF5 file and exit |
| `--dataset NAME` | HDF5 dataset name (auto-detected if not specified) |
| `--start INDEX` | Start index for slice (default: 0) |
| `--end INDEX` | End index for slice (default: all) |

### Processing

| Option | Description |
|--------|-------------|
| `--batch-size N` | Batch size for inference (default: 32) |
| `--parallel` | Run multiple models in parallel (one process per model). Useful with `--all`. |

---

## Examples

### Basic Scoring

```bash
# Score memorability of images
viz2psy resmem images/*.jpg

# Extract CLIP embeddings and save
viz2psy clip images/*.png -o clip_embeddings.csv

# Run multiple models
viz2psy resmem emonet aesthetics images/*.jpg -o scores.csv

# Run all models (comprehensive feature extraction)
viz2psy --all images/*.jpg -o all_features.csv

# Run all models in parallel (faster on multi-core systems)
viz2psy --all --parallel images/*.jpg -o all_features.csv
```

### Video Processing

```bash
# Score video every 0.5 seconds (default)
viz2psy resmem emonet movie.mp4 -o video_scores.csv

# Score every 2 seconds for long videos
viz2psy clip movie.mp4 --frame-interval 2.0 -o embeddings.csv

# Save frames for inspection
viz2psy resmem movie.mp4 --save-frames ./extracted_frames -o scores.csv
```

### HDF5 Processing

```bash
# Score first 100 images from NSD dataset
viz2psy resmem nsd_stimuli.hdf5 --dataset imgBrick --end 100 -o nsd_scores.csv

# Process large dataset in chunks
viz2psy clip data.hdf5 --start 0 --end 10000 -o chunk1.csv
viz2psy clip data.hdf5 --start 10000 --end 20000 -o chunk2.csv
```

### Device Selection

```bash
# Force CPU (useful for debugging)
viz2psy resmem images/*.jpg --device cpu

# Use specific GPU
viz2psy clip images/*.jpg --device cuda:1

# Use Apple Silicon GPU
viz2psy clip images/*.jpg --device mps
```

---

## Output Format

### CSV Structure

Output is a CSV with one row per image/frame:

| Column | Description |
|--------|-------------|
| `filename` | Image filename (for image inputs) |
| `time` | Timestamp in seconds (for video inputs) |
| `image_idx` | Index into dataset (for HDF5 inputs) |
| `<feature>` | Model output values |

Feature column names follow model conventions:
- Scalar features: `resmem_memorability`, `aesthetics_score`
- Embeddings: `clip_000`, `clip_001`, ..., `clip_511`
- Grids: `saliency_00_00`, `saliency_00_01`, ...
- Categories: `emonet_adoration`, `emonet_amusement`, `places_000`, `yolo_person`

### Metadata Sidecar

When saving to file, a `.meta.json` sidecar is created with:

```json
{
  "viz2psy_version": "0.1.0",
  "created_at": "2024-01-15T10:30:00",
  "input": {
    "type": "video",
    "path": "/path/to/movie.mp4",
    "frame_interval_sec": 0.5
  },
  "output": {
    "path": "/path/to/scores.csv",
    "rows": 120,
    "columns": 533
  },
  "device": "cuda",
  "device_name": "NVIDIA H100 80GB HBM3",
  "total_runtime_sec": 45.2,
  "models": {
    "resmem": {
      "features": {"columns": ["resmem_memorability"]},
      "runtime_sec": 12.3
    },
    "clip": {
      "features": {"pattern": "clip_{NNN}", "range": [0, 511], "count": 512},
      "runtime_sec": 32.9
    }
  }
}
```

`device_name` is optional: the GPU model behind a `cuda` device, written when
it can be read and left out otherwise (CPU, MPS, or a name the driver will not
report). It matters for models whose output depends on the GPU's arithmetic:
`caption_vlm`'s greedy bfloat16 captions reword some frames between GPU types,
so compare or pool such outputs within one `device_name`.

This sidecar is used by `viz2psy-viz` for:
- Path resolution (finding source images/video)
- Semantic labels (displaying feature names)
- Input type detection (image folder vs video vs HDF5)

---

## Framing Tables (`viz2psy framing`)

A framing table holds one letterbox/pillarbox crop box per video file. It is
detected once per file, not frame by frame, because a dark scene or a fade
looks like a bar in any single frame. Boxes are in the pixel coordinates of
the frames the models score (the same decode loop, `viz2psy.video.iter_frames`).

```bash
# 1. Detect: strict + robust box per file, flags, band levels (1 frame/s)
viz2psy framing detect --list videos.txt -o detected.csv -j 8

# 2. Review: contact sheets (max-over-all-frames panel + 11 frames, box drawn)
viz2psy framing sheet detected.csv --out-dir sheets/ --select flagged

# 3. Finalize: merge reviewed decisions; refuses while a flagged file is unreviewed
viz2psy framing finalize detected.csv --overrides overrides.csv -o framing_table.csv
```

- **Strict box** (the one used): lines never lit (luma > `--dark`, default 24)
  in any non-black frame, so it never crops a pixel that was ever shown.
- **Robust box** (flag only): lines lit in at most `--sparse` (25 %) of their
  pixels in at least `--robust` (50 %) of lit frames. When it disagrees with
  the strict box, subtitles or credits probably cross a bar.
- **Flags**: `strict_robust_disagree`, `large_crop` (> 30 % removed),
  `lopsided`, `dim_band` (a cropped band brighter than near-black: a dark
  picture edge that never exceeds `--dark` is cropped like a bar but still
  carries detail; each band's level is in `band_<edge>`), `no_frames`.
- **Overrides CSV**: columns `path, decision, reason, reviewer, crop_left,
  crop_right, crop_top, crop_bottom`. `decision` is `confirm` or `override`
  (box columns required for `override`); every row needs a reason.
- **Keys**: rows are keyed on the absolute path as given, never a resolved
  symlink (git-annex trees key on the stimulus path, not the annex object).
  Lookup fails loudly on a missing path or a changed file size.
- Every table is written with a `.json` parameter record that pins its
  sha256; a table edited by hand refuses to load.

---

## Performance Tips

1. **Use GPU**: CUDA is fastest, MPS (Apple Silicon) is good, CPU is slowest
2. **Batch size**: Increase `--batch-size` for faster throughput (limited by GPU memory)
3. **Parallel mode**: Use `--parallel` with multiple models to run them concurrently
4. **Video frame interval**: Use larger intervals for long videos
5. **HDF5 slicing**: Process large datasets in chunks with `--start`/`--end`
6. **Model selection**: Only run models you need; `--all` is comprehensive but slow

---

## See Also

- [Models](models.md) - Detailed model documentation
- [Visualization](visualization.md) - `viz2psy-viz` CLI for visualizing results
- [API](api.md) - Python API reference
