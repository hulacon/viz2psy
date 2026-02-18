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
| `--save-frames DIR` | Save extracted frames to directory |

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
- Scalar features: `memorability`, `aesthetic_score`
- Embeddings: `clip_000`, `clip_001`, ..., `clip_511`
- Grids: `saliency_00_00`, `saliency_00_01`, ...
- Categories: `Adoration`, `Amusement`, `places_000`, `yolo_person`

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
  "total_runtime_sec": 45.2,
  "models": {
    "resmem": {
      "features": {"columns": ["memorability"]},
      "runtime_sec": 12.3
    },
    "clip": {
      "features": {"pattern": "clip_{NNN}", "range": [0, 511], "count": 512},
      "runtime_sec": 32.9
    }
  }
}
```

This sidecar is used by `viz2psy-viz` for:
- Path resolution (finding source images/video)
- Semantic labels (displaying feature names)
- Input type detection (image folder vs video vs HDF5)

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
