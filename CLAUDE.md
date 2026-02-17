# viz2psy — Visual Image Psychological Feature Extraction

## Project Goal

A Python package that takes any image(s) and produces psychological/perceptual
feature scores from multiple computational models. The package is model-agnostic
and extensible — each model is a thin wrapper with a consistent interface. The
primary use case is the MMMData neuroimaging study (1,000 NSD shared images),
but all models work on arbitrary images via `scripts/score.py`.

---

## Input Data

- **Location**: `/projects/hulacon/shared/mmmdata/stimuli/shared1000/images/`
- **Count**: 1,000 images
- **Format**: PNG, 425x425 pixels, 8-bit RGB, non-interlaced
- **Naming**: `shared{NNNN}_nsd{NNNNN}.png`
  - `shared` index: 1-based, zero-padded to 4 digits (0001–1000)
  - `nsd` ID: NSD image identifier (variable digit count, e.g., `02951`, `72949`)
- **Source**: Natural Scenes Dataset (NSD) shared images, cropped from MS-COCO
  `train2017`

### Existing Metadata

Located in `/projects/hulacon/shared/mmmdata/stimuli/shared1000/`:

| File | Description |
|------|-------------|
| `nsd_stim_info.csv` | Full NSD metadata (73K rows); `shared1000=True` for our subset. Key columns: `nsdId` (0-based), `cocoId`, `cocoSplit`, `cropBox` |
| `coco_annotations.csv` | Per-image COCO annotations (1,000 rows): 5 captions, object categories, supercategories, instance counts. Keyed by `nsdId` (0-based) and `cocoId` |
| `coco_captions.csv` | Per-caption long format (5,003 rows): `nsdId`, `cocoId`, `caption_index`, `caption` |

**Important ID note**: `nsdId` in the CSVs is 0-based. The `_nsd{NNNNN}` in
filenames is 1-based (i.e., filename nsd ID = CSV nsdId + 1).

---

## Usage

### Score arbitrary images

```bash
# Single image
python scripts/score.py resmem photo.jpg

# Multiple images, save to CSV
python scripts/score.py clip /path/to/images/*.png -o scores.csv
```

### Score MMMData shared1000 (via pipeline.run_model)

```bash
python scripts/run_resmem.py   # outputs resmem_scores.csv
python scripts/run_all.py      # runs all models
```

## Output Requirements

### MMMData shared1000 output
- One CSV per model, saved in `/projects/hulacon/shared/mmmdata/stimuli/shared1000/`
- Each CSV keyed by `nsdId` (0-based, consistent with existing metadata CSVs)
  and `cocoId`
- Include the image filename as a column for easy cross-referencing
- Naming convention: `{model_name}_scores.csv` (e.g., `resmem_scores.csv`)

### General output (score.py)
- One CSV with columns: `filename`, `filepath`, and all model score columns
- Works with any image format PIL can read

---

## Implemented Models

All models follow the `BaseModel` interface (`load()`, `predict(image)`,
`predict_batch(images)`) and are registered in `scripts/score.py`.

| Model | Module | Output columns | Description |
|-------|--------|---------------|-------------|
| `resmem` | `models/resmem.py` | `memorability` (1) | Image memorability score (0–1). ResMem, Needell & Bainbridge 2022. |
| `emonet` | `models/emonet.py` | `Adoration`, `Awe`, `Fear`, etc. (20) | Emotion category probabilities. EmoNet, Kragel et al. 2019. |
| `clip` | `models/clip.py` | `clip_000`–`clip_511` (512) | CLIP ViT-B-32 L2-normalized embeddings. Semantic image representations. |
| `gist` | `models/gist.py` | `gist_000`–`gist_511` (512) | Gabor-based spatial envelope descriptor. Oliva & Torralba 2001. |
| `llstat` | `models/llstat.py` | `luminance_mean`, `rms_contrast`, etc. (17) | Luminance, RGB, LAB, saturation, spectral energy, edge density, colorfulness. |
| `saliency` | `models/saliency.py` | `saliency_XX_YY` (576) | DeepGaze IIE 24x24 fixation density grid. |
| `dinov2` | `models/dinov2.py` | `dinov2_000`–`dinov2_767` (768) | DINOv2 ViT-B/14 self-supervised embeddings. Visual structure features. |
| `aesthetics` | `models/aesthetics.py` | `aesthetic_score` (1) | LAION Aesthetics V2 MLP on CLIP ViT-L/14. Score ~1–10. |
| `places` | `models/places.py` | `places_*` (365) + `sunattr_*` (102) | Places365 scene categories + 102 SUN scene attributes. |
| `yolo` | `models/yolo.py` | `yolo_*` (80) + aggregates (5) | YOLOv8 object detection: per-category counts + summary stats. |

### Future Models

- **SAM** — Segment Anything for visual complexity (segment counts, size distribution)
- **RetinaFace** — Face detection (count, area, centrality) via insightface
- **ViTPose** — Pose estimation (person count, body area, pose features) via HuggingFace

---

## Compute Environment

### Cluster: University of Oregon HPC (Talapas)

- **OS**: RHEL 8 (Linux 4.18)
- **Default Python**: 3.6.8 (too old — use modules or conda)
- **Available Python modules**: 3.6.x, 3.7.5, 3.10.13, 3.11.4
- **Available CUDA modules**: 11.8.0, 12.4.1, 13.0
- **Conda**: Available (FSL's conda at `/packages/fsl/6.0.7.9/fsl/bin/conda`)
- **Existing conda env**: `neuroconda3` at `/home/bhutch/.conda/envs/neuroconda3`
- **Singularity**: Available at `/usr/bin/singularity`
- **Job scheduler**: SLURM

### GPU Partitions

| Partition | Time limit | GPU type |
|-----------|-----------|----------|
| `gpu` | 1 day | NVIDIA A100 80GB (full, 3g.40gb, 1g.10gb MIG slices) |
| `gpulong` | 14 days | NVIDIA A100 80GB (various MIG configs) |
| `interactivegpu` | 8 hours | NVIDIA A100 (1 node) |

For 1,000 images through 2 models, a single GPU job should be sufficient
(likely < 1 hour total).

### Environment Setup

PyTorch and PIL are NOT installed in the base system Python. You will need to
either:
1. Create a new conda environment with Python 3.10+, PyTorch, torchvision, PIL
2. Use an existing environment if it has the right packages
3. Use a Singularity container

A conda environment is probably simplest. Check if `neuroconda3` already has
PyTorch before creating a new env.

---

## Design Notes

### Architecture

```
viz2psy/
├── CLAUDE.md              # This file
├── pyproject.toml         # Package metadata + dependencies
├── src/
│   └── viz2psy/
│       ├── __init__.py
│       ├── models/        # One module per model
│       │   ├── __init__.py
│       │   ├── base.py    # Abstract base class (load/predict/predict_batch)
│       │   ├── resmem.py
│       │   ├── emonet.py
│       │   ├── clip.py
│       │   ├── gist.py
│       │   ├── llstat.py
│       │   ├── saliency.py
│       │   ├── dinov2.py
│       │   ├── aesthetics.py
│       │   ├── places.py
│       │   └── yolo.py
│       ├── pipeline.py    # Batch inference: load images, run model, save CSV
│       └── utils.py       # Shared utilities (image loading, ID parsing, etc.)
├── scripts/
│   ├── score.py           # CLI: score arbitrary images with any model
│   ├── run_all.py         # Run all models on shared1000
│   └── run_{model}.py     # Per-model entry points for shared1000
└── slurm/
    └── score_images.sbatch  # SLURM job script for GPU inference
```

### Key Design Principles

1. **Each model is a thin wrapper** with a consistent interface:
   - `load()` — load weights
   - `predict(image) -> dict` — returns named scores for a single image
   - `predict_batch(images) -> list[dict]` — optional batched inference
2. **Pipeline handles I/O**: image loading, ID extraction, CSV writing
3. **Batch processing with progress**: use tqdm or similar for progress bars
4. **GPU-aware**: auto-detect CUDA, move models/tensors to GPU when available
5. **Resumable**: if a run is interrupted, skip already-processed images

### ID Handling

The pipeline must correctly handle the 0-based/1-based nsdId discrepancy:
- Parse filenames to extract both the `shared` index and `nsd` ID (1-based)
- Convert to 0-based `nsdId` (subtract 1) for CSV output
- Include `cocoId` by joining with `nsd_stim_info.csv`

---

## Related Project

This pipeline produces metadata for the MMMData neuroimaging study. The main
project code lives at:
- **Agent code**: `/gpfs/projects/hulacon/shared/mmmdata/code/mmmdata-agents/`
- **Dataset docs**: `mmmdata-agents/docs/dataset_reference.md`
- **Stimuli docs**: `mmmdata-agents/docs/stimuli.md`
- **COCO extraction script**: `mmmdata-agents/scripts/extract_coco_metadata.py`
  (reference for how the existing metadata CSVs were produced)
