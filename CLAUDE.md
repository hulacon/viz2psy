# viz2psych — Visual Image Psychological Feature Extraction

## Project Goal

Build a Python pipeline that takes images and produces psychological/perceptual
feature scores from multiple computational models. The immediate use case is
scoring 1,000 images from the MMMData neuroimaging study, but the code should
be model-agnostic and extensible to new models.

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

## Output Requirements

- One CSV per model, saved in `/projects/hulacon/shared/mmmdata/stimuli/shared1000/`
- Each CSV keyed by `nsdId` (0-based, consistent with existing metadata CSVs)
  and `cocoId`
- Include the image filename as a column for easy cross-referencing
- Naming convention: `{model_name}_scores.csv` (e.g., `resmem_scores.csv`,
  `emonet_scores.csv`)

---

## Models to Implement

### 1. ResMem (Image Memorability)

- **What it predicts**: How memorable an image is to humans (scalar 0–1)
- **Paper**: Needell & Bainbridge, "Embracing New Techniques in Deep Learning
  for Estimating Image Memorability" (2022)
- **Code**: https://github.com/Brain-Bridge-Lab/resmem
- **Install**: `pip install resmem` (PyPI package)
- **Architecture**: ResNet-based with regression head
- **Input**: 227x227 RGB images (built-in `transformer` handles resizing)
- **Output**: Single float (memorability score, 0–1)
- **Dependencies**: PyTorch, torchvision, PIL

**Usage example**:
```python
from resmem import ResMem, transformer
from PIL import Image

model = ResMem(pretrained=True)
model.eval()

img = Image.open('path/to/image.png').convert('RGB')
image_x = transformer(img)
prediction = model(image_x.view(-1, 3, 227, 227))
# prediction is a tensor with the memorability score
```

### 2. EmoNet (Emotion Classification)

- **What it predicts**: Emotion category probabilities from images
- **Paper**: Kragel et al., "Emotion schemas are embedded in the human visual
  system" (2019, Science Advances)
- **Code**: https://github.com/ecco-laboratory/emonet-pytorch
- **Install**: Clone repo, download weights from OSF via built-in method
- **Architecture**: Fine-tuned AlexNet (only last FC layer retrained for
  emotion categories)
- **Input**: RGB images (AlexNet standard: 227x227 or 224x224 — check repo)
- **Output**: Probabilities over ~20 emotion categories (from Cowen & Keltner
  2017 taxonomy — includes amusement, anger, awe, contentment, disgust,
  excitement, fear, sadness, etc.)
- **Dependencies**: PyTorch

**Usage example**:
```python
import torch
from models import EmoNet

model = EmoNet()
model.load_state_dict_from_web()  # downloads weights from OSF
model.eval()
# Process image through model...
```

The repo includes `replicate-kragel2019.py` as a reference for how to process
images.

### Future Models (not yet, but design for extensibility)

- Additional memorability models (AMNet, MemNet via Caffe-to-PyTorch)
- Scene recognition (Places365)
- Aesthetic quality prediction
- Visual complexity metrics
- Low-level features (spatial frequency, color statistics, edge density)

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

### Recommended Architecture

```
viz2psych/
├── CLAUDE.md              # This file
├── pyproject.toml         # Package metadata + dependencies
├── src/
│   └── viz2psych/
│       ├── __init__.py
│       ├── models/        # One module per model
│       │   ├── __init__.py
│       │   ├── base.py    # Abstract base class for all models
│       │   ├── resmem.py  # ResMem wrapper
│       │   └── emonet.py  # EmoNet wrapper
│       ├── pipeline.py    # Batch inference: load images, run model, save CSV
│       └── utils.py       # Shared utilities (image loading, ID parsing, etc.)
├── scripts/
│   ├── run_resmem.py      # CLI entry point for ResMem scoring
│   ├── run_emonet.py      # CLI entry point for EmoNet scoring
│   └── run_all.py         # Run all models
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
