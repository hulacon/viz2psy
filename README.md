# viz2psy

Psychological and perceptual feature extraction from images using pretrained computational models.

Built for scoring the 1,000 shared images from the [MMMData](https://github.com/jhutchin) neuroimaging study (Natural Scenes Dataset subset), but usable on any image set.

## Models

| Model | Output | Dimensions | Reference |
|-------|--------|------------|-----------|
| **ResMem** | Image memorability | 1 (scalar 0--1) | [Needell & Bainbridge 2022](https://github.com/Brain-Bridge-Lab/resmem) |
| **EmoNet** | Emotion category probabilities | 20 (Cowen & Keltner taxonomy) | [Kragel et al. 2019](https://github.com/ecco-laboratory/emonet-pytorch) |
| **CLIP** | Semantic image embedding | 512 (ViT-B-32, L2-normalized) | [Radford et al. 2021](https://github.com/mlfoundations/open_clip) |
| **GIST** | Spatial envelope descriptor | 512 (Gabor filter bank) | [Oliva & Torralba 2001](https://doi.org/10.1016/S0079-6123(06)55002-2) |
| **LLStat** | Low-level image statistics | 17 (luminance, color, frequency, edges) | Custom |
| **Saliency** | Visual saliency map (24x24 grid) | 576 (spatial attention density) | [DeepGaze IIE](https://github.com/matthias-k/DeepGaze) |

### EmoNet emotion categories

Adoration, Aesthetic Appreciation, Amusement, Anxiety, Awe, Boredom, Confusion, Craving, Disgust, Empathic Pain, Entrancement, Excitement, Fear, Horror, Interest, Joy, Romance, Sadness, Sexual Desire, Surprise

### LLStat features

luminance_mean, luminance_std, rms_contrast, r_mean, r_std, g_mean, g_std, b_mean, b_std, lab_l_mean, lab_a_mean, lab_b_mean, saturation_mean, hf_energy, lf_energy, edge_density, colorfulness

## Setup

Designed for the University of Oregon Talapas HPC cluster.

```bash
# Create the conda environment with PyTorch + CUDA
bash setup_env.sh

# Activate
conda activate viz2psy
```

This creates a `viz2psy` conda environment with Python 3.10, PyTorch (CUDA 12.4), and all dependencies. The package is installed in editable mode.

### Dependencies

- Python >= 3.10
- PyTorch >= 2.0
- torchvision >= 0.15
- Pillow >= 9.0
- pandas >= 1.5
- tqdm >= 4.60
- resmem >= 1.1
- open-clip-torch >= 2.24
- scikit-image >= 0.21
- deepgaze-pytorch >= 1.0

## Usage

### Score all shared1000 images (SLURM)

```bash
# Create log directory
mkdir -p slurm/logs

# Submit GPU job (runs all models)
sbatch slurm/score_images.sbatch
```

This runs all models (ResMem, EmoNet, CLIP, GIST, LLStat, Saliency) over all 1,000 images and saves CSVs to the shared stimuli directory.

### Run individual models

```bash
# ResMem only
python scripts/run_resmem.py

# EmoNet only
python scripts/run_emonet.py

# Both models
python scripts/run_all.py
```

All scripts accept `--image-dir`, `--stim-info`, `--output-dir`, and `--batch-size` arguments.

### Score arbitrary images

```bash
# Single image (prints to stdout)
python scripts/score.py resmem photo.jpg

# Multiple images, save to CSV
python scripts/score.py emonet img1.png img2.png -o scores.csv
```

### Python API

```python
from viz2psy.models import ResMemModel, EmoNetModel
from viz2psy import score_images

# Score a list of images with any model
model = ResMemModel()
df = score_images(model, ["img1.png", "img2.png"])

# Or use model wrappers directly
model = EmoNetModel()
model.load()
scores = model.predict(some_pil_image)  # {"Adoration": 0.02, "Awe": 0.15, ...}
```

## Output

Each model produces a CSV keyed by `nsdId` (0-based) and `cocoId`, consistent with the existing MMMData metadata files:

```
nsdId,cocoId,filename,memorability
2950,391895,shared0001_nsd02951.png,0.7823
```

Output CSVs are saved to `/projects/hulacon/shared/mmmdata/stimuli/shared1000/` as `{model_name}_scores.csv`.

Processing is **resumable** -- if a job is interrupted, rerunning it will skip already-scored images.

## Project structure

```
viz2psy/
├── CLAUDE.md                  # Detailed project specification
├── README.md                  # This file
├── pyproject.toml             # Package metadata and dependencies
├── setup_env.sh               # Conda environment setup for Talapas
├── src/
│   └── viz2psy/
│       ├── __init__.py        # Exports score_images()
│       ├── pipeline.py        # Batch inference and CSV I/O
│       ├── utils.py           # Image loading, filename parsing, ID mapping
│       └── models/
│           ├── __init__.py    # Exports all model classes
│           ├── base.py        # Abstract base class (BaseModel)
│           ├── resmem.py      # ResMem memorability wrapper
│           ├── emonet.py      # EmoNet emotion wrapper
│           ├── clip.py        # CLIP semantic embedding wrapper
│           ├── gist.py        # GIST spatial envelope descriptor
│           ├── llstat.py      # Low-level image statistics
│           └── saliency.py    # DeepGaze IIE saliency (24x24 grid)
├── scripts/
│   ├── run_resmem.py          # CLI: run ResMem on shared1000
│   ├── run_emonet.py          # CLI: run EmoNet on shared1000
│   ├── run_all.py             # CLI: run all models on shared1000
│   └── score.py               # CLI: score arbitrary images
└── slurm/
    └── score_images.sbatch    # SLURM GPU job script
```

## Adding a new model

1. Create `src/viz2psy/models/yourmodel.py` implementing `BaseModel`:

```python
from .base import BaseModel

class YourModel(BaseModel):
    name = "yourmodel"

    def load(self):
        # Load weights, set self.model, move to self.device
        ...

    def predict(self, image):
        # Return dict of named scores
        return {"score_name": 0.5}
```

2. Register it in `src/viz2psy/models/__init__.py`
3. Add a script in `scripts/` or add it to `scripts/run_all.py`

## Future models

The following models are planned or under consideration:

- **MOSAIC brain-optimized encoders** -- Shared-core encoder networks (ResNet18, AlexNet, Swin-T, etc.) trained end-to-end on fMRI visual cortex data. These learn a low-dimensional image representation optimized for predicting human brain responses, making them psychologically grounded feature spaces. Model weights will be released upon peer-reviewed publication of [Lahner et al. 2025](https://www.biorxiv.org/content/10.64898/2025.11.28.690060v1.full) ([project page](https://blahner.github.io/MOSAICfmri/)).
- **Places365** -- Scene category classification (365 categories, indoor/outdoor)
- **Aesthetic quality** -- NIMA or TANet for predicted aesthetic ratings
- **VGG-16 / AlexNet layer features** -- Intermediate DNN representations at multiple levels of the visual hierarchy

## Related data

The input images and output CSVs live in `/projects/hulacon/shared/mmmdata/stimuli/shared1000/`:

| File | Description |
|------|-------------|
| `images/` | 1,000 PNG images (425x425, RGB) |
| `nsd_stim_info.csv` | Full NSD metadata (73K rows; `shared1000=True` for our subset) |
| `coco_annotations.csv` | Per-image COCO annotations (categories, captions) |
| `coco_captions.csv` | Per-caption long format (5,003 rows) |
| `resmem_scores.csv` | ResMem memorability scores (generated by this pipeline) |
| `emonet_scores.csv` | EmoNet emotion probabilities (generated by this pipeline) |
| `clip_scores.csv` | CLIP 512-d semantic embeddings (generated by this pipeline) |
| `gist_scores.csv` | GIST 512-d spatial envelope descriptors (generated by this pipeline) |
| `llstat_scores.csv` | Low-level image statistics (generated by this pipeline) |
| `saliency_scores.csv` | DeepGaze IIE 24x24 saliency grids (generated by this pipeline) |

## License

Internal research use -- University of Oregon HPC.
