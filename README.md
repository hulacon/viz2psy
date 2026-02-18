# viz2psy

Extract psychological and perceptual features from images using multiple computational models.

## Features

- **11 pre-integrated models** covering memorability, emotion, semantics, captioning, saliency, and more
- **Unified CLI** for images, videos, and HDF5 image bricks
- **Interactive visualizations** with Plotly-based dashboards
- **Metadata sidecar files** documenting outputs and feature definitions

## Installation

```bash
pip install viz2psy
```

Or from source:

```bash
git clone https://github.com/bhutch/viz2psy.git
cd viz2psy
pip install -e .
```

## Quick Start

```bash
# Score images with multiple models
viz2psy resmem clip emonet images/*.jpg -o scores.csv

# Score video frames
viz2psy resmem movie.mp4 --frame-interval 1.0 -o scores.csv

# Visualize results
viz2psy-viz image scores.csv --browse --image-root ./images -o viewer.html
```

```python
from viz2psy.models.resmem import ResMemModel
from viz2psy.pipeline import score_images

model = ResMemModel()
df = score_images(model, ["photo1.jpg", "photo2.jpg"])
```

## Documentation

| Document | Description |
|----------|-------------|
| [CLI](docs/cli.md) | `viz2psy` command line reference |
| [Models](docs/models.md) | Available models, outputs, and references |
| [Visualization](docs/visualization.md) | `viz2psy-viz` CLI and interactive features |
| [API](docs/api.md) | Python API reference |
| [Known Issues](KNOWN_ISSUES.md) | Current limitations |

## Available Models

| Model | Output | Description |
|-------|--------|-------------|
| `resmem` | 1 score | Image memorability |
| `emonet` | 20 scores | Emotion probabilities |
| `clip` | 512 dims | Vision-language embeddings |
| `caption` | 1 caption | Natural language image captions (BLIP) |
| `dinov2` | 768 dims | Self-supervised features |
| `gist` | 512 dims | Spatial envelope |
| `places` | 467 scores | Scene categories + attributes |
| `llstat` | 17 scores | Low-level statistics |
| `saliency` | 576 dims | Fixation density grid |
| `aesthetics` | 1 score | Aesthetic quality |
| `yolo` | 85 scores | Object detection counts |

See [docs/models.md](docs/models.md) for detailed documentation.

## Citation

If you use viz2psy in your research, please cite the relevant model papers:

- **ResMem**: Needell & Bainbridge (2022). *Computational Brain & Behavior*.
- **EmoNet**: Kragel et al. (2019). *Science Advances*.
- **CLIP**: Radford et al. (2021). *ICML*.
- **BLIP**: Li et al. (2022). *ICML*.
- **DINOv2**: Oquab et al. (2023). *arXiv*.

## License

MIT License. See [LICENSE](LICENSE) for details.
