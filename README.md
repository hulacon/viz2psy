# viz2psy

Extract psychological and perceptual features from images using multiple computational models.

## Features

- **10 pre-integrated models** covering memorability, emotion, semantics, low-level statistics, and more
- **Consistent interface** across all models with batch processing support
- **CLI and Python API** for flexible integration
- **GPU acceleration** with automatic device detection

## Installation

```bash
pip install viz2psy
```

Or install from source:

```bash
git clone https://github.com/bhutch/viz2psy.git
cd viz2psy
pip install -e .
```

## Quick Start

### Command Line

```bash
# List available models
viz2psy --list-models

# Score a single image
viz2psy resmem photo.jpg

# Score multiple images and save to CSV
viz2psy clip images/*.png -o embeddings.csv

# Run multiple models at once
viz2psy resmem clip emonet images/*.png -o scores.csv

# Run all models
viz2psy --all images/*.png -o all_scores.csv

# Use CPU explicitly
viz2psy emonet images/*.jpg --device cpu --quiet

# Score video frames (default: every 0.5s)
viz2psy resmem movie.mp4 -o scores.csv

# Custom frame interval (1 second)
viz2psy resmem movie.mp4 --frame-interval 1.0 -o scores.csv

# Save extracted frames to disk (for large videos)
viz2psy resmem movie.mp4 --save-frames ./frames -o scores.csv
```

### Python API

```python
from viz2psy.models.resmem import ResMemModel
from viz2psy.pipeline import score_images

# Initialize and score images
model = ResMemModel()
df = score_images(model, ["photo1.jpg", "photo2.jpg"])
print(df)
```

## Available Models

| Model | Output | Description |
|-------|--------|-------------|
| `resmem` | 1 score | Image memorability (0-1). ResMem, Needell & Bainbridge 2022. |
| `emonet` | 20 scores | Emotion category probabilities. EmoNet, Kragel et al. 2019. |
| `clip` | 512 dims | CLIP ViT-B-32 L2-normalized embeddings. |
| `gist` | 512 dims | Gabor-based spatial envelope descriptor. Oliva & Torralba 2001. |
| `llstat` | 17 scores | Low-level statistics: luminance, contrast, color, edges, etc. |
| `saliency` | 576 dims | DeepGaze IIE 24x24 fixation density grid. |
| `dinov2` | 768 dims | DINOv2 ViT-B/14 self-supervised embeddings. |
| `aesthetics` | 1 score | LAION Aesthetics V2 quality score (1-10). |
| `places` | 467 scores | Places365 scene categories + 102 SUN attributes. |
| `yolo` | 85 scores | YOLOv8 object counts (80 classes) + summary statistics. |

## Adding Custom Models

Create a new model by inheriting from `BaseModel`:

```python
from viz2psy.models.base import BaseModel
from PIL import Image

class MyModel(BaseModel):
    name = "mymodel"

    def load(self):
        # Load your model weights here
        self.model = ...

    def predict(self, image: Image.Image) -> dict:
        # Return a dict of feature names -> values
        return {"score": 0.5}
```

Then use it with the pipeline:

```python
from viz2psy.pipeline import score_images

model = MyModel()
df = score_images(model, image_paths)
```

## Citation

If you use viz2psy in your research, please cite the relevant model papers:

- **ResMem**: Needell, C. D., & Bainbridge, W. A. (2022). Embracing new techniques in deep learning for estimating image memorability. *Computational Brain & Behavior*.
- **EmoNet**: Kragel, P. A., et al. (2019). Emotion schemas are embedded in the human visual system. *Science Advances*.
- **CLIP**: Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *ICML*.
- **DINOv2**: Oquab, M., et al. (2023). DINOv2: Learning robust visual features without supervision. *arXiv*.

## License

MIT License. See [LICENSE](LICENSE) for details.
