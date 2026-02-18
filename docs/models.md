# Models

viz2psy provides wrappers for 11 computational models covering memorability, emotion, semantics, captioning, low-level statistics, and visual attention.

## Overview

| Model | Output | Category | Description |
|-------|--------|----------|-------------|
| `resmem` | 1 score | Memory | Image memorability prediction |
| `emonet` | 20 scores | Emotion | Emotion category probabilities |
| `clip` | 512 dims | Semantics | Vision-language embeddings |
| `caption` | 1 caption | Captioning | Natural language image captions (BLIP) |
| `dinov2` | 768 dims | Semantics | Self-supervised visual features |
| `gist` | 512 dims | Scene | Spatial envelope descriptor |
| `places` | 467 scores | Scene | Scene categories + attributes |
| `llstat` | 17 scores | Low-level | Color, contrast, edges, etc. |
| `saliency` | 576 dims | Attention | Fixation density prediction |
| `aesthetics` | 1 score | Quality | Image aesthetic quality |
| `yolo` | 85 scores | Objects | Object detection counts |

---

## Memory

### resmem

Predicts how memorable an image is to human observers.

- **Output**: Single `memorability` score (0-1)
- **Model**: ResMem CNN trained on LaMem dataset
- **Reference**: Needell, C. D., & Bainbridge, W. A. (2022). Embracing new techniques in deep learning for estimating image memorability. *Computational Brain & Behavior*.

```python
from viz2psy.models.resmem import ResMemModel
model = ResMemModel()
```

---

## Emotion

### emonet

Predicts emotion category probabilities for images.

- **Output**: 20 emotion probabilities (sum to 1)
- **Categories**: Adoration, Aesthetic Appreciation, Amusement, Anxiety, Awe, Boredom, Confusion, Craving, Disgust, Empathic Pain, Entrancement, Excitement, Fear, Horror, Interest, Joy, Romance, Sadness, Sexual Desire, Surprise
- **Model**: EmoNet CNN trained on emotion-labeled images
- **Reference**: Kragel, P. A., et al. (2019). Emotion schemas are embedded in the human visual system. *Science Advances*.

```python
from viz2psy.models.emonet import EmoNetModel
model = EmoNetModel()
```

---

## Semantics

### clip

Extracts vision-language embeddings using OpenAI's CLIP model.

- **Output**: 512-dimensional L2-normalized embedding (`clip_000` to `clip_511`)
- **Model**: CLIP ViT-B/32
- **Use cases**: Semantic similarity, zero-shot classification, cross-modal retrieval
- **Reference**: Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *ICML*.

```python
from viz2psy.models.clip import CLIPModel
model = CLIPModel()
```

---

## Captioning

### caption

Generates natural language captions describing image content using BLIP.

- **Output**: Single `caption` string column
- **Model**: Salesforce BLIP (large by default)
- **Use cases**: Image description, accessibility, content understanding, metadata
- **Reference**: Li, J., et al. (2022). BLIP: Bootstrapping Language-Image Pre-training for Unified Vision-Language Understanding and Generation. *ICML*.

```python
from viz2psy.models.caption import CaptionModel

# Default (large model)
model = CaptionModel()

# Use smaller/faster base model
model = CaptionModel(model_name="Salesforce/blip-image-captioning-base")
```

Example output:
```
caption
"a man playing tennis on a tennis court"
"a sunset over the ocean with palm trees"
"a close up of a red flower with water droplets"
```

### dinov2

Extracts self-supervised visual features using Meta's DINOv2.

- **Output**: 768-dimensional embedding (`dinov2_000` to `dinov2_767`)
- **Model**: DINOv2 ViT-B/14
- **Use cases**: Visual similarity, transfer learning, scene understanding
- **Reference**: Oquab, M., et al. (2023). DINOv2: Learning robust visual features without supervision. *arXiv*.

```python
from viz2psy.models.dinov2 import DINOv2Model
model = DINOv2Model()
```

---

## Scene

### gist

Computes Gabor-based spatial envelope descriptors.

- **Output**: 512-dimensional GIST descriptor (`gist_000` to `gist_511`)
- **Model**: Gabor filter banks at multiple scales and orientations
- **Use cases**: Scene categorization, spatial layout analysis
- **Reference**: Oliva, A., & Torralba, A. (2001). Modeling the shape of the scene: A holistic representation of the spatial envelope. *IJCV*.

```python
from viz2psy.models.gist import GISTModel
model = GISTModel()
```

### places

Predicts scene categories and attributes.

- **Output**:
  - 365 scene category probabilities (`places_000` to `places_364`)
  - 102 SUN attribute scores (`sunattr_000` to `sunattr_101`)
- **Model**: Places365 CNN + SUN Attributes
- **Categories**: Indoor/outdoor scenes (kitchen, beach, office, etc.)
- **Attributes**: Natural, open, enclosed, rugged, etc.
- **Reference**: Zhou, B., et al. (2017). Places: A 10 million image database for scene recognition. *TPAMI*.

```python
from viz2psy.models.places import PlacesModel
model = PlacesModel()
```

---

## Low-level Statistics

### llstat

Computes low-level image statistics.

- **Output**: 17 statistics including:
  - `luminance_mean`, `luminance_std` - Brightness
  - `rms_contrast` - Root-mean-square contrast
  - `r_mean`, `r_std`, `g_mean`, `g_std`, `b_mean`, `b_std` - RGB channel stats
  - `lab_l_mean`, `lab_a_mean`, `lab_b_mean` - CIELAB color space
  - `saturation_mean` - Color saturation
  - `hf_energy`, `lf_energy` - High/low frequency energy (FFT)
  - `edge_density` - Canny edge density
  - `colorfulness` - Hasler & Süsstrunk metric

```python
from viz2psy.models.llstat import LowLevelStatModel
model = LowLevelStatModel()
```

---

## Visual Attention

### saliency

Predicts where humans are likely to fixate in an image.

- **Output**: 576-dimensional grid (`saliency_00_00` to `saliency_23_23`)
  - 24x24 spatial grid of fixation densities
  - Column naming: `saliency_X_Y` where X=column, Y=row
- **Model**: DeepGaze IIE
- **Note**: Falls back to CPU on Apple Silicon (MPS doesn't support float64)
- **Reference**: Kümmerer, M., et al. (2022). DeepGaze IIE: Calibrated prediction in and out-of-domain for state-of-the-art saliency modeling. *ICCV*.

```python
from viz2psy.models.saliency import SaliencyModel
model = SaliencyModel()
```

---

## Image Quality

### aesthetics

Predicts aesthetic quality using the LAION Aesthetics model.

- **Output**: Single `aesthetic_score` (1-10 scale)
- **Model**: CLIP-based aesthetic predictor trained on human ratings
- **Reference**: Schuhmann, C., et al. (2022). LAION-5B: An open large-scale dataset for training next generation image-text models. *NeurIPS*.

```python
from viz2psy.models.aesthetics import AestheticsModel
model = AestheticsModel()
```

---

## Object Detection

### yolo

Detects and counts objects using YOLOv8.

- **Output**: 85 values including:
  - 80 object class counts (`yolo_person`, `yolo_car`, etc.)
  - `object_count` - Total objects detected
  - `category_count` - Unique categories present
  - `object_coverage` - Fraction of image covered by detections
  - `largest_object_ratio` - Size of largest object relative to image
  - `mean_confidence` - Average detection confidence
- **Model**: YOLOv8n (nano)
- **Classes**: COCO 80-class object categories

```python
from viz2psy.models.yolo import YOLOModel
model = YOLOModel()
```

---

## Adding Custom Models

Create a new model by inheriting from `BaseModel`:

```python
from viz2psy.models.base import BaseModel
from PIL import Image

class MyModel(BaseModel):
    name = "mymodel"

    def load(self):
        """Load model weights. Called once before inference."""
        self.model = load_weights()
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image) -> dict:
        """Score a single image. Returns {feature_name: value}."""
        tensor = self.preprocess(image)
        with torch.no_grad():
            output = self.model(tensor.to(self.device))
        return {"score": output.item()}

    def predict_batch(self, images: list[Image.Image]) -> list[dict]:
        """Optional: Override for efficient batch processing."""
        return [self.predict(img) for img in images]
```

Key conventions:
- `self.device` is auto-detected (CUDA > MPS > CPU)
- `load()` is called lazily before first inference
- Return flat dicts; use numbered keys for vectors (`feat_000`, `feat_001`, etc.)
