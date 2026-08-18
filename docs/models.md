# Models

viz2psy provides wrappers for 12 computational models covering memorability, emotion, semantics, captioning, low-level statistics, and visual attention.

## Overview

| Model | Output | Category | Description |
|-------|--------|----------|-------------|
| `resmem` | 1 score | Memory | Image memorability prediction |
| `emonet` | 20 scores | Emotion | Emotion category probabilities |
| `clip` | 512 dims | Semantics | Vision-language embeddings (OpenCLIP) |
| `ebind` | 1024 dims | Semantics | Cross-modal EBind embeddings (shared image–text–audio space) |
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

- **Output**: Single `resmem_memorability` score (0-1)
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

- **Output**: 20 emotion probabilities (`emonet_adoration` to `emonet_surprise`, sum to 1)
- **Categories**: Adoration, Aesthetic Appreciation, Amusement, Anxiety, Awe, Boredom, Confusion, Craving, Disgust, Empathic Pain, Entrancement, Excitement, Fear, Horror, Interest, Joy, Romance, Sadness, Sexual Desire, Surprise (columns are lowercase with the `emonet_` prefix, e.g. `emonet_empathic_pain`)
- **Model**: EmoNet CNN trained on emotion-labeled images
- **Reference**: Kragel, P. A., et al. (2019). Emotion schemas are embedded in the human visual system. *Science Advances*.

```python
from viz2psy.models.emonet import EmoNetModel
model = EmoNetModel()
```

---

## Semantics

### clip

Extracts vision-language embeddings using OpenCLIP.

- **Output**: 512-dimensional L2-normalized embedding (`clip_000` to `clip_511`)
- **Model**: OpenCLIP ViT-B/32, LAION-2B weights (`laion2b_s34b_b79k`) — not the original OpenAI checkpoint
- **Use cases**: Semantic similarity, zero-shot classification, cross-modal retrieval
- **Cross-modal**: same space as word2psy `clip_text` (identical checkpoint), so image and text embeddings are directly comparable
- **Reference**: Radford, A., et al. (2021). Learning transferable visual models from natural language supervision. *ICML*. Weights: Cherti, M., et al. (2023). Reproducible scaling laws for contrastive language-image learning. *CVPR*.

```python
from viz2psy.models.clip import CLIPModel
model = CLIPModel()
```

### ebind

Extracts cross-modal embeddings from EBind's Perception Encoder vision arm.
Images, video frames, text (word2psy `ebind_text`), and audio (aud2psy
`ebind_audio`) all land in one shared 1024-d space.

- **Output**: 1024-dimensional L2-normalized embedding (`ebind_0000` to `ebind_1023`; fixed-width 4-digit indices)
- **Model**: EBind (`encord-team/ebind-full`, revision-pinned); vision arm is Perception Encoder `PE-Core-L14-336`
- **Install**: requires the `ebind` extra (`pip install "viz2psy[ebind]"` from source); weights are CC-BY-NC-SA 4.0 (non-commercial)
- **Use cases**: cross-modal similarity between images, text, and audio without a projection step
- **Reference**: Broadbent, J., et al. (2025). EBind: A practical approach to space binding. *arXiv:2511.14229*. Bolya, D., et al. (2025). Perception Encoder: The best visual embeddings are not at the output of the network. *NeurIPS*.

```python
from viz2psy.models.ebind import EBindModel
model = EBindModel()
```

---

## Captioning

### caption

Generates natural language captions describing image content using BLIP.

- **Output**: Single `caption_text` string column
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
caption_text
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
- **Reference**: Zhou, B., et al. (2018). Places: A 10 million image database for scene recognition. *IEEE TPAMI, 40*(6), 1452–1464.

```python
from viz2psy.models.places import PlacesModel
model = PlacesModel()
```

---

## Low-level Statistics

### llstat

Computes low-level image statistics.

- **Output**: 17 statistics including:
  - `llstat_luminance_mean`, `llstat_luminance_std` - Brightness
  - `llstat_rms_contrast` - Root-mean-square contrast
  - `llstat_r_mean`, `llstat_r_std`, `llstat_g_mean`, `llstat_g_std`, `llstat_b_mean`, `llstat_b_std` - RGB channel stats
  - `llstat_lab_l_mean`, `llstat_lab_a_mean`, `llstat_lab_b_mean` - CIELAB color space
  - `llstat_saturation_mean` - Color saturation
  - `llstat_hf_energy`, `llstat_lf_energy` - High/low frequency energy (FFT)
  - `llstat_edge_density` - Canny edge density
  - `llstat_colorfulness` - Hasler & Süsstrunk metric
- **Reference** (colorfulness): Hasler, D., & Suesstrunk, S. E. (2003). Measuring colorfulness in natural images. *SPIE Human Vision and Electronic Imaging VIII*.

```python
from viz2psy.models.llstat import LLStatModel
model = LLStatModel()
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
- **Reference**: Linardos, A., Kümmerer, M., Press, O., & Bethge, M. (2021). DeepGaze IIE: Calibrated prediction in and out-of-domain for state-of-the-art saliency modeling. *ICCV 2021*.

```python
from viz2psy.models.saliency import SaliencyModel
model = SaliencyModel()
```

---

## Image Quality

### aesthetics

Predicts aesthetic quality using the LAION Aesthetics model.

- **Output**: Single `aesthetics_score` (1-10 scale)
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
  - `yolo_object_count` - Total objects detected
  - `yolo_category_count` - Unique categories present
  - `yolo_object_coverage` - Fraction of image covered by detections
  - `yolo_largest_object_ratio` - Size of largest object relative to image
  - `yolo_mean_confidence` - Average detection confidence
- **Model**: YOLOv8n (nano)
- **Classes**: COCO 80-class object categories
- **Reference**: Jocher, G., Chaurasia, A., & Qiu, J. (2023). Ultralytics YOLOv8 (software). https://github.com/ultralytics/ultralytics

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
