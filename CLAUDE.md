# viz2psy Developer Guide

A Python package for extracting psychological and perceptual features from images using multiple computational models.

## Project Structure

```
viz2psy/
├── CLAUDE.md              # This developer guide
├── LICENSE                # MIT license
├── README.md              # User documentation
├── pyproject.toml         # Package metadata + dependencies
└── src/
    └── viz2psy/
        ├── __init__.py
        ├── cli.py         # Command-line interface
        ├── pipeline.py    # Batch inference: score_images()
        ├── utils.py       # Shared utilities (image loading)
        └── models/        # One module per model
            ├── __init__.py
            ├── base.py    # Abstract base class
            ├── resmem.py
            ├── emonet.py
            ├── clip.py
            ├── gist.py
            ├── llstat.py
            ├── saliency.py
            ├── dinov2.py
            ├── aesthetics.py
            ├── places.py
            └── yolo.py
```

## Model Interface

All models inherit from `BaseModel` and implement three methods:

```python
from viz2psy.models.base import BaseModel
from PIL import Image

class ExampleModel(BaseModel):
    name = "example"  # Used in CLI and output

    def load(self):
        """Load model weights. Called once before inference."""
        self.model = load_weights()

    def predict(self, image: Image.Image) -> dict:
        """Score a single image. Returns {feature_name: value}."""
        return {"score": self.model(image)}

    def predict_batch(self, images: list[Image.Image]) -> list[dict]:
        """Score multiple images. Default impl calls predict() in loop."""
        return [self.predict(img) for img in images]
```

### Key Conventions

1. **Device handling**: Models auto-detect CUDA availability via `self.device` (set in `BaseModel.__init__`). Accept `device` parameter to override.

2. **Lazy loading**: `load()` is called by `score_images()`, not in `__init__`. This allows model inspection without loading weights.

3. **Output format**: `predict()` returns a flat dict. For vector outputs, use numbered keys like `clip_000`, `clip_001`, etc.

4. **Normalization**: Document any output normalization (e.g., L2-normalized embeddings, probability distributions).

## Adding a New Model

1. Create `src/viz2psy/models/yourmodel.py`:

```python
from .base import BaseModel
from PIL import Image

class YourModel(BaseModel):
    name = "yourmodel"

    def load(self):
        import your_library
        self.model = your_library.load()
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image: Image.Image) -> dict:
        tensor = self._preprocess(image)
        with torch.no_grad():
            output = self.model(tensor.to(self.device))
        return {"yourmodel_score": output.item()}
```

2. Register in `src/viz2psy/cli.py`:

```python
MODEL_REGISTRY = {
    ...
    "yourmodel": ("viz2psy.models.yourmodel", "YourModel", "Description here"),
}
```

3. Add any new dependencies to `pyproject.toml`.

4. Update `README.md` with the new model in the Available Models table.

## Development Setup

```bash
git clone https://github.com/bhutch/viz2psy.git
cd viz2psy
pip install -e ".[dev]"
```

## Running Tests

```bash
pytest
```

## Code Style

- Use type hints for function signatures
- Follow PEP 8
- Keep model wrappers thin; complex logic belongs in the upstream library
- Prefer explicit device handling over global state
