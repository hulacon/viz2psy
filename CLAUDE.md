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
        ├── cli.py         # Unified CLI (images, video, HDF5)
        ├── pipeline.py    # Batch inference: score_images()
        ├── metadata.py    # Sidecar JSON generation
        ├── video.py       # Video frame extraction
        ├── utils.py       # Shared utilities (image loading)
        ├── models/        # One module per model
        │   ├── __init__.py
        │   ├── base.py    # Abstract base class
        │   ├── resmem.py
        │   ├── emonet.py
        │   ├── clip.py
        │   ├── gist.py
        │   ├── llstat.py
        │   ├── saliency.py
        │   ├── dinov2.py
        │   ├── aesthetics.py
        │   ├── places.py
        │   └── yolo.py
        └── viz/           # Visualization tools
            ├── __init__.py
            └── cli.py
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

## Commit Conventions

This project uses [Conventional Commits](https://www.conventionalcommits.org/) with [Commitizen](https://commitizen-tools.github.io/commitizen/) for automated changelog generation and version bumping.

### Commit Message Format

```
<type>(<scope>): <description>

[optional body]

[optional footer(s)]
```

### Common Types

| Type | Description |
|------|-------------|
| `feat` | New feature |
| `fix` | Bug fix |
| `docs` | Documentation only |
| `style` | Formatting, no code change |
| `refactor` | Code change that neither fixes nor adds |
| `perf` | Performance improvement |
| `test` | Adding or updating tests |
| `chore` | Build process, dependencies, etc. |

### Using Commitizen

```bash
# Interactive commit (guides you through the format)
cz commit

# Check if commits follow convention
cz check --rev-range HEAD~5..HEAD

# Bump version and update CHANGELOG automatically
cz bump

# Dry run to see what version bump would do
cz bump --dry-run
```

### Version Files

Version is tracked in two places (kept in sync by `cz bump`):
- `pyproject.toml`: `version = "X.Y.Z"`
- `src/viz2psy/__init__.py`: `__version__ = "X.Y.Z"`

## Code Style

- Use type hints for function signatures
- Follow PEP 8
- Keep model wrappers thin; complex logic belongs in the upstream library
- Prefer explicit device handling over global state

## CLI Input Types

The unified CLI (`viz2psy`) handles three input types with automatic detection:

| Input Type | Detection | Index Column | Specific Options |
|------------|-----------|--------------|------------------|
| Images | File extensions (.jpg, .png, etc.) | `filename` | — |
| Video | File extensions (.mp4, .mov, etc.) | `time` | `--frame-interval`, `--save-frames` |
| HDF5 brick | File extensions (.h5, .hdf5) | `image_idx` | `--dataset`, `--start`, `--end` |

## Metadata Sidecar

When output is saved to file (`-o`), a `.meta.json` sidecar is created with:

```json
{
  "viz2psy_version": "0.1.0",
  "created_at": "2024-...",
  "input": { "type": "hdf5_brick", "path": "...", ... },
  "output": { "path": "...", "rows": 1000, "columns": 2909 },
  "index_column": "image_idx",
  "device": "mps",
  "total_runtime_sec": 123.4,
  "models": {
    "clip": { "pattern": "clip_{NNN}", "range": [0, 511], "count": 512 },
    "emonet": { "columns": [...], "definition": "feature_definitions.emonet" }
  },
  "feature_definitions": { "emonet": ["Adoration", ...] }
}
```

Feature naming:
- **Pattern-based** (clip, dinov2, gist, saliency): `"pattern": "clip_{NNN}"` with range
- **Named columns** (emonet, places, yolo, etc.): Full column list in `feature_definitions`
