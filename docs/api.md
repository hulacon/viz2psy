# Python API

## Pipeline

### score_images

The main entry point for batch image scoring.

```python
from viz2psy.pipeline import score_images

def score_images(
    models: BaseModel | list[BaseModel],
    inputs: list[str | Path] | str | Path,
    batch_size: int = 16,
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Score images with one or more models.

    Parameters
    ----------
    models : BaseModel or list of BaseModel
        Model instance(s) to run.
    inputs : list of paths, or single path
        Image file paths, video file, or HDF5 file.
    batch_size : int
        Batch size for inference (default: 16).
    show_progress : bool
        Show progress bar (default: True).

    Returns
    -------
    pd.DataFrame
        Scores with one row per image, columns for each feature.
    """
```

**Example:**

```python
from viz2psy.models.resmem import ResMemModel
from viz2psy.models.clip import CLIPModel
from viz2psy.pipeline import score_images

models = [ResMemModel(), CLIPModel()]
df = score_images(models, ["img1.jpg", "img2.jpg"])
```

---

## Models

### BaseModel

Abstract base class for all models.

```python
from viz2psy.models.base import BaseModel

class BaseModel:
    name: str  # Model identifier
    device: torch.device  # Auto-detected (CUDA > MPS > CPU)

    def load(self) -> None:
        """Load model weights. Called lazily before first inference."""
        ...

    def predict(self, image: Image.Image) -> dict[str, float]:
        """
        Score a single image.

        Parameters
        ----------
        image : PIL.Image.Image
            Input image (RGB).

        Returns
        -------
        dict
            Feature names mapped to values.
        """
        ...

    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, float]]:
        """
        Score multiple images. Default calls predict() in loop.
        Override for efficient batched inference.
        """
        ...
```

### Available Models

```python
from viz2psy.models.resmem import ResMemModel
from viz2psy.models.emonet import EmoNetModel
from viz2psy.models.clip import CLIPModel
from viz2psy.models.gist import GISTModel
from viz2psy.models.llstat import LowLevelStatModel
from viz2psy.models.saliency import SaliencyModel
from viz2psy.models.dinov2 import DINOv2Model
from viz2psy.models.aesthetics import AestheticsModel
from viz2psy.models.places import PlacesModel
from viz2psy.models.yolo import YOLOModel
```

All models accept an optional `device` parameter:

```python
model = CLIPModel(device="cpu")  # Force CPU
model = CLIPModel(device="cuda:1")  # Specific GPU
```

---

## Visualization

### Interactive Plots

```python
from viz2psy.viz.interactive import (
    plot_scatter_interactive,
    plot_timeseries_interactive,
    create_linked_explorer,
    create_single_image_viewer,
    create_browsable_viewer,
    save_figure,
)
```

#### plot_scatter_interactive

```python
def plot_scatter_interactive(
    df: pd.DataFrame,
    features: list[str] | None = None,
    method: str = "pca",  # "pca", "umap", "tsne"
    color_by: str | None = None,
    max_points: int = 5000,
    sidecar: SidecarMetadata | None = None,
) -> go.Figure:
    """
    Create interactive 2D scatter projection.

    Parameters
    ----------
    df : pd.DataFrame
        Feature data.
    features : list of str
        Features to project. Supports glob patterns ("clip_*").
    method : str
        Dimensionality reduction method.
    color_by : str
        Column for point coloring.
    max_points : int
        Sample if more points than this.

    Returns
    -------
    plotly.graph_objects.Figure
    """
```

#### plot_timeseries_interactive

```python
def plot_timeseries_interactive(
    df: pd.DataFrame,
    features: list[str] | None = None,
    time_col: str = "time",
    normalize: bool = False,
    sidecar: SidecarMetadata | None = None,
) -> go.Figure:
    """
    Create interactive timeseries plot.

    Parameters
    ----------
    df : pd.DataFrame
        Feature data with time column.
    features : list of str
        Features to plot. If None, uses all numeric.
    time_col : str
        Name of time column.
    normalize : bool
        Normalize features to [0, 1].

    Returns
    -------
    plotly.graph_objects.Figure
    """
```

#### create_single_image_viewer

```python
def create_single_image_viewer(
    image: str | Path | Image.Image,
    scores_df: pd.DataFrame,
    row_idx: int = 0,
    panels: list[str] | None = None,
    width: int = 1200,
    height: int = 600,
    sidecar: SidecarMetadata | None = None,
    normalize_scalars: bool = True,
    image_title: str | None = None,
) -> go.Figure:
    """
    Create two-panel viewer with image and feature visualizations.

    Parameters
    ----------
    image : path or PIL.Image
        Image to display.
    scores_df : pd.DataFrame
        Feature scores.
    row_idx : int
        Row index for this image.
    panels : list of str
        Panel types: "emotions_bar", "emotions_spider", "scalars",
        "saliency", "yolo", "places", "wordcloud".
        If None, auto-detects available features.
    normalize_scalars : bool
        Normalize scalar values to [0, 1].

    Returns
    -------
    plotly.graph_objects.Figure
    """
```

#### create_browsable_viewer

```python
def create_browsable_viewer(
    scores_df: pd.DataFrame,
    image_resolver: UnifiedImageResolver,
    panels: list[str] | None = None,
    width: int = 1200,
    height: int = 700,
    max_rows: int = 100,
    sidecar: SidecarMetadata | None = None,
    normalize_scalars: bool = True,
) -> go.Figure:
    """
    Create viewer with slider to browse multiple images.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Feature scores with multiple rows.
    image_resolver : UnifiedImageResolver
        Resolver for getting images from rows.
    panels : list of str
        Panel types to include.
    max_rows : int
        Maximum rows to pre-render.

    Returns
    -------
    plotly.graph_objects.Figure
    """
```

#### save_figure

```python
def save_figure(fig: go.Figure, path: str | Path) -> None:
    """
    Save Plotly figure to file.

    Supports .html, .png, .svg, .pdf, .json.
    """
```

---

## Sidecar Metadata

### load_sidecar

```python
from viz2psy.viz.sidecar import load_sidecar, SidecarMetadata

def load_sidecar(csv_path: str | Path) -> SidecarMetadata | None:
    """
    Load .meta.json sidecar for a CSV file.

    Returns None if sidecar doesn't exist.
    """
```

### SidecarMetadata

```python
class SidecarMetadata:
    @property
    def index_column(self) -> str | None:
        """Index column name (time, filename, image_idx)."""

    @property
    def input_type(self) -> str | None:
        """Input type (video, image_folder, hdf5_brick)."""

    def get_semantic_label(self, column: str) -> str:
        """Get human-readable label for a column."""

    def get_model_for_column(self, column: str) -> str | None:
        """Determine which model produced a column."""
```

### Image Resolution

```python
from viz2psy.viz.sidecar import (
    create_image_resolver,
    create_unified_resolver,
    ImagePathResolver,
    UnifiedImageResolver,
)

# For image folders
resolver = create_image_resolver(csv_path, image_root=Path("./images"))
image_path = resolver.resolve(df.iloc[0])

# For any input type (images, video, HDF5)
resolver = create_unified_resolver(
    csv_path,
    image_root=Path("./images"),  # For image folders
    video_path=Path("movie.mp4"),  # For video
    hdf5_path=Path("data.h5"),     # For HDF5
    hdf5_dataset="stimuli",
)
image = resolver.resolve(df, row_idx=0)  # Returns Path or PIL.Image
```

---

## Static Plots

Matplotlib-based static visualizations:

```python
from viz2psy.viz import (
    plot_timeseries,
    plot_heatmap,
    plot_scatter,
    plot_composite,
)
from viz2psy.viz.wordcloud import make_wordcloud
```

These functions return `matplotlib.figure.Figure` objects.
