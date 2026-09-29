"""DeepGaze IIE wrapper — visual saliency prediction.

Produces a 24x24 spatial saliency grid per image by pooling the full
log-density saliency map from DeepGaze IIE into coarse grid cells.
Output keys use x_y coordinates: saliency_00_00 (top-left) through
saliency_23_23 (bottom-right), where x is the column and y is the row.

Every input is first resized to a canonical pixel area (NSD's 425 x 425) at
its display aspect ratio. DeepGaze's output depends on absolute pixel scale:
the same film frame scored at its native 1920 x 800 has ~5x the saliency
dimensionality of NSD images, and at NSD's pixel area it falls back to NSD's.
The grid stays in frame coordinates (the whole frame, undistorted, is pooled
to 24 x 24).
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.special import logsumexp

from .base import BaseModel

_DEFAULT_GRID_SIZE = 24

#: Pixel budget per forward pass. DeepGaze IIE runs four full-resolution
#: backbones, so its activation footprint scales with pixels x batch: a
#: 64-frame batch of SD video (720x360) fits a 40 GB device, the same batch
#: of 1920x800 film does not. predict_batch() splits the caller's batch so
#: that batch_size x H x W stays under this budget (about 46 SD frames, 7
#: at 1920x800, 5 at 1080p). Sub-batching does not change any value.
_PIXEL_BUDGET = 12_000_000

#: Pixel area every input is resized to before DeepGaze (NSD's 425 x 425).
CANONICAL_AREA = 425 * 425


def canonical_size(width: int, height: int, pixel_aspect: float = 1.0,
                   area: int | None = CANONICAL_AREA) -> tuple[int, int]:
    """(width, height) holding ``area`` pixels at the display aspect ratio.

    The display width is ``width * pixel_aspect`` (the stream's sample aspect
    ratio; 1.0 for square pixels). ``area=None`` keeps the stored pixel count
    and only corrects the aspect.
    """
    display_w = width * pixel_aspect
    target = width * height if area is None else area
    scale = (target / (display_w * height)) ** 0.5
    return max(1, round(display_w * scale)), max(1, round(height * scale))


class SaliencyModel(BaseModel):
    """DeepGaze IIE saliency model with 24x24 spatial grid output.

    Predicts where humans are likely to fixate in an image, pooled
    into a coarse spatial grid (576 values by default).
    """

    name = "saliency"
    checkpoint = "DeepGazeIIE-pretrained"

    def __init__(self, grid_size: int = _DEFAULT_GRID_SIZE, device: str | None = None,
                 canonical_area: int | None = CANONICAL_AREA, pixel_aspect: float = 1.0):
        super().__init__(device=device)
        self.grid_size = grid_size
        self.canonical_area = canonical_area
        self.pixel_aspect = float(pixel_aspect)
        self._centerbias: np.ndarray | None = None

    @staticmethod
    def describe_preprocessing(canonical_area: int | None = CANONICAL_AREA,
                               pixel_aspect: float = 1.0) -> dict:
        """The input resize, as recorded in the sidecar."""
        return {
            "resize": "canonical pixel area at display aspect ratio"
                      if canonical_area is not None else "display aspect ratio, stored pixel count",
            "canonical_area": canonical_area,
            "pixel_aspect": pixel_aspect,
            "interpolation": "bicubic",
        }

    def preprocessing(self) -> dict:
        return self.describe_preprocessing(self.canonical_area, self.pixel_aspect)

    def _prepare(self, image: Image.Image) -> np.ndarray:
        """RGB array at the canonical size (unchanged when already there)."""
        image = image.convert("RGB")
        size = canonical_size(image.width, image.height, self.pixel_aspect, self.canonical_area)
        if size != image.size:
            image = image.resize(size, Image.BICUBIC)
        return np.array(image)

    def load(self) -> None:
        import deepgaze_pytorch
        import torch

        # DeepGaze uses float64 buffers which MPS doesn't support
        if self.device.type == "mps":
            import warnings
            warnings.warn("Saliency model using CPU (MPS doesn't support float64)")
            self.device = torch.device("cpu")

        self.model = deepgaze_pytorch.DeepGazeIIE(pretrained=True)
        self.model.eval()
        self.model = self.model.to(self.device)

    def _get_centerbias(self, h: int, w: int) -> torch.Tensor:
        """Return a Gaussian centerbias log-density matching (h, w)."""
        if self._centerbias is not None and self._centerbias.shape == (h, w):
            return torch.tensor(self._centerbias, dtype=torch.float32)

        # Gaussian centerbias (sigma = 1/3 of image extent).
        cy, cx = h / 2.0, w / 2.0
        sigma_y, sigma_x = h / 3.0, w / 3.0
        Y, X = np.mgrid[:h, :w]
        log_density = -0.5 * (((Y - cy) / sigma_y) ** 2 + ((X - cx) / sigma_x) ** 2)
        log_density -= logsumexp(log_density)
        self._centerbias = log_density.astype(np.float32)
        return torch.tensor(self._centerbias, dtype=torch.float32)

    def _map_to_grid(self, log_density: torch.Tensor) -> np.ndarray:
        """Pool a (1, 1, H, W) log-density map to a (grid, grid) probability grid."""
        # Convert log-density to probability.
        prob = torch.exp(log_density)
        # Pool to grid_size x grid_size.
        grid = F.adaptive_avg_pool2d(prob, self.grid_size)
        # Normalize so values sum to 1.
        grid = grid / grid.sum()
        return grid.squeeze().cpu().numpy()

    def predict(self, image: Image.Image) -> dict[str, float]:
        img = self._prepare(image)
        h, w = img.shape[:2]

        image_tensor = torch.tensor(img.transpose(2, 0, 1)[None], dtype=torch.float32).to(self.device)
        centerbias = self._get_centerbias(h, w).unsqueeze(0).to(self.device)

        with torch.no_grad():
            log_density = self.model(image_tensor, centerbias)

        grid = self._map_to_grid(log_density)
        gs = self.grid_size
        return {
            f"saliency_{x:02d}_{y:02d}": float(grid[y, x])
            for y in range(gs)
            for x in range(gs)
        }

    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, float]]:
        # DeepGaze expects all images in a batch to have the same resolution.
        arrays = [self._prepare(img) for img in images]

        # Check if all images have the same shape
        shapes = [a.shape[:2] for a in arrays]
        if len(set(shapes)) > 1:
            # Different sizes - fall back to single-image processing
            return [self._predict_arrays([a], *a.shape[:2])[0] for a in arrays]

        h, w = arrays[0].shape[:2]
        per = max(1, _PIXEL_BUDGET // (h * w))
        results: list[dict[str, float]] = []
        for start in range(0, len(arrays), per):
            results.extend(self._predict_arrays(arrays[start : start + per], h, w))
        return results

    def _predict_arrays(self, arrays: list[np.ndarray], h: int, w: int) -> list[dict[str, float]]:
        """One forward pass over same-shaped (H, W, 3) uint8 arrays."""
        batch = torch.tensor(
            np.stack([a.transpose(2, 0, 1) for a in arrays]),
            dtype=torch.float32,
        ).to(self.device)
        centerbias = self._get_centerbias(h, w).unsqueeze(0).expand(len(arrays), -1, -1).to(self.device)

        with torch.no_grad():
            log_density = self.model(batch, centerbias)

        results = []
        gs = self.grid_size
        for i in range(len(arrays)):
            grid = self._map_to_grid(log_density[i : i + 1])
            results.append({
                f"saliency_{x:02d}_{y:02d}": float(grid[y, x])
                for y in range(gs)
                for x in range(gs)
            })
        return results
