"""DeepGaze IIE wrapper — visual saliency prediction.

Produces a 24x24 spatial saliency grid per image by pooling the full
log-density saliency map from DeepGaze IIE into coarse grid cells.
Output keys use x_y coordinates: saliency_00_00 (top-left) through
saliency_23_23 (bottom-right), where x is the column and y is the row.
"""

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.special import logsumexp

from .base import BaseModel

_DEFAULT_GRID_SIZE = 24


class SaliencyModel(BaseModel):
    """DeepGaze IIE saliency model with 24x24 spatial grid output.

    Predicts where humans are likely to fixate in an image, pooled
    into a coarse spatial grid (576 values by default).
    """

    name = "saliency"

    def __init__(self, grid_size: int = _DEFAULT_GRID_SIZE):
        super().__init__()
        self.grid_size = grid_size
        self._centerbias: np.ndarray | None = None

    def load(self) -> None:
        import deepgaze_pytorch

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
        img = np.array(image.convert("RGB"))
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
        # Our shared1000 images are all 425x425, so this should work.
        arrays = [np.array(img.convert("RGB")) for img in images]
        h, w = arrays[0].shape[:2]

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
