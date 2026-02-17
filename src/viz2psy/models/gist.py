"""GIST wrapper — spatial envelope descriptor via Gabor filter banks.

Implements the GIST descriptor from Oliva & Torralba (2001). Applies a bank
of Gabor filters at multiple scales and orientations, then pools mean
magnitude over a spatial grid to produce a fixed-length descriptor.

Default parameters: 4 scales x 8 orientations x 4x4 grid = 512 dimensions.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import convolve
from skimage.filters import gabor_kernel

from .base import BaseModel

_DEFAULT_N_ORIENTATIONS = 8
_DEFAULT_N_SCALES = 4
_DEFAULT_GRID_SIZE = 4
_DEFAULT_FREQUENCIES = (0.05, 0.1, 0.2, 0.4)


def _build_gabor_bank(
    n_orientations: int,
    frequencies: tuple[float, ...],
) -> list[np.ndarray]:
    """Build a bank of Gabor filter kernels (real part)."""
    kernels = []
    for freq in frequencies:
        for i in range(n_orientations):
            theta = i * np.pi / n_orientations
            kernel = gabor_kernel(freq, theta=theta)
            kernels.append(np.real(kernel))
    return kernels


def _compute_gist(
    gray: np.ndarray,
    kernels: list[np.ndarray],
    grid_size: int,
) -> np.ndarray:
    """Compute GIST descriptor for a grayscale image.

    For each Gabor filter, convolve with the image, then compute mean
    magnitude in each spatial grid cell.
    """
    h, w = gray.shape
    cell_h = h // grid_size
    cell_w = w // grid_size
    descriptor = []

    for kernel in kernels:
        filtered = convolve(gray, kernel, mode="reflect")
        magnitude = np.abs(filtered)
        for row in range(grid_size):
            for col in range(grid_size):
                r0 = row * cell_h
                r1 = (row + 1) * cell_h if row < grid_size - 1 else h
                c0 = col * cell_w
                c1 = (col + 1) * cell_w if col < grid_size - 1 else w
                descriptor.append(magnitude[r0:r1, c0:c1].mean())

    return np.array(descriptor, dtype=np.float64)


class GISTModel(BaseModel):
    """Spatial envelope descriptor (Oliva & Torralba 2001).

    Computes a 512-d GIST descriptor using Gabor filter banks:
    4 scales x 8 orientations x (4x4 grid) = 512 dimensions.
    """

    name = "gist"

    def __init__(
        self,
        n_orientations: int = _DEFAULT_N_ORIENTATIONS,
        n_scales: int = _DEFAULT_N_SCALES,
        grid_size: int = _DEFAULT_GRID_SIZE,
        frequencies: tuple[float, ...] | None = None,
        image_size: int = 256,
    ):
        super().__init__()
        self.n_orientations = n_orientations
        self.n_scales = n_scales
        self.grid_size = grid_size
        self.frequencies = frequencies or _DEFAULT_FREQUENCIES[:n_scales]
        self.image_size = image_size
        self._kernels: list[np.ndarray] = []

    def load(self) -> None:
        """Pre-compute the Gabor filter bank (no learned weights)."""
        self._kernels = _build_gabor_bank(self.n_orientations, self.frequencies)
        self.model = None

    def predict(self, image: Image.Image) -> dict[str, float]:
        img = image.resize(
            (self.image_size, self.image_size), Image.Resampling.LANCZOS
        )
        gray = np.array(img.convert("L"), dtype=np.float64) / 255.0
        desc = _compute_gist(gray, self._kernels, self.grid_size)
        return {f"gist_{i:03d}": float(v) for i, v in enumerate(desc)}
