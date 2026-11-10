"""GIST wrapper — spatial envelope descriptor via Gabor filter banks.

Implements the GIST descriptor from Oliva & Torralba (2001). Applies a bank
of Gabor filters at multiple scales and orientations, then pools mean
magnitude over a spatial grid to produce a fixed-length descriptor.

Default parameters: 4 scales x 8 orientations x 4x4 grid = 512 dimensions.
"""

import numpy as np
from PIL import Image
from scipy import fft as sfft
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


def _fft_bank(kernels: list[np.ndarray], shape: tuple[int, int]) -> dict:
    """Precompute the kernels' spectra for images of one ``shape``.

    The image is padded by ``r`` (the largest kernel half-width) the way
    ``scipy.ndimage``'s ``reflect`` mode extends it (numpy ``symmetric``), and
    the FFT is sized for a full linear convolution of the padded image, so no
    circular wrap reaches the cropped output.
    """
    h, w = shape
    r = max(max(k.shape) for k in kernels) // 2
    size = (sfft.next_fast_len(h + 4 * r, real=True), sfft.next_fast_len(w + 4 * r, real=True))
    return {
        "shape": (h, w),
        "pad": r,
        "size": size,
        "spectra": np.stack([sfft.rfft2(k, s=size) for k in kernels]),
        "centres": [(k.shape[0] // 2, k.shape[1] // 2) for k in kernels],
    }


def _compute_gist(
    gray: np.ndarray,
    kernels: list[np.ndarray],
    grid_size: int,
    bank: dict | None = None,
) -> np.ndarray:
    """Compute GIST descriptor for a grayscale image.

    For each Gabor filter, convolve with the image (``reflect`` boundary),
    then compute mean magnitude in each spatial grid cell.

    The convolution runs in the Fourier domain, all filters at once. It equals
    ``scipy.ndimage.convolve(gray, kernel, mode="reflect")`` to floating-point
    rounding (relative error ~1e-16) and is ~60x faster at 256 px, where the
    largest kernel is 69 x 69 (viz2psy <= 0.19.0 convolved directly).
    """
    h, w = gray.shape
    if bank is None or bank["shape"] != (h, w):
        bank = _fft_bank(kernels, (h, w))
    r, size = bank["pad"], bank["size"]
    padded = np.pad(gray, r, mode="symmetric")
    full = sfft.irfft2(sfft.rfft2(padded, s=size)[None] * bank["spectra"], s=size, axes=(-2, -1))

    cell_h = h // grid_size
    cell_w = w // grid_size
    descriptor = []
    for i, (cy, cx) in enumerate(bank["centres"]):
        magnitude = np.abs(full[i, r + cy:r + cy + h, r + cx:r + cx + w])
        for row in range(grid_size):
            for col in range(grid_size):
                r0 = row * cell_h
                r1 = (row + 1) * cell_h if row < grid_size - 1 else h
                c0 = col * cell_w
                c1 = (col + 1) * cell_w if col < grid_size - 1 else w
                descriptor.append(magnitude[r0:r1, c0:c1].mean())

    return np.array(descriptor, dtype=np.float64)


def _column_descriptions(
    n_orientations: int = _DEFAULT_N_ORIENTATIONS,
    frequencies: tuple[float, ...] = _DEFAULT_FREQUENCIES,
    grid_size: int = _DEFAULT_GRID_SIZE,
) -> dict[str, str]:
    """What each ``gist_NNN`` column measures, in emission order.

    Mirrors :func:`_build_gabor_bank` (frequency-major, then orientation) and
    :func:`_compute_gist` (grid row-major within each filter).
    """
    out = {}
    i = 0
    for freq in frequencies:
        for o in range(n_orientations):
            theta = o * 180.0 / n_orientations
            for row in range(grid_size):
                for col in range(grid_size):
                    out[f"gist_{i:03d}"] = (
                        f"Mean Gabor magnitude, frequency {freq:g} cycles/pixel (256 px "
                        f"image), orientation theta {theta:g} deg, grid cell row {row} "
                        f"col {col} of {grid_size} x {grid_size}"
                    )
                    i += 1
    return out


class GISTModel(BaseModel):
    """Spatial envelope descriptor (Oliva & Torralba 2001).

    Computes a 512-d GIST descriptor using Gabor filter banks:
    4 scales x 8 orientations x (4x4 grid) = 512 dimensions.
    """

    name = "gist"
    labels = {
        "default": {
            "interpretability": "nameable",
            "derivation": "formula",
            "Description": "GIST spatial-envelope descriptor (Oliva & Torralba 2001): mean "
                           "Gabor magnitude at one scale and orientation in one grid cell",
        },
        "columns": {c: {"Description": d} for c, d in _column_descriptions().items()},
    }

    def __init__(
        self,
        n_orientations: int = _DEFAULT_N_ORIENTATIONS,
        n_scales: int = _DEFAULT_N_SCALES,
        grid_size: int = _DEFAULT_GRID_SIZE,
        frequencies: tuple[float, ...] | None = None,
        image_size: int = 256,
        device: str | None = None,
    ):
        super().__init__(device=device)
        self.n_orientations = n_orientations
        self.n_scales = n_scales
        self.grid_size = grid_size
        self.frequencies = frequencies or _DEFAULT_FREQUENCIES[:n_scales]
        self.image_size = image_size
        self._kernels: list[np.ndarray] = []
        self._bank: dict | None = None

    @staticmethod
    def describe_preprocessing() -> dict:
        """The input transform, as recorded in the sidecar."""
        return {'square': 'squash to 256 x 256 (aspect not kept), then grayscale; 4 x 4 grid over the whole picture', 'interpolation': 'lanczos'}

    def load(self) -> None:
        """Pre-compute the Gabor filter bank (no learned weights)."""
        self._kernels = _build_gabor_bank(self.n_orientations, self.frequencies)
        self._bank = _fft_bank(self._kernels, (self.image_size, self.image_size))
        self.model = None

    def predict(self, image: Image.Image) -> dict[str, float]:
        img = image.resize(
            (self.image_size, self.image_size), Image.Resampling.LANCZOS
        )
        gray = np.array(img.convert("L"), dtype=np.float64) / 255.0
        desc = _compute_gist(gray, self._kernels, self.grid_size, self._bank)
        return {f"gist_{i:03d}": float(v) for i, v in enumerate(desc)}
