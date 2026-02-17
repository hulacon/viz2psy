"""LLStat wrapper — low-level image statistics."""

import numpy as np
from PIL import Image
from scipy.fft import fft2, fftshift
from skimage.color import rgb2lab
from skimage.feature import canny

from .base import BaseModel


def _luminance(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB (0-1 float) to relative luminance (ITU-R BT.601)."""
    return 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]


def _spectral_energy_ratio(gray: np.ndarray) -> tuple[float, float]:
    """Compute high-freq and low-freq energy fractions via 2D FFT.

    Splits at the median radial frequency.
    """
    f_transform = fftshift(fft2(gray))
    power = np.abs(f_transform) ** 2
    h, w = gray.shape
    cy, cx = h // 2, w // 2
    Y, X = np.ogrid[:h, :w]
    radius = np.sqrt((Y - cy) ** 2 + (X - cx) ** 2)
    median_r = np.median(radius)
    total = power.sum()
    if total == 0:
        return 0.5, 0.5
    lf = power[radius <= median_r].sum() / total
    hf = 1.0 - lf
    return float(hf), float(lf)


def _edge_density(gray: np.ndarray) -> float:
    """Fraction of pixels classified as edges by Canny detector."""
    edges = canny(gray, sigma=1.0)
    return float(edges.mean())


def _colorfulness(rgb: np.ndarray) -> float:
    """Hasler & Suesstrunk (2003) colorfulness metric."""
    R, G, B = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    rg = R - G
    yb = 0.5 * (R + G) - B
    sigma = np.sqrt(rg.std() ** 2 + yb.std() ** 2)
    mu = np.sqrt(rg.mean() ** 2 + yb.mean() ** 2)
    return float(sigma + 0.3 * mu)


class LLStatModel(BaseModel):
    """Low-level image statistics: luminance, contrast, color, frequency, edges."""

    name = "llstat"

    def load(self) -> None:
        """No model to load — all features are computed analytically."""
        self.model = None

    def predict(self, image: Image.Image) -> dict[str, float]:
        rgb = np.array(image.convert("RGB"), dtype=np.float64) / 255.0

        # Luminance (BT.601).
        lum = _luminance(rgb)
        lum_mean = float(lum.mean())
        lum_std = float(lum.std())
        rms_contrast = lum_std / lum_mean if lum_mean > 0 else 0.0

        # Per-channel RGB stats.
        r_mean, r_std = float(rgb[:, :, 0].mean()), float(rgb[:, :, 0].std())
        g_mean, g_std = float(rgb[:, :, 1].mean()), float(rgb[:, :, 1].std())
        b_mean, b_std = float(rgb[:, :, 2].mean()), float(rgb[:, :, 2].std())

        # CIE LAB (perceptually uniform).
        lab = rgb2lab(rgb)
        lab_l_mean = float(lab[:, :, 0].mean())
        lab_a_mean = float(lab[:, :, 1].mean())
        lab_b_mean = float(lab[:, :, 2].mean())

        # Saturation (HSV space via PIL).
        hsv = np.array(image.convert("HSV"), dtype=np.float64) / 255.0
        saturation_mean = float(hsv[:, :, 1].mean())

        # Spectral energy (FFT on luminance).
        hf_energy, lf_energy = _spectral_energy_ratio(lum)

        # Edge density (Canny on luminance).
        edge_dens = _edge_density(lum)

        # Colorfulness (Hasler & Suesstrunk 2003).
        color_metric = _colorfulness(rgb)

        return {
            "luminance_mean": lum_mean,
            "luminance_std": lum_std,
            "rms_contrast": rms_contrast,
            "r_mean": r_mean,
            "r_std": r_std,
            "g_mean": g_mean,
            "g_std": g_std,
            "b_mean": b_mean,
            "b_std": b_std,
            "lab_l_mean": lab_l_mean,
            "lab_a_mean": lab_a_mean,
            "lab_b_mean": lab_b_mean,
            "saturation_mean": saturation_mean,
            "hf_energy": hf_energy,
            "lf_energy": lf_energy,
            "edge_density": edge_dens,
            "colorfulness": color_metric,
        }
