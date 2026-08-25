"""Monocular-depth scene-layout statistics — Depth Anything V2 (small).

Continuous depth/spatial-layout structure, complementing gist's spatial
envelope and places' coarse SUN layout attributes (open/enclosed/horizon/
cluttered). The backbone predicts *relative* depth (disparity convention:
larger = nearer), which is normalized per image — so cross-stimulus
"mean depth" would be meaningless, and every statistic here is designed
to be invariant to that per-image normalization:

- ``depth_fg_fraction`` — fraction of pixels in the near field
  (normalized depth > ``NEAR_THRESHOLD``)
- ``depth_openness`` — fraction in the far field (< ``FAR_THRESHOLD``):
  open vistas score high, closed interiors low
- ``depth_iqr`` — interquartile range of the normalized depth
  distribution (foreground/background separation vs flat relief)
- ``depth_skew`` — skewness of that distribution (near-dominated
  close-ups negative, far-dominated vistas positive)
- ``depth_edge_density`` — mean gradient magnitude of the normalized
  depth map (depth discontinuities; spatial clutter)
- ``depth_center_near`` — mean normalized depth in the center third
  minus the periphery (positive = near content centered, the standard
  portrait/object composition)

Normalization uses the 2nd/98th percentiles (robust to speckle
outliers), then clips to [0, 1]; 1 = nearest. Depth models are trained
on real scenes — treat animated/stylized clips as out-of-domain (the
faces lesson), a covariate rather than ground truth. Checkpoint is
Apache-2.0 (the Small variant specifically; the larger V2 variants are
CC-BY-NC and deliberately not used).
"""

import numpy as np
from PIL import Image

from viz2psy.models.base import BaseModel

CHECKPOINT = "depth-anything/Depth-Anything-V2-Small-hf"
NEAR_THRESHOLD = 0.66
FAR_THRESHOLD = 0.33
ROBUST_PCT = (2.0, 98.0)

FEATURE_NAMES = [
    "depth_fg_fraction",
    "depth_openness",
    "depth_iqr",
    "depth_skew",
    "depth_edge_density",
    "depth_center_near",
]


class DepthModel(BaseModel):
    """Scene-layout statistics from Depth Anything V2 relative depth."""

    name = "depth"
    checkpoint = CHECKPOINT

    def load(self) -> None:
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(CHECKPOINT)
        self.model = (
            AutoModelForDepthEstimation.from_pretrained(CHECKPOINT)
            .eval()
            .to(self.device)
        )

    def predict(self, image: Image.Image) -> dict[str, float]:
        return self.predict_batch([image])[0]

    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, float]]:
        import torch

        rgb = [img.convert("RGB") for img in images]
        inputs = self.processor(images=rgb, return_tensors="pt").to(self.device)
        with torch.no_grad():
            depth = self.model(**inputs).predicted_depth  # (N, h, w), larger = nearer
        return [depth_stats(d.float().cpu().numpy()) for d in depth]


def depth_stats(dmap: np.ndarray) -> dict[str, float]:
    """The six layout statistics from one relative-depth map.

    Pure numpy, so the geometry is testable without the backbone.
    A constant map (no depth structure the normalization can resolve)
    scores 0 fractions/spread and 0 composition terms.
    """
    dmap = np.asarray(dmap, dtype=np.float64)
    lo, hi = np.percentile(dmap, ROBUST_PCT)
    if hi - lo <= 0:
        return {
            "depth_fg_fraction": 0.0,
            "depth_openness": 0.0,
            "depth_iqr": 0.0,
            "depth_skew": 0.0,
            "depth_edge_density": 0.0,
            "depth_center_near": 0.0,
        }
    d = np.clip((dmap - lo) / (hi - lo), 0.0, 1.0)  # 1 = nearest

    q25, q75 = np.percentile(d, (25, 75))
    mean = d.mean()
    std = d.std()
    skew = float(((d - mean) ** 3).mean() / std**3) if std > 0 else 0.0

    gy, gx = np.gradient(d)
    edge_density = float(np.hypot(gx, gy).mean())

    h, w = d.shape
    center = d[h // 3 : h - h // 3, w // 3 : w - w // 3]
    periphery_sum = d.sum() - center.sum()
    periphery_n = d.size - center.size
    center_near = float(center.mean() - periphery_sum / periphery_n) if periphery_n else 0.0

    return {
        "depth_fg_fraction": float((d > NEAR_THRESHOLD).mean()),
        "depth_openness": float((d < FAR_THRESHOLD).mean()),
        "depth_iqr": float(q75 - q25),
        "depth_skew": skew,
        "depth_edge_density": edge_density,
        "depth_center_near": center_near,
    }
