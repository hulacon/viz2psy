"""Face count/size/configuration statistics — OpenCV YuNet detector.

Social-visual structure beyond yolo's person count: how many faces are
on screen, how much of the frame they claim (shot scale), and how they
are arranged. Detection is OpenCV's bundled FaceDetectorYN (YuNet) — a
~230 KB ONNX checkpoint downloaded on first use to ``~/.cache/viz2psy/``
and verified by SHA-256, so no new Python dependency (deliberately not
mediapipe: its dependency tree is disproportionate to five scalars).

Per image:

- ``faces_count`` — detections at score >= 0.9 (the YuNet default)
- ``faces_total_area`` — summed face-box area as a fraction of the image
  (overlaps counted twice; a crowd of close-ups can exceed 1)
- ``faces_max_area`` — largest face as a fraction of the image (close-up
  vs long-shot scale)
- ``faces_center_dist`` — mean distance of face centers from the image
  center, normalized by the half-diagonal (0 = centered framing);
  NaN when no face
- ``faces_mutual_dist`` — mean pairwise distance between face centers,
  same normalization (small = faces clustered, the two-shot /
  conversation configuration); NaN with fewer than two faces

A frame with no faces scores count 0 and areas 0.0 — an explicit result,
not missing data; only the configuration features are NaN there.
"""

import hashlib
from pathlib import Path

import numpy as np
from PIL import Image

from viz2psy.exceptions import ModelLoadError
from viz2psy.models.base import BaseModel

_WEIGHTS_NAME = "face_detection_yunet_2023mar.onnx"
_DEFAULT_WEIGHTS_PATH = Path.home() / ".cache" / "viz2psy" / _WEIGHTS_NAME
# opencv_zoo distributes via Git LFS; this is the LFS media endpoint.
_WEIGHTS_URL = (
    "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
_WEIGHTS_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"

SCORE_THRESHOLD = 0.9  # the YuNet default

FEATURE_NAMES = [
    "faces_count",
    "faces_total_area",
    "faces_max_area",
    "faces_center_dist",
    "faces_mutual_dist",
]


class FacesModel(BaseModel):
    """Face count/size/configuration via OpenCV FaceDetectorYN (YuNet)."""

    name = "faces"
    checkpoint = "opencv_zoo/face_detection_yunet_2023mar"

    def __init__(self, weights_path: Path | None = None, device: str | None = None):
        super().__init__(device=device)  # detector is CPU; kept for interface parity
        self.weights_path = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS_PATH

    def load(self) -> None:
        import cv2

        self._ensure_weights()
        self._cv2 = cv2
        # Input size is set per image in predict(); (320, 320) is a placeholder.
        self.model = cv2.FaceDetectorYN_create(
            str(self.weights_path), "", (320, 320),
            score_threshold=SCORE_THRESHOLD,
        )

    def _ensure_weights(self) -> None:
        """Download the ONNX checkpoint if not cached; always verify SHA-256."""
        if not self.weights_path.exists():
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"Downloading YuNet weights to {self.weights_path} ...")
            import urllib.request

            urllib.request.urlretrieve(_WEIGHTS_URL, self.weights_path)
        digest = hashlib.sha256(self.weights_path.read_bytes()).hexdigest()
        if digest != _WEIGHTS_SHA256:
            raise ModelLoadError(
                self.name,
                f"checkpoint hash mismatch at {self.weights_path}: got {digest}, "
                f"expected {_WEIGHTS_SHA256}. Delete the file to re-download "
                f"(a 131-byte file is the Git-LFS pointer, not the model).",
            )

    def predict(self, image: Image.Image) -> dict[str, float]:
        rgb = np.asarray(image.convert("RGB"))
        bgr = rgb[..., ::-1].copy()
        h, w = bgr.shape[:2]
        self.model.setInputSize((w, h))
        _, faces = self.model.detect(bgr)
        boxes = np.asarray(faces)[:, :4] if faces is not None else np.empty((0, 4))
        return face_stats(boxes, w, h)


def face_stats(boxes: np.ndarray, w: int, h: int) -> dict[str, float]:
    """The five statistics from detected boxes (Nx4: x, y, w, h in pixels).

    Pure numpy, so the geometry is testable without a detector.
    """
    if len(boxes) == 0:
        return {
            "faces_count": 0.0,
            "faces_total_area": 0.0,
            "faces_max_area": 0.0,
            "faces_center_dist": float("nan"),
            "faces_mutual_dist": float("nan"),
        }

    boxes = np.asarray(boxes, dtype=float)
    areas = (boxes[:, 2] * boxes[:, 3]) / (w * h)
    centers = boxes[:, :2] + boxes[:, 2:4] / 2.0
    half_diag = np.hypot(w, h) / 2.0
    center_dist = (
        np.hypot(centers[:, 0] - w / 2.0, centers[:, 1] - h / 2.0) / half_diag
    )

    n = len(boxes)
    if n >= 2:
        diffs = centers[:, None, :] - centers[None, :, :]
        pair = np.hypot(diffs[..., 0], diffs[..., 1])
        mutual = float(pair[np.triu_indices(n, k=1)].mean() / half_diag)
    else:
        mutual = float("nan")

    return {
        "faces_count": float(n),
        "faces_total_area": float(areas.sum()),
        "faces_max_area": float(areas.max()),
        "faces_center_dist": float(center_dist.mean()),
        "faces_mutual_dist": mutual,
    }
