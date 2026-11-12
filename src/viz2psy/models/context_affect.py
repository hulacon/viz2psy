"""Context-based affect — a valence / arousal probe on a frozen CLIP ViT-L/14.

A linear (ridge) map from the L2-normalised CLIP ViT-L/14 (OpenAI) image
embedding to the per-frame consensus ratings of the VEATIC dataset (Ren et
al., WACV 2024): observers continuously rated the valence and arousal of a
cued on-screen character in silent film and home-video clips, inferring it
from context plus character. Applied to uncued footage, the probe reads the
salient character or the scene context. It is not the viewer's felt emotion.

Weights are not shipped with the package. VEATIC's ratings are released for
research use, so the probe is rebuilt from the public data with
``scripts/train_context_affect.py`` and read from :func:`weights_path`
(``$VIZ2PSY_CONTEXT_AFFECT_WEIGHTS``, else ``<torch hub dir>/viz2psy/``).
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import SQUASH_RECORD, BaseModel, squash_to_square

_CLIP_MODEL = "ViT-L-14"
_CLIP_PRETRAINED = "openai"
BACKBONE = f"{_CLIP_MODEL}/{_CLIP_PRETRAINED}"
EMBED_DIM = 768

#: The emitted columns, in the order of the weights' output dimension.
FEATURE_NAMES = ("context_affect_valence", "context_affect_arousal")

WEIGHTS_ENV = "VIZ2PSY_CONTEXT_AFFECT_WEIGHTS"
WEIGHTS_NAME = "context_affect_v1.npz"
WEIGHTS_FORMAT = "1"


def weights_path() -> Path:
    """The weights file: ``$VIZ2PSY_CONTEXT_AFFECT_WEIGHTS``, else the shared torch hub cache."""
    env = os.environ.get(WEIGHTS_ENV)
    if env:
        return Path(env)
    return Path(torch.hub.get_dir()) / "viz2psy" / WEIGHTS_NAME


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_weights(path: Path) -> dict:
    """Load and validate a weights file written by ``scripts/train_context_affect.py``.

    Raises ``FileNotFoundError`` naming the path and how to build it: a model
    without its weights must fail loudly, never emit a silent default.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"context_affect weights not found at {path}. Build them with "
            f"scripts/train_context_affect.py (VEATIC ratings, research use) and "
            f"point ${WEIGHTS_ENV} at the file, or place it at "
            f"<torch hub dir>/viz2psy/{WEIGHTS_NAME}."
        )
    with np.load(path, allow_pickle=False) as z:
        w = {k: z[k] for k in z.files}
    fmt = str(w.get("format_version", ""))
    if fmt != WEIGHTS_FORMAT:
        raise ValueError(f"{path}: weights format {fmt!r}, expected {WEIGHTS_FORMAT!r}")
    if str(w["backbone"]) != BACKBONE:
        raise ValueError(f"{path}: trained on {w['backbone']!s}, this model loads {BACKBONE}")
    if tuple(str(c) for c in w["columns"]) != FEATURE_NAMES:
        raise ValueError(f"{path}: columns {list(w['columns'])} != {list(FEATURE_NAMES)}")
    if w["W"].shape != (EMBED_DIM, len(FEATURE_NAMES)) or w["b"].shape != (len(FEATURE_NAMES),):
        raise ValueError(f"{path}: W {w['W'].shape} / b {w['b'].shape} do not fit "
                         f"{EMBED_DIM} -> {len(FEATURE_NAMES)}")
    return w


def apply_head(embeddings: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """``(n, 768) -> (n, 2)``: the folded ridge map (standardisation is inside W and b)."""
    return np.asarray(embeddings, dtype=np.float64) @ W + b


class ContextAffectModel(BaseModel):
    """Visually inferred valence / arousal (VEATIC probe on CLIP ViT-L/14)."""

    name = "context_affect"
    checkpoint = f"{BACKBONE}+context_affect_v1"
    _DESCRIPTION = (
        "Visually inferred {dim} of an on-screen character, as observers judge it from "
        "context plus character: a ridge probe on frozen CLIP ViT-L-14 (OpenAI) image "
        "embeddings, trained on VEATIC's per-frame consensus ratings (silent video, cued "
        "character; inferential emotion tracking). On uncued footage it reads the salient "
        "character or the scene context; it is not the viewer's felt emotion. Ratings were "
        "shifted for the raters' reaction lag before training (delay in the sidecar's "
        "preprocessing record). Linear output on VEATIC's [-1, 1] scale, not clipped."
    )
    labels = {
        "default": {
            "interpretability": "nameable",
            "derivation": "probe",
            "derived_from": BACKBONE,
        },
        "columns": {
            "context_affect_valence": {
                "LongName": "Context-inferred valence",
                "Description": _DESCRIPTION.format(dim="valence"),
            },
            "context_affect_arousal": {
                "LongName": "Context-inferred arousal",
                "Description": _DESCRIPTION.format(dim="arousal"),
            },
        },
    }

    def __init__(self, device: str | None = None):
        super().__init__(device=device)
        self._preprocess = None
        self._W = None
        self._b = None

    @staticmethod
    def describe_preprocessing() -> dict:
        """The input transform and the probe it feeds, as recorded in the sidecar."""
        rec = {**SQUASH_RECORD,
               "then": "the model's own transform: resize to its input size, then centre "
                       "crop (removes nothing on a square); L2-normalised CLIP ViT-L/14 "
                       "embedding into the context_affect ridge head"}
        path = weights_path()
        if path.is_file():
            w = read_weights(path)
            rec["probe"] = {
                "weights_file": path.name,
                "weights_sha256": sha256(path),
                "target_lag_sec": dict(zip(FEATURE_NAMES, map(float, w["lag_sec"]))),
                "alpha": dict(zip(FEATURE_NAMES, map(float, w["alpha"]))),
                "trained_on": json.loads(str(w["trained_on"])),
            }
        else:
            rec["probe"] = {"weights_file": None, "note": f"no weights at {path}"}
        return rec

    def load_backbone(self) -> None:
        """Load only the CLIP backbone (training embeds with exactly this path)."""
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            _CLIP_MODEL, pretrained=_CLIP_PRETRAINED,
        )
        self.model = model.eval().to(self.device)
        self._preprocess = preprocess

    def load(self) -> None:
        w = read_weights(weights_path())  # before the backbone: fail fast, and loudly
        self._W, self._b = w["W"], w["b"]
        self.load_backbone()

    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        """``(n, 768)`` L2-normalised embeddings of squashed RGB images."""
        tensors = [self._preprocess(squash_to_square(img.convert("RGB"))) for img in images]
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            f = self.model.encode_image(batch)
            f = f / f.norm(dim=-1, keepdim=True)
        return f.float().cpu().numpy()

    def predict(self, image: Image.Image) -> dict[str, float]:
        return self.predict_batch([image])[0]

    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, float]]:
        out = apply_head(self.embed_batch(images), self._W, self._b)
        return [dict(zip(FEATURE_NAMES, map(float, row))) for row in out]
