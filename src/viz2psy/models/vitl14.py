"""CLIP ViT-L/14 (OpenAI) image embeddings: the backbone the probe heads read.

The L2-normalised 768-d embedding that ``context_affect`` and ``experiential``
feed their ridge heads, through exactly their squash-and-embed path. Emitting it
once lets any head on this backbone be applied, or fitted, on CPU from the
stored table (``table @ W + b`` reproduces the head's columns), instead of
re-running the backbone per head.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

from .base import SQUASH_RECORD
from .context_affect import BACKBONE, EMBED_DIM, ContextAffectModel

FEATURE_NAMES = tuple(f"vitl14_{i:03d}" for i in range(EMBED_DIM))


class ViTL14Model(ContextAffectModel):
    """768-d L2-normalised CLIP ViT-L/14 (OpenAI) image embeddings."""

    name = "vitl14"
    checkpoint = BACKBONE
    labels = {
        "default": {
            "interpretability": "opaque",
            "derivation": "embedding",
            "Description": ("One dimension of the L2-normalised CLIP ViT-L-14 (OpenAI) image "
                            "embedding, on the squashed image (the input context_affect and "
                            "experiential read)"),
        },
        "columns": {},
    }

    @staticmethod
    def describe_preprocessing() -> dict:
        return {**SQUASH_RECORD,
                "then": "the model's own transform: resize to its input size, then centre "
                        "crop (removes nothing on a square); L2-normalised CLIP ViT-L/14 "
                        "embedding (no head)"}

    def load(self) -> None:
        self.load_backbone()

    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, float]]:
        out = np.asarray(self.embed_batch(images), dtype=np.float64)
        return [dict(zip(FEATURE_NAMES, map(float, row))) for row in out]
