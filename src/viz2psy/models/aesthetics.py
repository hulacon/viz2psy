"""LAION Aesthetics wrapper — aesthetic quality prediction.

An MLP on top of CLIP ViT-L/14 embeddings, trained on human aesthetic
ratings (SAC + LAION-Logos + AVA). Produces a single score on an
approximate 1–10 scale.

Architecture from: github.com/christophschuhmann/improved-aesthetic-predictor
"""

import torch
import torch.nn as nn
import open_clip
from PIL import Image

from .base import BaseModel

_CLIP_MODEL = "ViT-L-14"
_CLIP_PRETRAINED = "openai"
_EMBED_DIM = 768

_WEIGHTS_URL = (
    "https://github.com/christophschuhmann/"
    "improved-aesthetic-predictor/raw/main/"
    "sac+logos+ava1-l14-linearMSE.pth"
)


class _AestheticHead(nn.Module):
    """MLP head matching the pretrained weight keys (layers.N.*)."""

    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(_EMBED_DIM, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        return self.layers(x)


class AestheticsModel(BaseModel):
    """LAION Aesthetics Predictor V2.

    Predicts aesthetic quality as a single float (~1–10 scale)
    using an MLP head on CLIP ViT-L/14 embeddings.
    """

    name = "aesthetics"

    def __init__(self, device: str | None = None):
        super().__init__(device=device)
        self._clip_model = None
        self._preprocess = None
        self._head = None

    def load(self) -> None:
        # Load CLIP backbone.
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            _CLIP_MODEL, pretrained=_CLIP_PRETRAINED,
        )
        self._clip_model = clip_model.eval().to(self.device)
        self._preprocess = preprocess

        # Load aesthetic MLP head.
        self._head = _AestheticHead()
        state_dict = torch.hub.load_state_dict_from_url(
            _WEIGHTS_URL, map_location=self.device,
        )
        self._head.load_state_dict(state_dict)
        self._head = self._head.eval().to(self.device)

        # BaseModel.to() moves self.model, so alias the clip model.
        self.model = self._clip_model

    def _embed(self, batch: torch.Tensor) -> torch.Tensor:
        """Get L2-normalized CLIP embeddings."""
        with torch.no_grad():
            features = self._clip_model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.float()

    def predict(self, image: Image.Image) -> dict[str, float]:
        x = self._preprocess(image.convert("RGB")).unsqueeze(0).to(self.device)
        emb = self._embed(x)
        with torch.no_grad():
            score = self._head(emb).item()
        return {"aesthetic_score": score}

    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, float]]:
        tensors = [self._preprocess(img.convert("RGB")) for img in images]
        batch = torch.stack(tensors).to(self.device)
        emb = self._embed(batch)
        with torch.no_grad():
            scores = self._head(emb).squeeze(1).tolist()
        return [{"aesthetic_score": s} for s in scores]
