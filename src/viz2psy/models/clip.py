"""CLIP wrapper — semantic image embeddings via OpenCLIP."""

import torch
import open_clip
from PIL import Image

from .base import BaseModel

_DEFAULT_MODEL_NAME = "ViT-B-32"
_DEFAULT_PRETRAINED = "laion2b_s34b_b79k"


class CLIPModel(BaseModel):
    """Extract L2-normalized CLIP image embeddings.

    Uses OpenCLIP's ViT-B-32 to produce a 512-d embedding per image.
    """

    name = "clip"

    def __init__(
        self,
        model_name: str = _DEFAULT_MODEL_NAME,
        pretrained: str = _DEFAULT_PRETRAINED,
    ):
        super().__init__()
        self.model_name = model_name
        self.pretrained = pretrained
        self._preprocess = None

    def load(self) -> None:
        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained,
        )
        self.model = model.eval().to(self.device)
        self._preprocess = preprocess

    def predict(self, image: Image.Image) -> dict[str, float]:
        x = self._preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(x)
            features = features / features.norm(dim=-1, keepdim=True)
        emb = features.squeeze(0).cpu().tolist()
        return {f"clip_{i:03d}": v for i, v in enumerate(emb)}

    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, float]]:
        tensors = [self._preprocess(img) for img in images]
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            features = self.model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        results = []
        for row in features.cpu().tolist():
            results.append({f"clip_{i:03d}": v for i, v in enumerate(row)})
        return results
