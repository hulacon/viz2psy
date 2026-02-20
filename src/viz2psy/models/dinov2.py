"""DINOv2 wrapper — self-supervised visual embeddings.

Extracts the CLS token embedding from DINOv2 ViT-B/14 (768-d).
Unlike CLIP, DINOv2 is trained without language supervision and
captures visual structure (texture, shape, layout) rather than
semantic/conceptual content.
"""

import torch
import torchvision.transforms as T
from PIL import Image

from .base import BaseModel

_DEFAULT_MODEL_NAME = "dinov2_vitb14"
_EMBED_DIMS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class DINOv2Model(BaseModel):
    """Extract CLS token embeddings from DINOv2.

    Uses ViT-B/14 by default, producing a 768-d embedding per image.
    """

    name = "dinov2"

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME, device: str | None = None):
        super().__init__(device=device)
        self.model_name = model_name
        self._transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def load(self) -> None:
        self.model = torch.hub.load(
            "facebookresearch/dinov2", self.model_name, trust_repo=True,
        )
        self.model.eval()
        self.model = self.model.to(self.device)

    def predict(self, image: Image.Image) -> dict[str, float]:
        x = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            emb = self.model(x)
        values = emb.squeeze(0).cpu().tolist()
        return {f"dinov2_{i:03d}": v for i, v in enumerate(values)}

    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, float]]:
        tensors = [self._transform(img.convert("RGB")) for img in images]
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            embs = self.model(batch)
        results = []
        for row in embs.cpu().tolist():
            results.append({f"dinov2_{i:03d}": v for i, v in enumerate(row)})
        return results
