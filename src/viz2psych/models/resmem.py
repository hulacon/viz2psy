"""ResMem wrapper — image memorability prediction."""

import torch
from PIL import Image
from resmem import ResMem, transformer

from .base import BaseModel


class ResMemModel(BaseModel):
    """Thin wrapper around the ResMem memorability model.

    Predicts a single memorability score (0–1) per image.
    """

    name = "resmem"

    def load(self) -> None:
        self.model = ResMem(pretrained=True)
        self.model.eval()
        self.model = self.model.to(self.device)

    def predict(self, image: Image.Image) -> dict[str, float]:
        x = transformer(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = self.model(x).item()
        return {"memorability": score}

    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, float]]:
        tensors = [transformer(img) for img in images]
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            scores = self.model(batch).squeeze(1).tolist()
        return [{"memorability": s} for s in scores]
