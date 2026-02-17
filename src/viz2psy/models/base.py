"""Abstract base class for all model wrappers."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from PIL import Image


class BaseModel(ABC):
    """Common interface for psychological feature extraction models."""

    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None

    @abstractmethod
    def load(self) -> None:
        """Load model weights and set to eval mode."""

    @abstractmethod
    def predict(self, image: Image.Image) -> dict[str, float]:
        """Return named scores for a single PIL image."""

    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, float]]:
        """Return named scores for a list of PIL images.

        Default implementation loops over predict(); subclasses can override
        with a more efficient batched version.
        """
        return [self.predict(img) for img in images]

    def to(self, device: torch.device) -> "BaseModel":
        """Move the underlying torch model to the given device."""
        self.device = device
        if self.model is not None:
            self.model = self.model.to(device)
        return self
