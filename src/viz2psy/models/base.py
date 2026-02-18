"""Abstract base class for all model wrappers."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from PIL import Image

from viz2psy.exceptions import DeviceError


def _get_default_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BaseModel(ABC):
    """Common interface for psychological feature extraction models."""

    def __init__(self, device: str | None = None):
        if device:
            try:
                self.device = torch.device(device)
            except RuntimeError as e:
                raise DeviceError(device, str(e)) from e
            if self.device.type == "cuda" and not torch.cuda.is_available():
                raise DeviceError(device, "CUDA is not available on this system")
            if self.device.type == "mps" and not torch.backends.mps.is_available():
                raise DeviceError(device, "MPS is not available on this system")
        else:
            self.device = _get_default_device()
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
