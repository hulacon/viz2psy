"""Abstract base class for all model wrappers."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch
from PIL import Image

from viz2psy.exceptions import DeviceError


#: Pixel area a model works at when its output depends on absolute pixel
#: scale (saliency, llstat): NSD's 425 x 425, so NSD stills are unchanged.
CANONICAL_AREA = 425 * 425

#: The ``preprocessing`` record of a square-input network that squashes.
SQUASH_RECORD = {
    "square": "squash to a short-side square (the whole picture; aspect not kept)",
    "interpolation": "bicubic",
}


def canonical_size(width: int, height: int, pixel_aspect: float = 1.0,
                   area: int | None = CANONICAL_AREA) -> tuple[int, int]:
    """(width, height) holding ``area`` pixels at the display aspect ratio.

    The display width is ``width * pixel_aspect`` (the stream's sample aspect
    ratio; 1.0 for square pixels). ``area=None`` keeps the stored pixel count
    and only corrects the aspect.
    """
    display_w = width * pixel_aspect
    target = width * height if area is None else area
    scale = (target / (display_w * height)) ** 0.5
    return max(1, round(display_w * scale)), max(1, round(height * scale))


def squash_to_square(image: Image.Image) -> Image.Image:
    """The whole picture resized to a short-side square (bicubic).

    The framing rule's stage 1 for square-input networks (psytwill-space
    framing D2 = squash): on a wide frame a centre crop would show the network
    only the middle of the picture, so every square-input network sees all of
    it instead, distorted rather than cut. A square input is returned as is.
    """
    s = min(image.size)
    return image if image.size == (s, s) else image.resize((s, s), Image.BICUBIC)


def _get_default_device() -> torch.device:
    """Auto-detect the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class BaseModel(ABC):
    """Common interface for psychological feature extraction models.

    Contract (constellation Contract B, §4.1): every feature column a model
    emits starts with ``{name}_`` (or a prefix listed in ``extra_prefixes``),
    and ``checkpoint`` identifies the exact architecture+weights for any
    model with learned parameters (None for analytic models).
    """

    checkpoint: str | None = None
    extra_prefixes: tuple[str, ...] = ()
    # Contract B §4.1 (schema 1.1): every column this model can set to NaN,
    # keyed by exact column name, with what the null means (``undefined`` /
    # ``undefinable`` / ``missing``) and when. ``{}`` is a positive claim that
    # no column can be null. Stamped into the sidecar by metadata.add_model.
    nulls: dict[str, dict[str, str]] = {}
    # Contract B §4.1 (schema 1.2): what each column is and how its value is
    # computed, in the sidecar's own shape: {"default": {...}, "columns":
    # {<exact column>: {...}}}, merged key-wise per column (viz2psy.labels).
    # Every column must resolve to `interpretability` + `derivation`; the
    # empty default here resolves nothing, so an undeclared model fails
    # tests/test_labels.py. Stamped into the sidecar by metadata.add_model.
    labels: dict[str, dict] = {}

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
