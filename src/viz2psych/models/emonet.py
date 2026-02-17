"""EmoNet wrapper — emotion classification from images."""

from pathlib import Path

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from .base import BaseModel

# 20 emotion categories from Cowen & Keltner (2017), in model output order.
EMOTION_CATEGORIES = [
    "Adoration",
    "Aesthetic Appreciation",
    "Amusement",
    "Anxiety",
    "Awe",
    "Boredom",
    "Confusion",
    "Craving",
    "Disgust",
    "Empathic Pain",
    "Entrancement",
    "Excitement",
    "Fear",
    "Horror",
    "Interest",
    "Joy",
    "Romance",
    "Sadness",
    "Sexual Desire",
    "Surprise",
]

# EmoNet expects 227x227, pixel values in 0-255 range (no normalization).
# PILToTensor keeps uint8 [0,255]; the model's forward() casts to float.
_emonet_transform = T.Compose([
    T.Resize((227, 227)),
    T.PILToTensor(),  # keeps 0-255 range (uint8)
])

# Default cache location for downloaded weights.
_DEFAULT_WEIGHTS_PATH = Path.home() / ".cache" / "viz2psych" / "emonet_pytorch_weights.pt"

# OSF download URL for EmoNet weights.
_WEIGHTS_URL = "https://osf.io/amdju/download"


class _EmoNetArch(nn.Module):
    """EmoNet architecture (custom AlexNet converted from MATLAB via ONNX).

    Reproduced from https://github.com/ecco-laboratory/emonet-pytorch/blob/main/models.py
    so we don't need to clone the repo at runtime.
    """

    def __init__(self, num_classes: int = 20) -> None:
        super().__init__()
        alpha = 9.999999747378752e-05
        self.conv_0 = nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=alpha, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv_1 = nn.Sequential(
            nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2, groups=2),
            nn.ReLU(),
            nn.LocalResponseNorm(size=5, alpha=alpha, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        self.conv_3 = nn.Sequential(
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU(),
        )
        self.conv_4 = nn.Sequential(
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1, groups=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.conv_5 = nn.Sequential(
            nn.Conv2d(256, 4096, kernel_size=6, stride=1),
            nn.ReLU(),
        )
        self.conv_6 = nn.Sequential(
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.Conv2d(4096, num_classes, kernel_size=1, stride=1),
            nn.Flatten(start_dim=-3, end_dim=-1),
            nn.Softmax(dim=-1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(torch.float)
        x = self.conv_0(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)
        return self.classifier(x)


class EmoNetModel(BaseModel):
    """Wrapper around EmoNet for emotion probability prediction.

    Outputs a probability for each of 20 Cowen & Keltner emotion categories.
    """

    name = "emonet"

    def __init__(self, weights_path: Path | str | None = None):
        super().__init__()
        self.weights_path = Path(weights_path) if weights_path else _DEFAULT_WEIGHTS_PATH

    def load(self) -> None:
        self.model = _EmoNetArch(num_classes=20)
        self._ensure_weights()
        state_dict = torch.load(self.weights_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model = self.model.to(self.device)

    def _ensure_weights(self) -> None:
        """Download weights from OSF if not already cached."""
        if self.weights_path.exists():
            return
        self.weights_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading EmoNet weights to {self.weights_path} ...")
        torch.hub.download_url_to_file(_WEIGHTS_URL, str(self.weights_path))

    def predict(self, image: Image.Image) -> dict[str, float]:
        x = _emonet_transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probs = self.model(x).squeeze(0).tolist()
        return dict(zip(EMOTION_CATEGORIES, probs))

    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, float]]:
        tensors = [_emonet_transform(img) for img in images]
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            probs = self.model(batch)  # (N, 20)
        return [
            dict(zip(EMOTION_CATEGORIES, row.tolist()))
            for row in probs
        ]
