"""Places365 + SUN attributes wrapper — scene classification and attributes.

Uses a WideResNet-18 trained on Places365 to produce 365 scene category
probabilities and 102 SUN scene attribute scores. The attribute scores
are derived by multiplying the scene probabilities by a learned
attribute weight matrix (W_sceneattribute).
"""

import io
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

from .base import BaseModel

# URLs for model weights and label files hosted by MIT CSAIL.
_WEIGHTS_URL = "http://places2.csail.mit.edu/models_places365/wideresnet18_places365.pth.tar"
_CATEGORIES_URL = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
_ATTRIBUTES_URL = "https://raw.githubusercontent.com/CSAILVision/places365/master/labels_sunattribute.txt"
_W_ATTR_URL = "http://places2.csail.mit.edu/models_places365/W_sceneattribute_wideresnet18.npy"

# WideResNet-18 architecture (from the Places365 repo).
# We define it inline to avoid cloning the full repo.

import torch.nn as nn
import torch.utils.model_zoo as model_zoo


def _conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class _BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)


class _WideResNet(nn.Module):
    def __init__(self, block, layers, num_classes=365):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # No maxpool in this architecture (disabled in the Places365 repo).
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(14)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


def _wideresnet18(num_classes=365):
    return _WideResNet(_BasicBlock, [2, 2, 2, 2], num_classes=num_classes)


class PlacesModel(BaseModel):
    """Places365 scene classification + 102 SUN attribute scores.

    Outputs 365 scene probabilities (places_*) and 102 SUN attribute
    scores (sunattr_*).
    """

    name = "places"

    def __init__(self, cache_dir: Path | None = None, device: str | None = None):
        super().__init__(device=device)
        self._cache_dir = cache_dir
        self._categories: list[str] = []
        self._attributes: list[str] = []
        self._w_attr: np.ndarray | None = None
        self._transform = T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    def _download_text(self, url: str) -> str:
        with urlopen(url) as resp:
            return resp.read().decode("utf-8")

    def _download_npy(self, url: str) -> np.ndarray:
        with urlopen(url) as resp:
            return np.load(io.BytesIO(resp.read()))

    def load(self) -> None:
        # Load model weights.
        model = _wideresnet18(num_classes=365)
        checkpoint = model_zoo.load_url(_WEIGHTS_URL, map_location="cpu")
        state_dict = {k.replace("module.", ""): v
                      for k, v in checkpoint["state_dict"].items()}
        model.load_state_dict(state_dict)
        self.model = model.eval().to(self.device)

        # Load category labels (strip /a/abbey -> abbey).
        text = self._download_text(_CATEGORIES_URL)
        self._categories = []
        for line in text.strip().splitlines():
            name = line.split(" ")[0]
            # Strip leading /x/ path prefix.
            parts = name.strip("/").split("/")
            self._categories.append(parts[-1])

        # Load SUN attribute labels.
        text = self._download_text(_ATTRIBUTES_URL)
        self._attributes = [line.strip() for line in text.strip().splitlines()]

        # Load attribute weight matrix (365 x 102).
        self._w_attr = self._download_npy(_W_ATTR_URL)

    def _forward_with_features(self, x: torch.Tensor):
        """Run forward pass, returning both logits and avgpool features."""
        m = self.model
        x = m.relu(m.bn1(m.conv1(x)))
        x = m.layer1(x)
        x = m.layer2(x)
        x = m.layer3(x)
        x = m.layer4(x)
        pooled = m.avgpool(x)
        features = pooled.view(pooled.size(0), -1)  # (batch, 512)
        logits = m.fc(features)  # (batch, 365)
        return logits, features

    def _score(self, logits: torch.Tensor, features: torch.Tensor) -> list[dict[str, float]]:
        probs = F.softmax(logits, dim=1).cpu().numpy()
        feats = features.cpu().numpy()
        # W_attr is (102, 512), features is (batch, 512) -> (batch, 102).
        attr_scores = feats @ self._w_attr.T

        results = []
        for i in range(probs.shape[0]):
            row: dict[str, float] = {}
            # Scene category probabilities.
            for j, cat in enumerate(self._categories):
                row[f"places_{cat}"] = float(probs[i, j])
            # SUN attribute scores.
            for j, attr in enumerate(self._attributes):
                key = attr.lower().replace(" ", "_").replace("/", "_")
                row[f"sunattr_{key}"] = float(attr_scores[i, j])
            results.append(row)
        return results

    def predict(self, image: Image.Image) -> dict[str, float]:
        x = self._transform(image.convert("RGB")).unsqueeze(0).to(self.device)
        with torch.no_grad():
            logits, features = self._forward_with_features(x)
        return self._score(logits, features)[0]

    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, float]]:
        tensors = [self._transform(img.convert("RGB")) for img in images]
        batch = torch.stack(tensors).to(self.device)
        with torch.no_grad():
            logits, features = self._forward_with_features(batch)
        return self._score(logits, features)
