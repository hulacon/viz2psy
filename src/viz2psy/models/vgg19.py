"""VGG19 wrapper — layerwise ImageNet CNN features.

A generic hierarchical-CNN control: per-layer activations of torchvision's
ImageNet-trained VGG19 (``IMAGENET1K_V1``), each layer reduced to one value
per channel by a spatial mean. Two layer sets are emitted, matching the two
published uses this model exists to reproduce:

* **All 19 layers, pre-ReLU, 224 px** (conv1_1 ... conv5_4, fc6, fc7, fc8):
  the layer set of Kamitani-lab feature decoding, which uses the outputs of
  every VGG19 layer "before rectification" on images rescaled to 224x224.
  Columns ``vgg19_<layer>_<NNNN>``.
* **Four block outputs, post-ReLU, 112 px** (conv1_2, conv2_2, conv3_3,
  conv4_3): the branches of the Wasserman et al. image-to-fMRI encoder, which
  reads VGG block 1-4 embeddings at 112x112. Columns
  ``vgg19_<layer>_relu112_<NNNN>``.

Inputs are resized to a square **without cropping** (the whole frame, aspect
ratio not preserved) and normalized with the ImageNet mean/std torchvision's
weights expect. The spatial mean is this wrapper's reduction: both source
methods consume full spatial maps, which cannot be stored per frame.

Column count: 5,504 conv + 4,096 + 4,096 + 1,000 fc = 14,696 (224 px set)
plus 64 + 128 + 256 + 512 = 960 (112 px set) = 15,656.
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image

from .base import BaseModel

_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

# VGG19 conv layers in order, named conv<block>_<index>.
_BLOCK_DEPTHS = (2, 2, 4, 4, 4)
CONV_LAYERS = [f"conv{b}_{i}" for b, n in enumerate(_BLOCK_DEPTHS, start=1)
               for i in range(1, n + 1)]
FC_LAYERS = ["fc6", "fc7", "fc8"]
#: Layers read pre-ReLU at 224 px.
LAYERS_224 = CONV_LAYERS + FC_LAYERS
#: Layers read post-ReLU at 112 px.
LAYERS_112 = ["conv1_2", "conv2_2", "conv3_3", "conv4_3"]

_CONV_CHANNELS = {1: 64, 2: 128, 3: 256, 4: 512, 5: 512}


def layer_width(layer: str) -> int:
    """Number of values one layer contributes (channels, or fc units)."""
    if layer.startswith("conv"):
        return _CONV_CHANNELS[int(layer[4])]
    return 1000 if layer == "fc8" else 4096


def column_names() -> list[str]:
    """Every column this model emits, in emission order."""
    cols = [f"vgg19_{layer}_{i:04d}"
            for layer in LAYERS_224 for i in range(layer_width(layer))]
    cols += [f"vgg19_{layer}_relu112_{i:04d}"
             for layer in LAYERS_112 for i in range(layer_width(layer))]
    return cols


class VGG19Model(BaseModel):
    """Layerwise VGG19 features (spatial mean per channel).

    See the module docstring for the two layer sets and their sources.
    """

    name = "vgg19"
    checkpoint = "torchvision/vgg19_IMAGENET1K_V1"
    # Every value is a finite activation mean; no column can be NaN.
    nulls: dict[str, dict[str, str]] = {}

    def __init__(self, device: str | None = None):
        super().__init__(device=device)
        self._columns = column_names()
        self._transforms = {
            size: T.Compose([
                T.Resize((size, size)),
                T.ToTensor(),
                T.Normalize(mean=_MEAN, std=_STD),
            ])
            for size in (224, 112)
        }

    def load(self) -> None:
        from torchvision.models import VGG19_Weights, vgg19

        model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        self.model = model.eval().to(self.device)

    def _conv_names(self) -> list[tuple[int, str]]:
        """(features index, layer name) for every Conv2d, in order."""
        convs = [i for i, m in enumerate(self.model.features) if isinstance(m, nn.Conv2d)]
        assert len(convs) == len(CONV_LAYERS), "unexpected VGG19 layout"
        return list(zip(convs, CONV_LAYERS))

    def _forward_224(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Pre-ReLU spatial means of all 16 conv layers, then fc6-fc8."""
        names = dict(self._conv_names())
        out = []
        for i, module in enumerate(self.model.features):
            x = module(x)
            if i in names:
                # Taken before the (in-place) ReLU that follows.
                out.append(x.mean(dim=(2, 3)))
        x = torch.flatten(self.model.avgpool(x), 1)
        cls = self.model.classifier  # Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear
        fc6 = cls[0](x)
        out.append(fc6)
        fc7 = cls[3](cls[2](cls[1](fc6.clone())))
        out.append(fc7)
        out.append(cls[6](cls[5](cls[4](fc7.clone()))))
        return out

    def _forward_112(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Post-ReLU spatial means of the four block outputs in LAYERS_112."""
        wanted = {i: name for i, name in self._conv_names() if name in LAYERS_112}
        last = max(wanted) + 1  # the ReLU after the deepest wanted conv
        out = []
        for i, module in enumerate(self.model.features[: last + 1]):
            x = module(x)
            if i - 1 in wanted and isinstance(module, nn.ReLU):
                out.append(x.mean(dim=(2, 3)))
        return out

    def _rows(self, images: list[Image.Image]) -> list[dict[str, float]]:
        rgb = [img.convert("RGB") for img in images]
        with torch.no_grad():
            b224 = torch.stack([self._transforms[224](im) for im in rgb]).to(self.device)
            b112 = torch.stack([self._transforms[112](im) for im in rgb]).to(self.device)
            feats = torch.cat(self._forward_224(b224) + self._forward_112(b112), dim=1)
        values = feats.float().cpu().numpy()
        assert values.shape[1] == len(self._columns)
        return [dict(zip(self._columns, row.tolist())) for row in values]

    def predict(self, image: Image.Image) -> dict[str, float]:
        return self._rows([image])[0]

    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, float]]:
        return self._rows(images)
