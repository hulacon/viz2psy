"""EBind vision arm — 1024-d embeddings in the cross-modal shared space.

EBind (checkpoint ``encord-team/ebind-full``) binds Perception Encoder
vision/text backbones with an ImageBind-huge audio arm in one 1024-d space.
The paired text encoder lives in word2psy as ``ebind_text`` and the audio
arm in aud2psy as ``ebind_audio`` — all three load the same revision-pinned
checkpoint (every sub-checkpoint is pinned in ``ebind/consts.py``), so
images, words, and soundtracks are directly comparable. psytwill's
COMPATIBLE_SPACES declares the pairings. Do not change the checkpoint, the
L2 normalization, or the ``ebind_{i:04d}`` naming without coordinating all
three repos.

First >999-d space in the family: embedding indices are fixed-width
4-digit (``ebind_0000`` .. ``ebind_1023``) per contracts §4.1.
"""

import torch
from PIL import Image

from .base import BaseModel

DEFAULT_CHECKPOINT = "encord-team/ebind-full"
EMBED_DIM = 1024
# EBind's config requires image+video+text at minimum; audio and points are
# excluded so their backbones (ImageBind, Uni3D) never load here.
_MODALITIES = ["image", "video", "text"]


class EBindModel(BaseModel):
    """Extract L2-normalized EBind image embeddings (PE-Core-L14-336 arm)."""

    name = "ebind"
    checkpoint = DEFAULT_CHECKPOINT

    def __init__(self, checkpoint: str = DEFAULT_CHECKPOINT, device: str | None = None):
        super().__init__(device=device)
        self.checkpoint = checkpoint
        self._transform = None

    def load(self) -> None:
        from ebind import EBindModel as _EBind
        from ebind.configuration import EBindConfig
        from ebind.consts import PERCEPTION_ENCODER_CHECKPOINT_ARGS
        from ebind.models.perception_encoder.models import PEImageProcessor

        config = EBindConfig(modalities=list(_MODALITIES))
        model = _EBind.from_pretrained(self.checkpoint, config=config)
        self.model = model.eval().to(self.device)
        pe_name = PERCEPTION_ENCODER_CHECKPOINT_ARGS["repo_id"].split("/")[1]
        # PEImageProcessor.__call__ wants a file path; its .transform takes
        # PIL directly (and converts to RGB itself), which is what the
        # viz2psy pipeline hands us.
        self._transform = PEImageProcessor.from_config(pe_name).transform

    def predict(self, image: Image.Image) -> dict[str, float]:
        return self.predict_batch([image])[0]

    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, float]]:
        batch = torch.stack([self._transform(img) for img in images]).to(self.device)
        with torch.no_grad():
            features = self.model.forward(image=batch)["image"].float()
            features = features / features.norm(dim=-1, keepdim=True)
        return [
            {f"ebind_{i:04d}": v for i, v in enumerate(row)}
            for row in features.cpu().tolist()
        ]
