"""Experiential attributes (Binder et al., 2016) of the object an image depicts.

A linear (ridge) map from the L2-normalised CLIP ViT-L/14 (OpenAI) image
embedding (the backbone ``aesthetics`` and ``context_affect`` load, through
the same squash-and-embed path) to Binder et al.'s 0-6 ratings of how strongly
a concept is associated with each of 65 experiential attributes (vision,
sound, upper-limb action, fear, ...). Training rows are THINGS object
photographs (Hebart et al., 2019) of the Binder concepts THINGS covers, each
image labelled with its concept's ratings.

On a scene or film frame it reads a mixture of what is depicted. It is not a
rating of the image, nor of the viewer's experience. Out of the training
domain: scenes, people-centred frames, text on screen.

Weights are not shipped: the ratings are user-supplied (Binder et al. ask
for a citation and give no licence) and THINGS images are licensed for
research use. Build them with ``scripts/train_experiential.py``; they are read
from :func:`weights_path` (``$VIZ2PSY_EXPERIENTIAL_WEIGHTS``, else
``<torch hub dir>/viz2psy/``). The model emits exactly the attributes its
weights hold.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .base import SQUASH_RECORD
from .context_affect import BACKBONE, EMBED_DIM, ContextAffectModel, sha256

#: Binder et al. (2016) attributes -> (domain, the definition raters were given),
#: in the ratings spreadsheet's column order (their Table 1); identical to
#: ``word2psy.norms.binder.ATTRIBUTES`` (tests check it when word2psy is installed).
ATTRIBUTES: dict[str, tuple[str, str]] = {
    "Vision": ("Vision", "something that you can easily see"),
    "Bright": ("Vision", "visually light or bright"),
    "Dark": ("Vision", "visually dark"),
    "Color": ("Vision", "having a characteristic or defining colour"),
    "Pattern": ("Vision", "having a characteristic or defining visual texture or surface pattern"),
    "Large": ("Vision", "large in size"),
    "Small": ("Vision", "small in size"),
    "Motion": ("Vision", "showing a lot of visually observable movement"),
    "Biomotion": ("Vision", "showing movement like that of a living thing"),
    "Fast": ("Vision", "showing visible movement that is fast"),
    "Slow": ("Vision", "showing visible movement that is slow"),
    "Shape": ("Vision", "having a characteristic or defining visual shape or form"),
    "Complexity": ("Vision", "visually complex"),
    "Face": ("Vision", "having a human or human-like face"),
    "Body": ("Vision", "having human or human-like body parts"),
    "Touch": ("Somatic", "something that you could easily recognize by touch"),
    "Temperature": ("Somatic", "hot or cold to the touch"),
    "Texture": ("Somatic", "having a smooth or rough texture to the touch"),
    "Weight": ("Somatic", "light or heavy in weight"),
    "Pain": ("Somatic", "associated with pain or physical discomfort"),
    "Audition": ("Audition", "something that you can easily hear"),
    "Loud": ("Audition", "making a loud sound"),
    "Low": ("Audition", "having a low-pitched sound"),
    "High": ("Audition", "having a high-pitched sound"),
    "Sound": ("Audition", "having a characteristic or recognizable sound or sounds"),
    "Music": ("Audition", "making a musical sound"),
    "Speech": ("Audition", "someone or something that talks"),
    "Taste": ("Gustation", "having a characteristic or defining taste"),
    "Smell": ("Olfaction", "having a characteristic or defining smell or smells"),
    "Head": ("Motor", "associated with actions using the face, mouth, or tongue"),
    "UpperLimb": ("Motor", "associated with actions using the arm, hand, or fingers"),
    "LowerLimb": ("Motor", "associated with actions using the leg or foot"),
    "Practice": ("Motor", "a physical object you have personal experience using"),
    "Landmark": ("Spatial", "having a fixed location, as on a map"),
    "Path": ("Spatial", "showing changes in location along a particular direction or path"),
    "Scene": ("Spatial", "bringing to mind a particular setting or physical location"),
    "Near": ("Spatial", "often physically near to you (within easy reach) in everyday life"),
    "Toward": ("Spatial", "associated with movement toward or into you"),
    "Away": ("Spatial", "associated with movement away from or out of you"),
    "Number": ("Spatial", "associated with a specific number or amount"),
    "Time": ("Temporal", "an event or occurrence that occurs at a typical or predictable time"),
    "Duration": ("Temporal", "an event that has a predictable duration, whether short or long"),
    "Long": ("Temporal", "an event that lasts for a long period of time"),
    "Short": ("Temporal", "an event that lasts for a short period of time"),
    "Caused": ("Causal", "caused by some clear preceding event, action, or situation"),
    "Consequential": ("Causal", "likely to have consequences (cause other things to happen)"),
    "Social": ("Social", "an activity or event that involves an interaction between people"),
    "Human": ("Social", "having human or human-like intentions, plans, or goals"),
    "Communication": ("Social", "a thing or action that people use to communicate"),
    "Self": ("Social", "related to your own view of yourself, a part of your self-image"),
    "Cognition": ("Cognition", "a form of mental activity or a function of the mind"),
    "Benefit": ("Emotion", "someone or something that could help or benefit you or others"),
    "Harm": ("Emotion", "someone or something that could cause harm to you or others"),
    "Pleasant": ("Emotion", "someone or something that you find pleasant"),
    "Unpleasant": ("Emotion", "someone or something that you find unpleasant"),
    "Happy": ("Emotion", "someone or something that makes you feel happy"),
    "Sad": ("Emotion", "someone or something that makes you feel sad"),
    "Angry": ("Emotion", "someone or something that makes you feel angry"),
    "Disgusted": ("Emotion", "someone or something that makes you feel disgusted"),
    "Fearful": ("Emotion", "someone or something that makes you feel afraid"),
    "Surprised": ("Emotion", "someone or something that makes you feel surprised"),
    "Drive": ("Drive", "someone or something that motivates you to do something"),
    "Needs": ("Drive", "someone or something that would be hard for you to live without"),
    "Attention": ("Attention", "someone or something that grabs your attention"),
    "Arousal": (
        "Attention",
        "someone or something that makes you feel alert, activated, excited, or keyed up in either a positive or negative way",
    ),
}

PREFIX = "experiential_"
#: Every column the probe can emit, in the spreadsheet's attribute order.
FEATURE_NAMES = tuple(PREFIX + a.lower() for a in ATTRIBUTES)

WEIGHTS_ENV = "VIZ2PSY_EXPERIENTIAL_WEIGHTS"
WEIGHTS_NAME = "experiential_v1.npz"
WEIGHTS_FORMAT = "1"

CITATION = (
    "Binder, J. R., Conant, L. L., Humphries, C. J., Fernandino, L., Simons, S. B., "
    "Aguilar, M., & Desai, R. H. (2016). Toward a brain-based componential semantic "
    "representation. Cognitive Neuropsychology, 33(3-4), 130-174."
)


def weights_path() -> Path:
    """``$VIZ2PSY_EXPERIENTIAL_WEIGHTS``, else the shared torch hub cache."""
    env = os.environ.get(WEIGHTS_ENV)
    if env:
        return Path(env)
    return Path(torch.hub.get_dir()) / "viz2psy" / WEIGHTS_NAME


def read_weights(path: Path) -> dict:
    """Load and validate a weights file written by ``scripts/train_experiential.py``.

    Raises ``FileNotFoundError`` naming the path and how to build it.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"experiential weights not found at {path}. Build them with "
            f"scripts/train_experiential.py (Binder et al. 2016 ratings, user-supplied; THINGS "
            f"images, research use) and point ${WEIGHTS_ENV} at the file, or place it at "
            f"<torch hub dir>/viz2psy/{WEIGHTS_NAME}."
        )
    with np.load(path, allow_pickle=False) as z:
        w = {k: z[k] for k in z.files}
    fmt = str(w.get("format_version", ""))
    if fmt != WEIGHTS_FORMAT:
        raise ValueError(f"{path}: weights format {fmt!r}, expected {WEIGHTS_FORMAT!r}")
    if str(w["backbone"]) != BACKBONE:
        raise ValueError(f"{path}: trained on {w['backbone']!s}, this model loads {BACKBONE}")
    cols = tuple(str(c) for c in w["columns"])
    unknown = [c for c in cols if c not in FEATURE_NAMES]
    if not cols or unknown or len(set(cols)) != len(cols):
        raise ValueError(f"{path}: bad columns {list(cols)[:5]} (unknown {unknown[:5]})")
    if w["W"].shape != (EMBED_DIM, len(cols)) or w["b"].shape != (len(cols),):
        raise ValueError(f"{path}: W {w['W'].shape} / b {w['b'].shape} do not fit "
                         f"{EMBED_DIM} -> {len(cols)}")
    w["columns"] = cols
    return w


def _labels() -> dict[str, dict]:
    cols = {}
    for attr, col in zip(ATTRIBUTES, FEATURE_NAMES):
        domain, definition = ATTRIBUTES[attr]
        cols[col] = {
            "LongName": f"Experiential attribute: {attr} ({domain}; depicted object)",
            "Description": (
                f"How strongly the depicted object's concept is associated with '{definition}' "
                f"(Binder et al. 2016 attribute {attr}, {domain} domain; 0-6 scale): a ridge probe "
                f"on frozen CLIP {BACKBONE} image embeddings, trained on THINGS photographs of the "
                "rated concepts, each labelled with its concept's ratings. On a scene or film frame "
                "it reads a mixture of what is depicted; not a rating of the image or of the "
                "viewer's experience. Linear output, not clipped"),
        }
    return cols


class ExperientialModel(ContextAffectModel):
    """Binder et al. (2016) experiential attributes of the depicted object (probe on CLIP ViT-L/14)."""

    name = "experiential"
    checkpoint = f"{BACKBONE}+experiential_v1"
    labels = {
        "default": {"interpretability": "nameable", "derivation": "probe", "derived_from": BACKBONE},
        "columns": _labels(),
    }

    def __init__(self, device: str | None = None):
        super().__init__(device=device)
        self._columns: tuple[str, ...] = ()

    @staticmethod
    def describe_preprocessing() -> dict:
        """The input transform and the probe it feeds, as recorded in the sidecar."""
        rec = {**SQUASH_RECORD,
               "then": "the model's own transform: resize to its input size, then centre "
                       "crop (removes nothing on a square); L2-normalised CLIP ViT-L/14 "
                       "embedding into the experiential ridge head"}
        path = weights_path()
        if path.is_file():
            w = read_weights(path)
            rec["probe"] = {
                "weights_file": path.name,
                "weights_sha256": sha256(path),
                "columns": list(w["columns"]),
                "alpha": dict(zip(w["columns"], map(float, w["alpha"]))),
                "trained_on": json.loads(str(w["trained_on"])),
            }
        else:
            rec["probe"] = {"weights_file": None, "note": f"no weights at {path}"}
        return rec

    def load(self) -> None:
        w = read_weights(weights_path())  # before the backbone: fail fast, and loudly
        self._W, self._b, self._columns = w["W"], w["b"], w["columns"]
        self.load_backbone()

    def predict_batch(self, images: list[Image.Image]) -> list[dict[str, float]]:
        out = np.asarray(self.embed_batch(images), dtype=np.float64) @ self._W + self._b
        return [dict(zip(self._columns, map(float, row))) for row in out]
