"""A weightless model for driving the CLI end to end in tests.

Registered into ``viz2psy.cli.MODEL_REGISTRY`` by monkeypatching, so the CLI's
routing, merging and sidecar code run for real without any checkpoint.
"""

import numpy as np

from viz2psy.models.base import BaseModel


class FakeModel(BaseModel):
    name = "fake"

    def load(self):
        self.model = None

    def predict(self, image):
        return {"fake_mean": float(np.asarray(image, dtype=float).mean())}


class OtherFakeModel(FakeModel):
    name = "otherfake"

    def predict(self, image):
        return {"otherfake_max": float(np.asarray(image, dtype=float).max())}


class TextureFakeModel(FakeModel):
    """Mean absolute horizontal gradient: JPEG's block smoothing moves it, so
    a lossy round trip anywhere on the scoring path shows up in this column."""

    name = "texture"

    def predict(self, image):
        a = np.asarray(image, dtype=float)
        return {"texture_grad": float(np.abs(np.diff(a, axis=1)).mean())}
