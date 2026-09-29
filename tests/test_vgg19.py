"""VGG19 model tests.

The layer bookkeeping and the hand-written forward pass run against a
randomly initialised VGG19 (no download), checked against torchvision's own
forward. Only the pretrained-weights test is `heavy` (~550 MB checkpoint).
"""

import math

import numpy as np
import pytest
import torch
from PIL import Image

from viz2psy.metadata import get_feature_info
from viz2psy.models.vgg19 import (
    LAYERS_112,
    LAYERS_224,
    VGG19Model,
    column_names,
    layer_width,
)

N_COLS = 15_656


def _image(w: int, h: int, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8))


@pytest.fixture(scope="module")
def model():
    from torchvision.models import vgg19

    torch.manual_seed(0)
    m = VGG19Model(device="cpu")
    m.model = vgg19(weights=None).eval()
    return m


class TestColumns:
    def test_count_and_uniqueness(self):
        cols = column_names()
        assert len(cols) == N_COLS == len(set(cols))
        assert all(c.startswith("vgg19_") for c in cols)

    def test_layer_sets(self):
        assert len(LAYERS_224) == 19
        assert LAYERS_224[0] == "conv1_1" and LAYERS_224[15] == "conv5_4"
        assert sum(layer_width(layer) for layer in LAYERS_224) == 14_696
        assert sum(layer_width(layer) for layer in LAYERS_112) == 960

    def test_sidecar_declares_pattern_with_layer_widths(self):
        info = get_feature_info("vgg19", column_names())
        assert info["pattern"] == "vgg19_{layer}_{NNNN}"
        assert info["count"] == N_COLS == sum(info["layers"].values())
        assert info["layers"]["conv4_3_relu112"] == 512


class TestForward:
    def test_rows_are_complete_and_finite(self, model):
        rows = model.predict_batch([_image(320, 180, 1), _image(200, 200, 2)])
        assert len(rows) == 2
        for row in rows:
            assert list(row) == column_names()
            assert all(math.isfinite(v) for v in row.values())

    def test_fc8_matches_torchvision_forward(self, model):
        img = _image(256, 144, 3)
        row = model.predict(img)
        x = model._transforms[224](img).unsqueeze(0)
        with torch.no_grad():
            logits = model.model(x)[0].numpy()
        ours = np.array([row[f"vgg19_fc8_{i:04d}"] for i in range(1000)])
        np.testing.assert_allclose(ours, logits, rtol=1e-4, atol=1e-4)

    def test_conv1_1_is_pre_relu(self, model):
        img = _image(224, 224, 4)
        row = model.predict(img)
        x = model._transforms[224](img).unsqueeze(0)
        with torch.no_grad():
            expect = model.model.features[0](x).mean(dim=(2, 3))[0].numpy()
        ours = np.array([row[f"vgg19_conv1_1_{i:04d}"] for i in range(64)])
        np.testing.assert_allclose(ours, expect, rtol=1e-4, atol=1e-5)
        assert (ours < 0).any()  # a post-ReLU mean could never be negative

    def test_relu112_set_is_post_relu_at_112(self, model):
        img = _image(300, 169, 5)
        row = model.predict(img)
        x = model._transforms[112](img).unsqueeze(0)
        with torch.no_grad():
            expect = model.model.features[:4](x).mean(dim=(2, 3))[0].numpy()  # conv1_2 + ReLU
        ours = np.array([row[f"vgg19_conv1_2_relu112_{i:04d}"] for i in range(64)])
        np.testing.assert_allclose(ours, expect, rtol=1e-4, atol=1e-5)
        relu_cols = [c for c in column_names() if "_relu112_" in c]
        assert min(row[c] for c in relu_cols) >= 0.0

    def test_batch_equals_single(self, model):
        imgs = [_image(160, 90, 6), _image(90, 160, 7)]
        batch = model.predict_batch(imgs)
        for img, row in zip(imgs, batch):
            single = model.predict(img)
            a = np.array(list(single.values()))
            b = np.array(list(row.values()))
            np.testing.assert_allclose(a, b, rtol=1e-4, atol=1e-4)


@pytest.mark.heavy
def test_pretrained_weights_load_and_score():
    m = VGG19Model(device="cpu")
    m.load()
    row = m.predict(_image(320, 180, 8))
    assert len(row) == N_COLS
    assert all(math.isfinite(v) for v in row.values())
