"""Depth model tests.

The layout statistics (depth_stats) are pure numpy against hand-built
depth maps. The backbone integration test loads the real checkpoint
(~100 MB, cached in the shared HF cache) and checks output sanity on the
skimage astronaut portrait.
"""

import math

import numpy as np
import pytest

from viz2psy.models.depth import FEATURE_NAMES, depth_stats

H, W = 90, 120


class TestDepthStats:
    def test_constant_map_scores_zero_structure(self):
        s = depth_stats(np.full((H, W), 5.0))
        assert list(s) == FEATURE_NAMES
        assert all(v == 0.0 for v in s.values())

    def test_half_near_half_far_plane(self):
        # left half far (0), right half near (1): clean bimodal split
        d = np.zeros((H, W))
        d[:, W // 2 :] = 1.0
        s = depth_stats(d)
        assert s["depth_fg_fraction"] == pytest.approx(0.5, abs=0.02)
        assert s["depth_openness"] == pytest.approx(0.5, abs=0.02)
        assert s["depth_iqr"] > 0.9  # quartiles straddle the two modes
        # edges only along the single vertical boundary: low mean density
        assert 0 < s["depth_edge_density"] < 0.05

    def test_near_center_composition_is_positive(self):
        # near blob in the center third, far elsewhere: portrait composition
        d = np.zeros((H, W))
        d[H // 3 : -H // 3, W // 3 : -W // 3] = 1.0
        s = depth_stats(d)
        assert s["depth_center_near"] > 0.5
        # far-dominated distribution skews positive
        assert s["depth_skew"] > 0

    def test_gradient_ramp_has_no_hard_edges(self):
        # smooth left-right ramp: spread without discontinuities
        d = np.tile(np.linspace(0, 1, W), (H, 1))
        s = depth_stats(d)
        assert s["depth_iqr"] == pytest.approx(0.5, abs=0.05)
        assert s["depth_edge_density"] < 0.02
        assert abs(s["depth_skew"]) < 0.2  # symmetric distribution

    def test_speckle_outliers_do_not_dominate(self):
        # robust percentile normalization: two hot pixels must not
        # compress the rest of the map into the far field
        d = np.full((H, W), 0.5)
        d[:, : W // 2] = 0.6  # genuine two-level structure
        d[0, 0] = 1000.0
        d[1, 1] = -1000.0
        s = depth_stats(d)
        assert s["depth_fg_fraction"] + s["depth_openness"] == pytest.approx(1.0, abs=0.01)


@pytest.fixture(scope="module")
def model():
    from viz2psy.models.depth import DepthModel

    m = DepthModel()
    m.load()  # downloads the checkpoint on a cold cache
    return m


class TestDepthModel:
    def test_astronaut_outputs_are_sane(self, model):
        from PIL import Image
        from skimage.data import astronaut

        s = model.predict(Image.fromarray(astronaut()))
        assert list(s) == FEATURE_NAMES
        assert 0.0 <= s["depth_fg_fraction"] <= 1.0
        assert 0.0 <= s["depth_openness"] <= 1.0
        assert s["depth_fg_fraction"] + s["depth_openness"] <= 1.0
        assert 0.0 < s["depth_iqr"] <= 1.0
        assert all(math.isfinite(v) for v in s.values())
        # composition term is bounded; its sign is image-specific (the
        # astronaut sits left-of-center, so it comes out slightly negative)
        assert -1.0 <= s["depth_center_near"] <= 1.0

    def test_batch_matches_single(self, model):
        from PIL import Image
        from skimage.data import astronaut

        img = Image.fromarray(astronaut())
        single = model.predict(img)
        batch = model.predict_batch([img, img])
        for row in batch:
            for k in FEATURE_NAMES:
                assert row[k] == pytest.approx(single[k], rel=1e-4)

    def test_registered_and_checkpointed(self):
        from viz2psy.cli import MODEL_REGISTRY, VIDEO_ONLY_MODELS
        from viz2psy.metadata import get_model_contract

        assert "depth" in MODEL_REGISTRY
        assert "depth" not in VIDEO_ONLY_MODELS  # per-image, runs on stills
        checkpoint, _ = get_model_contract("depth")
        assert checkpoint == "depth-anything/Depth-Anything-V2-Small-hf"
