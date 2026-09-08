"""Saliency model tests.

The sub-batching logic is exercised with a stub in place of DeepGaze IIE
(no weights needed): a fake network that records the batch sizes it sees
and returns a valid log-density map.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from viz2psy.models import saliency as sal


class _FakeNet:
    """Stands in for DeepGaze IIE: records batch sizes, returns a
    log-density whose peak depends on the image so rows are distinguishable."""

    def __init__(self):
        self.batch_sizes: list[int] = []

    def __call__(self, batch: torch.Tensor, centerbias: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(batch.shape[0]))
        n, _, h, w = batch.shape
        out = torch.full((n, 1, h, w), -float(np.log(h * w)))
        for i in range(n):
            # brighten a cell keyed on the image's mean so rows differ
            col = int(batch[i].mean().item()) % w
            out[i, 0, :, col] += 1.0
        return out


@pytest.fixture
def model():
    m = sal.SaliencyModel(device="cpu")
    m.model = _FakeNet()
    return m


def _images(n: int, h: int, w: int) -> list[Image.Image]:
    rng = np.random.default_rng(0)
    return [Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)) for _ in range(n)]


class TestSubBatching:
    def test_small_frames_go_through_in_one_pass(self, model):
        imgs = _images(10, 36, 72)  # 2.6k pixels each: far under budget
        rows = model.predict_batch(imgs)
        assert len(rows) == 10
        assert model.model.batch_sizes == [10]

    def test_large_frames_are_split_under_the_pixel_budget(self, model, monkeypatch):
        monkeypatch.setattr(sal, "_PIXEL_BUDGET", 5 * 40 * 60)  # 5 frames of 40x60
        imgs = _images(12, 40, 60)
        rows = model.predict_batch(imgs)
        assert len(rows) == 12
        assert model.model.batch_sizes == [5, 5, 2]
        assert all(b * 40 * 60 <= sal._PIXEL_BUDGET for b in model.model.batch_sizes)

    def test_split_matches_single_image_predictions(self, model, monkeypatch):
        monkeypatch.setattr(sal, "_PIXEL_BUDGET", 3 * 40 * 60)
        imgs = _images(7, 40, 60)
        batched = model.predict_batch(imgs)
        singles = [model.predict(img) for img in imgs]
        assert len(batched) == len(singles) == 7
        for b, s in zip(batched, singles):
            assert list(b) == list(s)
            assert np.allclose(list(b.values()), list(s.values()), rtol=1e-6)
        # rows are genuinely different images, not copies of one result
        assert not np.allclose(list(batched[0].values()), list(batched[1].values()))

    def test_frame_larger_than_budget_still_runs_one_at_a_time(self, model, monkeypatch):
        monkeypatch.setattr(sal, "_PIXEL_BUDGET", 100)  # smaller than one frame
        rows = model.predict_batch(_images(3, 40, 60))
        assert len(rows) == 3
        assert model.model.batch_sizes == [1, 1, 1]

    def test_grid_rows_sum_to_one(self, model):
        rows = model.predict_batch(_images(2, 48, 48))
        for r in rows:
            assert len(r) == 24 * 24
            assert sum(r.values()) == pytest.approx(1.0, abs=1e-5)
