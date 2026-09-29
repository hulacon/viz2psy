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
    # canonical_area=None keeps the stored pixel count, so the sub-batching
    # tests see the frame sizes they set up
    m = sal.SaliencyModel(device="cpu", canonical_area=None)
    m.model = _FakeNet()
    return m


class _ShapeNet(_FakeNet):
    """Also records the (H, W) of every batch."""

    def __init__(self):
        super().__init__()
        self.shapes: list[tuple[int, int]] = []

    def __call__(self, batch, centerbias):
        self.shapes.append(tuple(batch.shape[2:]))
        return super().__call__(batch, centerbias)


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


class TestCanonicalSize:
    def test_square_nsd_image_is_unchanged(self):
        assert sal.canonical_size(425, 425) == (425, 425)

    def test_area_is_nsd_and_aspect_is_kept(self):
        w, h = sal.canonical_size(1920, 800)
        assert w / h == pytest.approx(2.4, rel=0.01)
        assert w * h == pytest.approx(425 * 425, rel=0.01)

    def test_anamorphic_frame_takes_its_display_aspect(self):
        # 720 x 480 at 8:9 is displayed 4:3, like a square-pixel 640 x 480
        assert sal.canonical_size(720, 480, pixel_aspect=8 / 9) == sal.canonical_size(640, 480)

    def test_area_none_keeps_the_stored_pixel_count(self):
        w, h = sal.canonical_size(720, 480, pixel_aspect=8 / 9, area=None)
        assert w / h == pytest.approx(4 / 3, rel=0.01)
        assert w * h == pytest.approx(720 * 480, rel=0.01)
        assert sal.canonical_size(60, 40, area=None) == (60, 40)


class TestCanonicalResize:
    def _model(self, **kw):
        m = sal.SaliencyModel(device="cpu", **kw)
        m.model = _ShapeNet()
        return m

    def test_network_sees_the_canonical_size(self):
        m = self._model(pixel_aspect=8 / 9)
        m.predict_batch(_images(2, 480, 720))
        assert m.model.shapes == [sal.canonical_size(640, 480)[::-1]]

    def test_nsd_sized_input_scores_as_before(self):
        imgs = _images(2, 425, 425)
        new, old = self._model(), self._model(canonical_area=None)
        assert new.predict_batch(imgs) == old.predict_batch(imgs)
        assert new.model.shapes == [(425, 425)]

    def test_mixed_sizes_share_one_canonical_shape_per_aspect(self):
        m = self._model()
        m.predict_batch(_images(1, 480, 640) + _images(1, 240, 320))
        assert m.model.shapes == [sal.canonical_size(640, 480)[::-1]]

    def test_preprocessing_is_described(self):
        d = self._model(pixel_aspect=0.5).preprocessing()
        assert d["canonical_area"] == 425 * 425 and d["pixel_aspect"] == 0.5


class TestVideoWiring:
    def test_only_saliency_gets_the_pixel_aspect(self):
        from viz2psy.cli import _video_model, _video_preprocessing

        assert _video_model("saliency", "cpu", 8 / 9).pixel_aspect == pytest.approx(8 / 9)
        assert _video_preprocessing("saliency", 8 / 9)["pixel_aspect"] == pytest.approx(8 / 9)
        assert _video_preprocessing("llstat", 8 / 9) is None

    def test_sidecar_records_saliency_preprocessing(self):
        from viz2psy.metadata import MetadataBuilder

        b = MetadataBuilder()
        b.add_model("saliency", ["saliency_00_00"], 1.0)
        assert b.models["saliency"]["preprocessing"]["canonical_area"] == 425 * 425
        b.add_model("saliency", ["saliency_00_00"], 1.0,
                    preprocessing=sal.SaliencyModel.describe_preprocessing(pixel_aspect=0.75))
        assert b.models["saliency"]["preprocessing"]["pixel_aspect"] == 0.75

    def test_get_pixel_aspect_reads_the_stream(self, tmp_path):
        from fractions import Fraction

        av = pytest.importorskip("av")
        from viz2psy.video import get_pixel_aspect, get_video_info

        path = tmp_path / "anamorphic.mkv"
        c = av.open(str(path), "w")
        st = c.add_stream("mpeg2video", rate=25)
        st.width, st.height, st.pix_fmt = 72, 48, "yuv420p"
        st.codec_context.sample_aspect_ratio = Fraction(8, 9)
        for _ in range(10):
            for pkt in st.encode(av.VideoFrame.from_ndarray(np.zeros((48, 72, 3), np.uint8), format="rgb24")):
                c.mux(pkt)
        for pkt in st.encode():
            c.mux(pkt)
        c.close()
        assert get_pixel_aspect(path) == pytest.approx(8 / 9)
        assert get_video_info(path)["pixel_aspect"] == pytest.approx(8 / 9)
