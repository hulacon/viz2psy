"""Motion model tests — synthetic videos with known ground truth, offline.

viz2psy's first test file; the synthetic-video helpers here are the
conftest seed if more model tests follow.
"""

import numpy as np
import pytest

FPS = 24
W, H = 320, 240


def write_video(path, frames, fps=FPS):
    import cv2

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, fps, (W, H))
    if not out.isOpened():
        pytest.skip("cv2.VideoWriter cannot encode mp4v in this environment")
    for f in frames:
        out.write(f)
    out.release()
    return path


def square_frame(x, y, size=40):
    """Black frame with a white square whose top-left corner is (x, y)."""
    f = np.zeros((H, W, 3), dtype=np.uint8)
    f[y : y + size, x : x + size] = 255
    return f


@pytest.fixture(scope="module")
def model():
    from viz2psy.models.motion import MotionModel

    m = MotionModel(device="cpu")
    m.load()
    return m


class TestMotionModel:
    def test_static_video_scores_zero_motion(self, model, tmp_path_factory):
        d = tmp_path_factory.mktemp("static")
        video = write_video(d / "static.mp4", [square_frame(100, 100)] * FPS)
        rows = model.predict_video(video, [0.0, 0.5], quiet=True)
        assert len(rows) == 2
        for row in rows:
            assert row["motion_energy"] == pytest.approx(0.0, abs=1e-3)
            assert row["motion_frame_diff"] == pytest.approx(0.0, abs=1e-2)
            assert np.isnan(row["motion_coherence"])  # nothing to be coherent about

    def test_rightward_translation_sign_and_coherence(self, model, tmp_path_factory):
        d = tmp_path_factory.mktemp("pan")
        # square moves right 4 px per native frame
        frames = [square_frame(50 + 4 * i, 100) for i in range(FPS)]
        video = write_video(d / "pan.mp4", frames)
        row = model.predict_video(video, [0.25], quiet=True)[0]
        assert row["motion_energy"] > 0.01
        assert row["motion_horizontal"] > 0  # rightward is +x
        assert abs(row["motion_horizontal"]) > abs(row["motion_vertical"])

    def test_downward_translation_is_positive_vertical(self, model, tmp_path_factory):
        d = tmp_path_factory.mktemp("tilt")
        frames = [square_frame(100, 50 + 4 * i) for i in range(FPS)]
        video = write_video(d / "tilt.mp4", frames)
        row = model.predict_video(video, [0.25], quiet=True)[0]
        assert row["motion_vertical"] > 0  # image coords: downward is +y
        assert abs(row["motion_vertical"]) > abs(row["motion_horizontal"])

    def test_hard_cut_spikes_frame_diff(self, model, tmp_path_factory):
        import cv2  # noqa: F401  (ensures the codec check ran)

        d = tmp_path_factory.mktemp("cut")
        black = np.zeros((H, W, 3), dtype=np.uint8)
        white = np.full((H, W, 3), 255, dtype=np.uint8)
        # cut exactly at the frame after t=0
        video = write_video(d / "cut.mp4", [black] + [white] * (FPS - 1))
        row = model.predict_video(video, [0.0], quiet=True)[0]
        assert row["motion_frame_diff"] > 0.5

    def test_past_end_scores_nan(self, model, tmp_path_factory):
        d = tmp_path_factory.mktemp("end")
        video = write_video(d / "short.mp4", [square_frame(100, 100)] * 4)
        rows = model.predict_video(video, [0.0, 10.0], quiet=True)
        assert not np.isnan(rows[0]["motion_energy"])
        assert all(np.isnan(v) for v in rows[1].values())

    def test_rows_align_with_requested_times(self, model, tmp_path_factory):
        d = tmp_path_factory.mktemp("align")
        video = write_video(d / "clip.mp4", [square_frame(100, 100)] * FPS)
        times = [0.0, 0.25, 0.5, 0.75]
        rows = model.predict_video(video, times, quiet=True)
        assert len(rows) == len(times)
        from viz2psy.models.motion import FEATURE_NAMES

        assert all(list(r) == FEATURE_NAMES for r in rows)

    def test_still_image_refused(self, model):
        from PIL import Image

        from viz2psy.exceptions import Viz2PsyError

        with pytest.raises(Viz2PsyError, match="video-only"):
            model.predict(Image.new("RGB", (64, 64)))

    def test_registered_and_analytic(self):
        from viz2psy.cli import MODEL_REGISTRY, VIDEO_ONLY_MODELS
        from viz2psy.metadata import get_model_contract

        assert "motion" in MODEL_REGISTRY
        assert "motion" in VIDEO_ONLY_MODELS
        checkpoint, _ = get_model_contract("motion")
        assert checkpoint is None


class TestVideoOnlyGuards:
    def test_process_images_rejects_motion(self):
        from viz2psy.cli import _process_images
        from viz2psy.exceptions import Viz2PsyError

        with pytest.raises(Viz2PsyError, match="video-only"):
            _process_images([], ["motion"], batch_size=1, device=None, quiet=True)
