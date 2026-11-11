"""Faces model tests.

The geometry (face_stats) is pure numpy against hand-built boxes. The
detector integration uses skimage's bundled astronaut portrait as a
deterministic real-face asset; the ~230 KB YuNet checkpoint downloads to
~/.cache/viz2psy/ on first use.
"""

import math

import numpy as np
import pytest

from viz2psy.models.faces import FEATURE_NAMES, SHORT_SIDE, FacesModel, face_stats, resize_short_side

W, H = 200, 100  # half-diagonal = sqrt(200^2+100^2)/2
HALF_DIAG = math.hypot(W, H) / 2


class TestFaceStats:
    def test_no_faces_is_explicit_zero_not_nan(self):
        s = face_stats(np.empty((0, 4)), W, H)
        assert list(s) == FEATURE_NAMES
        assert s["faces_count"] == 0.0
        assert s["faces_total_area"] == 0.0
        assert s["faces_max_area"] == 0.0
        assert math.isnan(s["faces_center_dist"])
        assert math.isnan(s["faces_mutual_dist"])

    def test_single_centered_face(self):
        # 20x20 box centered at image center
        s = face_stats(np.array([[90, 40, 20, 20]]), W, H)
        assert s["faces_count"] == 1.0
        assert s["faces_total_area"] == pytest.approx(400 / (W * H))
        assert s["faces_max_area"] == pytest.approx(400 / (W * H))
        assert s["faces_center_dist"] == pytest.approx(0.0)
        assert math.isnan(s["faces_mutual_dist"])

    def test_two_faces_hand_computed(self):
        # centers at (50,50) and (150,50): each 50 px from the image
        # center horizontally; 100 px apart
        boxes = np.array([[40, 40, 20, 20], [140, 40, 20, 20]])
        s = face_stats(boxes, W, H)
        assert s["faces_count"] == 2.0
        assert s["faces_total_area"] == pytest.approx(800 / (W * H))
        assert s["faces_center_dist"] == pytest.approx(50 / HALF_DIAG)
        assert s["faces_mutual_dist"] == pytest.approx(100 / HALF_DIAG)

    def test_max_area_tracks_largest(self):
        boxes = np.array([[0, 0, 10, 10], [50, 20, 40, 40]])
        s = face_stats(boxes, W, H)
        assert s["faces_max_area"] == pytest.approx(1600 / (W * H))


class TestResizeShortSide:
    @pytest.mark.parametrize("h,w,expect", [
        (360, 640, (540, 960)),    # landscape, upsampled
        (800, 1920, (540, 1296)),  # landscape, downsampled
        (1080, 720, (810, 540)),   # portrait
        (540, 540, (540, 540)),    # already canonical: returned as is
    ])
    def test_short_side_and_aspect(self, h, w, expect):
        import cv2

        img = np.zeros((h, w, 3), np.uint8)
        out = resize_short_side(img, SHORT_SIDE, cv2)
        assert out.shape[:2] == expect
        if (h, w) == expect:
            assert out is img


@pytest.fixture(scope="module")
def model():
    m = FacesModel(device="cpu")
    m.load()  # downloads the checkpoint on a cold cache
    return m


class TestFacesModel:
    def test_astronaut_portrait_has_one_face(self, model):
        from PIL import Image
        from skimage.data import astronaut

        s = model.predict(Image.fromarray(astronaut()))
        assert s["faces_count"] == 1.0
        assert 0.0 < s["faces_max_area"] < 0.5
        assert not math.isnan(s["faces_center_dist"])
        assert math.isnan(s["faces_mutual_dist"])

    def test_blank_image_has_no_faces(self, model):
        from PIL import Image

        s = model.predict(Image.new("RGB", (320, 240), (128, 128, 128)))
        assert s["faces_count"] == 0.0
        assert s["faces_total_area"] == 0.0

    def test_detector_sees_the_canonical_size(self, model):
        # The point of the canonical size: whatever the picture's pixels, YuNet
        # runs on a short side of SHORT_SIDE, so its pixel floor is a fixed
        # fraction of the frame.
        from PIL import Image
        from skimage.data import astronaut

        im = Image.fromarray(astronaut())
        seen = []
        real = model.model

        class Spy:  # cv2 objects take no attribute assignment, so wrap instead
            def setInputSize(self, size):
                seen.append(tuple(size))
                real.setInputSize(size)

            def detect(self, img):
                return real.detect(img)

        model.model = Spy()
        try:
            for w, h in [(256, 256), (1024, 1024), (640, 360)]:
                s = model.predict(im.resize((w, h), Image.BILINEAR))
                assert s["faces_count"] == 1.0
        finally:
            model.model = real
        assert seen == [(540, 540), (540, 540), (960, 540)]

    def test_preprocessing_records_size_and_threshold(self):
        pre = FacesModel.describe_preprocessing()
        assert str(SHORT_SIDE) in pre["resize"]
        assert pre["score_threshold"] == 0.8

    def test_count_description_states_the_animation_limit(self):
        desc = FacesModel.labels["columns"]["faces_count"]["Description"]
        assert "animated" in desc and "does not mean no face" in desc

    def test_hash_mismatch_names_the_fix(self, tmp_path):
        from viz2psy.exceptions import ModelLoadError

        bad = tmp_path / "face_detection_yunet_2023mar.onnx"
        bad.write_bytes(b"not an onnx file")
        with pytest.raises(ModelLoadError, match="hash mismatch"):
            FacesModel(weights_path=bad, device="cpu").load()

    def test_registered_and_checkpointed(self):
        from viz2psy.cli import MODEL_REGISTRY
        from viz2psy.metadata import get_model_contract

        assert "faces" in MODEL_REGISTRY
        checkpoint, _ = get_model_contract("faces")
        assert checkpoint == "opencv_zoo/face_detection_yunet_2023mar"
