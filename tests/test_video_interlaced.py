"""Interlaced video: OpenCV returns a blank frame with ok=True; viz2psy deinterlaces.

Synthetic, offline. The interlaced clip is field-coded MPEG-2 (every FFmpeg
build has the encoder), which reproduces the failure seen on interlaced
broadcast sources: OpenCV "reads" it without error and returns a frame that
is not the picture.
"""

import numpy as np
import pytest

av = pytest.importorskip("av")

FPS = 25
W, H = 128, 96
N_FRAMES = 50
BG = 40


def square_frames(n=N_FRAMES):
    """A bright square on a grey field, moving right 1 px per native frame."""
    frames = []
    for i in range(n):
        f = np.full((H, W, 3), BG, dtype=np.uint8)
        x = 10 + i
        f[30:60, x : x + 30] = (220, 180, 60)
        frames.append(f)
    return frames


def write_mpeg2(path, frames, interlaced):
    options = {"flags": "+ildct+ilme", "top": "1"} if interlaced else {}
    container = av.open(str(path), "w")
    stream = container.add_stream("mpeg2video", rate=FPS, options=options)
    stream.width, stream.height, stream.pix_fmt = W, H, "yuv420p"
    stream.bit_rate = 4_000_000
    for f in frames:
        for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


@pytest.fixture(scope="module")
def videos(tmp_path_factory):
    d = tmp_path_factory.mktemp("interlace")
    frames = square_frames()
    return {
        "progressive": write_mpeg2(d / "progressive.mkv", frames, interlaced=False),
        "interlaced": write_mpeg2(d / "interlaced.mkv", frames, interlaced=True),
        "blank": write_mpeg2(d / "blank.mkv", [np.full((H, W, 3), BG, np.uint8)] * N_FRAMES, False),
    }


def square_x(rgb):
    """Left edge of the bright square (first column whose middle row is bright)."""
    row = rgb[45, :, 0].astype(int)
    return int(np.argmax(row > 150))


class TestInterlacedDecode:
    def test_is_interlaced(self, videos):
        from viz2psy.video import is_interlaced

        assert is_interlaced(videos["interlaced"])
        assert not is_interlaced(videos["progressive"])

    def test_extract_frames_recovers_the_picture(self, videos):
        from viz2psy.video import extract_frames

        frames = extract_frames(videos["interlaced"], frame_interval=0.4, quiet=True)
        assert len(frames) >= 4
        for t, img in frames:
            rgb = np.asarray(img)
            assert rgb.min() != rgb.max()  # not a blank frame
            assert abs(square_x(rgb) - (10 + int(t * FPS))) <= 2

    def test_deinterlaced_matches_progressive_encode(self, videos):
        from viz2psy.video import extract_frames

        a = extract_frames(videos["interlaced"], frame_interval=0.4, quiet=True)
        b = extract_frames(videos["progressive"], frame_interval=0.4, quiet=True)
        assert [t for t, _ in a] == [t for t, _ in b]
        for (_, ia), (_, ib) in zip(a, b):
            diff = np.abs(np.asarray(ia, float) - np.asarray(ib, float)).mean()
            assert diff < 6.0  # codec + deinterlace blur, not a different picture

    def test_progressive_path_is_opencv_unchanged(self, videos):
        import cv2

        from viz2psy.video import extract_frames

        frames = extract_frames(videos["progressive"], frame_interval=0.4, quiet=True)
        cap = cv2.VideoCapture(str(videos["progressive"]))
        fps = cap.get(cv2.CAP_PROP_FPS)
        for t, img in frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, bgr = cap.read()
            assert ok
            np.testing.assert_array_equal(np.asarray(img), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        cap.release()

    def test_blank_clip_is_refused(self, videos):
        from viz2psy.exceptions import VideoError
        from viz2psy.video import extract_frames

        with pytest.raises(VideoError, match="uniform colour"):
            extract_frames(videos["blank"], frame_interval=0.4, quiet=True)

    def test_frozen_decode_is_refused(self, videos, monkeypatch):
        """The guard, on the real failure: the interlaced clip forced through OpenCV."""
        import viz2psy.video as video
        from viz2psy.exceptions import VideoError

        monkeypatch.setattr(video, "is_interlaced", lambda *a, **k: False)
        with pytest.raises(VideoError, match="bit-identical"):
            video.extract_frames(videos["interlaced"], frame_interval=0.1, quiet=True)

    def test_moving_clip_is_not_frozen(self, videos):
        from viz2psy.video import extract_frames

        assert len(extract_frames(videos["progressive"], frame_interval=0.1, quiet=True)) >= 10

    def test_motion_on_interlaced_sees_rightward_motion(self, videos):
        from viz2psy.models.motion import MotionModel

        m = MotionModel(device="cpu")
        m.load()
        rows = m.predict_video(videos["interlaced"], [0.4, 0.8], quiet=True)
        for row in rows:
            assert row["motion_energy"] > 0.001
            assert row["motion_horizontal"] > 0
