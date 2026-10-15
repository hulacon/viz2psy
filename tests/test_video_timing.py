"""Frame sampling: the `time` column must name the frame that was scored."""

import math
from fractions import Fraction

import numpy as np
import pytest

from viz2psy.video import extract_frames, frame_index, sample_times

from test_motion import square_frame, write_video


@pytest.mark.parametrize("fps, fps_exact", [(24, Fraction(24)), (25, Fraction(25)),
                                            (30, Fraction(30)),
                                            (30000 / 1001, Fraction(30000, 1001))])
@pytest.mark.parametrize("interval, interval_exact", [(0.1, Fraction(1, 10)),
                                                      (0.04, Fraction(1, 25)),
                                                      (0.5, Fraction(1, 2))])
def test_sampled_frame_is_the_one_on_screen(fps, fps_exact, interval, interval_exact):
    # Accumulating `t += interval` and truncating `int(t * fps)` read the
    # frame before the right one at about half the samples (0.1 s at 30 fps:
    # sample 10 is t=0.9999999999999999 -> frame 29, not 30).
    times = sample_times(100.0, interval)
    assert len(times) == int(100 / interval_exact) + 1
    for i, t in enumerate(times):
        assert t == pytest.approx(float(i * interval_exact), abs=1e-12)
        assert frame_index(t, fps) == math.floor(i * interval_exact * fps_exact)


def test_extract_frames_reads_the_frame_at_each_time(tmp_path):
    # Frame k carries a square at x = 5k; decode it back from each sample.
    fps = 25
    video = write_video(tmp_path / "v.mp4", [square_frame(5 * k, 100) for k in range(50)], fps=fps)
    frames = extract_frames(video, frame_interval=0.1, quiet=True)
    for t, img in frames:
        x = int(np.argmax(np.asarray(img)[120, :, 0] > 128))
        assert round(x / 5) == math.floor(round(t * 10) * 2.5), t
