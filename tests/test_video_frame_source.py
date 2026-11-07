"""Video features are scored on the decoded frames, never on a lossy copy.

Before 0.15.0 the CLI scored JPEG q85 copies whenever frames went to disk:
on ``--save-frames`` (the default unless ``--no-save-frames``), on
``--parallel`` multi-model runs, and silently when the estimated frame memory
exceeded 80 % of available. The same video and model then emitted features
differing at r ~ .95, depending on the flags and on node memory, and the
sidecar did not say which.
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from viz2psy import cli
from viz2psy.video import extract_frames

from test_motion import H, W, write_video

INTERVAL = 0.25


@pytest.fixture
def texture_models(monkeypatch):
    monkeypatch.setitem(cli.MODEL_REGISTRY, "texture", ("fake_model", "TextureFakeModel", "test"))
    monkeypatch.setitem(cli.MODEL_REGISTRY, "otherfake", ("fake_model", "OtherFakeModel", "test"))


@pytest.fixture
def noisy_video(tmp_path):
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 256, (H, W, 3), dtype=np.uint8) for _ in range(24)]
    return write_video(tmp_path / "noise.mp4", frames, fps=12)


def _reference(video):
    """The statistic on frames decoded straight into memory."""
    return [float(np.abs(np.diff(np.asarray(img, dtype=float), axis=1)).mean())
            for _, img in extract_frames(video, frame_interval=INTERVAL, quiet=True)]


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["viz2psy", *map(str, argv)])
    cli.main()


ROUTES = {
    "in_memory": ([], False),
    "save_frames_jpg_default": (None, False),  # no flag: frames saved as jpg by default
    "save_frames_png": (["--frame-format", "png"], False),
    "memory_pressure": (["--no-save-frames"], True),
    "memory_pressure_save_jpg": ([], True),
    "parallel": (["--parallel", "--no-save-frames"], False),
    "parallel_save_jpg": (["--parallel"], False),
}


@pytest.mark.parametrize("route", list(ROUTES))
def test_every_route_scores_the_decoded_frames(route, tmp_path, monkeypatch,
                                               texture_models, noisy_video):
    extra, squeeze = ROUTES[route]
    if extra == []:
        extra = ["--no-save-frames"]
    elif extra is None:
        extra = []
    if squeeze:
        import viz2psy.video as video

        monkeypatch.setattr(video, "get_available_memory", lambda: 0)
    models = ["texture", "otherfake"] if "parallel" in route else ["texture"]
    out = tmp_path / "out" / "feat.csv"
    _run(monkeypatch, *models, noisy_video, "-o", out, "--frame-interval", INTERVAL,
         "--no-viz", "-q", *extra)

    got = pd.read_csv(out)["texture_grad"].to_numpy()
    np.testing.assert_allclose(got, _reference(noisy_video), rtol=1e-5)

    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["schema_version"] == "1.3"
    framing = meta["input"]["framing"]
    assert framing["frame_source"] == "raw decode"
    assert framing["frame_cache"] in {"memory", "png"}
    if squeeze or "parallel" in route:
        assert framing["frame_cache"] == "png"


def test_saved_frames_are_viewing_copies_in_the_requested_format(tmp_path, monkeypatch,
                                                                  texture_models, noisy_video):
    out = tmp_path / "feat.csv"
    _run(monkeypatch, "texture", noisy_video, "-o", out, "--frame-interval", INTERVAL,
         "--no-viz", "-q")
    saved = sorted((tmp_path / "feat_frames").iterdir())
    assert saved and all(p.suffix == ".jpg" for p in saved)
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["input"]["saved_frames_format"] == "jpg"


def test_no_temp_frames_left_behind(tmp_path, monkeypatch, texture_models, noisy_video):
    import tempfile

    import viz2psy.video as video

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    (tmp_path / "tmp").mkdir()
    monkeypatch.setattr(video, "get_available_memory", lambda: 0)
    _run(monkeypatch, "texture", noisy_video, "-o", tmp_path / "feat.csv",
         "--frame-interval", INTERVAL, "--no-viz", "-q", "--no-save-frames")
    assert list((tmp_path / "tmp").iterdir()) == []


def test_refresh_does_not_claim_framing_it_cannot_know(tmp_path):
    """A pre-1.3 video sidecar has no record of its frame source: refresh stops at 1.2."""
    from viz2psy.metadata import MetadataBuilder
    from viz2psy.sidecar import refresh_sidecar

    csv = tmp_path / "scores.csv"
    pd.DataFrame({"time": [0.0], "llstat_luminance_mean": [0.2]}).to_csv(csv, index=False)
    b = MetadataBuilder()
    b.set_input_video(csv, 0.5, 1)
    b.set_output(csv, 1, 2)
    b.add_model("llstat", ["llstat_luminance_mean"], 1.0)
    meta = b.build()
    meta["schema_version"] = "1.1"
    del meta["input"]["framing"]
    side = tmp_path / "scores.meta.json"
    side.write_text(json.dumps(meta))

    assert refresh_sidecar(side).status == "refreshed"
    new = json.loads(side.read_text())
    assert new["schema_version"] == "1.2"
    assert "framing" not in new["input"]
