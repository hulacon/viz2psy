"""Several videos in one call (``input.type: "video_set"``).

A set must score every video exactly as one call per video would (same stage
0, same frame grid, same features), load each model once, and record the
shared framing rule once with the per-file values under ``input.videos``.
Synthetic and offline: lossless FFV1 so a crop is exact.
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

pytest.importorskip("av")

from viz2psy import cli, framing  # noqa: E402

from fake_model import FakeModel  # noqa: E402
from test_framing_reader import H, W, letterboxed, table_for, write_ffv1  # noqa: E402

FEATURES = ["fake_mean", "texture_grad"]


class CountingFake(FakeModel):
    name = "counting"
    loads = 0

    def load(self):
        type(self).loads += 1
        self.model = None

    def predict(self, image):
        return {"counting_sum": float(np.asarray(image, dtype=float).sum())}


@pytest.fixture
def fake_models(monkeypatch):
    for name, cls in (("fake", "FakeModel"), ("texture", "TextureFakeModel")):
        monkeypatch.setitem(cli.MODEL_REGISTRY, name, ("fake_model", cls, "test"))
    monkeypatch.setitem(cli.MODEL_REGISTRY, "counting", ("test_video_set", "CountingFake", "test"))


@pytest.fixture(scope="module")
def clips(tmp_path_factory):
    d = tmp_path_factory.mktemp("set")
    a = write_ffv1(d / "clip_a.mkv", letterboxed(n=30))
    b = write_ffv1(d / "clip_b.mkv", letterboxed(n=17, top=0, bottom=0))  # no bars
    c = write_ffv1(d / "clip_c.mkv", letterboxed(n=23, top=20, bottom=20))
    table = table_for([a, b, c], d / "framing_table.csv")
    return {"videos": [a, b, c], "table": table, "dir": d}


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["viz2psy", *map(str, argv)])
    cli.main()


def test_set_matches_one_call_per_video(clips, fake_models, monkeypatch, tmp_path):
    out = tmp_path / "set.csv"
    _run(monkeypatch, "fake", "texture", *clips["videos"], "--framing-table", clips["table"],
         "--id-pattern", "ext-test-{stem}", "--no-viz", "--quiet", "-o", out)
    got = pd.read_csv(out)
    assert list(got.columns[:2]) == ["stimulus_id", "time"]

    want = []
    for v in clips["videos"]:
        one = tmp_path / f"{v.stem}.csv"
        _run(monkeypatch, "fake", "texture", v, "--framing-table", clips["table"],
             "--stimulus-id", f"ext-test-{v.stem}", "--no-save-frames", "--no-viz", "--quiet",
             "-o", one)
        want.append(pd.read_csv(one))
    want = pd.concat(want, ignore_index=True)
    pd.testing.assert_frame_equal(got[["stimulus_id", "time", *FEATURES]],
                                  want[["stimulus_id", "time", *FEATURES]])
    # The bars were cropped: clip_c's picture is not clip_b's size.
    assert got["stimulus_id"].tolist()[0] == "ext-test-clip_a"


def test_set_sidecar_records_rule_once_and_files_each(clips, fake_models, monkeypatch, tmp_path):
    out = tmp_path / "set.csv"
    _run(monkeypatch, "fake", *clips["videos"], "--framing-table", clips["table"],
         "--no-viz", "--quiet", "-o", out)
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    inp = meta["input"]
    assert inp["type"] == "video_set" and inp["count"] == 3
    fr = inp["framing"]
    params = framing.read_table(clips["table"])[1]
    assert fr["frame_source"] == "raw decode" and fr["stage0"] == "framing table"
    assert fr["framing_table_sha256"] == params["sha256"] and fr["frame_cache"] == "memory"
    assert set(fr) == {"frame_source", "frame_cache", *framing.RULE_RECORD_KEYS}
    crops = [v["crop"]["top"] for v in inp["videos"]]
    assert crops == [12, 0, 20]
    assert [v["stimulus_id"] for v in inp["videos"]] == ["clip_a", "clip_b", "clip_c"]
    rows = pd.read_csv(out)
    assert inp["n_frames"] == len(rows) == sum(v["n_frames"] for v in inp["videos"])
    assert rows.groupby("stimulus_id", sort=False).size().tolist() == \
        [v["n_frames"] for v in inp["videos"]]


def test_each_model_loads_once(clips, fake_models, monkeypatch, tmp_path):
    CountingFake.loads = 0
    _run(monkeypatch, "counting", *clips["videos"], "--no-viz", "--quiet",
         "-o", tmp_path / "c.csv")
    assert CountingFake.loads == 1


def test_motion_in_a_set_matches_one_call_per_video(clips, monkeypatch, tmp_path):
    out = tmp_path / "set.csv"
    _run(monkeypatch, "motion", *clips["videos"][:2], "--framing-table", clips["table"],
         "--no-viz", "--quiet", "-o", out)
    got = pd.read_csv(out)
    for v in clips["videos"][:2]:
        one = tmp_path / f"{v.stem}.csv"
        _run(monkeypatch, "motion", v, "--framing-table", clips["table"],
             "--no-save-frames", "--no-viz", "--quiet", "-o", one)
        want = pd.read_csv(one)
        part = got[got["stimulus_id"] == v.stem].reset_index(drop=True)
        cols = [c for c in want.columns if c.startswith("motion")]
        assert cols
        pd.testing.assert_frame_equal(part[["time", *cols]], want[["time", *cols]])


def test_a_video_missing_from_the_table_fails_before_any_model(clips, fake_models,
                                                                monkeypatch, tmp_path):
    stray = write_ffv1(tmp_path / "stray.mkv", letterboxed(n=10))
    CountingFake.loads = 0
    with pytest.raises(framing.FramingError, match="not in the framing table"):
        _run(monkeypatch, "counting", clips["videos"][0], stray, "--framing-table", clips["table"],
             "--no-viz", "--quiet", "-o", tmp_path / "x.csv")
    assert CountingFake.loads == 0


@pytest.mark.parametrize("extra, message", [
    (["--stimulus-id", "x"], "--id-pattern"),
    (["--save-frames", "frames"], "--save-frames"),
    (["--parallel"], "--parallel"),
    (["--id-pattern", "{idx}"], "{stem} only"),
])
def test_refused_flags(clips, fake_models, monkeypatch, tmp_path, capsys, extra, message):
    with pytest.raises(SystemExit):
        _run(monkeypatch, "fake", *clips["videos"][:2], *extra, "--no-viz", "--quiet",
             "-o", tmp_path / "x.csv")
    assert message in capsys.readouterr().err


def test_mixed_videos_and_images_refused(clips, fake_models, monkeypatch, tmp_path, capsys):
    img = tmp_path / "a.png"
    Image.fromarray(np.zeros((H, W, 3), np.uint8)).save(img)
    with pytest.raises(SystemExit):
        _run(monkeypatch, "fake", clips["videos"][0], img, "--no-viz", "--quiet",
             "-o", tmp_path / "x.csv")
    assert "not both" in capsys.readouterr().err


def test_duplicate_ids_refused(clips, fake_models, monkeypatch, tmp_path, capsys):
    other = tmp_path / "again"
    other.mkdir()
    twin = write_ffv1(other / clips["videos"][0].name, letterboxed(n=10))
    with pytest.raises(SystemExit):
        _run(monkeypatch, "fake", clips["videos"][0], twin, "--no-viz", "--quiet",
             "-o", tmp_path / "x.csv")
    assert "share a stimulus_id" in capsys.readouterr().err
