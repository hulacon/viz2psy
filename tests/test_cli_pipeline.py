"""End-to-end CLI runs with weightless models (no checkpoints, no GPU)."""

import json
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from viz2psy import cli


@pytest.fixture
def fake_models(monkeypatch):
    monkeypatch.setitem(cli.MODEL_REGISTRY, "fake", ("fake_model", "FakeModel", "test"))
    monkeypatch.setitem(cli.MODEL_REGISTRY, "otherfake", ("fake_model", "OtherFakeModel", "test"))


def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["viz2psy", *map(str, argv)])
    cli.main()


def _img(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.full((8, 8, 3), value, np.uint8)).save(path)
    return path


def test_duplicate_basenames_keep_one_row_per_input(tmp_path, monkeypatch, fake_models):
    # Two inputs that share a basename (sub-##/ses-## layouts do this). The
    # sequential path merged models on `filename`, which is a cartesian join on
    # a duplicated key: 2 inputs x 2 models came out as 4 rows.
    a = _img(tmp_path / "sub-01" / "stim.png", 10)
    b = _img(tmp_path / "sub-02" / "stim.png", 200)
    out = tmp_path / "out.csv"
    _run(monkeypatch, "fake", "otherfake", a, b, "-o", out, "--no-viz", "-q")

    df = pd.read_csv(out)
    assert len(df) == 2
    assert df["fake_mean"].tolist() == [10.0, 200.0]
    assert df["otherfake_max"].tolist() == [10.0, 200.0]


def test_hdf5_without_output_writes_csv_and_sidecar(tmp_path, monkeypatch, fake_models):
    # HDF5 is the one input with a default output path, but the metadata
    # builder was only created when `-o` was given, so the run scored every
    # image, wrote the CSV, then died on `None.set_output` with no sidecar.
    h5py = pytest.importorskip("h5py")
    brick = tmp_path / "brick.h5"
    with h5py.File(brick, "w") as f:
        f["images"] = np.stack([np.full((8, 8, 3), v, np.uint8) for v in (0, 50, 100)])

    _run(monkeypatch, "fake", brick, "--no-viz", "-q")

    out = tmp_path / "brick_scores.csv"
    assert pd.read_csv(out)["fake_mean"].tolist() == [0.0, 50.0, 100.0]
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["index_column"] == "image_idx"
    assert meta["output"]["rows"] == 3


def test_missing_plotly_does_not_crash_default_run(tmp_path, monkeypatch, fake_models):
    # The dashboard is on by default but plotly is only in the `interactive`
    # extra. Without it the run wrote the CSV and sidecar, then raised
    # ModuleNotFoundError out of main() as a traceback.
    monkeypatch.setitem(sys.modules, "plotly", None)
    monkeypatch.setitem(sys.modules, "plotly.graph_objects", None)
    monkeypatch.delitem(sys.modules, "viz2psy.viz.dashboard", raising=False)
    a = _img(tmp_path / "a.png", 10)
    out = tmp_path / "out.csv"

    _run(monkeypatch, "fake", a, "-o", out, "-q")

    assert out.exists() and out.with_suffix(".meta.json").exists()
