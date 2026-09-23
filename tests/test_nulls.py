"""Contract B §4.1 (schema 1.1) producer duty: every NaN viz2psy can emit is declared.

A fixture per declared condition (no face, one face, a static video pair, a
timestamp past the video's end), each asserting both directions: the NaN
columns are a subset of the declared ones, and every declared key is a real
column of the model.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest

from viz2psy.cli import MODEL_REGISTRY
from viz2psy.metadata import SCHEMA_VERSION, MetadataBuilder, declared_nulls
from viz2psy.models.faces import FEATURE_NAMES as FACE_COLS, face_stats
from viz2psy.models.motion import FEATURE_NAMES as MOTION_COLS
from viz2psy.sidecar import refresh_sidecar

from test_motion import model as motion_model  # noqa: F401  (module-scoped fixture)
from test_motion import square_frame, write_video

KINDS = {"undefined", "undefinable", "missing"}


def _nan_keys(row: dict) -> set[str]:
    return {k for k, v in row.items() if isinstance(v, float) and math.isnan(v)}


@pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
def test_every_model_declares_well_formed_nulls(model):
    # `nulls` lives on the model class, so reading it imports the model's
    # backend. CI installs only a subset (see ci.yml "Model registry is
    # importable"): skip a missing third-party backend, but fail a missing
    # viz2psy module -- that is a broken registry entry, not a missing dep.
    try:
        nulls = declared_nulls(model)
    except ImportError as e:
        if (getattr(e, "name", "") or "").startswith("viz2psy"):
            raise
        pytest.skip(f"{model}: backend not installed ({e})")
    for col, entry in nulls.items():
        assert set(entry) == {"means", "when"}, col
        assert entry["means"] in KINDS, col
        assert entry["when"].strip(), col


@pytest.mark.parametrize("boxes, expect", [
    (np.empty((0, 4)), {"faces_center_dist", "faces_mutual_dist"}),  # no face
    (np.array([[90, 40, 20, 20]]), {"faces_mutual_dist"}),           # one face
    (np.array([[40, 40, 20, 20], [140, 40, 20, 20]]), set()),        # two faces
])
def test_faces_nulls_are_declared(boxes, expect):
    nan = _nan_keys(face_stats(boxes, 200, 100))
    assert nan == expect
    assert nan <= set(declared_nulls("faces")) <= set(FACE_COLS)


def test_motion_static_pair_and_video_end_are_declared(motion_model, tmp_path):  # noqa: F811
    video = write_video(tmp_path / "static.mp4", [square_frame(100, 100)] * 4)
    static, past_end = motion_model.predict_video(video, [0.0, 10.0], quiet=True)
    declared = declared_nulls("motion")
    assert set(declared) == set(MOTION_COLS)
    assert _nan_keys(static) == {"motion_coherence"}
    assert declared["motion_coherence"]["means"] == "undefined"
    assert _nan_keys(past_end) == set(MOTION_COLS)
    assert {declared[c]["means"] for c in MOTION_COLS if c != "motion_coherence"} == {"undefinable"}


def test_sidecar_is_1_1_with_every_model_carrying_nulls(tmp_path):
    b = MetadataBuilder()
    b.add_model("faces", FACE_COLS, 1.0)
    b.add_model("llstat", ["llstat_luminance_mean"], 1.0)
    meta = b.build()
    assert meta["schema_version"] == SCHEMA_VERSION == "1.1"
    assert set(meta["models"]["faces"]["nulls"]) == {"faces_center_dist", "faces_mutual_dist"}
    assert meta["models"]["llstat"]["nulls"] == {}


class TestRefresh:
    def _family_1_0(self, tmp_path, faces_center_dist):
        csv = tmp_path / "scores.csv"
        pd.DataFrame({"filename": ["a.jpg", "b.jpg"], "faces_count": [0.0, 1.0],
                      "faces_center_dist": faces_center_dist,
                      "llstat_luminance_mean": [0.2, 0.3]}).to_csv(csv, index=False)
        b = MetadataBuilder()
        b.set_output(csv, 2, 4)
        b.add_model("faces", ["faces_count", "faces_center_dist"], 1.0)
        b.add_model("llstat", ["llstat_luminance_mean"], 1.0)
        meta = b.build()
        meta["schema_version"] = "1.0"
        for e in meta["models"].values():
            e.pop("nulls")
        side = tmp_path / "scores.meta.json"
        side.write_text(json.dumps(meta, indent=2))
        return csv, side

    def test_refresh_adds_nulls_and_never_touches_the_csv(self, tmp_path):
        csv, side = self._family_1_0(tmp_path, [np.nan, 0.1])
        before = csv.read_bytes()
        assert refresh_sidecar(side).status == "refreshed"
        meta = json.loads(side.read_text())
        assert meta["schema_version"] == "1.1"
        assert set(meta["models"]["faces"]["nulls"]) == {"faces_center_dist"}
        assert meta["models"]["llstat"]["nulls"] == {}
        assert csv.read_bytes() == before
        assert refresh_sidecar(side).status == "unchanged"

    def test_refresh_refuses_an_undeclared_nan(self, tmp_path):
        csv, side = self._family_1_0(tmp_path, [0.2, 0.1])
        df = pd.read_csv(csv)
        df.loc[0, "llstat_luminance_mean"] = np.nan
        df.to_csv(csv, index=False)
        before = side.read_text()
        r = refresh_sidecar(side)
        assert r.status == "refused" and r.undeclared == {"llstat": ["llstat_luminance_mean"]}
        assert side.read_text() == before
