"""viz.sidecar.load_sidecar: absent and unreadable are different outcomes."""

import pytest

from viz2psy.viz.sidecar import load_sidecar


def test_absent_sidecar_is_none(tmp_path):
    assert load_sidecar(tmp_path / "scores.csv") is None


def test_corrupt_sidecar_warns(tmp_path):
    # A truncated .meta.json was indistinguishable from no sidecar at all: the
    # dashboard silently lost feature labels and index metadata.
    (tmp_path / "scores.meta.json").write_text('{"models": {')
    with pytest.warns(UserWarning, match="scores.meta.json"):
        assert load_sidecar(tmp_path / "scores.csv") is None
