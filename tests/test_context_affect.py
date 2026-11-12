"""context_affect: weights resolution, validation, the folded head, and the trainer's maths.

No backbone is loaded here: the CLIP ViT-L-14 forward pass is open_clip's, and
the probe is a linear map whose weights are built locally (not shipped).
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from viz2psy.models import context_affect as ca

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_context_affect.py"


def _trainer():
    spec = importlib.util.spec_from_file_location("train_context_affect", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(path, **override):
    fields = dict(W=np.zeros((ca.EMBED_DIM, 2)), b=np.zeros(2), lag_sec=np.array([1.0, 1.5]),
                  alpha=np.array([10.0, 100.0]), columns=np.array(ca.FEATURE_NAMES),
                  backbone=np.array(ca.BACKBONE), format_version=np.array(ca.WEIGHTS_FORMAT),
                  trained_on=np.array(json.dumps({"dataset": "VEATIC"})))
    fields.update(override)
    np.savez(path, **fields)
    return path


def test_weights_path_follows_the_env(monkeypatch, tmp_path):
    monkeypatch.setenv(ca.WEIGHTS_ENV, str(tmp_path / "w.npz"))
    assert ca.weights_path() == tmp_path / "w.npz"
    monkeypatch.delenv(ca.WEIGHTS_ENV)
    assert ca.weights_path().name == ca.WEIGHTS_NAME
    assert ca.weights_path().parent.name == "viz2psy"


def test_missing_weights_fail_loudly_and_name_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match=ca.WEIGHTS_ENV) as e:
        ca.read_weights(tmp_path / "absent.npz")
    assert "train_context_affect.py" in str(e.value)


def test_load_fails_before_touching_the_backbone(monkeypatch, tmp_path):
    monkeypatch.setenv(ca.WEIGHTS_ENV, str(tmp_path / "absent.npz"))
    with pytest.raises(FileNotFoundError):
        ca.ContextAffectModel(device="cpu").load()


@pytest.mark.parametrize("override, match", [
    ({"backbone": np.array("ViT-B-32/laion2b_s34b_b79k")}, "trained on"),
    ({"format_version": np.array("0")}, "format"),
    ({"columns": np.array(["a", "b"])}, "columns"),
    ({"W": np.zeros((512, 2))}, "do not fit"),
])
def test_mismatched_weights_are_refused(tmp_path, override, match):
    with pytest.raises(ValueError, match=match):
        ca.read_weights(_write(tmp_path / "w.npz", **override))


def test_sidecar_records_the_weights(monkeypatch, tmp_path):
    p = _write(tmp_path / "w.npz")
    monkeypatch.setenv(ca.WEIGHTS_ENV, str(p))
    rec = ca.ContextAffectModel.describe_preprocessing()
    assert rec["square"].startswith("squash")
    assert rec["probe"]["weights_sha256"] == ca.sha256(p)
    assert rec["probe"]["target_lag_sec"] == {"context_affect_valence": 1.0,
                                              "context_affect_arousal": 1.5}
    monkeypatch.setenv(ca.WEIGHTS_ENV, str(tmp_path / "absent.npz"))
    assert ca.ContextAffectModel.describe_preprocessing()["probe"]["weights_file"] is None


def test_folded_head_equals_the_standardised_ridge():
    t = _trainer()
    rng = np.random.default_rng(0)
    X = rng.normal(3.0, 2.0, (200, 12)) * rng.uniform(0.1, 5, 12)
    Y = X[:, :2] @ np.array([[1.0], [-2.0]]) + rng.normal(0, 0.1, (200, 1))
    r = t.Ridge(X, Y)
    W, b = r.folded(5.0)
    np.testing.assert_allclose(ca.apply_head(X, W, b), r.predict(X, 5.0), atol=1e-10)


def test_ccc_is_one_only_for_identity_and_penalises_bias():
    t = _trainer()
    x = np.linspace(-1, 1, 50)
    assert t.ccc(x, x) == pytest.approx(1.0)
    assert t.ccc(x, x + 0.5) < 1.0 and t.ccc(x, 2 * x) < 1.0


def test_group_folds_keep_each_group_whole():
    t = _trainer()
    groups = np.repeat(np.arange(23), np.arange(1, 24))
    folds = t.group_folds(groups, 5, seed=1)
    assert np.array_equal(np.sum(folds, axis=0), np.ones(len(groups)))
    for f in folds:
        assert not set(groups[f]) & set(groups[~f])
