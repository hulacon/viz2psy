"""experiential: weights resolution and validation, the head, and the trainer's plumbing.

No backbone is loaded: the forward pass is context_affect's (open_clip), and the
probe is a linear map whose weights are built locally. The trainer's end-to-end
fit needs word2psy (the ratings loader and the probe fit); it skips without it.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from viz2psy.models import experiential as ex

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "train_experiential.py"


def _trainer():
    spec = importlib.util.spec_from_file_location("train_experiential", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write(path, cols=("experiential_vision", "experiential_pain"), **override):
    k = len(cols)
    fields = dict(W=np.arange(ex.EMBED_DIM * k, dtype=float).reshape(ex.EMBED_DIM, k) * 1e-4,
                  b=np.arange(k, dtype=float), alpha=np.full(k, 10.0), columns=np.array(cols),
                  backbone=np.array(ex.BACKBONE), format_version=np.array(ex.WEIGHTS_FORMAT),
                  trained_on=np.array(json.dumps({"ratings_sha256": "x"})))
    fields.update(override)
    np.savez(path, **fields)
    return path


def test_columns_cover_the_65_attributes_in_sheet_order():
    assert len(ex.FEATURE_NAMES) == 65 and ex.FEATURE_NAMES[0] == "experiential_vision"
    assert "experiential_upperlimb" in ex.FEATURE_NAMES


def test_attribute_table_matches_word2psy():
    binder = pytest.importorskip("word2psy.norms.binder")
    assert ex.ATTRIBUTES == binder.ATTRIBUTES and list(ex.ATTRIBUTES) == list(binder.ATTRIBUTES)


def test_weights_path_follows_the_env(monkeypatch, tmp_path):
    monkeypatch.setenv(ex.WEIGHTS_ENV, str(tmp_path / "w.npz"))
    assert ex.weights_path() == tmp_path / "w.npz"
    monkeypatch.delenv(ex.WEIGHTS_ENV)
    assert ex.weights_path().name == ex.WEIGHTS_NAME and ex.weights_path().parent.name == "viz2psy"


def test_missing_weights_fail_loudly_and_name_the_fix(tmp_path):
    with pytest.raises(FileNotFoundError, match=ex.WEIGHTS_ENV) as e:
        ex.read_weights(tmp_path / "absent.npz")
    assert "train_experiential.py" in str(e.value)


def test_load_fails_before_touching_the_backbone(monkeypatch, tmp_path):
    monkeypatch.setenv(ex.WEIGHTS_ENV, str(tmp_path / "absent.npz"))
    m = ex.ExperientialModel(device="cpu")
    monkeypatch.setattr(m, "load_backbone", lambda: pytest.fail("backbone loaded first"))
    with pytest.raises(FileNotFoundError):
        m.load()


@pytest.mark.parametrize("override, match", [
    ({"backbone": np.array("ViT-B-32/laion2b_s34b_b79k")}, "trained on"),
    ({"format_version": np.array("0")}, "format"),
    ({"columns": np.array(["experiential_vision", "experiential_telepathy"])}, "bad columns"),
    ({"columns": np.array(["experiential_vision", "experiential_vision"])}, "bad columns"),
    ({"W": np.zeros((512, 2))}, "do not fit"),
])
def test_mismatched_weights_are_refused(tmp_path, override, match):
    with pytest.raises(ValueError, match=match):
        ex.read_weights(_write(tmp_path / "w.npz", **override))


def test_model_emits_exactly_the_weights_columns(monkeypatch, tmp_path):
    p = _write(tmp_path / "w.npz", cols=("experiential_pain", "experiential_scene", "experiential_sad"))
    monkeypatch.setenv(ex.WEIGHTS_ENV, str(p))
    m = ex.ExperientialModel(device="cpu")
    monkeypatch.setattr(m, "load_backbone", lambda: None)
    m.load()
    emb = np.random.default_rng(0).standard_normal((2, ex.EMBED_DIM)).astype(np.float32)
    monkeypatch.setattr(m, "embed_batch", lambda images: emb)
    rows = m.predict_batch([None, None])
    w = ex.read_weights(p)
    assert list(rows[0]) == ["experiential_pain", "experiential_scene", "experiential_sad"]
    np.testing.assert_allclose(list(rows[1].values()), emb[1].astype(np.float64) @ w["W"] + w["b"])


def test_sidecar_records_the_weights(monkeypatch, tmp_path):
    p = _write(tmp_path / "w.npz")
    monkeypatch.setenv(ex.WEIGHTS_ENV, str(p))
    rec = ex.ExperientialModel.describe_preprocessing()
    assert rec["square"].startswith("squash")
    assert rec["probe"]["weights_sha256"] == ex.sha256(p)
    assert rec["probe"]["columns"] == ["experiential_vision", "experiential_pain"]
    assert rec["probe"]["trained_on"] == {"ratings_sha256": "x"}
    monkeypatch.setenv(ex.WEIGHTS_ENV, str(tmp_path / "absent.npz"))
    assert ex.ExperientialModel.describe_preprocessing()["probe"]["weights_file"] is None


def test_labels_are_a_nameable_probe_on_the_vitl14_backbone():
    assert ex.ExperientialModel.labels["default"] == {
        "interpretability": "nameable", "derivation": "probe", "derived_from": ex.BACKBONE}
    assert set(ex.ExperientialModel.labels["columns"]) == set(ex.FEATURE_NAMES)


class TestTrainer:
    def test_password_is_the_last_marked_line(self, tmp_path):
        t = _trainer()
        p = tmp_path / "pw.txt"
        p.write_text("licence text\nmore text\nPassword for images_THINGS.zip: s3cret \n")
        assert t.read_password(p) == b"s3cret"
        p.write_text("no password here\n")
        with pytest.raises(SystemExit):
            t.read_password(p)

    def test_members_match_rated_words_only(self):
        t = _trainer()
        names = ["object_images/", "object_images/hot_dog/", "object_images/hot_dog/hot_dog_01b.jpg",
                 "object_images/apple/apple_02s.jpg", "object_images/zebra/zebra_01b.jpg",
                 "object_images/apple/notes.txt", "other/apple/apple_01b.jpg"]
        assert t.matched_members(names, {"hot dog", "apple"}) == [
            "object_images/apple/apple_02s.jpg", "object_images/hot_dog/hot_dog_01b.jpg"]

    def test_fit_writes_weights_the_model_accepts(self, tmp_path):
        pytest.importorskip("word2psy.norms.experiential")
        from word2psy.norms import binder
        import pandas as pd

        words = [f"concept{chr(97 + i)}" for i in range(20)]
        recs = []
        for i, w in enumerate(words):
            rec = {"No": i + 1, "Word": w, "WC": 1, "N": 30, "Mean R": 0.8, "Super Category": "A"}
            rec.update({a: (i + j) % 7 for j, a in enumerate(binder.ATTRIBUTES)})
            recs.append(rec)
        ratings = tmp_path / "WordSet1_Ratings.xlsx"
        pd.DataFrame(recs).to_excel(ratings, index=False)
        rng = np.random.default_rng(0)
        concept = np.repeat(words, 5)
        emb = rng.standard_normal((len(concept), ex.EMBED_DIM)).astype(np.float32)
        embf = tmp_path / "emb.npz"
        np.savez(embf, emb=emb, image=np.array([f"{c}/{c}_{k}.jpg" for k, c in enumerate(concept)]),
                 concept=concept)
        keep = tmp_path / "keep.txt"
        keep.write_text("pain\n# a comment\nexperiential_vision\n")
        out = tmp_path / "w.weights"
        assert _trainer().main(["fit", "--ratings", str(ratings), "--embeddings", str(embf),
                                "--attributes", str(keep), "--write-weights", str(out)]) == 0
        w = ex.read_weights(out)
        assert w["columns"] == ("experiential_vision", "experiential_pain")
        meta = json.loads(str(w["trained_on"]))
        assert meta["n_images"] == 100 and meta["n_concepts"] == 20
        assert meta["ratings_sha256"] == binder.file_sha256(ratings)


def test_vitl14_shares_the_probe_backbone_and_emits_the_raw_embedding(monkeypatch):
    from viz2psy.models import context_affect as ca
    from viz2psy.models import vitl14

    assert vitl14.ViTL14Model.checkpoint == ex.BACKBONE == ca.BACKBONE
    assert len(vitl14.FEATURE_NAMES) == ex.EMBED_DIM and vitl14.FEATURE_NAMES[0] == "vitl14_000"
    m = vitl14.ViTL14Model(device="cpu")
    emb = np.random.default_rng(1).standard_normal((1, ex.EMBED_DIM)).astype(np.float32)
    monkeypatch.setattr(m, "embed_batch", lambda images: emb)
    row = m.predict_batch([None])[0]
    np.testing.assert_array_equal(np.array(list(row.values())), emb[0].astype(np.float64))
    assert vitl14.ViTL14Model.describe_preprocessing()["square"].startswith("squash")
