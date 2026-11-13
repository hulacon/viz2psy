"""Contract B §4.1 (schema 1.2) producer duty: every column carries labels.

For every registered model, each column of its inventory resolves (default,
then a key-wise ``columns`` override) to ``interpretability`` and
``derivation`` from the vocabularies, every ``columns`` key is a real column,
and ``derived_from`` is present exactly when the derivation names a parent
(formula-on-model, probe, zero-shot). The sidecar writer and
``viz2psy sidecar refresh`` stamp the labels into each model entry.
"""

import json
import re

import numpy as np
import pandas as pd
import pytest

from viz2psy.cli import MODEL_REGISTRY, _load_model_class
from viz2psy.labels import DERIVATION, INTERPRETABILITY, NEEDS_PARENT, problems, resolve
from viz2psy.metadata import SCHEMA_VERSION, MetadataBuilder, get_feature_info
from viz2psy.sidecar import refresh_sidecar


def _numbered(prefix: str, n: int, width: int = 3) -> list[str]:
    return [f"{prefix}_{i:0{width}d}" for i in range(n)]


def _llstat_columns() -> list[str]:
    from PIL import Image

    from viz2psy.models.llstat import LLStatModel

    img = np.full((16, 16, 3), 128, np.uint8)
    return list(LLStatModel(device="cpu").predict(Image.fromarray(img)))


def _places_columns() -> list[str]:
    # Scene names are downloaded at load(); a few real ones stand in for the
    # 365 (all take the default). The SUN attributes are the exact 102.
    from viz2psy.models.places import SUN_ATTRIBUTES

    scenes = ["abbey", "church_indoor", "church_outdoor", "swimming_pool_outdoor"]
    return [f"places_{s}" for s in scenes] + [f"sunattr_{a}" for a in SUN_ATTRIBUTES]


def _yolo_columns() -> list[str]:
    from viz2psy.models.yolo import _COCO_NAMES

    return [f"yolo_{n}" for n in _COCO_NAMES] + [
        "yolo_object_count", "yolo_category_count", "yolo_object_coverage",
        "yolo_largest_object_ratio", "yolo_mean_confidence",
    ]


def _module_attr(module: str, attr: str):
    def get():
        import importlib

        value = getattr(importlib.import_module(f"viz2psy.models.{module}"), attr)
        return list(value() if callable(value) else value)
    return get


#: Every column each model emits under the CLI's defaults, built without
#: loading weights. A new model must add its inventory here.
INVENTORY = {
    "resmem": lambda: ["resmem_memorability"],
    "emonet": _module_attr("emonet", "EMONET_COLUMNS"),
    "clip": lambda: _numbered("clip", 512),
    "ebind": lambda: _numbered("ebind", 1024, width=4),
    "caption": lambda: ["caption_text"],
    "caption_vlm": lambda: ["caption_vlm_text"],
    "gist": lambda: _numbered("gist", 512),
    "llstat": _llstat_columns,
    "saliency": lambda: [f"saliency_{x:02d}_{y:02d}" for y in range(24) for x in range(24)],
    "dinov2": lambda: _numbered("dinov2", 768),
    "vitl14": lambda: _numbered("vitl14", 768),
    "aesthetics": lambda: ["aesthetics_score"],
    "context_affect": _module_attr("context_affect", "FEATURE_NAMES"),
    "experiential": _module_attr("experiential", "FEATURE_NAMES"),  # weights emit a subset
    "places": _places_columns,
    "yolo": _yolo_columns,
    "motion": _module_attr("motion", "FEATURE_NAMES"),
    "faces": _module_attr("faces", "FEATURE_NAMES"),
    "depth": _module_attr("depth", "FEATURE_NAMES"),
    "vgg19": _module_attr("vgg19", "column_names"),
}


def _model_class(model):
    # As in test_nulls: labels live on the class, so reading them imports the
    # model's backend. Skip a third-party backend CI does not install; fail a
    # missing viz2psy module, which is a broken registry entry.
    try:
        return _load_model_class(model)
    except ImportError as e:
        if (getattr(e, "name", "") or "").startswith("viz2psy"):
            raise
        pytest.skip(f"{model}: backend not installed ({e})")


def _pattern_regex(pattern: str) -> re.Pattern:
    """A sidecar ``pattern`` (``clip_{NNN}``, ``saliency_{XX}_{YY}``) as a regex."""
    return re.compile("^" + re.sub(r"\\\{([A-Z]+)\\\}",
                                   lambda m: rf"\d{{{len(m.group(1))}}}",
                                   re.escape(pattern)) + "$")


def test_inventory_covers_the_registry():
    assert set(INVENTORY) == set(MODEL_REGISTRY)


@pytest.mark.parametrize("model", sorted(MODEL_REGISTRY))
def test_every_column_resolves_to_valid_labels(model):
    labels = _model_class(model).labels
    columns = INVENTORY[model]()
    assert set(labels) == {"default", "columns"}, model
    assert problems(labels, columns) == []


@pytest.mark.parametrize("model", ["clip", "dinov2", "vitl14", "ebind", "gist", "saliency"])
def test_pattern_inventories_match_the_sidecar_pattern(model):
    # The inventories above are generated; check them against the pattern the
    # sidecar records, so `columns` keys are tested against real column names.
    columns = INVENTORY[model]()
    info = get_feature_info(model, columns)
    if "pattern" in info:
        rx = _pattern_regex(info["pattern"])
        assert all(rx.match(c) for c in columns), info["pattern"]
    else:  # ebind lists its columns
        assert info["columns"] == columns


@pytest.mark.parametrize("model", ["clip", "dinov2", "vitl14", "ebind"])
def test_embeddings_are_opaque_embeddings(model):
    labels = _model_class(model).labels
    for col in INVENTORY[model]():
        r = resolve(labels, col)
        assert (r["interpretability"], r["derivation"]) == ("opaque", "embedding"), col


@pytest.mark.parametrize("model", ["gist", "saliency", "depth"])
def test_spatial_and_layout_models_are_nameable(model):
    labels = _model_class(model).labels
    assert {resolve(labels, c)["interpretability"] for c in INVENTORY[model]()} == {"nameable"}


def test_yolo_mean_confidence_is_bookkeeping():
    labels = _model_class("yolo").labels
    assert resolve(labels, "yolo_mean_confidence")["derivation"] == "bookkeeping"
    others = {resolve(labels, c)["derivation"] for c in INVENTORY["yolo"]()
              if c != "yolo_mean_confidence"}
    assert others == {"trained-on-labels"}


def test_sun_attributes_are_probes_on_places_own_checkpoint():
    cls = _model_class("places")
    for col in INVENTORY["places"]():
        r = resolve(cls.labels, col)
        if col.startswith("sunattr_"):
            assert r["derivation"] == "probe" and r["derived_from"] == cls.checkpoint, col
        else:
            assert r["derivation"] == "trained-on-labels" and "derived_from" not in r, col


def test_aesthetics_is_a_probe_on_the_clip_backbone_it_loads():
    labels = _model_class("aesthetics").labels  # skips when open_clip is absent
    from viz2psy.models import aesthetics

    r = resolve(labels, "aesthetics_score")
    assert r["derivation"] == "probe"
    assert r["derived_from"] == f"{aesthetics._CLIP_MODEL}/{aesthetics._CLIP_PRETRAINED}"
    # A different CLIP from the `clip` model, so not circular with it.
    assert r["derived_from"] != _model_class("clip").checkpoint


def test_vgg19_fc8_units_name_imagenet_classes():
    labels = _model_class("vgg19").labels
    fc8 = [c for c in INVENTORY["vgg19"]() if c.startswith("vgg19_fc8_")]
    assert len(fc8) == 1000 and set(labels["columns"]) == set(fc8)
    assert resolve(labels, "vgg19_fc8_0000")["interpretability"] == "nameable"
    assert resolve(labels, "vgg19_conv1_1_0000")["derivation"] == "embedding"


class TestRules:
    """viz2psy.labels.problems enforces each 1.2 rule."""

    BASE = {"interpretability": "nameable", "derivation": "formula"}

    def test_override_merges_key_wise(self):
        labels = {"default": {**self.BASE, "Description": "d"},
                  "columns": {"a": {"derivation": "bookkeeping"}}}
        assert resolve(labels, "a") == {"interpretability": "nameable",
                                        "derivation": "bookkeeping", "Description": "d"}
        assert resolve(labels, "b") == labels["default"]

    @pytest.mark.parametrize("derivation", sorted(NEEDS_PARENT))
    def test_parent_required(self, derivation):
        labels = {"default": {**self.BASE, "derivation": derivation}, "columns": {}}
        assert any("requires derived_from" in p for p in problems(labels, ["a"]))
        labels["default"]["derived_from"] = "parent/ckpt"
        assert problems(labels, ["a"]) == []

    @pytest.mark.parametrize("derivation", sorted(DERIVATION - NEEDS_PARENT))
    def test_parent_forbidden(self, derivation):
        labels = {"default": {**self.BASE, "derivation": derivation, "derived_from": "x"},
                  "columns": {}}
        assert any("forbidden" in p for p in problems(labels, ["a"]))

    def test_vocabularies_and_unknown_column(self):
        bad = {"default": {"interpretability": "readable", "derivation": "learned"},
               "columns": {"ghost": {}}}
        found = " ".join(problems(bad, ["a"]))
        assert "interpretability 'readable'" in found
        assert "derivation 'learned'" in found
        assert "ghost" in found
        assert INTERPRETABILITY == {"nameable", "opaque"}

    def test_missing_required_key(self):
        labels = {"default": {"derivation": "formula"}, "columns": {}}
        assert problems(labels, ["a"]) == ["a: missing interpretability"]

    def test_source_validation_items(self):
        item = {"citation": "doi:x", "eval_dataset": "d", "metric": "m", "value": 0.5}
        ok = {"default": {**self.BASE, "source_validation": [item]}, "columns": {}}
        assert problems(ok, ["a"]) == []
        bad = {"default": {**self.BASE, "source_validation": [{"citation": "doi:x"}]},
               "columns": {}}
        assert problems(bad, ["a"])


def test_sidecar_writes_labels_for_emitted_columns_only():
    b = MetadataBuilder()
    b.add_model("faces", ["faces_count", "faces_center_dist"], 1.0)
    b.add_model("caption", ["caption_text"], 1.0)
    meta = b.build()
    assert meta["schema_version"] == SCHEMA_VERSION == "1.3"
    faces = meta["models"]["faces"]["labels"]
    assert set(faces) == {"default", "columns"}
    assert set(faces["columns"]) == {"faces_count", "faces_center_dist"}
    caption = meta["models"]["caption"]
    assert caption["features"]["dtype"] == "string"
    assert resolve(caption["labels"], "caption_text")["interpretability"] == "nameable"


def test_refresh_adds_labels_and_never_touches_the_csv(tmp_path):
    csv = tmp_path / "scores.csv"
    pd.DataFrame({"filename": ["a.jpg"], "faces_count": [1.0],
                  "llstat_luminance_mean": [0.2]}).to_csv(csv, index=False)
    b = MetadataBuilder()
    b.set_output(csv, 1, 3)
    b.add_model("faces", ["faces_count"], 1.0)
    b.add_model("llstat", ["llstat_luminance_mean"], 1.0)
    meta = b.build()
    meta["schema_version"] = "1.1"
    for e in meta["models"].values():
        e.pop("labels")
    side = tmp_path / "scores.meta.json"
    side.write_text(json.dumps(meta, indent=2))
    before = csv.read_bytes()

    assert refresh_sidecar(side).status == "refreshed"
    new = json.loads(side.read_text())
    assert new["schema_version"] == "1.3"  # not a video sidecar: no framing needed
    assert set(new["models"]["faces"]["labels"]["columns"]) == {"faces_count"}
    assert set(new["models"]["llstat"]["labels"]["columns"]) == {"llstat_luminance_mean"}
    assert "models.*.labels" in new["refreshed"][-1]["fields"]
    assert csv.read_bytes() == before
    assert refresh_sidecar(side).status == "unchanged"


def test_context_affect_is_a_probe_on_the_aesthetics_backbone_not_on_clip():
    labels = _model_class("context_affect").labels  # open_clip is imported only at load()
    from viz2psy.models import context_affect

    for col in context_affect.FEATURE_NAMES:
        r = resolve(labels, col)
        assert (r["interpretability"], r["derivation"]) == ("nameable", "probe"), col
        assert r["derived_from"] == context_affect.BACKBONE
    # Not circular with `clip`; the same backbone `aesthetics` loads.
    assert context_affect.BACKBONE != _model_class("clip").checkpoint  # skips without open_clip
    from viz2psy.models import aesthetics

    assert context_affect.BACKBONE == f"{aesthetics._CLIP_MODEL}/{aesthetics._CLIP_PRETRAINED}"
