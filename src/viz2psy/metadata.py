"""Metadata generation for viz2psy output files.

Generates a sidecar JSON file that describes:
- Code version and runtime info
- Input source details
- Output structure
- Feature definitions for each model
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def get_version() -> str:
    """Get viz2psy package version."""
    # Prefer __version__ from package (always up-to-date in editable installs)
    try:
        from viz2psy import __version__
        return __version__
    except Exception:
        pass
    # Fall back to importlib.metadata
    try:
        from importlib.metadata import version
        return version("viz2psy")
    except Exception:
        return "unknown"


def device_name(device: str | None) -> str | None:
    """The GPU model behind a ``cuda[:N]`` device (e.g. "NVIDIA H100 80GB HBM3"), else None.

    Optional provenance (Contract B §4.1): a model whose output depends on the
    GPU's arithmetic (greedy bf16 decoding in ``caption_vlm`` rewords some
    captions between GPU types) can be grouped by it. Never required, and a
    failure to read it never fails a run: the key is then left out.
    """
    if not device or not str(device).startswith("cuda"):
        return None
    try:
        import torch

        if not torch.cuda.is_available():
            return None
        index = torch.device(device).index
        return torch.cuda.get_device_name(torch.cuda.current_device() if index is None else index)
    except Exception:
        return None


def get_model_version(model_name: str) -> str:
    """Get version of underlying model package."""
    version_map = {
        "resmem": ("resmem", None),
        "emonet": (None, "1.0"),  # No package, bundled
        "clip": ("open_clip_torch", None),
        "ebind": ("ebind", None),
        "caption": ("transformers", None),
        "caption_vlm": ("transformers", None),  # checkpoint pins the Hub revision
        "gist": (None, "1.0"),  # Custom implementation
        "llstat": (None, "1.0"),  # Custom implementation
        "saliency": ("deepgaze_pytorch", None),
        "dinov2": ("torch", "facebookresearch/dinov2"),
        "aesthetics": ("open_clip_torch", "LAION"),
        "places": ("torch", None),  # inline arch; checkpoint field names the weights
        "yolo": ("ultralytics", None),
        "motion": ("opencv-python", None),  # analytic Farnebäck flow
        "faces": ("opencv-python", None),  # YuNet via cv2.FaceDetectorYN
        "depth": ("transformers", None),
        "vgg19": ("torchvision", None),  # checkpoint field names the weights
    }

    pkg, fallback = version_map.get(model_name, (None, "unknown"))
    if pkg:
        try:
            from importlib.metadata import version
            return version(pkg)
        except Exception:
            pass
    return fallback or "unknown"


def get_model_contract(model_name: str) -> tuple[str | None, list[str]]:
    """Return (checkpoint, extra_prefixes) from the model class.

    Class-level defaults only — accurate for CLI runs, which never override
    checkpoint arguments. API users constructing models with non-default
    checkpoints should amend the sidecar themselves.
    """
    try:
        from viz2psy.cli import _load_model_class
        cls = _load_model_class(model_name)
        return (
            getattr(cls, "checkpoint", None),
            list(getattr(cls, "extra_prefixes", ())),
        )
    except Exception:
        return None, []


def default_preprocessing(model_name: str) -> dict | None:
    """The class-default input transform a model declares, or None."""
    from viz2psy.cli import _load_model_class
    describe = getattr(_load_model_class(model_name), "describe_preprocessing", None)
    return describe() if describe else None


SCHEMA_VERSION = "1.3"  # Contract B §4.1 extractor output convention


def declared_nulls(model_name: str) -> dict[str, dict[str, str]]:
    """The model's Contract B 1.1 ``nulls`` map, read from its class.

    Unlike :func:`get_model_contract` this never swallows an error: a silent
    ``{}`` would be a false claim that the model cannot emit NaN.
    """
    from viz2psy.cli import _load_model_class

    return {col: dict(entry) for col, entry in _load_model_class(model_name).nulls.items()}


def declared_labels(model_name: str, columns: list[str]) -> dict:
    """The model's Contract B 1.2 ``labels`` for the emitted ``columns``.

    Read from the model class like :func:`declared_nulls`, and like it never
    swallows an error: a sidecar without labels must be a visible failure.
    """
    from viz2psy.cli import _load_model_class
    from viz2psy.labels import for_columns

    return for_columns(_load_model_class(model_name).labels, columns)


def get_feature_info(model_name: str, feature_names: list[str]) -> dict[str, Any]:
    """Get feature pattern/definition info for a model."""
    count = len(feature_names)

    # Check for numbered patterns
    if model_name == "clip" and feature_names and feature_names[0].startswith("clip_"):
        return {
            "pattern": "clip_{NNN}",
            "range": [0, count - 1],
            "count": count
        }

    if model_name == "gist" and feature_names and feature_names[0].startswith("gist_"):
        return {
            "pattern": "gist_{NNN}",
            "range": [0, count - 1],
            "count": count
        }

    if model_name == "dinov2" and feature_names and feature_names[0].startswith("dinov"):
        return {
            "pattern": "dinov2_{NNN}",
            "range": [0, count - 1],
            "count": count
        }

    if model_name == "saliency" and feature_names and feature_names[0].startswith("saliency_"):
        # saliency_XX_YY format (24x24 grid)
        return {
            "pattern": "saliency_{XX}_{YY}",
            "range": [[0, 23], [0, 23]],
            "count": count
        }

    if model_name == "vgg19" and feature_names and feature_names[0].startswith("vgg19_"):
        # One {NNNN}-numbered block per layer; see viz2psy.models.vgg19.
        from viz2psy.models.vgg19 import LAYERS_112, LAYERS_224, layer_width
        layers = {f"{layer}": layer_width(layer) for layer in LAYERS_224}
        layers.update({f"{layer}_relu112": layer_width(layer) for layer in LAYERS_112})
        return {
            "pattern": "vgg19_{layer}_{NNNN}",
            "layers": layers,
            "count": count
        }

    if model_name in ("caption", "caption_vlm"):
        return {
            "columns": feature_names,
            "count": count,
            "dtype": "string"
        }

    # Named features - use definition reference
    if model_name in ("emonet", "places", "yolo", "llstat", "resmem", "aesthetics"):
        return {
            "columns": feature_names,
            "count": count,
            "definition": f"feature_definitions.{model_name}"
        }

    # Fallback - list all columns
    return {
        "columns": feature_names,
        "count": count
    }


def build_feature_definitions(model_features: dict[str, list[str]]) -> dict[str, Any]:
    """Build the feature_definitions section."""
    definitions = {}

    for model_name, features in model_features.items():
        if model_name == "emonet":
            definitions["emonet"] = features

        elif model_name == "places":
            places_scenes = [f.replace("places_", "") for f in features if f.startswith("places_")]
            sunattr = [f.replace("sunattr_", "") for f in features if f.startswith("sunattr_")]
            definitions["places"] = {
                "scenes": places_scenes,
                "attributes": sunattr
            }

        elif model_name == "yolo":
            stat_names = {
                "yolo_object_count", "yolo_category_count",
                "yolo_object_coverage", "yolo_largest_object_ratio",
                "yolo_mean_confidence",
            }
            objects = [
                f.replace("yolo_", "")
                for f in features
                if f.startswith("yolo_") and f not in stat_names
            ]
            stats = [f for f in features if f in stat_names]
            definitions["yolo"] = {
                "objects": objects,
                "stats": stats
            }

        elif model_name == "llstat":
            definitions["llstat"] = features

        elif model_name == "resmem":
            definitions["resmem"] = features

        elif model_name == "aesthetics":
            definitions["aesthetics"] = features

    return definitions


def _still_framing() -> dict:
    from viz2psy.framing import STILL_FRAMING
    return dict(STILL_FRAMING)


class MetadataBuilder:
    """Builder for output metadata."""

    def __init__(self):
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.input_info = {}
        self.output_info = {}
        self.index_column = None
        self.device = None
        self.device_name = None
        self.models = {}
        self.model_features = {}
        self.total_runtime_sec = 0.0

    def set_input_hdf5(self, path: Path, dataset: str, start_idx: int, end_idx: int):
        """Set input info for HDF5 brick."""
        self.input_info = {
            "type": "hdf5_brick",
            "path": str(path.resolve()),
            "dataset": dataset,
            "indices": [start_idx, end_idx],
            "framing": _still_framing(),
        }
        self.index_column = "image_idx"

    def set_input_images(self, paths: list[Path]):
        """Set input info for image files."""
        self.input_info = {
            "type": "image_folder",
            "paths": [str(p.resolve()) for p in paths],
            "count": len(paths),
            "framing": _still_framing(),
        }
        self.index_column = "filename"

    def set_input_video(
        self,
        path: Path,
        frame_interval: float,
        n_frames: int,
        saved_frames_dir: Path | None = None,
        frame_format: str = "jpg",
        frame_cache: str = "memory",
        stage0: dict | None = None,
    ):
        """Set input info for video file.

        ``framing`` (schema 1.3) records what the models scored: always the
        decoded frames (``frame_source``), held in memory or cached on disk
        as lossless PNG (``frame_cache``), after stage 0 of the framing rule
        (``stage0``: :meth:`viz2psy.framing.Stage0.record`). ``saved_frames_*``
        describe viewing copies only; models never read them.
        """
        self.input_info = {
            "type": "video",
            "path": str(path.resolve()),
            "frame_interval_sec": frame_interval,
            "n_frames": n_frames,
            "framing": {"frame_source": "raw decode", "frame_cache": frame_cache,
                        **(stage0 or {})},
        }
        if saved_frames_dir is not None:
            self.input_info["saved_frames_dir"] = str(saved_frames_dir.resolve())
            self.input_info["saved_frames_format"] = frame_format
        self.index_column = "time"

    def set_input_video_set(
        self,
        videos: list[dict],
        frame_interval: float,
        rule: dict,
    ):
        """Set input info for several videos scored in one call (``type: "video_set"``).

        ``rule`` is the stage-0 rule every video shares (``frame_source`` and
        :data:`viz2psy.framing.RULE_RECORD_KEYS`), recorded once as
        ``input.framing`` so a consumer reads the rule where it reads a single
        video's. Each entry of ``videos`` carries what differs per file: its
        path, ``stimulus_id``, ``n_frames``, ``frame_cache`` and the per-file
        stage-0 values (box, sizes). Rows are keyed (``stimulus_id``, ``time``).
        """
        caches = {v["frame_cache"] for v in videos}
        self.input_info = {
            "type": "video_set",
            "count": len(videos),
            "frame_interval_sec": frame_interval,
            "n_frames": sum(int(v["n_frames"]) for v in videos),
            "framing": {**rule,
                        "frame_cache": caches.pop() if len(caches) == 1 else "per video"},
            "videos": videos,
        }
        self.index_column = "time"

    def set_output(self, path: Path, rows: int, columns: int):
        """Set output info."""
        self.output_info = {
            "path": str(path.resolve()),
            "rows": rows,
            "columns": columns
        }

    def set_device(self, device: str):
        """Set device used for inference, and the GPU's model name when it can be read."""
        self.device = str(device)
        self.device_name = device_name(self.device)

    def add_model(self, model_name: str, feature_names: list[str], runtime_sec: float,
                  preprocessing: dict | None = None):
        """Add model info after it completes.

        ``preprocessing`` describes an input transform the model applies
        (e.g. saliency's canonical resize); when omitted, the model class's
        defaults are recorded if it declares any.
        """
        checkpoint, extra_prefixes = get_model_contract(model_name)
        package_version = get_model_version(model_name)
        entry = {
            "version": package_version,  # legacy key, one deprecation cycle
            "package_version": package_version,
            "checkpoint": checkpoint,
            "runtime_sec": round(runtime_sec, 3),
            "features": get_feature_info(model_name, feature_names)
        }
        if extra_prefixes:
            entry["prefixes"] = [model_name] + extra_prefixes
        if preprocessing is None:
            preprocessing = default_preprocessing(model_name)
        if preprocessing is not None:
            entry["preprocessing"] = preprocessing
        emitted = set(feature_names)
        entry["nulls"] = {c: e for c, e in declared_nulls(model_name).items() if c in emitted}
        entry["labels"] = declared_labels(model_name, feature_names)
        self.models[model_name] = entry
        self.model_features[model_name] = feature_names
        self.total_runtime_sec += runtime_sec

    def build(self) -> dict[str, Any]:
        """Build the final metadata dict."""
        # Determine which models need definitions
        needs_definition = {"emonet", "places", "yolo", "llstat", "resmem", "aesthetics"}
        models_with_defs = {k: v for k, v in self.model_features.items() if k in needs_definition}

        metadata = {
            "schema_version": SCHEMA_VERSION,
            "extractor": "viz2psy",
            "extractor_version": get_version(),
            "viz2psy_version": get_version(),  # legacy key, one deprecation cycle
            "created_at": self.created_at,
            "input": self.input_info,
            "output": self.output_info,
            "index_column": self.index_column,
            "device": self.device,
            **({"device_name": self.device_name} if self.device_name else {}),
            "total_runtime_sec": round(self.total_runtime_sec, 3),
            "models": self.models,
        }

        if models_with_defs:
            metadata["feature_definitions"] = build_feature_definitions(models_with_defs)

        return metadata

    def save(self, output_path: Path):
        """Save metadata to sidecar JSON file."""
        meta_path = output_path.with_suffix(".meta.json")
        metadata = self.build()

        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        return meta_path
