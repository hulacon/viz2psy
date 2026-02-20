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


def get_model_version(model_name: str) -> str:
    """Get version of underlying model package."""
    version_map = {
        "resmem": ("resmem", None),
        "emonet": (None, "1.0"),  # No package, bundled
        "clip": ("open_clip_torch", None),
        "gist": (None, "1.0"),  # Custom implementation
        "llstat": (None, "1.0"),  # Custom implementation
        "saliency": ("deepgaze_pytorch", None),
        "dinov2": ("torch", "facebookresearch/dinov2"),
        "aesthetics": ("open_clip_torch", "LAION"),
        "places": (None, "wideresnet18-places365"),
        "yolo": ("ultralytics", None),
    }

    pkg, fallback = version_map.get(model_name, (None, "unknown"))
    if pkg:
        try:
            from importlib.metadata import version
            return version(pkg)
        except Exception:
            pass
    return fallback or "unknown"


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
            "pattern": "dinov_{NNN}",
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
            objects = [f.replace("yolo_", "") for f in features if f.startswith("yolo_")]
            stats = [f for f in features if not f.startswith("yolo_")]
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


class MetadataBuilder:
    """Builder for output metadata."""

    def __init__(self):
        self.created_at = datetime.now(timezone.utc).isoformat()
        self.input_info = {}
        self.output_info = {}
        self.index_column = None
        self.device = None
        self.models = {}
        self.model_features = {}
        self.total_runtime_sec = 0.0

    def set_input_hdf5(self, path: Path, dataset: str, start_idx: int, end_idx: int):
        """Set input info for HDF5 brick."""
        self.input_info = {
            "type": "hdf5_brick",
            "path": str(path.resolve()),
            "dataset": dataset,
            "indices": [start_idx, end_idx]
        }
        self.index_column = "image_idx"

    def set_input_images(self, paths: list[Path]):
        """Set input info for image files."""
        self.input_info = {
            "type": "image_folder",
            "paths": [str(p.resolve()) for p in paths],
            "count": len(paths)
        }
        self.index_column = "filename"

    def set_input_video(
        self,
        path: Path,
        frame_interval: float,
        n_frames: int,
        saved_frames_dir: Path | None = None,
        frame_format: str = "jpg",
    ):
        """Set input info for video file."""
        self.input_info = {
            "type": "video",
            "path": str(path.resolve()),
            "frame_interval_sec": frame_interval,
            "n_frames": n_frames
        }
        if saved_frames_dir is not None:
            self.input_info["saved_frames_dir"] = str(saved_frames_dir.resolve())
            self.input_info["saved_frames_format"] = frame_format
        self.index_column = "time"

    def set_output(self, path: Path, rows: int, columns: int):
        """Set output info."""
        self.output_info = {
            "path": str(path.resolve()),
            "rows": rows,
            "columns": columns
        }

    def set_device(self, device: str):
        """Set device used for inference."""
        self.device = str(device)

    def add_model(self, model_name: str, feature_names: list[str], runtime_sec: float):
        """Add model info after it completes."""
        self.models[model_name] = {
            "version": get_model_version(model_name),
            "runtime_sec": round(runtime_sec, 3),
            "features": get_feature_info(model_name, feature_names)
        }
        self.model_features[model_name] = feature_names
        self.total_runtime_sec += runtime_sec

    def build(self) -> dict[str, Any]:
        """Build the final metadata dict."""
        # Determine which models need definitions
        needs_definition = {"emonet", "places", "yolo", "llstat", "resmem", "aesthetics"}
        models_with_defs = {k: v for k, v in self.model_features.items() if k in needs_definition}

        metadata = {
            "viz2psy_version": get_version(),
            "created_at": self.created_at,
            "input": self.input_info,
            "output": self.output_info,
            "index_column": self.index_column,
            "device": self.device,
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
