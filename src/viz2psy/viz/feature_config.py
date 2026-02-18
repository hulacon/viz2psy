"""Feature-to-visualization mapping configuration.

Defines which visualization types are appropriate for each model's output,
enabling smart defaults in the CLI and automatic visualization suggestions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class FeatureConfig:
    """Configuration for a model's feature visualization options."""

    # Model name (matches MODEL_REGISTRY key)
    name: str

    # Human-readable description
    description: str

    # Feature type classification
    feature_type: Literal["scalar", "named_distribution", "embedding", "spatial"]

    # Number of output dimensions
    n_dims: int

    # Visualization appropriateness
    timeseries: bool = False
    mds_clustering: bool = False
    trajectories: bool = False

    # Special handling options
    timeseries_mode: Literal["all", "top_k", "aggregate", "none"] = "none"
    top_k: int = 5  # For top_k mode

    # Column patterns for matching
    column_patterns: list[str] = field(default_factory=list)

    # Scalar feature names (for models with named outputs)
    scalar_features: list[str] = field(default_factory=list)


# Feature configurations for all viz2psy models
FEATURE_CONFIGS: dict[str, FeatureConfig] = {
    "resmem": FeatureConfig(
        name="resmem",
        description="Image memorability (0-1)",
        feature_type="scalar",
        n_dims=1,
        timeseries=True,
        mds_clustering=False,
        trajectories=False,
        timeseries_mode="all",
        column_patterns=["memorability"],
        scalar_features=["memorability"],
    ),
    "aesthetics": FeatureConfig(
        name="aesthetics",
        description="Aesthetic quality score (1-10)",
        feature_type="scalar",
        n_dims=1,
        timeseries=True,
        mds_clustering=False,
        trajectories=False,
        timeseries_mode="all",
        column_patterns=["aesthetic*"],
        scalar_features=["aesthetic_score"],
    ),
    "emonet": FeatureConfig(
        name="emonet",
        description="20 emotion category probabilities",
        feature_type="named_distribution",
        n_dims=20,
        timeseries=True,
        mds_clustering=True,
        trajectories=True,
        timeseries_mode="top_k",
        top_k=5,
        # Emonet columns are direct emotion names (no prefix)
        column_patterns=[
            "Adoration", "Aesthetic Appreciation", "Amusement", "Anxiety",
            "Awe", "Boredom", "Confusion", "Craving", "Disgust",
            "Empathic Pain", "Entrancement", "Excitement", "Fear",
            "Horror", "Interest", "Joy", "Romance", "Sadness",
            "Sexual Desire", "Surprise",
        ],
        scalar_features=[
            "Adoration", "Aesthetic Appreciation", "Amusement", "Anxiety",
            "Awe", "Boredom", "Confusion", "Craving", "Disgust",
            "Empathic Pain", "Entrancement", "Excitement", "Fear",
            "Horror", "Interest", "Joy", "Romance", "Sadness",
            "Sexual Desire", "Surprise",
        ],
    ),
    "llstat": FeatureConfig(
        name="llstat",
        description="17 low-level image statistics",
        feature_type="named_distribution",
        n_dims=17,
        timeseries=True,
        mds_clustering=True,
        trajectories=True,
        timeseries_mode="all",
        # llstat columns use various naming patterns (no llstat_ prefix)
        column_patterns=[
            "luminance_*", "rms_contrast",
            "r_mean", "r_std", "g_mean", "g_std", "b_mean", "b_std",
            "lab_l_mean", "lab_a_mean", "lab_b_mean",
            "saturation_mean", "hf_energy", "lf_energy",
            "edge_density", "colorfulness",
        ],
        scalar_features=[
            "luminance_mean", "luminance_std",
            "rms_contrast",
            "r_mean", "r_std", "g_mean", "g_std", "b_mean", "b_std",
            "lab_l_mean", "lab_a_mean", "lab_b_mean",
            "saturation_mean",
            "hf_energy", "lf_energy",
            "edge_density",
            "colorfulness",
        ],
    ),
    "places": FeatureConfig(
        name="places",
        description="365 scene + 102 attribute probabilities",
        feature_type="named_distribution",
        n_dims=467,
        timeseries=True,
        mds_clustering=True,
        trajectories=True,
        timeseries_mode="top_k",
        top_k=5,
        column_patterns=["places_*", "sunattr_*"],
        scalar_features=[],  # Too many to list, use patterns
    ),
    "yolo": FeatureConfig(
        name="yolo",
        description="80 object counts + 5 summary statistics",
        feature_type="named_distribution",
        n_dims=85,
        timeseries=True,
        mds_clustering=True,
        trajectories=True,
        timeseries_mode="top_k",
        top_k=10,
        # yolo_ prefix for object classes, plus summary stats
        column_patterns=[
            "yolo_*",
            "object_count", "category_count", "object_coverage",
            "largest_object_ratio", "mean_confidence",
        ],
        scalar_features=[
            "object_count", "category_count", "object_coverage",
            "largest_object_ratio", "mean_confidence",
        ],
    ),
    "clip": FeatureConfig(
        name="clip",
        description="512-dim CLIP ViT-B-32 embeddings",
        feature_type="embedding",
        n_dims=512,
        timeseries=False,
        mds_clustering=True,
        trajectories=True,
        timeseries_mode="none",
        column_patterns=["clip_*"],
        scalar_features=[],
    ),
    "gist": FeatureConfig(
        name="gist",
        description="512-dim Gabor spatial envelope features",
        feature_type="embedding",
        n_dims=512,
        timeseries=False,
        mds_clustering=True,
        trajectories=True,
        timeseries_mode="none",
        column_patterns=["gist_*"],
        scalar_features=[],
    ),
    "dinov2": FeatureConfig(
        name="dinov2",
        description="768-dim DINOv2 ViT-B/14 embeddings",
        feature_type="embedding",
        n_dims=768,
        timeseries=False,
        mds_clustering=True,
        trajectories=True,
        timeseries_mode="none",
        column_patterns=["dinov2_*"],
        scalar_features=[],
    ),
    "saliency": FeatureConfig(
        name="saliency",
        description="576-dim fixation density grid (24x24)",
        feature_type="spatial",
        n_dims=576,
        timeseries=True,  # Via aggregates
        mds_clustering=True,
        trajectories=True,
        timeseries_mode="aggregate",
        column_patterns=["saliency_*"],
        scalar_features=[],  # Could add center_bias, spread, etc.
    ),
}


def get_timeseries_features(df_columns: list[str]) -> list[str]:
    """Get columns appropriate for timeseries visualization.

    Returns scalar features and top-k candidates from named distributions.
    Excludes high-dimensional embeddings.
    """
    timeseries_cols = []

    for config in FEATURE_CONFIGS.values():
        if not config.timeseries:
            continue

        if config.timeseries_mode == "all":
            # Include all matching columns
            for pattern in config.column_patterns:
                import fnmatch
                for col in df_columns:
                    if fnmatch.fnmatch(col, pattern) and col not in timeseries_cols:
                        timeseries_cols.append(col)

        elif config.timeseries_mode == "top_k":
            # Include named scalars if present
            for feat in config.scalar_features:
                if feat in df_columns and feat not in timeseries_cols:
                    timeseries_cols.append(feat)

    return timeseries_cols


def get_mds_features(df_columns: list[str]) -> dict[str, list[str]]:
    """Get column groups appropriate for MDS/clustering visualization.

    Returns dict mapping model name to list of columns.
    """
    mds_groups = {}

    for config in FEATURE_CONFIGS.values():
        if not config.mds_clustering:
            continue

        matching_cols = []
        for pattern in config.column_patterns:
            import fnmatch
            for col in df_columns:
                if fnmatch.fnmatch(col, pattern):
                    matching_cols.append(col)

        if len(matching_cols) >= 2:  # Need at least 2 dims for MDS
            mds_groups[config.name] = matching_cols

    return mds_groups


def get_trajectory_features(df_columns: list[str]) -> dict[str, list[str]]:
    """Get column groups appropriate for trajectory/animation visualization.

    Returns dict mapping model name to list of columns.
    Best for temporal/sequential data.
    """
    traj_groups = {}

    for config in FEATURE_CONFIGS.values():
        if not config.trajectories:
            continue

        matching_cols = []
        for pattern in config.column_patterns:
            import fnmatch
            for col in df_columns:
                if fnmatch.fnmatch(col, pattern):
                    matching_cols.append(col)

        if len(matching_cols) >= 2:
            traj_groups[config.name] = matching_cols

    return traj_groups


def detect_models_in_dataframe(df_columns: list[str]) -> list[str]:
    """Detect which viz2psy models produced columns in a DataFrame."""
    detected = []

    for config in FEATURE_CONFIGS.values():
        for pattern in config.column_patterns:
            import fnmatch
            for col in df_columns:
                if fnmatch.fnmatch(col, pattern):
                    if config.name not in detected:
                        detected.append(config.name)
                    break

    return detected


def get_visualization_recommendations(df_columns: list[str]) -> dict:
    """Get visualization recommendations for a DataFrame.

    Returns a dict with recommended visualizations and the features to use.
    """
    detected_models = detect_models_in_dataframe(df_columns)
    timeseries_features = get_timeseries_features(df_columns)
    mds_groups = get_mds_features(df_columns)
    trajectory_groups = get_trajectory_features(df_columns)

    recommendations = {
        "detected_models": detected_models,
        "timeseries": {
            "available": len(timeseries_features) > 0,
            "features": timeseries_features,
            "description": "Scalar features over time/sequence",
        },
        "mds_clustering": {
            "available": len(mds_groups) > 0,
            "groups": mds_groups,
            "description": "2D projection to visualize similarity structure",
        },
        "trajectories": {
            "available": len(trajectory_groups) > 0,
            "groups": trajectory_groups,
            "description": "Animated state-space evolution (requires sequential data)",
        },
    }

    return recommendations


# Summary table for documentation/CLI help
VISUALIZATION_MATRIX = """
Model        | Timeseries | MDS/Cluster | Trajectories | Notes
-------------|------------|-------------|--------------|-------------------------------
resmem       | Yes        | No          | No           | Single scalar (memorability)
aesthetics   | Yes        | No          | No           | Single scalar (1-10)
emonet       | Top-5      | Yes         | Yes          | 20 emotions, emotional state evolution
llstat       | Yes        | Yes         | Yes          | 17 interpretable visual stats
places       | Top-5      | Yes         | Yes          | 467 scene/attribute probs
yolo         | Top-10     | Yes         | Yes          | Object counts + summary stats
clip         | No         | Yes         | Yes          | 512-dim semantic embeddings
gist         | No         | Yes         | Yes          | 512-dim spatial frequency
dinov2       | No         | Yes         | Yes          | 768-dim visual semantics
saliency     | Aggregate  | Yes         | Yes          | 576-dim attention grid
"""
