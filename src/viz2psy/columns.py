"""Column-name contract helpers.

viz2psy 0.6.0 renamed all un-prefixed feature columns to carry their
model's registry-name prefix (constellation Contract B, §4.1). Scores CSVs
written by earlier versions keep the old names on disk; consumers inside
viz2psy (the viz CLI) normalize them at load time via
``apply_legacy_renames``.
"""

import pandas as pd

# 20 EmoNet emotion categories (Cowen & Keltner 2017), pre-0.6.0 spelling.
_EMONET_LEGACY = [
    "Adoration", "Aesthetic Appreciation", "Amusement", "Anxiety", "Awe",
    "Boredom", "Confusion", "Craving", "Disgust", "Empathic Pain",
    "Entrancement", "Excitement", "Fear", "Horror", "Interest", "Joy",
    "Romance", "Sadness", "Sexual Desire", "Surprise",
]

_LLSTAT_LEGACY = [
    "luminance_mean", "luminance_std", "rms_contrast",
    "r_mean", "r_std", "g_mean", "g_std", "b_mean", "b_std",
    "lab_l_mean", "lab_a_mean", "lab_b_mean", "saturation_mean",
    "hf_energy", "lf_energy", "edge_density", "colorfulness",
]

_YOLO_STATS_LEGACY = [
    "object_count", "category_count", "object_coverage",
    "largest_object_ratio", "mean_confidence",
]

LEGACY_RENAMES: dict[str, str] = {
    **{c: "emonet_" + c.lower().replace(" ", "_") for c in _EMONET_LEGACY},
    **{c: f"llstat_{c}" for c in _LLSTAT_LEGACY},
    **{c: f"yolo_{c}" for c in _YOLO_STATS_LEGACY},
    "memorability": "resmem_memorability",
    "aesthetic_score": "aesthetics_score",
    "caption": "caption_text",
}


def apply_legacy_renames(df: pd.DataFrame) -> pd.DataFrame:
    """Rename pre-0.6.0 feature columns to their prefixed equivalents.

    No-op on already-conformant frames; never overwrites an existing
    new-style column.
    """
    renames = {
        old: new
        for old, new in LEGACY_RENAMES.items()
        if old in df.columns and new not in df.columns
    }
    return df.rename(columns=renames) if renames else df
