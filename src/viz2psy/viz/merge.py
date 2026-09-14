"""Merge standalone per-model feature CSVs into one dashboard DataFrame.

Extractors write one CSV per model (``clip.csv``, ``emonet.csv``, ...), all
keyed on ``stimulus_id`` and — for video input — ``time``. The dashboard
renders every model it finds in a single DataFrame, so viewing all models at
once is purely a merge problem, solved here.

Two merge regimes, chosen automatically:

- **static** (no ``time`` column anywhere): exact merge on ``stimulus_id``.
- **time-indexed** (``time`` everywhere): frames whose time grid matches the
  spine (the input with the most rows) merge exactly; frames on a shifted
  grid (e.g. audio frames centered between video frames) merge to the
  nearest spine row within half the spine's sampling step.

Mixing the two regimes in one call is an error: per-stimulus and per-frame
rows have no shared unit. Identity/carrier columns (``filename``,
``chunk_idx``, ...) are kept from the first input that carries them; a
feature column appearing in two inputs (the same model run on two caption
sources, say) gets a ``__<source>`` suffix derived from the filenames.
"""

from __future__ import annotations

import fnmatch
from pathlib import Path

import pandas as pd

# Columns that identify or describe the stimulus rather than carry a model's
# output. Duplicates across inputs are expected; the first occurrence wins.
CARRIER_COLUMNS = frozenset({
    "filename", "chunk_idx", "chunk_label", "n_words", "sentence_idx",
    "word", "word_idx", "onset", "offset",
})

KEY_COLUMNS = ("stimulus_id", "time")

# Per-word scaffold files: one row per word, no per-stimulus feature columns.
EXCLUDED_PATTERNS = ("*_words.csv", "*_transcript*.csv")


def collect_feature_csvs(directory: Path) -> list[Path]:
    """Model feature CSVs directly inside ``directory``, merge candidates only.

    Excludes per-word scaffolds (``*_words.csv``) and transcripts, which are
    per-word tables, not per-stimulus/per-frame feature tables.
    """
    out = []
    for p in sorted(directory.glob("*.csv")):
        if p.name.startswith("."):
            continue
        if any(fnmatch.fnmatch(p.name, pat) for pat in EXCLUDED_PATTERNS):
            continue
        # Header sniff: directory mode only picks up stimulus-keyed tables;
        # explicitly-passed files still fail loudly in merge_feature_frames.
        with open(p) as fh:
            header = fh.readline()
        if "stimulus_id" not in header.strip().split(","):
            header_cols = header.strip().split(",")
            print(f"skipping {p.name}: no stimulus_id column "
                  f"(has {', '.join(header_cols[:4])}...)")
            continue
        out.append(p)
    if not out:
        raise FileNotFoundError(
            f"no feature CSVs found in {directory} "
            f"(looked for *.csv, excluding {', '.join(EXCLUDED_PATTERNS)})"
        )
    return out


def _spine_time_step(spine: pd.DataFrame) -> float:
    """Median sampling step of the spine's time grid (per stimulus)."""
    steps = (
        spine.sort_values("time", kind="stable")
        .groupby("stimulus_id")["time"]
        .diff()
        .dropna()
    )
    if steps.empty:
        return 0.0
    return float(steps.median())


def _distinguishing_tokens(name: str, others: list[str]) -> str:
    """The part of a file's stem that its colliding partners lack.

    ``caption_clap_text_chunks`` vs ``humancap_clap_text_chunks`` →
    ``caption``. Falls back to the full stem when nothing distinguishes.
    """
    stem = Path(name).stem
    other_tokens = set()
    for o in others:
        other_tokens.update(Path(o).stem.split("_"))
    unique = [t for t in stem.split("_") if t not in other_tokens]
    return "_".join(unique) if unique else stem


def _disambiguate_collisions(frames: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Rename feature columns that appear in more than one input.

    The same model run on two inputs (e.g. ``clap_text`` on model captions
    and on human captions) yields identical column names; each copy gets a
    ``__<source>`` suffix derived from its filename. A suffix (not a prefix)
    keeps the model-detection glob patterns matching.
    """
    owners: dict[str, list[str]] = {}
    for name, df in frames.items():
        for col in df.columns:
            if col in KEY_COLUMNS or col in CARRIER_COLUMNS:
                continue
            owners.setdefault(col, []).append(name)

    renames: dict[str, dict[str, str]] = {}
    for col, files in owners.items():
        if len(files) < 2:
            continue
        for f in files:
            tag = _distinguishing_tokens(f, [o for o in files if o != f])
            renames.setdefault(f, {})[col] = f"{col}__{tag}"

    if not renames:
        return frames

    out = {}
    for name, df in frames.items():
        if name in renames:
            tags = sorted({v.rsplit("__", 1)[1]
                           for v in renames[name].values()})
            print(f"note: {Path(name).name} shares feature columns with "
                  f"another input; suffixing its copies with __{tags[0]}")
            df = df.rename(columns=renames[name])
        out[name] = df

    # A failed disambiguation (identical stems) would silently drop columns
    # in the merge — refuse instead.
    all_cols: dict[str, str] = {}
    for name, df in out.items():
        for col in df.columns:
            if col in KEY_COLUMNS or col in CARRIER_COLUMNS:
                continue
            if col in all_cols:
                raise ValueError(
                    f"feature column {col!r} appears in both {all_cols[col]} "
                    f"and {name} even after filename-based renaming; rename "
                    "one file to give it a distinguishing name"
                )
            all_cols[col] = name
    return out


def _drop_duplicate_carriers(base: pd.DataFrame, incoming: pd.DataFrame,
                             keys: list[str]) -> pd.DataFrame:
    dupes = [c for c in incoming.columns
             if c not in keys and c in base.columns]
    return incoming.drop(columns=dupes)


def merge_feature_frames(
    frames: dict[str, pd.DataFrame],
    tolerance: float | None = None,
    lenient: set[str] | None = None,
) -> pd.DataFrame:
    """Merge per-model feature DataFrames (name -> frame) into one.

    ``tolerance`` overrides the nearest-time match window for time-indexed
    input (default: half the spine's median sampling step). Names in
    ``lenient`` (auto-collected from a directory, not named by the user)
    are skipped with a note instead of raising when they cannot merge
    (e.g. several rows per stimulus_id).
    """
    lenient = lenient or set()
    if not frames:
        raise ValueError("no input frames to merge")

    for name, df in frames.items():
        if "stimulus_id" not in df.columns:
            raise ValueError(
                f"{name} has no stimulus_id column; every mergeable feature "
                "CSV is keyed on stimulus_id"
            )

    # Auto-collected tables that cannot merge 1:1 (several rows per key —
    # e.g. one row per human caption) are dropped with a note.
    def _dup_keys(df):
        keys = ["stimulus_id"] + (["time"] if "time" in df.columns else [])
        return df.duplicated(subset=keys).any()

    for name in [n for n in frames if n in lenient and _dup_keys(frames[n])]:
        print(f"skipping {Path(name).name}: several rows per stimulus, "
              "cannot merge 1:1")
        del frames[name]

    timed = {n for n, df in frames.items() if "time" in df.columns}
    if timed and timed != set(frames):
        static = set(frames) - timed
        minority, majority = sorted((timed, static), key=len)
        if minority <= lenient:
            for name in sorted(minority):
                print(f"skipping {Path(name).name}: "
                      f"{'time-indexed' if name in timed else 'static'} "
                      "table in a merge of the other kind")
                del frames[name]
            timed &= set(frames)
        else:
            raise ValueError(
                "cannot merge time-indexed and static feature files in one "
                f"call: {sorted(timed)} have a time column, "
                f"{sorted(static)} do not"
            )

    frames = _disambiguate_collisions(frames)

    if not timed:
        return _merge_static(frames)
    return _merge_timed(frames, tolerance)


def _merge_static(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    merged: pd.DataFrame | None = None
    for name, df in frames.items():
        if df["stimulus_id"].duplicated().any():
            raise ValueError(
                f"{name} has duplicate stimulus_id rows; merge on "
                "stimulus_id would multiply rows — exclude this file"
            )
        if merged is None:
            merged = df
            continue
        incoming = _drop_duplicate_carriers(merged, df, ["stimulus_id"])
        merged = merged.merge(incoming, on="stimulus_id", how="outer")
    return merged


def _merge_timed(frames: dict[str, pd.DataFrame],
                 tolerance: float | None) -> pd.DataFrame:
    # Spine: the densest input; everything else aligns to its grid.
    spine_name = max(frames, key=lambda n: len(frames[n]))
    spine = frames[spine_name].sort_values("time", kind="stable")
    spine = spine.reset_index(drop=True)

    if tolerance is None:
        tolerance = _spine_time_step(spine) / 2

    spine_grid = set(zip(spine["stimulus_id"], spine["time"]))
    merged = spine
    for name, df in frames.items():
        if name == spine_name:
            continue
        incoming = _drop_duplicate_carriers(
            merged, df, ["stimulus_id", "time"])
        grid = set(zip(df["stimulus_id"], df["time"]))
        if grid <= spine_grid:
            merged = merged.merge(
                incoming, on=["stimulus_id", "time"], how="left")
        else:
            incoming = incoming.sort_values("time", kind="stable")
            merged = pd.merge_asof(
                merged, incoming,
                on="time", by="stimulus_id",
                direction="nearest",
                tolerance=tolerance if tolerance > 0 else None,
            )
    # Row order for the dashboard: per stimulus, time ascending.
    return (merged.sort_values(["stimulus_id", "time"], kind="stable")
            .reset_index(drop=True))


def load_and_merge(paths: list[Path],
                   tolerance: float | None = None) -> pd.DataFrame:
    """Read feature CSVs (expanding directories) and merge them.

    Import-light entry point for CLIs: applies the same legacy column
    renames as single-CSV loading.
    """
    from viz2psy.columns import apply_legacy_renames

    expanded: list[Path] = []
    lenient: set[str] = set()
    for p in paths:
        if p.is_dir():
            found = collect_feature_csvs(p)
            expanded.extend(found)
            lenient.update(str(f) for f in found)
        else:
            expanded.append(p)

    frames = {}
    for p in expanded:
        frames[str(p)] = apply_legacy_renames(pd.read_csv(p))
    return merge_feature_frames(frames, tolerance=tolerance, lenient=lenient)
