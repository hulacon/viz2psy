"""Contract B schema 1.2 per-column ``labels``: vocabularies, merge and checks.

Each model class declares ``labels`` (see :class:`viz2psy.models.base.BaseModel`)
in exactly the shape the sidecar carries::

    {"default": {<keys for every column>},
     "columns": {<exact column name>: {<keys replacing the default's>}}}

A column's labels are the default with its ``columns`` entry merged in key by
key (not a whole-object replacement). Labels say what a column is and how its
value is computed. They describe the code, never a check of the column on a
particular stimulus set.
"""

from __future__ import annotations

import copy
from collections.abc import Iterable

INTERPRETABILITY = frozenset({"nameable", "opaque"})
DERIVATION = frozenset({
    "formula",            # DSP, image statistics, closed word lists, published standards
    "formula-on-model",   # a formula over another model's output
    "trained-on-labels",  # trained or fine-tuned end to end on labels for the quantity
    "probe",              # a small head trained on a frozen general-purpose representation
    "zero-shot",          # similarity to prompt text, no training on labels for the quantity
    "embedding",          # a general-purpose representation
    "bookkeeping",        # describes the extraction, not the stimulus
})
#: Derivations that name a parent model; ``derived_from`` is required for
#: these and forbidden for every other derivation.
NEEDS_PARENT = frozenset({"formula-on-model", "probe", "zero-shot"})
REQUIRED = ("interpretability", "derivation")
#: Every key a label may carry: the Contract B keys plus the optional BIDS
#: tabular-sidecar column keys.
ALLOWED_KEYS = frozenset({
    "interpretability", "derivation", "derived_from", "source_validation",
    "LongName", "Description", "Units", "Levels", "TermURL",
})
SOURCE_VALIDATION_KEYS = frozenset({"citation", "eval_dataset", "metric", "value"})


def resolve(labels: dict, column: str) -> dict:
    """One column's labels: the default, with its ``columns`` entry merged key-wise."""
    out = dict(labels.get("default", {}))
    out.update(labels.get("columns", {}).get(column, {}))
    return out


def for_columns(labels: dict, columns: Iterable[str]) -> dict:
    """The sidecar form for the emitted ``columns``.

    ``columns`` entries for columns this run did not emit are dropped, so
    every key in the written map is a real column of the table.
    """
    emitted = set(columns)
    return {
        "default": copy.deepcopy(labels.get("default", {})),
        "columns": {c: copy.deepcopy(e) for c, e in labels.get("columns", {}).items()
                    if c in emitted},
    }


def problems(labels: dict, columns: Iterable[str] | None = None) -> list[str]:
    """Every violation of the 1.2 rules, as readable strings ([] = valid).

    With ``columns`` (the model's inventory) every column is resolved and
    checked, and every ``columns`` key must be one of them. Without it, the
    default and each override are checked as resolved for their own column.
    """
    out: list[str] = []
    if set(labels) - {"default", "columns"}:
        out.append(f"unknown top-level keys {sorted(set(labels) - {'default', 'columns'})}")
    overrides = labels.get("columns", {})
    if columns is not None:
        inventory = list(columns)
        unknown = sorted(set(overrides) - set(inventory))
        if unknown:
            out.append(f"`columns` keys that are not columns of the model: {unknown[:5]}"
                       + (f" (+{len(unknown) - 5} more)" if len(unknown) > 5 else ""))
    else:
        inventory = list(overrides) or [None]
    for col in inventory:
        resolved = resolve(labels, col) if col is not None else dict(labels.get("default", {}))
        out.extend(f"{col}: {p}" for p in _entry_problems(resolved))
    return out


def _entry_problems(entry: dict) -> list[str]:
    out = [f"missing {k}" for k in REQUIRED if k not in entry]
    extra = set(entry) - ALLOWED_KEYS
    if extra:
        out.append(f"unknown keys {sorted(extra)}")
    if "interpretability" in entry and entry["interpretability"] not in INTERPRETABILITY:
        out.append(f"interpretability {entry['interpretability']!r} not in {sorted(INTERPRETABILITY)}")
    derivation = entry.get("derivation")
    if derivation is not None and derivation not in DERIVATION:
        out.append(f"derivation {derivation!r} not in {sorted(DERIVATION)}")
    if derivation in NEEDS_PARENT:
        if not (isinstance(entry.get("derived_from"), str) and entry["derived_from"].strip()):
            out.append(f"derivation {derivation!r} requires derived_from")
    elif "derived_from" in entry:
        out.append(f"derived_from is forbidden for derivation {derivation!r}")
    for item in entry.get("source_validation", []):
        if not isinstance(item, dict) or set(item) != SOURCE_VALIDATION_KEYS:
            out.append(f"source_validation item must have exactly {sorted(SOURCE_VALIDATION_KEYS)}")
    for key in ("LongName", "Description", "Units", "TermURL"):
        if key in entry and not (isinstance(entry[key], str) and entry[key].strip()):
            out.append(f"{key} must be a non-empty string")
    return out
