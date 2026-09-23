"""`viz2psy sidecar refresh`: bring existing sidecars up to schema 1.1.

Contract B 1.1 adds a `nulls` map to every model entry (what a NaN in each
column means). Feature values do not change between 1.0 and 1.1, so an old
scores file is brought forward by rewriting its `.meta.json` only; the CSV is
never written.

A refreshed sidecar is checked against the table it describes: each model's
columns are read back from the recorded `output` CSV, and a column holding
NaN without a declaration refuses the refresh for that sidecar (nothing is
written), so a producer defect surfaces here rather than in a downstream fit.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from viz2psy.exceptions import Viz2PsyError
from viz2psy.metadata import SCHEMA_VERSION, declared_nulls


@dataclass
class RefreshResult:
    path: Path
    status: str  # "refreshed" | "unchanged" | "refused" | "skipped"
    undeclared: dict[str, list[str]] = field(default_factory=dict)  # model -> NaN columns
    note: str = ""


def find_sidecars(paths: list[str | Path]) -> list[Path]:
    """Sidecar files named directly, plus every `*.meta.json` under a directory."""
    found: list[Path] = []
    for p in map(Path, paths):
        if p.is_dir():
            found.extend(sorted(p.rglob("*.meta.json")))
        elif p.name.endswith(".meta.json") and p.is_file():
            found.append(p)
        else:
            raise Viz2PsyError(f"{p} is neither a directory nor an existing .meta.json sidecar")
    return found


def _table_path(sidecar: Path, recorded: str) -> Path:
    """The recorded CSV, or the same filename beside the sidecar if the tree moved."""
    p = Path(recorded)
    if p.is_file():
        return p
    beside = sidecar.parent / p.name
    if beside.is_file():
        return beside
    raise Viz2PsyError(
        f"{sidecar}: output table {recorded} is missing (also not beside the sidecar). "
        "The refresh checks nulls against the data, so it cannot proceed without it."
    )


def _model_columns(name: str, entry: dict, table_columns: list[str]) -> set[str]:
    """A model's columns: its listed inventory, else the table columns under its prefixes."""
    features = entry.get("features") or {}
    if "columns" in features:
        return set(features["columns"]) & set(table_columns)
    prefixes = [p + "_" for p in entry.get("prefixes", [name])]
    if "pattern" in features:
        prefixes.append(features["pattern"].split("{")[0])
    return {c for c in table_columns if any(c.startswith(p) for p in prefixes)}


def refresh_sidecar(path: str | Path, *, dry_run: bool = False) -> RefreshResult:
    import pandas as pd

    path = Path(path)
    meta = json.loads(path.read_text())
    if meta.get("extractor") != "viz2psy":
        return RefreshResult(path, "skipped", note=f"extractor {meta.get('extractor')!r}")
    table = pd.read_csv(_table_path(path, meta["output"]["path"]))
    # numeric columns only: a string column's empty cell reads back as NaN
    num = table.select_dtypes(include="number")
    nan_cols = {c for c in num.columns if num[c].isna().any()}

    new = copy.deepcopy(meta)
    undeclared: dict[str, list[str]] = {}
    for name, entry in new["models"].items():
        cols = _model_columns(name, entry, list(table.columns))
        entry["nulls"] = {c: e for c, e in declared_nulls(name).items() if c in cols}
        bad = sorted((cols & nan_cols) - set(entry["nulls"]))
        if bad:
            undeclared[name] = bad
    if undeclared:
        return RefreshResult(path, "refused", undeclared,
                             note="NaN in undeclared column(s): a producer defect; nothing written")
    new["schema_version"] = SCHEMA_VERSION
    if new == meta:
        return RefreshResult(path, "unchanged")
    from viz2psy import __version__

    new.setdefault("refreshed", []).append({
        "by": f"viz2psy {__version__}",
        "at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fields": ["schema_version", "models.*.nulls"],
        "from_schema_version": meta.get("schema_version"),
    })
    if not dry_run:
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(new, indent=2))
        os.replace(tmp, path)
    return RefreshResult(path, "refreshed")


def main(argv: list[str]) -> int:
    """`viz2psy sidecar refresh PATH... [--dry-run]`."""
    import argparse
    import sys
    from collections import Counter

    parser = argparse.ArgumentParser(prog="viz2psy sidecar",
                                     description="Maintain existing .meta.json sidecars.")
    sub = parser.add_subparsers(dest="sidecar_cmd")
    p_rf = sub.add_parser(
        "refresh",
        help="Bring sidecars to the current Contract B schema (1.1: per-model `nulls`). "
             "Rewrites JSON only, never a CSV; refuses a sidecar whose table holds NaN in "
             "an undeclared column.",
    )
    p_rf.add_argument("paths", nargs="+",
                      help=".meta.json files, or directories searched recursively "
                           "(sidecars from other extractors are skipped)")
    p_rf.add_argument("--dry-run", action="store_true", help="Check and report; write nothing.")
    args = parser.parse_args(argv)
    if args.sidecar_cmd is None:
        parser.print_help()
        return 1
    counts: Counter = Counter()
    for sidecar in find_sidecars(args.paths):
        r = refresh_sidecar(sidecar, dry_run=args.dry_run)
        counts[r.status] += 1
        if r.status == "refused":
            cols = "; ".join(f"{m}: {', '.join(c)}" for m, c in r.undeclared.items())
            print(f"REFUSED {sidecar}: {cols}", file=sys.stderr)
    verb = "would refresh" if args.dry_run else "refreshed"
    print(f"viz2psy sidecar refresh: {counts['refreshed']} {verb}, {counts['unchanged']} unchanged, "
          f"{counts['refused']} refused, {counts['skipped']} skipped (other extractors)")
    return 1 if counts["refused"] else 0
