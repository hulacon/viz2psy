"""`viz2psy framing`: one letterbox/pillarbox crop box per video file, reviewed before use.

Black bars are a property of the file, not of the picture, and every model
reads them as content: frame-relative features (gist's grid, saliency's map,
face and object coverage, depth fractions, low-level statistics, motion) are
rescaled by them, and square-input networks squeeze black into the square. So
before extraction each video gets one crop box: its *active picture*.

The box is detected once per file and written to a **framing table**, never
detected on the fly while scoring, because a dark scene or a fade looks like a
bar in any single frame. The table is then reviewed by a person (contact
sheets, `viz2psy framing sheet`) and corrections are merged from a hand-edited
overrides file (`viz2psy framing finalize`), so the detector stays re-runnable
and every manual change stays visible.

Detection, per file, on the frames viz2psy itself scores (:func:`viz2psy.video.iter_frames`,
so the box is in the same pixel coordinates the models see):

* sample every ``interval`` seconds (default 1 s);
* luma = BT.601 weights over the decoded RGB; a pixel is *dark* at luma <= ``dark``;
* frames that are dark everywhere (fades, black cards) are skipped;
* **strict box** (the one used): a row/column is a bar when it is dark in every
  remaining frame, counted inward from each edge. It never crops a pixel that
  was lit in any sampled frame;
* **robust box** (flagging only): in one frame a row/column is *sparse* when
  at most ``sparse`` (default 25 %) of its pixels are lit; it is a bar when it
  is sparse in at least ``robust`` (default half) of the remaining frames.
  Subtitles, credits or a logo printed over a bar shrink the strict box but
  not the robust one, so a disagreement flags the file for review. The robust
  box can over-crop a very dark film; it is never used, only flagged.

Flags: ``strict_robust_disagree`` (any edge differs by more than
max(2 px, 1 % of that dimension)); ``large_crop`` (the box removes more than
30 % of the frame); ``lopsided`` (opposite bars differ by more than
max(2 px, 1 %) and both are non-zero); ``no_frames`` (nothing lit was decoded).

The table stores the box in decoded pixel coordinates. Squaring the pixels
(the stream's sample aspect ratio, ``sar``) happens after the crop; the
active picture's display aspect is ``active_w * sar / active_h``.

A table is keyed on the source's resolved absolute path, and records its size
in bytes, so a file that changed or moved after detection is a loud error, not
a stale box.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from viz2psy.exceptions import Viz2PsyError

#: Bump when the detection rule (not just a default) changes meaning.
RULE_VERSION = "1"
DEFAULT_INTERVAL = 1.0
DEFAULT_DARK = 24.0
DEFAULT_ROBUST = 0.5
DEFAULT_SPARSE = 0.25
LARGE_CROP = 0.30
LUMA = np.array([0.299, 0.587, 0.114])
EDGES = ("left", "right", "top", "bottom")

#: Columns of a detected table, in order. `finalize` adds the review columns.
DETECT_COLUMNS = [
    "path", "size_bytes", "width", "height", "sar",
    "n_sampled", "n_lit",
    "crop_left", "crop_right", "crop_top", "crop_bottom",
    "robust_left", "robust_right", "robust_top", "robust_bottom",
    "active_w", "active_h", "display_aspect", "removed_frac",
    "flags", "error",
]
REVIEW_COLUMNS = ["box_source", "review", "review_reason", "reviewer"]
FINAL_COLUMNS = DETECT_COLUMNS + [f"detected_{e}" for e in EDGES] + REVIEW_COLUMNS


class FramingError(Viz2PsyError):
    """A framing table is missing, stale, or does not cover a source."""


def _edge_run(is_bar: np.ndarray) -> tuple[int, int]:
    """Lengths of the leading and trailing runs of True."""
    n = len(is_bar)
    not_bar = np.flatnonzero(~is_bar)
    if len(not_bar) == 0:
        return n, 0
    return int(not_bar[0]), int(n - 1 - not_bar[-1])


def _tol(dim: int) -> float:
    return max(2.0, 0.01 * dim)


@dataclass
class LitCounts:
    """Per row and column: frames in which the line was lit at all, and frames
    in which it was sparse (at most ``sparse`` of its pixels lit)."""

    rows: np.ndarray
    cols: np.ndarray
    rows_sparse: np.ndarray
    cols_sparse: np.ndarray
    n_sampled: int = 0
    n_lit: int = 0

    @classmethod
    def accumulate(cls, frames, dark: float = DEFAULT_DARK,
                   sparse: float = DEFAULT_SPARSE) -> "LitCounts":
        acc = None
        for frame in frames:
            luma = frame.astype(np.float32) @ LUMA.astype(np.float32)
            if acc is None:
                h, w = luma.shape
                acc = cls(*(np.zeros(n, np.int64) for n in (h, w, h, w)))
            elif luma.shape != (len(acc.rows), len(acc.cols)):
                raise FramingError(f"frame size changed mid-stream: {luma.shape} after "
                                   f"{(len(acc.rows), len(acc.cols))}")
            acc.n_sampled += 1
            lit = luma > dark
            if not lit.any():
                continue  # a fade or a black card says nothing about the bars
            acc.n_lit += 1
            acc.rows += lit.any(axis=1)
            acc.cols += lit.any(axis=0)
            acc.rows_sparse += lit.mean(axis=1) <= sparse
            acc.cols_sparse += lit.mean(axis=0) <= sparse
        if acc is None:
            raise FramingError("no frames decoded")
        return acc

    def strict_box(self) -> dict[str, int]:
        """Bars: lines never lit in any lit frame."""
        top, bottom = _edge_run(self.rows == 0)
        left, right = _edge_run(self.cols == 0)
        return {"left": left, "right": right, "top": top, "bottom": bottom}

    def robust_box(self, robust: float = DEFAULT_ROBUST) -> dict[str, int]:
        """Bars: lines sparse in at least ``robust`` of the lit frames."""
        need = robust * self.n_lit
        top, bottom = _edge_run(self.rows_sparse >= need)
        left, right = _edge_run(self.cols_sparse >= need)
        return {"left": left, "right": right, "top": top, "bottom": bottom}


def flags_for(strict: dict, robust: dict, width: int, height: int, n_lit: int) -> list[str]:
    if n_lit == 0:
        return ["no_frames"]
    flags = []
    dims = {"left": width, "right": width, "top": height, "bottom": height}
    if any(abs(strict[e] - robust[e]) > _tol(dims[e]) for e in EDGES):
        flags.append("strict_robust_disagree")
    aw = width - strict["left"] - strict["right"]
    ah = height - strict["top"] - strict["bottom"]
    if aw <= 0 or ah <= 0 or 1 - (aw * ah) / (width * height) > LARGE_CROP:
        flags.append("large_crop")
    for a, b, dim in (("top", "bottom", height), ("left", "right", width)):
        if strict[a] and strict[b] and abs(strict[a] - strict[b]) > _tol(dim):
            flags.append("lopsided")
            break
    return flags


def detect(video_path: str | Path, interval: float = DEFAULT_INTERVAL,
           dark: float = DEFAULT_DARK, robust: float = DEFAULT_ROBUST,
           sparse: float = DEFAULT_SPARSE) -> dict:
    """One detected-table row for ``video_path``. Errors are recorded, never raised."""
    from viz2psy.video import get_pixel_aspect, iter_frames

    path = Path(video_path).resolve()
    row: dict = {"path": str(path)}
    try:
        row["size_bytes"] = path.stat().st_size
        row["sar"] = round(get_pixel_aspect(path), 6)
        acc = LitCounts.accumulate((f for _, f in iter_frames(path, interval)), dark, sparse)
        h, w = len(acc.rows), len(acc.cols)
        strict = acc.strict_box()
        rob = acc.robust_box(robust)
        aw = w - strict["left"] - strict["right"]
        ah = h - strict["top"] - strict["bottom"]
        row.update(width=w, height=h, n_sampled=acc.n_sampled, n_lit=acc.n_lit,
                   **{f"crop_{e}": strict[e] for e in EDGES},
                   **{f"robust_{e}": rob[e] for e in EDGES},
                   active_w=aw, active_h=ah,
                   display_aspect=round(aw * row["sar"] / ah, 4) if ah > 0 else "",
                   removed_frac=round(1 - max(aw, 0) * max(ah, 0) / (w * h), 4),
                   flags=";".join(flags_for(strict, rob, w, h, acc.n_lit)))
    except Exception as e:  # noqa: BLE001 -- recorded per row, and a table with errors cannot finalize
        row["error"] = f"{type(e).__name__}: {e}"[:300]
    return row


# --------------------------------------------------------------------------- table I/O

def _params_path(table: Path) -> Path:
    return table.with_name(table.name + ".json")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_table(rows: list[dict], out: Path, columns: list[str], params: dict) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".tmp")
    with open(tmp, "w", newline="") as fh:
        wr = csv.DictWriter(fh, columns, extrasaction="raise")
        wr.writeheader()
        for r in sorted(rows, key=lambda r: r["path"]):
            wr.writerow({c: r.get(c, "") for c in columns})
    os.replace(tmp, out)
    _params_path(out).write_text(json.dumps({**params, "sha256": sha256(out), "n_rows": len(rows)},
                                            indent=2))


def read_table(table: str | Path) -> tuple[dict[str, dict], dict]:
    """Rows keyed on resolved path, and the table's parameter record."""
    table = Path(table)
    if not table.is_file():
        raise FramingError(f"framing table not found: {table}")
    pp = _params_path(table)
    if not pp.is_file():
        raise FramingError(f"framing table {table} has no parameter record {pp}")
    params = json.loads(pp.read_text())
    if params.get("sha256") != sha256(table):
        raise FramingError(f"framing table {table} does not match the sha256 in {pp}: "
                           "it was edited after it was written. Put corrections in an "
                           "overrides file and re-run `viz2psy framing finalize`.")
    with open(table, newline="") as fh:
        rows = {r["path"]: r for r in csv.DictReader(fh)}
    return rows, params


def lookup(rows: dict[str, dict], video_path: str | Path) -> dict:
    """The table row for ``video_path``; a missing or changed source is a loud error."""
    path = Path(video_path).resolve()
    r = rows.get(str(path))
    if r is None:
        raise FramingError(f"{path} is not in the framing table; run `viz2psy framing detect` "
                           "on it and review it before extracting")
    if r.get("error"):
        raise FramingError(f"{path}: the framing table records a detection error: {r['error']}")
    if int(r["size_bytes"]) != path.stat().st_size:
        raise FramingError(f"{path} changed size since detection ({r['size_bytes']} -> "
                           f"{path.stat().st_size} bytes); re-detect and re-review it")
    return r


# --------------------------------------------------------------------------- finalize

def finalize(detected: Path, overrides: Path | None, out: Path,
             allow_unreviewed: bool = False) -> dict:
    """Merge a reviewed overrides file into a detected table.

    Overrides CSV columns: ``path, decision, reason, reviewer`` plus, for
    ``decision=override``, ``crop_left, crop_right, crop_top, crop_bottom``.
    ``decision=confirm`` records that the detected box was looked at and kept.
    Every flagged row needs a decision unless ``allow_unreviewed``; a row with
    a detection error always refuses.
    """
    rows, params = read_table(detected)
    if params.get("stage") != "detected":
        raise FramingError(f"{detected} is not a detected table (stage={params.get('stage')!r})")
    ov: dict[str, dict] = {}
    if overrides is not None:
        with open(overrides, newline="") as fh:
            for o in csv.DictReader(fh):
                p = str(Path(o["path"]).resolve())
                if p in ov:
                    raise FramingError(f"overrides name {p} twice")
                if p not in rows:
                    raise FramingError(f"overrides name {p}, which is not in {detected}")
                if o.get("decision") not in ("confirm", "override"):
                    raise FramingError(f"{p}: decision must be 'confirm' or 'override', "
                                       f"not {o.get('decision')!r}")
                if not (o.get("reason") or "").strip():
                    raise FramingError(f"{p}: every review decision needs a reason")
                ov[p] = o
    problems = []
    out_rows = []
    for p, r in rows.items():
        r = dict(r)
        if r.get("error"):
            problems.append(f"{p}: detection error ({r['error']})")
            continue
        for e in EDGES:
            r[f"detected_{e}"] = r[f"crop_{e}"]
        o = ov.get(p)
        if o is None:
            r.update(box_source="detector", review="")
            if r["flags"] and not allow_unreviewed:
                problems.append(f"{p}: flagged ({r['flags']}) but not reviewed")
        else:
            r.update(review=o["decision"], review_reason=o["reason"].strip(),
                     reviewer=(o.get("reviewer") or "").strip())
            if o["decision"] == "override":
                w, h = int(r["width"]), int(r["height"])
                box = {e: int(o[f"crop_{e}"]) for e in EDGES}
                aw, ah = w - box["left"] - box["right"], h - box["top"] - box["bottom"]
                if min(box.values()) < 0 or aw <= 0 or ah <= 0:
                    raise FramingError(f"{p}: override box {box} leaves no picture in {w}x{h}")
                sar = float(r["sar"])
                r.update(box_source="review", **{f"crop_{e}": box[e] for e in EDGES},
                         active_w=aw, active_h=ah, display_aspect=round(aw * sar / ah, 4),
                         removed_frac=round(1 - aw * ah / (w * h), 4))
            else:
                r["box_source"] = "detector"
        out_rows.append(r)
    if problems:
        raise FramingError(f"{len(problems)} row(s) cannot be finalized:\n  " + "\n  ".join(problems))
    from viz2psy import __version__
    write_table(out_rows, out, FINAL_COLUMNS, {
        **{k: v for k, v in params.items() if k not in ("sha256", "n_rows", "stage")},
        "stage": "final",
        "detected_table": str(Path(detected).resolve()),
        "detected_sha256": params["sha256"],
        "overrides": str(Path(overrides).resolve()) if overrides else None,
        "overrides_sha256": sha256(Path(overrides)) if overrides else None,
        "finalized_by": f"viz2psy {__version__}",
        "finalized_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "allow_unreviewed": allow_unreviewed,
    })
    return {"rows": len(out_rows), "overridden": sum(r["box_source"] == "review" for r in out_rows),
            "confirmed": sum(r["review"] == "confirm" for r in out_rows)}


# --------------------------------------------------------------------------- contact sheets

def contact_sheet(row: dict, out: Path, n_frames: int = 11, panel_w: int = 360,
                  interval: float = DEFAULT_INTERVAL) -> Path:
    """One PNG: the per-pixel max over every sampled frame, then ``n_frames``
    evenly spaced frames, each with the box drawn (red) and the cropped region
    dimmed. The max panel shows anything that ever appeared inside a bar
    (subtitles, credits, a logo)."""
    from PIL import Image, ImageDraw

    from viz2psy.video import iter_frames

    path = Path(row["path"])
    n_total = int(row["n_sampled"])
    pick = set(np.linspace(0, n_total - 1, n_frames).round().astype(int).tolist())
    vmax, picked = None, []
    for i, (t, f) in enumerate(iter_frames(path, interval)):  # streamed: films are long
        vmax = f.copy() if vmax is None else np.maximum(vmax, f, out=vmax)
        if i in pick:
            picked.append((f"t={t:.0f}s", f))
    if vmax is None:
        raise FramingError(f"{path}: no frames decoded for the contact sheet")
    panels = [("max over all", vmax)] + picked

    h, w = vmax.shape[:2]
    box = {e: int(row[f"crop_{e}"]) for e in EDGES}
    x0, y0, x1, y1 = box["left"], box["top"], w - box["right"], h - box["bottom"]
    scale = panel_w / w
    ph = max(1, round(h * scale))
    cols = 4
    nrows = -(-len(panels) // cols)
    head = 54
    sheet = Image.new("RGB", (cols * (panel_w + 6) + 6, head + nrows * (ph + 22) + 6), (60, 60, 60))
    d = ImageDraw.Draw(sheet)
    d.text((8, 6), f"{path.name}  {w}x{h} sar {row['sar']}", fill=(255, 255, 255))
    d.text((8, 22), f"crop L{box['left']} R{box['right']} T{box['top']} B{box['bottom']} "
                    f"({row.get('box_source') or 'detector'})   robust L{row.get('robust_left')} "
                    f"R{row.get('robust_right')} T{row.get('robust_top')} B{row.get('robust_bottom')}   "
                    f"removed {float(row['removed_frac']):.1%}   aspect {row['display_aspect']}",
           fill=(255, 255, 255))
    d.text((8, 38), f"flags: {row.get('flags') or '-'}", fill=(255, 200, 80))
    for k, (label, arr) in enumerate(panels):
        img = Image.fromarray(arr).convert("RGB")
        dim = Image.new("RGB", img.size, (0, 0, 0))
        mask = Image.new("L", img.size, 110)
        ImageDraw.Draw(mask).rectangle([x0, y0, x1 - 1, y1 - 1], fill=0)
        img = Image.composite(dim, img, mask).resize((panel_w, ph), Image.BILINEAR)
        ImageDraw.Draw(img).rectangle(
            [round(x0 * scale), round(y0 * scale), round(x1 * scale) - 1, round(y1 * scale) - 1],
            outline=(255, 0, 0))
        cx = 6 + (k % cols) * (panel_w + 6)
        cy = head + (k // cols) * (ph + 22)
        sheet.paste(img, (cx, cy))
        d.text((cx, cy + ph + 4), label, fill=(220, 220, 220))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out)
    return out


# --------------------------------------------------------------------------- CLI

def _read_list(paths: list[str], list_file: str | None) -> list[str]:
    items = list(paths)
    if list_file:
        items += [ln.strip() for ln in open(list_file) if ln.strip() and not ln.startswith("#")]
    if not items:
        raise Viz2PsyError("no video paths given (positional or --list)")
    return items


def main(argv: list[str]) -> int:
    """`viz2psy framing {detect,finalize,sheet}`."""
    import argparse
    import sys
    from concurrent.futures import ProcessPoolExecutor

    from viz2psy import __version__

    parser = argparse.ArgumentParser(prog="viz2psy framing",
                                     description="Per-file letterbox/pillarbox crop boxes "
                                                 "(the framing table), reviewed before use.")
    sub = parser.add_subparsers(dest="framing_cmd")

    p_d = sub.add_parser("detect", help="Detect one bar box per video; write a detected table.")
    p_d.add_argument("paths", nargs="*", help="video files")
    p_d.add_argument("--list", help="file with one video path per line")
    p_d.add_argument("-o", "--output", required=True, type=Path, help="table CSV to write")
    p_d.add_argument("--interval", type=float, default=DEFAULT_INTERVAL,
                     help=f"seconds between sampled frames (default {DEFAULT_INTERVAL})")
    p_d.add_argument("--dark", type=float, default=DEFAULT_DARK,
                     help=f"luma at or below which a pixel is dark (default {DEFAULT_DARK})")
    p_d.add_argument("--robust", type=float, default=DEFAULT_ROBUST,
                     help=f"robust box: a line is a bar when sparse in this share of lit frames "
                          f"(default {DEFAULT_ROBUST}; flags only)")
    p_d.add_argument("--sparse", type=float, default=DEFAULT_SPARSE,
                     help=f"robust box: a line is sparse in a frame when at most this share of "
                          f"its pixels is lit (default {DEFAULT_SPARSE})")
    p_d.add_argument("-j", "--jobs", type=int, default=1, help="parallel processes")

    p_f = sub.add_parser("finalize", help="Merge reviewed overrides into a detected table.")
    p_f.add_argument("detected", type=Path)
    p_f.add_argument("--overrides", type=Path, default=None)
    p_f.add_argument("-o", "--output", required=True, type=Path)
    p_f.add_argument("--allow-unreviewed", action="store_true",
                     help="finalize even though some flagged rows have no review decision")

    p_s = sub.add_parser("sheet", help="Contact sheets (PNG) for review.")
    p_s.add_argument("table", type=Path)
    p_s.add_argument("--out-dir", required=True, type=Path)
    p_s.add_argument("--select", default="flagged",
                     help="'all', 'flagged', or a substring every chosen path contains "
                          "(repeatable as comma list); default 'flagged'")
    p_s.add_argument("-j", "--jobs", type=int, default=1)

    args = parser.parse_args(argv)
    if args.framing_cmd is None:
        parser.print_help()
        return 1

    if args.framing_cmd == "detect":
        items = _read_list(args.paths, args.list)
        kw = dict(interval=args.interval, dark=args.dark, robust=args.robust, sparse=args.sparse)
        rows = []
        with ProcessPoolExecutor(args.jobs) as ex:
            futs = [ex.submit(detect, p, **kw) for p in items]
            for i, f in enumerate(futs):
                rows.append(f.result())
                if (i + 1) % 50 == 0:
                    print(f"{i + 1}/{len(items)}", file=sys.stderr, flush=True)
        write_table(rows, args.output, DETECT_COLUMNS, {
            "stage": "detected", "rule_version": RULE_VERSION, **kw,
            "luma": "BT.601 (0.299, 0.587, 0.114) over decoded RGB",
            "large_crop": LARGE_CROP, "detected_by": f"viz2psy {__version__}",
            "detected_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        n_err = sum(bool(r.get("error")) for r in rows)
        n_flag = sum(bool(r.get("flags")) for r in rows)
        print(f"viz2psy framing detect: {len(rows)} rows, {n_flag} flagged, {n_err} errors "
              f"-> {args.output}")
        return 1 if n_err else 0

    if args.framing_cmd == "finalize":
        try:
            s = finalize(args.detected, args.overrides, args.output, args.allow_unreviewed)
        except FramingError as e:
            print(f"viz2psy framing finalize: REFUSED: {e}", file=sys.stderr)
            return 1
        print(f"viz2psy framing finalize: {s['rows']} rows, {s['overridden']} overridden, "
              f"{s['confirmed']} confirmed -> {args.output}")
        return 0

    rows, params = read_table(args.table)
    sel = args.select.split(",")
    chosen = [r for p, r in rows.items() if not r.get("error") and (
        "all" in sel or ("flagged" in sel and r.get("flags")) or any(s in p for s in sel))]
    interval = float(params.get("interval", DEFAULT_INTERVAL))
    with ProcessPoolExecutor(args.jobs) as ex:
        futs = {ex.submit(contact_sheet, r, args.out_dir / (Path(r["path"]).stem + ".png"),
                          interval=interval): r["path"] for r in chosen}
        for f in futs:
            f.result()
    print(f"viz2psy framing sheet: {len(chosen)} sheets -> {args.out_dir}")
    return 0
