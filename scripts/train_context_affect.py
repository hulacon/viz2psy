#!/usr/bin/env python3
"""Train and validate the ``context_affect`` probe on VEATIC.

Developer script. Its weights are not package data, so a user who wants the
model runs it once (see ``viz2psy.models.context_affect``).

Data: VEATIC (Ren et al., WACV 2024; veatic.github.io, research use): 124 silent
clips at 25 fps, ``videos/<id>.mp4``, and per-frame consensus ratings,
``rating_averaged/<id>_{valence,arousal}.csv`` (frame index, value in [-1, 1];
one row per video frame). Some clips are byte-identical copies rated for a
second character; copies form one group here (one video, the targets averaged)
and always share a CV fold.

Subcommands
-----------
embed   The exact runtime path: viz2psy's frame extraction (stage 0 without a
        framing table, 0.5 s grid) into ``ContextAffectModel.embed_batch``.
        One parquet per unique video (``stimulus_id``, ``time``, ``ctx_000``…).
fit     Ridge probe, nested CV grouped by video, on any per-clip feature table
        (the backbone embeddings, or a baseline such as ``emonet`` / ``clip``
        output). Per dimension, the rating delay D and the ridge alpha are
        chosen together by CCC in the inner folds, as in the AVEC 2016
        baseline (Valstar et al. 2016, sec. 4.2): the rating at ``t + D`` is
        the target for the frame at ``t``. Rows are kept only where
        ``t + max(D)`` is still inside the clip, so every D is scored on the
        same rows. Writes ``report.json`` and out-of-fold predictions; with
        ``--write-weights``, also the all-clip fit, with the standardisation
        folded into ``W`` and ``b``.

Nothing here encodes a pass bar: the report states numbers, the caller judges.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

DIMS = ("valence", "arousal")


# ---------------------------------------------------------------------------
# data
# ---------------------------------------------------------------------------

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def video_groups(videos: Path, exclude: set[int]) -> dict[int, list[int]]:
    """``{representative id: [member ids]}`` over byte-identical videos; the rep is the smallest id."""
    by_hash: dict[str, list[int]] = {}
    for p in sorted(videos.glob("*.mp4"), key=lambda p: int(p.stem)):
        by_hash.setdefault(_md5(p), []).append(int(p.stem))
    groups = {min(ids): sorted(ids) for ids in by_hash.values()}
    hit = {rep for rep, ids in groups.items() if exclude & set(ids)}
    if exclude - {i for ids in groups.values() for i in ids}:
        raise SystemExit(f"--exclude names ids with no video: {sorted(exclude)}")
    return {rep: ids for rep, ids in groups.items() if rep not in hit}


def read_rating(ratings: Path, vid: int, dim: str) -> np.ndarray:
    path = ratings / f"{vid}_{dim}.csv"
    df = pd.read_csv(path, header=None, names=["frame", "value"])
    if not np.array_equal(df["frame"].to_numpy(), np.arange(len(df))):
        raise SystemExit(f"{path}: frame column is not 0..n-1")
    return df["value"].to_numpy(dtype=np.float64)


def ratings_sha256(ratings: Path, ids: list[int]) -> str:
    h = hashlib.sha256()
    for vid in sorted(ids):
        for dim in DIMS:
            h.update((ratings / f"{vid}_{dim}.csv").read_bytes())
    return h.hexdigest()


def load_features(features: Path, rep: int, members: list[int], prefix: str) -> pd.DataFrame:
    for vid in [rep, *members]:
        for ext in (".parquet", ".csv"):
            p = features / f"{vid}{ext}"
            if p.is_file():
                df = pd.read_parquet(p) if ext == ".parquet" else pd.read_csv(p)
                cols = [c for c in df.columns if c.startswith(prefix)]
                if not cols:
                    raise SystemExit(f"{p}: no columns start with {prefix!r}")
                return df[["time", *cols]]
    raise SystemExit(f"no feature file for video {rep} (members {members}) in {features}")


def build_rows(args, groups: dict[int, list[int]], lags: np.ndarray):
    """X, per-dimension target matrices over lags, group labels, times, columns."""
    from viz2psy.video import frame_index, get_video_info

    xs, ys, grp, times = [], {d: [] for d in DIMS}, [], []
    cols = None
    for rep, members in groups.items():
        df = load_features(args.features, rep, members, args.prefix)
        cols = cols or [c for c in df.columns if c != "time"]
        if [c for c in df.columns if c != "time"] != cols:
            raise SystemExit(f"video {rep}: feature columns differ from the first video's")
        fps = float(get_video_info(args.videos / f"{rep}.mp4")["fps"])
        rat = {d: np.mean([read_rating(args.ratings, v, d) for v in members], axis=0) for d in DIMS}
        n = len(rat["valence"])
        if len(rat["arousal"]) != n:
            raise SystemExit(f"video {rep}: valence and arousal lengths differ")
        t = df["time"].to_numpy(dtype=np.float64)
        keep = np.array([frame_index(ti + lags.max(), fps) < n for ti in t])
        if not keep.any():
            continue
        t = t[keep]
        xs.append(df.loc[keep, cols].to_numpy(dtype=np.float64))
        idx = np.array([[frame_index(ti + D, fps) for D in lags] for ti in t])
        for d in DIMS:
            ys[d].append(rat[d][idx])
        grp.append(np.full(len(t), rep))
        times.append(t)
    X = np.vstack(xs)
    if not np.isfinite(X).all():
        raise SystemExit("features carry NaN / inf; this probe takes defined rows only")
    return X, {d: np.vstack(ys[d]) for d in DIMS}, np.concatenate(grp), np.concatenate(times), cols


# ---------------------------------------------------------------------------
# ridge + CCC
# ---------------------------------------------------------------------------

def ccc(x: np.ndarray, y: np.ndarray) -> float:
    mx, my = x.mean(), y.mean()
    vx, vy = x.var(), y.var()
    return float(2 * ((x - mx) * (y - my)).mean() / (vx + vy + (mx - my) ** 2))


def group_folds(groups: np.ndarray, k: int, seed: int) -> list[np.ndarray]:
    """k folds of whole groups, balanced by row count (greedy over a seeded order)."""
    uniq, counts = np.unique(groups, return_counts=True)
    order = np.random.default_rng(seed).permutation(len(uniq))
    order = order[np.argsort(-counts[order], kind="stable")]
    load, members = np.zeros(k), [[] for _ in range(k)]
    for i in order:
        f = int(np.argmin(load))
        members[f].append(uniq[i])
        load[f] += counts[i]
    return [np.isin(groups, m) for m in members]


class Ridge:
    """Ridge over every alpha at once from one SVD of the standardised training X."""

    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.mu, sd = X.mean(0), X.std(0)
        self.sd = np.where(sd > 0, sd, 1.0)
        self.ym = Y.mean(0)
        U, self.s, self.Vt = np.linalg.svd((X - self.mu) / self.sd, full_matrices=False)
        self.UtY = U.T @ (Y - self.ym)

    def coef(self, alpha: float) -> np.ndarray:
        return self.Vt.T @ ((self.s / (self.s ** 2 + alpha))[:, None] * self.UtY)

    def predict(self, X: np.ndarray, alpha: float) -> np.ndarray:
        return ((X - self.mu) / self.sd) @ self.coef(alpha) + self.ym

    def folded(self, alpha: float) -> tuple[np.ndarray, np.ndarray]:
        """``W, b`` with the standardisation folded in: ``X @ W + b``."""
        c = self.coef(alpha)
        W = c / self.sd[:, None]
        return W, self.ym - self.mu @ W


def inner_grid(X, Y, groups, k, seed, alphas) -> np.ndarray:
    """Pooled out-of-fold CCC, shape (n_lags, n_alphas)."""
    pred = np.empty((len(alphas), *Y.shape))
    for test in group_folds(groups, k, seed):
        r = Ridge(X[~test], Y[~test])
        for a, alpha in enumerate(alphas):
            pred[a][test] = r.predict(X[test], alpha)
    return np.array([[ccc(pred[a][:, j], Y[:, j]) for a in range(len(alphas))]
                     for j in range(Y.shape[1])])


def summarise(pred, target, groups) -> dict:
    g = pd.DataFrame({"p": pred, "t": target, "g": groups})
    means = g.groupby("g")[["p", "t"]].mean()
    centred = g[["p", "t"]] - g.groupby("g")[["p", "t"]].transform("mean")
    return {"ccc": ccc(pred, target), "r": float(np.corrcoef(pred, target)[0, 1]),
            "ccc_between_clip": ccc(means["p"].to_numpy(), means["t"].to_numpy()),
            "ccc_within_clip": ccc(centred["p"].to_numpy(), centred["t"].to_numpy()),
            "n_rows": int(len(pred)), "n_clips": int(len(means))}


# ---------------------------------------------------------------------------
# subcommands
# ---------------------------------------------------------------------------

def cmd_embed(args) -> int:
    from viz2psy import __version__
    from viz2psy.framing import stage0_for
    from viz2psy.models.context_affect import BACKBONE, ContextAffectModel
    from viz2psy.video import extract_frames

    args.out.mkdir(parents=True, exist_ok=True)
    groups = video_groups(args.videos, set())
    model = ContextAffectModel(device=args.device)
    model.load_backbone()
    for rep, members in groups.items():
        dst = args.out / f"{rep}.parquet"
        if dst.exists():
            continue
        path = args.videos / f"{rep}.mp4"
        frames = extract_frames(path, frame_interval=args.frame_interval, quiet=True,
                                stage0=stage0_for(path, None))
        emb = np.vstack([model.embed_batch([im for _, im in frames[i:i + args.batch]])
                         for i in range(0, len(frames), args.batch)])
        df = pd.DataFrame(emb, columns=[f"ctx_{i:03d}" for i in range(emb.shape[1])])
        df.insert(0, "time", [t for t, _ in frames])
        df.insert(0, "stimulus_id", f"veatic-{rep}")
        tmp = dst.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        tmp.rename(dst)
        print(f"{rep}: {len(df)} frames (members {members})", flush=True)
    (args.out / "embed_manifest.json").write_text(json.dumps({
        "backbone": BACKBONE, "viz2psy_version": __version__,
        "frame_interval_sec": args.frame_interval, "stage0": "no framing table",
        "groups": {str(k): v for k, v in groups.items()},
        "created_at": datetime.now(timezone.utc).isoformat()}, indent=2))
    return 0


def cmd_fit(args) -> int:
    from viz2psy import __version__

    lags = np.round(np.arange(args.lag_min, args.lag_max + 1e-9, args.lag_step), 6)
    alphas = np.logspace(args.alpha_min, args.alpha_max, args.n_alphas)
    groups_map = video_groups(args.videos, set(args.exclude))
    X, Y, groups, times, cols = build_rows(args, groups_map, lags)
    print(f"{len(X)} rows, {len(np.unique(groups))} videos, {X.shape[1]} features", flush=True)
    args.out.mkdir(parents=True, exist_ok=True)

    report = {"features": str(args.features), "prefix": args.prefix, "n_features": X.shape[1],
              "excluded": sorted(args.exclude), "lags_sec": lags.tolist(),
              "alphas": alphas.tolist(), "outer_folds": args.outer, "inner_folds": args.inner,
              "seed": args.seed, "viz2psy_version": __version__, "dims": {}}
    oof_rows = []
    outer = group_folds(groups, args.outer, args.seed)
    for d in DIMS:
        Yd = Y[d]
        chosen = np.empty(len(X), dtype=int)  # lag index per row (its outer fold's choice)
        pred_by_lag = np.empty_like(Yd)        # outer predictions at every lag (curve)
        folds = []
        for f, test in enumerate(outer):
            grid = inner_grid(X[~test], Yd[~test], groups[~test], args.inner, args.seed + 1 + f, alphas)
            j, a = np.unravel_index(np.argmax(grid), grid.shape)
            r = Ridge(X[~test], Yd[~test])
            for jj in range(len(lags)):
                pred_by_lag[test, jj] = r.predict(X[test], alphas[int(np.argmax(grid[jj]))])[:, jj]
            chosen[test] = j
            folds.append({"fold": f, "lag_sec": float(lags[j]), "alpha": float(alphas[a]),
                          "inner_ccc": float(grid[j, a]), "n_test_rows": int(test.sum())})
        rows = np.arange(len(X))
        pred, target = pred_by_lag[rows, chosen], Yd[rows, chosen]
        report["dims"][d] = {
            **summarise(pred, target, groups), "folds": folds,
            "curve_by_lag": [{"lag_sec": float(L), **summarise(pred_by_lag[:, jj], Yd[:, jj], groups)}
                             for jj, L in enumerate(lags)]}
        fold_of = np.empty(len(X), dtype=int)
        for f, test in enumerate(outer):
            fold_of[test] = f
        oof_rows.append(pd.DataFrame({"video": groups, "time": times, "dim": d, "fold": fold_of,
                                      "lag_sec": lags[chosen], "pred": pred, "target": target}))
        print(f"{d}: pooled CCC {report['dims'][d]['ccc']:.3f}", flush=True)

    pd.concat(oof_rows).to_parquet(args.out / "oof_predictions.parquet", index=False)

    if args.write_weights:
        from viz2psy.models.context_affect import (BACKBONE, EMBED_DIM, FEATURE_NAMES,
                                                   WEIGHTS_FORMAT)
        manifest = json.loads((args.features / "embed_manifest.json").read_text())
        if manifest["backbone"] != BACKBONE or X.shape[1] != EMBED_DIM:
            raise SystemExit("--write-weights needs the context_affect backbone embeddings")
        W, b, lag_sec, alpha_sel, final = [], [], [], [], {}
        for d in DIMS:
            grid = inner_grid(X, Y[d], groups, args.inner, args.seed + 1000, alphas)
            j, a = np.unravel_index(np.argmax(grid), grid.shape)
            w_d, b_d = Ridge(X, Y[d][:, [j]]).folded(alphas[a])
            W.append(w_d[:, 0]); b.append(b_d[0])
            lag_sec.append(lags[j]); alpha_sel.append(alphas[a])
            final[d] = {"lag_sec": float(lags[j]), "alpha": float(alphas[a]),
                        "cv_ccc_all_clips": float(grid[j, a])}
        members = [v for ids in groups_map.values() for v in ids]
        trained_on = {"dataset": "VEATIC", "videos": sorted(groups_map),
                      "copies_averaged": {str(k): v for k, v in groups_map.items() if len(v) > 1},
                      "excluded": sorted(args.exclude),
                      "ratings_sha256": ratings_sha256(args.ratings, members),
                      "frame_interval_sec": manifest["frame_interval_sec"], "n_rows": int(len(X)),
                      "lags_sec": lags.tolist(), "viz2psy_version": __version__,
                      "created_at": datetime.now(timezone.utc).isoformat()}
        args.write_weights.parent.mkdir(parents=True, exist_ok=True)
        tmp = args.write_weights.with_name(args.write_weights.name + ".tmp.npz")
        np.savez(tmp, W=np.stack(W, 1), b=np.array(b), lag_sec=np.array(lag_sec),
                 alpha=np.array(alpha_sel), columns=np.array(FEATURE_NAMES),
                 backbone=np.array(BACKBONE), format_version=np.array(WEIGHTS_FORMAT),
                 trained_on=np.array(json.dumps(trained_on)))
        tmp.rename(args.write_weights)
        report["final_fit"] = {**final, "weights": str(args.write_weights)}
        print(f"weights -> {args.write_weights}", flush=True)

    (args.out / "report.json").write_text(json.dumps(report, indent=2))
    return 0


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("embed", help="backbone embeddings at the runtime frame grid")
    e.add_argument("--videos", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    e.add_argument("--frame-interval", type=float, default=0.5)
    e.add_argument("--batch", type=int, default=64)
    e.add_argument("--device", default=None)

    f = sub.add_parser("fit", help="nested grouped CV, optional all-clip weights")
    f.add_argument("--features", type=Path, required=True, help="dir of <id>.parquet|csv per video")
    f.add_argument("--prefix", default="ctx_", help="feature column prefix (e.g. emonet_, clip_)")
    f.add_argument("--videos", type=Path, required=True)
    f.add_argument("--ratings", type=Path, required=True, help="rating_averaged/")
    f.add_argument("--exclude", type=int, nargs="*", default=[], help="video ids to drop")
    f.add_argument("--out", type=Path, required=True)
    f.add_argument("--lag-min", type=float, default=0.0)
    f.add_argument("--lag-max", type=float, default=6.0)
    f.add_argument("--lag-step", type=float, default=0.5)
    f.add_argument("--alpha-min", type=float, default=0.0, help="log10")
    f.add_argument("--alpha-max", type=float, default=6.0, help="log10")
    f.add_argument("--n-alphas", type=int, default=13)
    f.add_argument("--outer", type=int, default=10)
    f.add_argument("--inner", type=int, default=5)
    f.add_argument("--seed", type=int, default=0)
    f.add_argument("--write-weights", type=Path, default=None)

    args = p.parse_args(argv)
    return {"embed": cmd_embed, "fit": cmd_fit}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
