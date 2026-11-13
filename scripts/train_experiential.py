#!/usr/bin/env python3
"""Train the ``experiential`` probe: THINGS images -> Binder et al. (2016) ratings.

Developer script. Its weights are not package data (see
``viz2psy.models.experiential``), so a user who wants the model runs it once.

Data
----
- **Ratings:** Binder et al. (2016), ``WordSet1_Ratings.xlsx``, user-supplied and
  read through word2psy's loader (``word2psy.norms.binder``, word2psy >= 0.14.0),
  so both packages apply one rule for duplicates and unrated attributes.
- **Images:** THINGS (Hebart et al., 2019), ``images_THINGS.zip``, read in place
  (nothing is extracted to disk). The archive is encrypted; its password is the
  last line of the password file distributed beside it ("Password for
  images_THINGS.zip: ..."). A THINGS concept trains the probe when its folder
  name, with ``_`` read as a space, is a rated Binder word.

Subcommands
-----------
embed   The exact runtime path (``ExperientialModel.load_backbone`` +
        ``embed_batch``: squash to square, OpenAI preprocessing, L2-normalised
        768-d) over every image of the matched concepts. Writes one ``.npz``:
        ``emb`` (float32, n x 768), ``image`` (path inside ``object_images/``),
        ``concept`` (folder name).
fit     One ridge probe per attribute on those embeddings (word2psy's
        ``fit_probes``: standardised features, alpha per attribute by 5-fold CV
        grouped by concept on ``logspace(-2, 6, 17)``, standardisation folded
        into ``W`` and ``b``). Every image is a row, labelled with its concept's
        ratings; an attribute trains on the concepts that rated it.
        ``--attributes`` keeps a subset (one per line). Writes the weights.

Nothing here encodes a pass bar: which attributes to keep is the caller's call.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = "object_images/"
PASSWORD_PREFIX = "Password for images_THINGS.zip:"


def read_password(path: Path) -> bytes:
    for line in reversed(Path(path).read_text().splitlines()):
        if line.startswith(PASSWORD_PREFIX):
            return line[len(PASSWORD_PREFIX):].strip().encode()
    raise SystemExit(f"{path}: no line starting {PASSWORD_PREFIX!r}")


def concept_word(folder: str) -> str:
    """THINGS folder name -> the Binder word it matches (``hot_dog`` -> ``hot dog``)."""
    return folder.replace("_", " ").lower()


def _binder():
    try:
        from word2psy.norms import binder
        from word2psy.norms.experiential import fit_probes
    except ImportError as e:
        raise SystemExit(f"this script reads the ratings through word2psy >= 0.14.0 ({e})") from e
    return binder, fit_probes


def matched_members(names: list[str], words: set[str]) -> list[str]:
    """Image members of the zip whose concept folder matches a rated word, sorted."""
    out = []
    for n in names:
        if not n.startswith(ROOT) or n.endswith("/"):
            continue
        parts = n[len(ROOT):].split("/")
        if len(parts) == 2 and Path(parts[1]).suffix.lower() in {".jpg", ".jpeg", ".png"} \
                and concept_word(parts[0]) in words:
            out.append(n)
    return sorted(out)


def cmd_embed(a) -> int:
    from PIL import Image

    from viz2psy.models.experiential import BACKBONE, ExperientialModel

    binder, _ = _binder()
    words = set(binder.load_binder(a.ratings)["word"])
    t0 = time.time()
    zf = zipfile.ZipFile(a.things_zip)
    zf.setpassword(read_password(a.password_file))
    members = matched_members(zf.namelist(), words)
    if not members:
        raise SystemExit("no THINGS concept matches a rated word")
    m = ExperientialModel(device=a.device)
    m.load_backbone()
    embs = []
    for i in range(0, len(members), a.batch):
        ims = [Image.open(io.BytesIO(zf.read(n))) for n in members[i:i + a.batch]]
        embs.append(m.embed_batch(ims).astype(np.float32))
    rel = [n[len(ROOT):] for n in members]
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "wb") as f:
        np.savez(f, emb=np.vstack(embs), image=np.array(rel),
                 concept=np.array([r.split("/")[0] for r in rel]))
    print(json.dumps({"backbone": BACKBONE, "n_images": len(rel),
                      "n_concepts": len({r.split('/')[0] for r in rel}),
                      "image_list_sha256": hashlib.sha256("\n".join(rel).encode()).hexdigest(),
                      "seconds": round(time.time() - t0, 1)}))
    return 0


def cmd_fit(a) -> int:
    import viz2psy
    from viz2psy.models import experiential as ex

    binder, fit_probes = _binder()
    df = binder.load_binder(a.ratings).set_index("word")
    with np.load(a.embeddings, allow_pickle=False) as z:
        emb, image, concept = z["emb"].astype(np.float64), z["image"], z["concept"]
    if emb.shape[1] != ex.EMBED_DIM:
        raise SystemExit(f"{a.embeddings}: {emb.shape[1]}-d embeddings, expected {ex.EMBED_DIM}")
    words = np.array([concept_word(c) for c in concept])
    unrated = sorted(set(words) - set(df.index))
    if unrated:
        raise SystemExit(f"concepts without ratings: {unrated[:5]}")
    known = [binder.column_name(x) for x in binder.ATTRIBUTES]
    if a.attributes:
        keep = {s.strip().removeprefix(ex.PREFIX).lower() for s in a.attributes.read_text().splitlines()
                if s.strip() and not s.lstrip().startswith("#")}
        bad = sorted(keep - set(known))
        if bad:
            raise SystemExit(f"unknown attributes {bad}")
        attrs = [x for x in known if x in keep]
    else:
        attrs = known
    Y = df.loc[words, attrs].to_numpy(np.float64)
    fit = fit_probes(emb, Y, groups=words)
    cols = [ex.PREFIX + x for x in attrs]
    trained_on = {
        "ratings": "Binder et al. (2016) WordSet1_Ratings.xlsx", "citation": ex.CITATION,
        "ratings_sha256": binder.file_sha256(a.ratings),
        "images": "THINGS (Hebart et al. 2019) object_images, concepts matching a rated word",
        "n_images": int(len(image)), "n_concepts": int(len(set(words))),
        "image_list_sha256": hashlib.sha256("\n".join(map(str, image)).encode()).hexdigest(),
        "embeddings_sha256": hashlib.sha256(Path(a.embeddings).read_bytes()).hexdigest(),
        "rows": "one per image, labelled with its concept's ratings",
        "n_train_images": dict(zip(cols, map(int, fit["n_train"]))),
        "alpha_grid": np.logspace(-2, 6, 17).tolist(),
        "alpha_selection": "5-fold CV grouped by concept, seed 0, pooled SSE",
        "viz2psy": viz2psy.__version__,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    a.write_weights.parent.mkdir(parents=True, exist_ok=True)
    with open(a.write_weights, "wb") as f:
        np.savez(f, format_version=np.array(ex.WEIGHTS_FORMAT), backbone=np.array(ex.BACKBONE),
                 columns=np.array(cols), W=fit["W"], b=fit["b"], alpha=fit["alpha"],
                 trained_on=np.array(json.dumps(trained_on)))
    ex.read_weights(a.write_weights)  # the file the model will read must validate
    edge = [c for c, e in zip(cols, fit["alpha_at_edge"]) if e]
    print(f"wrote {a.write_weights}: {len(cols)} attributes, {len(image)} images, "
          f"{len(set(words))} concepts" + (f"; alpha at a grid edge for {edge}" if edge else ""))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("embed", help="ViT-L-14 embeddings of the matched THINGS images")
    e.add_argument("--ratings", required=True)
    e.add_argument("--things-zip", required=True, type=Path)
    e.add_argument("--password-file", required=True, type=Path)
    e.add_argument("--out", required=True, type=Path)
    e.add_argument("--batch", type=int, default=64)
    e.add_argument("--device", default=None)
    f = sub.add_parser("fit", help="ridge probes -> weights file")
    f.add_argument("--ratings", required=True)
    f.add_argument("--embeddings", required=True, type=Path)
    f.add_argument("--attributes", type=Path, help="file: one attribute per line (default all 65)")
    f.add_argument("--write-weights", required=True, type=Path)
    a = ap.parse_args(argv)
    return {"embed": cmd_embed, "fit": cmd_fit}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
