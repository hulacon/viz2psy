"""Framing table: per-file bar boxes, review merge, and the loud lookup.

Synthetic and offline. Videos are written losslessly (FFV1) so a bar's edge
is exact: a lossy codec rings at the bar boundary and the expected box would
then be approximate.
"""

import csv
import json

import numpy as np
import pytest

av = pytest.importorskip("av")

from viz2psy import framing  # noqa: E402
from viz2psy.video import extract_frames, iter_frames  # noqa: E402

FPS = 10
W, H = 128, 96
N = 60  # 6 s -> 13 samples at 0.5 s
RNG = np.random.default_rng(0)


def write_ffv1(path, frames):
    container = av.open(str(path), "w")
    stream = container.add_stream("ffv1", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = W, H, "bgr0"
    for f in frames:
        for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def letterboxed(top, bottom, left=0, right=0, n=N):
    """Bright noise inside the box, black bars outside."""
    frames = []
    for _ in range(n):
        f = np.zeros((H, W, 3), np.uint8)
        f[top:H - bottom, left:W - right] = RNG.integers(60, 256, (H - top - bottom, W - left - right, 3))
        frames.append(f)
    return frames


@pytest.fixture(scope="module")
def vids(tmp_path_factory):
    d = tmp_path_factory.mktemp("framing")
    out = {}
    out["letterbox"] = write_ffv1(d / "letterbox.mkv", letterboxed(12, 12))
    out["pillarbox"] = write_ffv1(d / "pillarbox.mkv", letterboxed(0, 0, 16, 16))
    out["none"] = write_ffv1(d / "none.mkv", letterboxed(0, 0))
    # Subtitles: a text-like strip inside the bottom bar in about 40 % of frames.
    subs = letterboxed(12, 12)
    for i, f in enumerate(subs):
        if i % 5 < 2:
            f[H - 8:H - 4, 40:48] = 230
    out["subtitles"] = write_ffv1(d / "subtitles.mkv", subs)
    # A fade: a third of the frames are black; the box must not move.
    fade = letterboxed(12, 12)
    for f in fade[:20]:
        f[:] = 0
    out["fade"] = write_ffv1(d / "fade.mkv", fade)
    out["large"] = write_ffv1(d / "large.mkv", letterboxed(20, 20))
    out["lopsided"] = write_ffv1(d / "lopsided.mkv", letterboxed(6, 18))
    return out


def box(row, kind="crop"):
    return tuple(int(row[f"{kind}_{e}"]) for e in framing.EDGES)


def test_iter_frames_is_what_extract_frames_scores(vids):
    a = [f for _, f in iter_frames(vids["letterbox"], 0.5)]
    b = [np.asarray(im) for _, im in extract_frames(vids["letterbox"], 0.5, quiet=True)]
    assert len(a) == len(b) == 12  # frame 60 of 0..59 does not exist
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.mark.parametrize("name,expected", [
    ("letterbox", (0, 0, 12, 12)),
    ("pillarbox", (16, 16, 0, 0)),
    ("none", (0, 0, 0, 0)),
])
def test_strict_box_exact_and_unflagged(vids, name, expected):
    r = framing.detect(vids[name], interval=0.5)
    assert not r.get("error")
    assert box(r) == expected
    assert box(r, "robust") == expected
    assert r["flags"] == ""
    assert r["active_w"] == W - expected[0] - expected[1]
    assert r["active_h"] == H - expected[2] - expected[3]


def test_subtitles_in_a_bar_shrink_strict_and_flag(vids):
    r = framing.detect(vids["subtitles"], interval=0.5)
    assert int(r["crop_bottom"]) == 4  # strict keeps every pixel ever lit
    assert int(r["robust_bottom"]) == 12  # sparse text in < half the frames is still bar
    assert "strict_robust_disagree" in r["flags"]


def test_fade_frames_are_skipped(vids):
    r = framing.detect(vids["fade"], interval=0.5)
    assert r["n_lit"] < r["n_sampled"]
    assert box(r) == (0, 0, 12, 12)
    assert r["flags"] == ""


def test_large_and_lopsided_flags(vids):
    assert "large_crop" in framing.detect(vids["large"], interval=0.5)["flags"]  # 40/96 rows
    assert "lopsided" in framing.detect(vids["lopsided"], interval=0.5)["flags"]


def test_missing_file_is_a_row_error(tmp_path):
    r = framing.detect(tmp_path / "nope.mkv")
    assert r["error"]


def _detected(vids, tmp_path, names):
    rows = [framing.detect(vids[n], interval=0.5) for n in names]
    t = tmp_path / "detected.csv"
    framing.write_table(rows, t, framing.DETECT_COLUMNS, {"stage": "detected", "interval": 0.5})
    return t


def _overrides(path, rows):
    cols = ["path", "decision", "reason", "reviewer"] + [f"crop_{e}" for e in framing.EDGES]
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, cols)
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    return path


def test_finalize_refuses_unreviewed_flags(vids, tmp_path):
    t = _detected(vids, tmp_path, ["letterbox", "subtitles"])
    with pytest.raises(framing.FramingError, match="flagged .* but not reviewed"):
        framing.finalize(t, None, tmp_path / "final.csv")
    s = framing.finalize(t, None, tmp_path / "final.csv", allow_unreviewed=True)
    assert s["rows"] == 2


def test_finalize_applies_override_and_keeps_detected(vids, tmp_path):
    t = _detected(vids, tmp_path, ["letterbox", "subtitles"])
    sub = str(vids["subtitles"].resolve())
    ov = _overrides(tmp_path / "ov.csv", [
        {"path": sub, "decision": "override", "reason": "subtitles in the bottom bar",
         "reviewer": "test", "crop_left": 0, "crop_right": 0, "crop_top": 12, "crop_bottom": 12},
        {"path": str(vids["letterbox"].resolve()), "decision": "confirm", "reason": "clean"},
    ])
    out = tmp_path / "final.csv"
    s = framing.finalize(t, ov, out)
    assert (s["overridden"], s["confirmed"]) == (1, 1)
    rows, params = framing.read_table(out)
    r = framing.lookup(rows, vids["subtitles"])
    assert box(r) == (0, 0, 12, 12) and box(r, "detected") == (0, 0, 12, 4)
    assert r["box_source"] == "review" and r["active_h"] == str(H - 24)
    assert params["stage"] == "final" and params["overrides_sha256"]


def test_finalize_needs_a_reason(vids, tmp_path):
    t = _detected(vids, tmp_path, ["letterbox"])
    ov = _overrides(tmp_path / "ov.csv", [
        {"path": str(vids["letterbox"].resolve()), "decision": "confirm", "reason": ""}])
    with pytest.raises(framing.FramingError, match="reason"):
        framing.finalize(t, ov, tmp_path / "final.csv")


def test_edited_table_refuses(vids, tmp_path):
    t = _detected(vids, tmp_path, ["letterbox"])
    t.write_text(t.read_text().replace(",12,12,", ",10,10,", 1))
    with pytest.raises(framing.FramingError, match="edited after it was written"):
        framing.read_table(t)


def test_lookup_is_loud(vids, tmp_path):
    t = _detected(vids, tmp_path, ["letterbox"])
    rows, _ = framing.read_table(t)
    with pytest.raises(framing.FramingError, match="not in the framing table"):
        framing.lookup(rows, vids["pillarbox"])
    rows[str(vids["letterbox"].resolve())]["size_bytes"] = "1"
    with pytest.raises(framing.FramingError, match="changed size"):
        framing.lookup(rows, vids["letterbox"])


def test_contact_sheet_writes_png(vids, tmp_path):
    r = framing.detect(vids["subtitles"], interval=0.5)
    out = framing.contact_sheet(r, tmp_path / "sheet.png", interval=0.5)
    from PIL import Image

    im = Image.open(out)
    assert im.size[0] > 4 * 360


def test_cli_detect_finalize(vids, tmp_path, capsys):
    t = tmp_path / "d.csv"
    rc = framing.main(["detect", str(vids["letterbox"]), str(vids["pillarbox"]),
                       "-o", str(t), "--interval", "0.5"])
    assert rc == 0
    params = json.loads((tmp_path / "d.csv.json").read_text())
    assert params["stage"] == "detected" and params["n_rows"] == 2
    assert params["rule_version"] == framing.RULE_VERSION
    assert framing.main(["finalize", str(t), "-o", str(tmp_path / "f.csv")]) == 0
