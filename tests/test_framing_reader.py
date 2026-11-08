"""The framing table's reader: stage 0 (bar crop + square pixels) and the
per-model stage 1 (squash for square-input networks, llstat's canonical area).

Synthetic and offline. Videos are lossless (FFV1) so a crop is exact.
"""

import json
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

av = pytest.importorskip("av")

from viz2psy import cli, framing  # noqa: E402
from viz2psy.models.base import CANONICAL_AREA, canonical_size, squash_to_square  # noqa: E402
from viz2psy.video import extract_frames, iter_frames  # noqa: E402

FPS = 10
W, H = 128, 96
TOP = BOTTOM = 12
RNG = np.random.default_rng(1)


def write_ffv1(path, frames, sar=None, w=W, h=H):
    container = av.open(str(path), "w")
    stream = container.add_stream("ffv1", rate=FPS)
    stream.width, stream.height, stream.pix_fmt = w, h, "bgr0"
    if sar is not None:
        stream.codec_context.sample_aspect_ratio = sar
    for f in frames:
        for packet in stream.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return path


def letterboxed(n=40, w=W, h=H, top=TOP, bottom=BOTTOM):
    frames = []
    for _ in range(n):
        f = np.zeros((h, w, 3), np.uint8)
        f[top:h - bottom] = RNG.integers(60, 256, (h - top - bottom, w, 3))
        frames.append(f)
    return frames


def table_for(videos, out):
    rows = [framing.detect(v) for v in videos]
    detected = out.with_name("detected.csv")
    framing.write_table(rows, detected, framing.DETECT_COLUMNS,
                        {"rule_version": framing.RULE_VERSION, "stage": "detected"})
    framing.finalize(detected, None, out, allow_unreviewed=True)
    return out


@pytest.fixture(scope="module")
def media(tmp_path_factory):
    d = tmp_path_factory.mktemp("reader")
    lb = write_ffv1(d / "letterbox.mkv", letterboxed())
    other = write_ffv1(d / "other.mkv", letterboxed())
    table = table_for([lb], d / "framing_table.csv")
    return {"dir": d, "letterbox": lb, "other": other, "table": table}


# ----------------------------------------------------------------- stage 0

def test_stage0_crops_then_squares_pixels():
    s0 = framing.Stage0(72, 48, 8 / 9, {"left": 3, "right": 3, "top": 6, "bottom": 6})
    assert s0.crop_wh == (66, 36)
    assert s0.active_wh == (int(round(66 * 8 / 9)), 36)
    frame = RNG.integers(0, 256, (48, 72, 3), dtype=np.uint8)
    out = s0.apply(frame)
    expected = np.asarray(Image.fromarray(frame[6:42, 3:69]).resize(s0.active_wh, Image.BICUBIC))
    assert out.shape == (36, s0.active_wh[0], 3)
    np.testing.assert_array_equal(out, expected)


def test_stage0_identity_and_wrong_frame_size():
    s0 = framing.Stage0(W, H, 1.0, {e: 0 for e in framing.EDGES})
    frame = RNG.integers(0, 256, (H, W, 3), dtype=np.uint8)
    assert s0.is_identity and s0.apply(frame) is frame
    with pytest.raises(framing.FramingError, match="decoded frame is"):
        s0.apply(frame[:, :-1])


def test_stage0_from_table(media):
    s0 = framing.stage0_for(media["letterbox"], media["table"])
    assert s0.crop == {"left": 0, "right": 0, "top": TOP, "bottom": BOTTOM}
    assert s0.active_wh == (W, H - TOP - BOTTOM)
    rec = s0.record()
    assert rec["stage0"] == "framing table" and rec["framing_table_sha256"]
    assert rec["active_wh"] == [W, H - TOP - BOTTOM] and rec["stage0_version"] == framing.STAGE0_VERSION


def test_stage0_without_table_crops_nothing(media):
    s0 = framing.stage0_for(media["letterbox"])
    assert s0.is_identity and s0.record()["framing_table"] is None
    assert s0.record()["stage0"].startswith("no framing table")


def test_source_missing_from_table_is_loud(media):
    with pytest.raises(framing.FramingError, match="not in the framing table"):
        framing.stage0_for(media["other"], media["table"])


def test_sar_mismatch_is_loud(media, tmp_path):
    rows, params = framing.read_table(media["table"])
    for r in rows.values():
        r["sar"] = "0.888889"
    bad = tmp_path / "bad_sar.csv"
    framing.write_table(list(rows.values()), bad, framing.FINAL_COLUMNS, params)
    with pytest.raises(framing.FramingError, match="pixel aspect"):
        framing.stage0_for(media["letterbox"], bad)


def test_non_square_pixels_are_squared(tmp_path):
    # mpeg2 (lossy) because FFV1 does not carry a sample aspect ratio here;
    # only the frame size is checked.
    from fractions import Fraction

    path = tmp_path / "anamorphic.mkv"
    c = av.open(str(path), "w")
    st = c.add_stream("mpeg2video", rate=25)
    st.width, st.height, st.pix_fmt = 72, 48, "yuv420p"
    st.codec_context.sample_aspect_ratio = Fraction(8, 9)
    for f in letterboxed(n=10, w=72, h=48, top=0, bottom=0):
        for pkt in st.encode(av.VideoFrame.from_ndarray(f, format="rgb24")):
            c.mux(pkt)
    for pkt in st.encode():
        c.mux(pkt)
    c.close()
    s0 = framing.stage0_for(path)
    assert s0.sar == pytest.approx(8 / 9) and s0.active_wh == (64, 48)
    frames = extract_frames(path, frame_interval=0.1, quiet=True, stage0=s0)
    assert frames[0][1].size == (64, 48)


def test_extract_frames_applies_stage0(media):
    s0 = framing.stage0_for(media["letterbox"], media["table"])
    raw = [f for _, f in iter_frames(media["letterbox"], 0.5)]
    got = extract_frames(media["letterbox"], frame_interval=0.5, quiet=True, stage0=s0)
    assert len(got) == len(raw)
    for (_, img), r in zip(got, raw):
        np.testing.assert_array_equal(np.asarray(img), r[TOP:H - BOTTOM])


def test_motion_scores_the_stage0_picture(media):
    from viz2psy.models.motion import MotionModel

    m = MotionModel(device="cpu")
    m.load()
    s0 = framing.stage0_for(media["letterbox"], media["table"])
    m.predict_video(media["letterbox"], [0.5], quiet=True, stage0=s0)
    gray = m._prepare(np.zeros((H, W, 3), np.uint8))
    assert gray.shape == (H - TOP - BOTTOM, W)


# ----------------------------------------------------------------- CLI + sidecar

def _run(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["viz2psy", *map(str, argv)])
    cli.main()


def test_cli_framing_table_end_to_end(media, tmp_path, monkeypatch):
    from viz2psy.models.llstat import LLStatModel

    out = tmp_path / "ll.csv"
    _run(monkeypatch, "llstat", media["letterbox"], "--framing-table", media["table"],
         "--no-save-frames", "--no-viz", "--quiet", "-o", out)
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    fr = meta["input"]["framing"]
    assert fr["frame_source"] == "raw decode" and fr["stage0"] == "framing table"
    assert fr["crop"] == {"left": 0, "right": 0, "top": TOP, "bottom": BOTTOM}
    assert meta["models"]["llstat"]["preprocessing"]["canonical_area"] == CANONICAL_AREA

    got = pd.read_csv(out)
    ref = LLStatModel()
    ref.load()
    frames = [Image.fromarray(f[TOP:H - BOTTOM]) for _, f in iter_frames(media["letterbox"], 0.5)]
    want = pd.DataFrame([ref.predict(im) for im in frames])
    np.testing.assert_allclose(got[want.columns].to_numpy(), want.to_numpy(), rtol=1e-9)


def test_cli_without_table_records_it(media, tmp_path, monkeypatch):
    out = tmp_path / "ll.csv"
    _run(monkeypatch, "llstat", media["letterbox"], "--no-save-frames", "--no-viz", "--quiet", "-o", out)
    fr = json.loads(out.with_suffix(".meta.json").read_text())["input"]["framing"]
    assert fr["stage0"].startswith("no framing table") and fr["crop"]["top"] == 0


def test_still_images_record_still_framing(tmp_path, monkeypatch):
    img = tmp_path / "a.png"
    Image.fromarray(RNG.integers(0, 256, (40, 60, 3), dtype=np.uint8)).save(img)
    out = tmp_path / "ll.csv"
    _run(monkeypatch, "llstat", img, "--no-viz", "--quiet", "-o", out)
    fr = json.loads(out.with_suffix(".meta.json").read_text())["input"]["framing"]
    assert fr == framing.STILL_FRAMING


# ----------------------------------------------------------------- stage 1

def test_squash_to_square():
    sq = Image.new("RGB", (50, 50))
    assert squash_to_square(sq) is sq
    assert squash_to_square(Image.new("RGB", (160, 90))).size == (90, 90)
    assert squash_to_square(Image.new("RGB", (90, 160))).size == (90, 90)


def test_llstat_canonical_area():
    from viz2psy.models.llstat import LLStatModel

    m = LLStatModel()
    m.load()
    nsd = Image.fromarray(RNG.integers(0, 256, (425, 425, 3), dtype=np.uint8))
    wide = Image.fromarray(RNG.integers(0, 256, (400, 1000, 3), dtype=np.uint8))
    assert canonical_size(425, 425) == (425, 425)
    # A wide picture is scored at the canonical area: same values as resizing it first.
    small = wide.resize(canonical_size(*wide.size), Image.BICUBIC)
    assert m.predict(wide) == pytest.approx(m.predict(small), nan_ok=True)
    assert m.predict(nsd)  # unchanged size path


def _model_class(name):
    """Load a model class; skip when its third-party backend is not installed.

    A missing viz2psy module is a real error and still fails.
    """
    try:
        return cli._load_model_class(name)
    except ModuleNotFoundError as e:
        if (e.name or "").split(".")[0] == "viz2psy":
            raise
        pytest.skip(f"{name}: backend {e.name!r} not installed")


@pytest.mark.parametrize("name", sorted(cli.MODEL_REGISTRY))
def test_every_model_records_its_preprocessing(name):
    describe = getattr(_model_class(name), "describe_preprocessing", None)
    assert describe is not None, f"{name} has no describe_preprocessing()"
    assert isinstance(describe(), dict) and describe()


@pytest.mark.parametrize("name", ["clip", "aesthetics", "ebind", "dinov2", "places"])
def test_crop_networks_squash(name):
    rec = _model_class(name).describe_preprocessing()
    assert rec["square"].startswith("squash")
