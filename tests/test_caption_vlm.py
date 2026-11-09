"""caption_vlm without weights: input sizing, decoding, the frozen prompt, the sidecar."""

import hashlib

import pytest
from PIL import Image

from viz2psy.labels import problems, resolve
from viz2psy.metadata import MetadataBuilder
from viz2psy.models import caption_vlm as cv

#: sha256 of the caption-prompt v1 text. Editing PROMPT without a new
#: PROMPT_VERSION would silently change the feature; this pins the pair.
PROMPT_V1_SHA256 = "96c334fe590717104ce9dc86846bcd55d573ee7fed1887b2a7c63823cf18a324"


def test_prompt_is_frozen_with_its_version():
    assert cv.PROMPT_VERSION == "caption-prompt v1"
    assert hashlib.sha256(cv.PROMPT.encode()).hexdigest() == PROMPT_V1_SHA256


@pytest.mark.parametrize("size, expected", [
    ((1920, 800), (768, 320)),    # wide film frame: long side to 768, aspect kept
    ((800, 1920), (320, 768)),    # tall
    ((425, 425), (425, 425)),     # NSD still: never enlarged
    ((768, 432), (768, 432)),     # already at the limit
])
def test_fit_long_side(size, expected):
    assert cv.fit_long_side(Image.new("RGB", size)).size == expected


def test_fit_long_side_copies_and_converts():
    src = Image.new("L", (1600, 900))
    out = cv.fit_long_side(src)
    assert out.mode == "RGB" and src.size == (1600, 900)


class _Tok:
    """Decodes token ids as characters; 0 is a special (pad/eos) token."""

    def decode(self, ids, skip_special_tokens=True):
        return "".join(chr(int(i)) for i in ids if not (skip_special_tokens and int(i) == 0))


def _ids(prompt: str, text: str, width: int):
    row = [ord(c) for c in prompt] + [ord(c) for c in text]
    return row + [0] * (width - len(row))


def test_decode_keeps_only_tokens_after_the_prompt():
    gen = [_ids("PP", " a dog runs ", 16), _ids("PP", "two cats", 16)]
    assert cv.decode_new_tokens(_Tok(), gen, 2) == ["a dog runs", "two cats"]


def test_empty_caption_is_an_error():
    gen = [_ids("PP", "a dog", 8), _ids("PP", "   ", 8)]
    with pytest.raises(ValueError, match="1 empty caption"):
        cv.decode_new_tokens(_Tok(), gen, 2)


def test_checkpoint_pins_the_revision():
    m = cv.CaptionVLMModel(device="cpu")
    assert m.checkpoint == f"{cv.CHECKPOINT}@{cv.REVISION}" == cv.CaptionVLMModel.checkpoint
    assert len(cv.REVISION) == 40


def test_preprocessing_records_prompt_and_decoding():
    rec = cv.CaptionVLMModel.describe_preprocessing()
    assert rec["prompt"] == cv.PROMPT and rec["prompt_version"] == cv.PROMPT_VERSION
    assert "greedy" in rec["decoding"] and str(cv.MAX_NEW_TOKENS) in rec["decoding"]


def test_sidecar_declares_a_string_column():
    b = MetadataBuilder()
    b.add_model("caption_vlm", [cv.COLUMN], 1.0)
    entry = b.build()["models"]["caption_vlm"]
    assert entry["features"]["dtype"] == "string"
    assert entry["preprocessing"]["prompt_version"] == cv.PROMPT_VERSION
    assert problems(entry["labels"], [cv.COLUMN]) == []
    r = resolve(entry["labels"], cv.COLUMN)
    assert (r["interpretability"], r["derivation"]) == ("nameable", "trained-on-labels")
