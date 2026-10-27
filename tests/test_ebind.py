"""ebind: what can be checked without the checkpoint."""

from viz2psy.models.ebind import DEFAULT_REVISION, EBindModel


def test_default_checkpoint_loads_the_pinned_revision():
    """The *2psy repos share one embedding space only while they load the same
    snapshot of the checkpoint. A floating ``main`` would split it silently."""
    assert EBindModel(device="cpu").revision == DEFAULT_REVISION
    assert EBindModel(checkpoint="someone/other-ebind", device="cpu").revision == "main"
