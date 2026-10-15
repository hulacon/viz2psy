"""Places365 category labels: one distinct column per class."""

from viz2psy.models.places import _parse_categories

# Shape of categories_places365.txt: `/letter/name[/qualifier] index`.
SAMPLE = """/a/abbey 0
/c/church/indoor 84
/c/church/outdoor 85
/s/swimming_pool/indoor 297
/s/swimming_pool/outdoor 298
"""


def test_qualified_categories_stay_distinct():
    # Keeping only the last path segment collapsed 51 of the 365 classes onto
    # `indoor`/`outdoor`/...: later classes overwrote earlier ones in the
    # output row, leaving 314 columns with mislabelled probabilities.
    cats = _parse_categories(SAMPLE)
    assert cats == ["abbey", "church_indoor", "church_outdoor",
                    "swimming_pool_indoor", "swimming_pool_outdoor"]
    assert len(set(cats)) == len(cats)
