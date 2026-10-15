"""Dashboard projection panels: degrade on bad data, never on a broken call."""

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("plotly")
pytest.importorskip("sklearn")

from viz2psy.viz import dashboard


@pytest.fixture
def emonet_df():
    rng = np.random.default_rng(0)
    cols = [f"emonet_e{i:02d}" for i in range(5)]
    return pd.DataFrame(rng.random((10, 5)), columns=cols)


PANELS = [
    lambda df: dashboard._create_cluster_trace(df, "emonet", np.arange(len(df))),
    lambda df: dashboard._create_trajectory_trace(df, "emonet", np.arange(len(df))),
    lambda df: dashboard._create_trajectory_animated(df, "emonet", np.arange(len(df))),
]


@pytest.mark.parametrize("panel", PANELS)
def test_broken_projection_call_is_not_swallowed(monkeypatch, emonet_df, panel):
    # The sklearn MDS kwarg rename surfaces as a TypeError. `except Exception`
    # turned that into a warning plus an empty panel: a broken dashboard that
    # looks like a model without enough data.
    def broken(*a, **k):
        raise TypeError("__init__() got an unexpected keyword argument 'metric'")

    monkeypatch.setattr(dashboard, "compute_projection", broken)
    with pytest.raises(TypeError):
        panel(emonet_df)


@pytest.mark.parametrize("panel", PANELS)
def test_insufficient_data_still_degrades(emonet_df, panel):
    # compute_projection raises ValueError for < 3 rows; that panel is skipped.
    with pytest.warns(UserWarning, match="failed"):
        result = panel(emonet_df.head(2))
    assert not result[0]
