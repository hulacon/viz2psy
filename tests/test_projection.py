"""viz.projection.compute_projection: the analytic projections behind the dashboard."""

import warnings

import numpy as np
import pytest

pytest.importorskip("sklearn")

from viz2psy.viz.projection import compute_projection


@pytest.fixture
def X():
    return np.random.default_rng(0).normal(size=(12, 5))


@pytest.mark.parametrize("method", ["pca", "ppca", "tsne", "mds", "mds_nonmetric"])
def test_methods_run_on_installed_sklearn(X, method):
    # MDS kwargs moved between sklearn majors; a projection that raises here
    # is the failure the dashboard used to turn into a warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        proj, info = compute_projection(X, method=method)
    assert proj.shape == (12, 2)
    assert np.isfinite(proj).all()
    assert info["method"] == method


def test_nan_imputation_does_not_mutate_caller_array(X):
    # Documented public API (docs/visualization.md). Mean-imputation wrote into
    # the caller's array, so their NaNs silently became column means.
    X[0, 0] = np.nan
    with pytest.warns(UserWarning, match="NaN"):
        compute_projection(X, method="pca")
    assert np.isnan(X[0, 0])
