"""GIST's Fourier-domain convolution equals the direct one it replaced.

viz2psy <= 0.19.0 convolved each Gabor kernel with ``scipy.ndimage.convolve``
(``reflect`` boundary); 0.19.1 does all 32 in one FFT. The descriptor must be
the same to floating-point rounding, including near the borders and on
non-square inputs.
"""

import numpy as np
import pytest
from PIL import Image
from scipy.ndimage import convolve

from viz2psy.models.gist import (
    _DEFAULT_FREQUENCIES, GISTModel, _build_gabor_bank, _compute_gist, _fft_bank,
)


def _direct(gray, kernels, grid_size):
    h, w = gray.shape
    ch, cw = h // grid_size, w // grid_size
    out = []
    for k in kernels:
        m = np.abs(convolve(gray, k, mode="reflect"))
        for row in range(grid_size):
            for col in range(grid_size):
                r1 = (row + 1) * ch if row < grid_size - 1 else h
                c1 = (col + 1) * cw if col < grid_size - 1 else w
                out.append(m[row * ch:r1, col * cw:c1].mean())
    return np.array(out)


@pytest.fixture(scope="module")
def kernels():
    return _build_gabor_bank(8, _DEFAULT_FREQUENCIES)


@pytest.mark.parametrize("shape", [(256, 256), (97, 130)])
def test_fft_equals_direct_convolution(kernels, shape):
    rng = np.random.default_rng(0)
    gray = rng.random(shape)
    gray[:, :5] = 1.0  # a hard edge at the border, where the boundary rule shows
    want = _direct(gray, kernels, 4)
    got = _compute_gist(gray, kernels, 4)
    np.testing.assert_allclose(got, want, rtol=1e-10, atol=1e-13)


def test_a_bank_for_another_shape_is_rebuilt(kernels):
    gray = np.random.default_rng(1).random((64, 80))
    stale = _fft_bank(kernels, (256, 256))
    np.testing.assert_allclose(_compute_gist(gray, kernels, 4, stale),
                               _direct(gray, kernels, 4), rtol=1e-10, atol=1e-13)


def test_model_predict_matches_direct(kernels):
    rng = np.random.default_rng(2)
    img = Image.fromarray(rng.integers(0, 256, (120, 200, 3), dtype=np.uint8))
    m = GISTModel()
    m.load()
    got = np.array(list(m.predict(img).values()))
    gray = np.array(img.resize((256, 256), Image.Resampling.LANCZOS).convert("L"),
                    dtype=np.float64) / 255.0
    assert got.shape == (512,)
    np.testing.assert_allclose(got, _direct(gray, kernels, 4), rtol=1e-10, atol=1e-13)
