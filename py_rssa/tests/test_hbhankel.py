import numpy as np
from py_rssa import (
    convolution_dims,
    convolven,
    factor_mask_2d,
    field_weights_2d,
    ball_mask,
    simplex_mask,
)


def test_convolution_dims_basic():
    dims = convolution_dims((4, 4), (2, 2), type="circular")
    assert dims["input_dim"] == (4, 4)
    assert dims["output_dim"] == (4, 4)
    dims = convolution_dims((4, 4), (2, 2), type="open")
    assert dims["output_dim"] == (5, 5)
    dims = convolution_dims((4, 4), (2, 2), type="filter")
    assert dims["output_dim"] == (3, 3)


def test_convolven_matches_numpy():
    x = np.array([[1, 2], [3, 4]], dtype=float)
    y = np.array([[0, 1], [1, 0]], dtype=float)
    res = convolven(x, y, conj=False, type="open")
    # naive convolution
    expected = np.zeros((3, 3))
    for i in range(2):
        for j in range(2):
            expected[i : i + 2, j : j + 2] += x * y[i, j]
    assert np.allclose(res, expected)


def test_factor_mask_and_weights():
    field_mask = np.ones((4, 4))
    window_mask = np.ones((2, 2))
    fmask = factor_mask_2d(field_mask, window_mask)
    weights = field_weights_2d(window_mask, fmask)
    assert weights.shape == (5, 5)
    assert np.all(weights > 0)


def test_ball_and_simplex_masks():
    b = ball_mask(2, 2)
    assert b.shape == (3, 3)
    assert b[1, 1]
    s = simplex_mask(2, 2)
    assert s.shape == (2, 2)
    assert s[0, 0]
