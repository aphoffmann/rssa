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
    dims = convolution_dims((4, 4), (2, 2), type="circular")
    assert dims["input_dim"] == (4, 4)
    assert dims["output_dim"] == (4, 4)
    dims = convolution_dims((4, 4), (2, 2), type="open")
    assert dims["output_dim"] == (5, 5)
    dims = convolution_dims((4, 4), (2, 2), type="filter")
    assert dims["output_dim"] == (3, 3)


# Failed assert weights.shape == (5, 5
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
