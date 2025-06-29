import numpy as np
from py_rssa import NDSSA, ndssa


def test_ndssa_reconstruction_rectangular():
    x = np.arange(9).reshape(3, 3).astype(float)
    s = NDSSA(x, L=(2, 2))
    rec = s.reconstruct([0])
    assert np.allclose(rec, x)


def test_ndssa_shaped_window():
    x = np.arange(9).reshape(3, 3).astype(float)
    wmask = np.array([[1, 0], [1, 1]], dtype=bool)
    s = ndssa(x, L=(2, 2), wmask=wmask)
    rec = s.reconstruct([0])
    assert np.allclose(rec, x)
