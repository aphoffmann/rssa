import numpy as np
from py_rssa import cssa


def test_complex_ssa_basic():
    t = np.arange(10)
    x = np.exp(1j * t)
    s = cssa(x, L=5)
    rec = s.reconstruct([0])
    assert np.allclose(rec, x)
