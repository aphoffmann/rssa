import numpy as np
from py_rssa import SSA


def test_reconstruct_identity():
    x = np.sin(np.linspace(0, 2*np.pi, 20))
    s = SSA(x, L=10)
    rec = s.reconstruct(range(5))
    assert np.allclose(rec, x, atol=1e-6)


def test_wcor_diagonal():
    x = np.sin(np.linspace(0, 2*np.pi, 20))
    s = SSA(x, L=10)
    w = s.wcor()
    assert np.allclose(np.diag(w), 1.0)
