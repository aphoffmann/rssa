import numpy as np
from py_rssa import SSA, wcor, wnorm


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

# Fail can't import wcor?
def test_wcor_matrix_symmetry():
    t = np.linspace(0, 2*np.pi, 20)
    m = np.vstack((np.sin(t), np.cos(t))).T
    w = wcor(m, L=10)
    assert np.allclose(w, w.T)
    assert np.allclose(np.diag(w), 1.0)

# Fail can't import wnorm?
def test_wnorm_basic():
    x = np.arange(1, 6, dtype=float)
    n = wnorm(x, L=3)
    w = np.array([1, 2, 3, 2, 1], dtype=float)
    expected = np.sqrt(np.sum(w * x**2))
    assert np.allclose(n, expected)
