from py_rssa import SSA, cadzow
import numpy as np

def test_cadzow_basic():
    x = np.sin(np.linspace(0, 2 * np.pi, 60)) + 0.1 * np.random.randn(60)
    s = SSA(x, L=30)
    res = cadzow(s, rank=1, maxiter=2)
    assert res.shape == x.shape
