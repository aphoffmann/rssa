import pytest

np = pytest.importorskip("numpy")
py_rssa = pytest.importorskip("py_rssa")


def test_cadzow_basic():
    x = np.sin(np.linspace(0, 2 * np.pi, 60)) + 0.1 * np.random.randn(60)
    s = py_rssa.SSA(x, L=30)
    res = py_rssa.cadzow(s, rank=1, maxiter=2)
    assert res.shape == x.shape
