import pytest

np = pytest.importorskip('numpy')
py_rssa = pytest.importorskip('py_rssa')


def test_basic_forecast():
    data = np.sin(np.linspace(0, 2 * np.pi, 60))
    ss = py_rssa.ssa(data, L=20)
    fc = py_rssa.forecast(ss, groups=[1], steps=10)
    assert fc.shape[0] == data.shape[0] + 10
