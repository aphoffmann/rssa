import pytest

np = pytest.importorskip('numpy')
py_rssa = pytest.importorskip('py_rssa')


def test_gap_fill():
    data = np.sin(np.linspace(0, 2 * np.pi, 60))
    data[10] = np.nan
    ss = py_rssa.ssa(data, L=20)
    filled = py_rssa.gapfill(ss, groups=[1])
    assert not np.isnan(filled[10])
