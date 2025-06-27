import pytest

np = pytest.importorskip('numpy')
py_rssa = pytest.importorskip('py_rssa')


def test_mssa_reconstruction():
    ts1 = np.sin(np.linspace(0, 4 * np.pi, 100))
    ts2 = np.cos(np.linspace(0, 4 * np.pi, 100))
    ss = py_rssa.mssa([ts1, ts2], L=30)
    rec = py_rssa.reconstruct(ss, groups=[1, 2])
    assert len(rec) == 2
    assert rec[0].shape == ts1.shape
    assert rec[1].shape == ts2.shape
