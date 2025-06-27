import pytest

np = pytest.importorskip('numpy')
py_rssa = pytest.importorskip('py_rssa')


def test_eossa_basic():
    data = np.sin(np.linspace(0, 2 * np.pi, 60))
    ss = py_rssa.ssa(data, L=20)
    es = py_rssa.eossa(ss, k=1)
    rec = es.reconstruct(range(es.U.shape[1]))
    assert rec.shape == data.shape
