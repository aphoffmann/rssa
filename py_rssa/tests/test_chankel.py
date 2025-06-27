import pytest

np = pytest.importorskip('numpy')
py_rssa = pytest.importorskip('py_rssa')


def test_chankel_mv():
    F = np.array([1+2j, 3+4j, 5+6j, 7+8j])
    v = np.array([2+1j, -1+0.5j])
    res = py_rssa.chankel_mv(F, v)
    L = len(F) - len(v) + 1
    expected = np.array([
        sum(F[i+j] * v[j] for j in range(len(v)))
        for i in range(L)
    ], dtype=complex)
    assert np.allclose(res, expected)


def test_chankelize():
    U = np.array([1+1j, 2-1j])
    V = np.array([3+2j, 4-2j, 5+0.5j])
    res = py_rssa.chankelize(U, V)
    weights = py_rssa.hankel_weights(len(U), len(V))
    expected = np.convolve(U, V) / weights
    assert np.allclose(res, expected)

