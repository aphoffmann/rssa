import numpy as np
from py_rssa import chankel_mv, chankelize, hankel_weights

def test_chankel_mv():
    F = np.array([1+2j, 3+4j, 5+6j, 7+8j])
    v = np.array([2+1j, -1+0.5j])
    res = chankel_mv(F, v)
    L = len(F) - len(v) + 1
    expected = np.array([
        sum(F[i+j] * v[j] for j in range(len(v)))
        for i in range(L)
    ], dtype=complex)
    assert np.allclose(res, expected)


def test_chankelize():
    U = np.array([1+1j, 2-1j])
    V = np.array([3+2j, 4-2j, 5+0.5j])
    res = chankelize(U, V)
    weights = hankel_weights(len(U), len(V))
    expected = np.convolve(U, V) / weights
    assert np.allclose(res, expected)

