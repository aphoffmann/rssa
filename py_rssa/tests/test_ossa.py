import numpy as np
import py_rssa


def test_cond():
    A = np.array([[1., 0.], [0., 2.]])
    c = py_rssa.cond(A)
    assert np.isclose(c, 2.0)


def test_pseudo_inverse():
    A = np.array([[1., 2.], [3., 4.]])
    pinv = py_rssa.pseudo_inverse(A)
    assert np.allclose(pinv, np.linalg.pinv(A))


def test_orthogonalize_bi():
    Y = np.array([[1., 0.], [0., 1.]])
    Z = np.array([[1., 0.], [0., 1.]])
    res = py_rssa.orthogonalize(Y, Z)
    U = res['u']
    V = res['v']
    assert np.allclose(U.T @ U, np.eye(2))
    assert np.allclose(V.T @ V, np.eye(2))


def test_owcor_basic():
    x = np.sin(np.linspace(0, 2 * np.pi, 20))
    ss = py_rssa.ssa(x, L=10)
    recs = [ss.reconstruct([0]), ss.reconstruct([1])]
    LM = py_rssa.pseudo_inverse(ss.U[:, :2])
    RM = py_rssa.pseudo_inverse(ss.Vt[:2].T)
    oc = py_rssa.owcor(recs, LM, RM)
    assert oc.shape == (2, 2)
    assert np.allclose(np.diag(oc), 1.0)
