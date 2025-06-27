import numpy as np
from py_rssa import SSA, WOSSA


def test_wossa_identity():
    x = np.sin(np.linspace(0, 2*np.pi, 40))
    s1 = SSA(x, L=20)
    s2 = WOSSA(x, L=20)
    rec1 = s1.reconstruct(range(5))
    rec2 = s2.reconstruct(range(5))
    assert np.allclose(rec1, rec2, atol=1e-6)


def test_wossa_weights_reconstruction():
    x = np.sin(np.linspace(0, 2*np.pi, 30))
    col_w = np.linspace(1, 2, 15)
    row_w = np.linspace(2, 1, 16)
    s = WOSSA(x, L=15, column_oblique=col_w, row_oblique=row_w)
    rec = s.reconstruct(range(len(s.s)))
    assert np.allclose(rec, x, atol=1e-6)
