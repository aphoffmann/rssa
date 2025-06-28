import numpy as np
from py_rssa import ssa, eossa

def test_eossa_basic():
    data = np.sin(np.linspace(0, 2 * np.pi, 60))
    ss = ssa(data, L=20)
    es = eossa(ss, k=1)
    rec = es.reconstruct(range(es.U.shape[1]))
    assert rec.shape == data.shape
