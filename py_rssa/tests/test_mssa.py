import numpy as np
from py_rssa import mssa, reconstruct


def test_mssa_reconstruction():
    ts1 = np.sin(np.linspace(0, 4 * np.pi, 100))
    ts2 = np.cos(np.linspace(0, 4 * np.pi, 100))
    ss = mssa([ts1, ts2], L=30)
    rec = reconstruct(ss, groups=[1, 2])
    assert len(rec) == 2
    assert rec[0].shape == ts1.shape
    assert rec[1].shape == ts2.shape
