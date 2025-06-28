import numpy as np
from py_rssa import ssa, gapfill

# Failed SVD did not converge
def test_gap_fill():
    data = np.sin(np.linspace(0, 2 * np.pi, 60))
    data[10] = np.nan
    ss = ssa(data, L=20)
    filled = gapfill(ss, groups=[1])
    assert not np.isnan(filled[10])
