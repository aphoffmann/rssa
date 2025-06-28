import numpy as np
from py_rssa import hmatr


#failed assert np.all(h >= 0)
def test_hmatr_shape_values():
    x = np.sin(np.linspace(0, 2 * np.pi, 40))
    h = hmatr(x)
    assert h.shape == (40 - 40 // 4, 40 - 40 // 4)
    assert np.all(h >= 0)
    assert np.all(h <= 1)
