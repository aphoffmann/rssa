import numpy as np
from py_rssa import ssa, forecast

# forecast and wcor seem to be mixed up?
def test_basic_forecast():
    data = np.sin(np.linspace(0, 2 * np.pi, 60))
    ss = ssa(data, L=20)
    fc = forecast(ss, groups=[1], steps=10)
    assert fc.shape[0] == data.shape[0] + 10
