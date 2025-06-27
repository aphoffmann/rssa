import numpy as np

from py_rssa import SSA, object_capabilities, fftw_available


def test_basic_capabilities():
    s = SSA(np.arange(10), L=5)
    caps = object_capabilities(s)
    assert caps["Decomposition"]
    assert caps["Reconstruction"]


def test_fftw_available_returns_bool():
    val = fftw_available()
    assert isinstance(val, bool)

