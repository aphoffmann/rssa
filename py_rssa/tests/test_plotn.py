import pytest

np = pytest.importorskip('numpy')
py_rssa = pytest.importorskip('py_rssa')


def test_do_slice_array_basic():
    arr = np.arange(24).reshape(2, 3, 4)
    slc = py_rssa.do_slice_array(arr, {"x": slice(None), "y": 1})
    assert np.all(slc == arr[:, 1])
