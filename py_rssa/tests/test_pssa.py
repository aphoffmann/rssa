import pytest

np = pytest.importorskip("numpy")
py_rssa = pytest.importorskip("py_rssa")


def test_pssa_basic_equivalence():
    data = np.sin(np.linspace(0, 2 * np.pi, 40))
    s1 = py_rssa.pssa(data, L=20, column_proj="centering")
    X = np.column_stack([data[i : i + 20] for i in range(21)])
    Xc = X - X.mean(axis=0)
    _, s, _ = np.linalg.svd(Xc, full_matrices=False)
    assert np.allclose(s1.s, s)


def test_pssa_equivalent_to_ssa_without_proj():
    rng = np.random.default_rng(123)
    data = rng.standard_normal(30)
    s1 = py_rssa.ssa(data, L=15)
    s2 = py_rssa.pssa(data, L=15)
    assert np.allclose(s1.s, s2.s)
