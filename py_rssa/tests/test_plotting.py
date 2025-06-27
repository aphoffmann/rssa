import pytest

np = pytest.importorskip('numpy')
plt = pytest.importorskip('matplotlib.pyplot')

from py_rssa import plot_2d_reconstructions, plot_2d_vectors


def test_plot_reconstructions_runs():
    data = [np.arange(9).reshape(3, 3)]
    fig = plot_2d_reconstructions(data)
    assert fig is not None
    plt.close(fig)


def test_plot_vectors_runs():
    vec = [np.arange(9)]
    fig = plot_2d_vectors(vec, shape=(3, 3))
    assert fig is not None
    plt.close(fig)
