import numpy as np
import matplotlib.pyplot as plt
from py_rssa import SSA, plot_values, plot_vectors, plot_series, plot_wcor


def test_plot_helpers_smoke():
    data = np.sin(np.linspace(0, 2 * np.pi, 60))
    ss = SSA(data, L=20)

    ax = plot_values(ss, numvalues=5)
    plt.close(ax.figure)

    axes = plot_vectors(ss, indices=[0, 1])
    plt.close(axes[0].figure)

    axes = plot_series(ss, groups=[[0], [1]])
    plt.close(axes[0].figure)

    ax = plot_wcor(ss, groups=[[0], [1]])
    plt.close(ax.figure)
