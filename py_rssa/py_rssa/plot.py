# Helper plotting routines for the SSA class
import numpy as np
import matplotlib.pyplot as plt


def plot_values(ssa_obj, numvalues=None, ax=None, **kwargs):
    """Plot the singular values of ``ssa_obj``.

    Parameters
    ----------
    ssa_obj : :class:`SSA`
        Decomposed SSA object.
    numvalues : int, optional
        Number of singular values to plot.  By default all available
        values are shown.
    ax : matplotlib axes, optional
        Axes to draw into.
    **kwargs : dict
        Additional keyword arguments passed to ``ax.plot``.
    """
    if numvalues is None:
        numvalues = len(ssa_obj.s)
    if ax is None:
        ax = plt.gca()
    x = np.arange(1, numvalues + 1)
    y = ssa_obj.s[:numvalues]
    ax.plot(x, y, marker="o", **kwargs)
    ax.set_xlabel("Index")
    ax.set_ylabel("norms")
    ax.set_title("Component norms")
    ax.set_yscale("log")
    ax.grid(True)
    return ax


def _ensure_axes(n, axes=None):
    if axes is None:
        fig, axes = plt.subplots(n, 1, squeeze=False)
        axes = axes.ravel()
    return axes


def plot_vectors(ssa_obj, indices=None, kind="eigen", axes=None, **kwargs):
    """Plot eigen or factor vectors."""
    if indices is None:
        indices = list(range(min(len(ssa_obj.s), 10)))
    axes = _ensure_axes(len(indices), axes)
    V = ssa_obj.Vt.T
    for ax, idx in zip(axes, indices):
        vec = ssa_obj.U[:, idx] if kind == "eigen" else V[:, idx]
        ax.plot(vec, **kwargs)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_ylabel(f"{kind[0].upper()}{idx+1}")
    axes[-1].set_xlabel("Index")
    axes[0].set_title("Eigenvectors" if kind == "eigen" else "Factor vectors")
    return axes


def plot_series(ssa_obj, groups=None, axes=None, **kwargs):
    """Plot reconstructed series for given ``groups``."""
    if groups is None:
        groups = [[i] for i in range(min(len(ssa_obj.s), ssa_obj.U.shape[1]))]
    series = [ssa_obj.reconstruct(g) for g in groups]
    axes = _ensure_axes(len(series), axes)
    for ax, s, g in zip(axes, series, groups):
        ax.plot(s, **kwargs)
        ax.set_ylabel(str(g))
        ax.grid(True)
    axes[-1].set_xlabel("Index")
    axes[0].set_title("Reconstructed series")
    return axes


def plot_wcor(ssa_obj, groups=None, ax=None, **kwargs):
    """Display weighted correlation matrix."""
    if groups is None:
        groups = [[i] for i in range(min(len(ssa_obj.s), ssa_obj.U.shape[1]))]
    wc = np.abs(ssa_obj.wcor(groups))
    if ax is None:
        ax = plt.gca()
    im = ax.imshow(wc, vmin=0.0, vmax=1.0, origin="lower", **kwargs)
    ax.set_xlabel("Component")
    ax.set_ylabel("Component")
    ax.set_title("W-correlation matrix")
    ax.figure.colorbar(im, ax=ax)
    return ax


def plot(ssa_obj, type="values", **kwargs):
    """High level plot similar to ``plot.ssa`` in R."""
    if type == "values":
        return plot_values(ssa_obj, **kwargs)
    if type == "vectors":
        return plot_vectors(ssa_obj, **kwargs)
    if type == "series":
        return plot_series(ssa_obj, **kwargs)
    if type == "wcor":
        return plot_wcor(ssa_obj, **kwargs)
    raise ValueError("Unsupported plot type")
