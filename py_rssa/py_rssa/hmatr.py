import numpy as np
from .ssa import SSA


def hmatr(F, B=None, T=None, L=None, neig=10, **kwargs):
    """Compute heterogeneity matrix for a one-dimensional series.

    Parameters
    ----------
    F : array_like
        Input sequence.
    B : int, optional
        Length of the base subseries. Defaults to ``len(F) // 4``.
    T : int, optional
        Length of the tested subseries. Defaults to ``len(F) // 4``.
    L : int, optional
        Window length for the decomposition. Defaults to ``B // 2``.
    neig : int, optional
        Number of eigentriples to consider. Only first ``neig`` left
        singular vectors of the base series are used.
    **kwargs : dict
        Additional arguments passed to :class:`SSA` constructor.

    Returns
    -------
    numpy.ndarray
        The heterogeneity matrix of shape ``(len(F)-T, len(F)-B)``.
    """
    F = np.asarray(F, dtype=float)
    N = F.size
    if B is None:
        B = N // 4
    if T is None:
        T = N // 4
    if L is None:
        L = B // 2

    K = N - L + 1
    # Lagged vectors of the original series
    th = np.column_stack([F[i:i+L] for i in range(K)]).T  # shape K x L

    # Squared norms of consecutive blocks of lagged vectors
    rows_sum = np.sum(th ** 2, axis=1)
    cth2 = np.concatenate(([0.0], np.cumsum(rows_sum)))
    idx = np.arange(N - T)
    cth2 = cth2[idx + (T - L + 1)] - cth2[idx]

    h = np.empty((N - T, N - B))

    for col, start in enumerate(range(N - B)):
        Fb = F[start:start + B + 1]
        s = SSA(Fb, L=L, **kwargs)
        U = s.U[:, :min(neig, s.U.shape[1])]
        XU = th @ U
        cXU2 = np.concatenate(([0.0], np.cumsum(np.sum(XU ** 2, axis=1))))
        proj = cXU2[idx + (T - L + 1)] - cXU2[idx]
        h[:, col] = 1 - proj / cth2

    return h


def plot_hmatr(h, ax=None, **kwargs):
    """Plot heterogeneity matrix using :mod:`matplotlib`."""
    import matplotlib.pyplot as plt

    if ax is None:
        ax = plt.gca()
    im = ax.imshow(h, origin="lower", aspect="auto", **kwargs)
    return im
