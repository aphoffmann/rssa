# Utilities and Cadzow iterations for SSA denoising
# Ported from R/cadzow.R

import numpy as np

from .ssa import SSA, hankel_weights


def _extend_series(x, alpha):
    """Scale series or list of series by ``alpha``."""
    if isinstance(x, (list, tuple)):
        return [_extend_series(v, alpha) for v in x]
    return alpha * np.asarray(x)


def _series_dist(f1, f2, norm_func, mask=None):
    """Distance between two series according to ``norm_func``."""
    a = np.asarray(f1).ravel()
    b = np.asarray(f2).ravel()
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        a = a[mask]
        b = b[mask]
    return norm_func(a - b)


def _series_winnerprod(f1, f2, weights=1):
    """Weighted inner product of two series."""
    a = np.asarray(f1).ravel()
    b = np.asarray(f2).ravel()
    w = np.asarray(weights)
    if w.ndim == 0:
        w = np.full_like(a, float(w))
    mask = w > 0
    return float(np.sum(w[mask] * a[mask] * b[mask]))


def cadzow(
    ssa_obj,
    rank,
    *,
    correct=True,
    tol=1e-6,
    maxiter=0,
    norm=lambda x: np.max(np.abs(x)),
    trace=False,
):
    """Perform Cadzow denoising on an :class:`SSA` object.

    Parameters
    ----------
    ssa_obj : :class:`SSA`
        Input SSA decomposition.
    rank : int
        Desired rank of the approximation.
    correct : bool, optional
        Whether to perform final correction as in the R implementation.
    tol : float, optional
        Convergence tolerance for the iterations.
    maxiter : int, optional
        Maximum number of iterations. ``0`` means no limit.
    norm : callable, optional
        Norm function used to check convergence.
    trace : bool, optional
        If ``True``, progress is printed during iterations.
    """
    weights = hankel_weights(ssa_obj.L, ssa_obj.K)
    mask = weights > 0

    F = ssa_obj.reconstruct(range(rank))

    it = 0
    while True:
        tmp = SSA(F, L=ssa_obj.L)
        rF = tmp.reconstruct(range(rank))
        it += 1
        dist = _series_dist(F, rF, norm, mask)
        if (maxiter > 0 and it >= maxiter) or dist < tol:
            break
        if trace:
            print(f"Iteration: {it}, distance: {dist}")
        F = rF

    if correct:
        alpha = _series_winnerprod(ssa_obj.x, F, weights) / _series_winnerprod(
            F, F, weights
        )
        F = _extend_series(F, alpha)
    return F
