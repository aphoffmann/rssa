"""Gap filling utilities based on SSA decomposition."""

import numpy as np


def lrr(U, reverse=False, eps=np.sqrt(np.finfo(float).eps)):
    """Estimate linear recurrence relation coefficients.

    Parameters
    ----------
    U : ndarray
        Orthonormal basis matrix with shape (L, r).
    reverse : bool, optional
        If ``True`` then coefficients for backward recurrence are
        calculated.  Defaults to ``False``.
    eps : float, optional
        Tolerance for verticality check.

    Returns
    -------
    ndarray
        Vector of length ``L - 1`` with recurrence coefficients.
    """
    U = np.asarray(U, dtype=float)
    L = U.shape[0]
    if U.size == 0:
        return np.zeros(L - 1)

    idx = 0 if reverse else L - 1
    lpf = U.conj() @ U[idx, :]
    divider = 1.0 - lpf[idx]
    if abs(divider) < eps:
        raise ValueError("Verticality coefficient equals to 1")
    return lpf[np.arange(L) != idx] / divider


def apply_lrr(F, coeffs, length, reverse=False):
    """Apply linear recurrence relation to extend a series."""
    F = np.asarray(F, dtype=float)
    r = len(coeffs)
    if r > len(F):
        raise ValueError("LRR order is larger than input sequence")

    if reverse:
        work = np.concatenate([np.full(length, np.nan), F])
        for i in range(length):
            pos = length - i - 1
            work[pos] = np.dot(work[pos + 1:pos + 1 + r], coeffs)
        return work[:length]
    else:
        work = np.concatenate([F, np.full(length, np.nan)])
        N = len(F)
        for i in range(length):
            work[N + i] = np.dot(work[N + i - r:N + i], coeffs)
        return work[-length:]


def _classify_gaps(na_idx):
    """Return list of (left, right) tuples for contiguous gaps."""
    if len(na_idx) == 0:
        return []
    na_idx = np.sort(na_idx)
    gaps = []
    start = na_idx[0]
    prev = start
    for idx in na_idx[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            gaps.append((start, prev))
            start = prev = idx
    gaps.append((start, prev))
    return gaps


def _fill_in(F, L, gap, flrr, blrr, alpha):
    left, right = gap
    length = right - left + 1
    if callable(alpha):
        alpha_vals = alpha(length)
    else:
        alpha_vals = np.full(length, alpha, dtype=float)

    left_pos = left < L - 1
    right_pos = right >= len(F) - L

    if left_pos:
        # gap touches the beginning
        seg = F[right + 1:right + 1 + L]
        return apply_lrr(seg, blrr, length, reverse=True)
    elif right_pos:
        seg = F[left - L:left]
        return apply_lrr(seg, flrr, length, reverse=False)
    else:
        seg_right = F[right + 1:right + 1 + L]
        seg_left = F[left - L:left]
        forward = apply_lrr(seg_left, flrr, length, reverse=False)
        backward = apply_lrr(seg_right, blrr, length, reverse=True)
        return alpha_vals * backward + (1 - alpha_vals) * forward


def gapfill(ssa_obj, groups=None, base="original", method="sequential", alpha=None):
    """Fill gaps (NaNs) in the original series using SSA reconstruction."""
    if groups is None:
        groups = [1]
    if alpha is None:
        alpha = lambda n: np.linspace(0, 1, n)

    if method != "sequential":
        raise NotImplementedError("Only sequential method is implemented")
    if base not in {"original", "reconstructed"}:
        raise ValueError("Invalid base option")

    L = ssa_obj.L
    N = ssa_obj.x.size
    result = []
    for g in groups:
        idx = [i - 1 for i in (g if isinstance(g, (list, tuple)) else [g])]
        series = ssa_obj.reconstruct(idx) if base == "reconstructed" else ssa_obj.x.copy()
        na_idx = np.flatnonzero(np.isnan(series))
        if len(na_idx) == 0:
            result.append(series)
            continue

        Ug = ssa_obj.U[:, idx]
        flrr = lrr(Ug, reverse=False)
        blrr = lrr(Ug, reverse=True)
        for gap in _classify_gaps(na_idx):
            fill = _fill_in(series, L, gap, flrr, blrr, alpha)
            series[gap[0]:gap[1] + 1] = fill
        result.append(series)

    return result[0] if len(result) == 1 else result

