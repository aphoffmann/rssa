import numpy as np
from numpy.linalg import svd, qr


def orthopoly(d, L):
    """Generate orthonormal polynomial basis of degree ``d``.

    Parameters
    ----------
    d : int or str
        Degree of the polynomial or one of ``"none"``, ``"constant"``,
        ``"centering"``, ``"linear"``, ``"quadratic"`` or ``"qubic"``.
    L : int
        Length of the window.

    Returns
    -------
    numpy.ndarray
        Matrix of shape ``(L, d)`` whose columns form an orthonormal basis
        for the requested polynomial space.  If ``d`` is ``0`` or ``"none"``
        an empty array with zero columns is returned.
    """

    if isinstance(d, str):
        mapping = {
            "none": 0,
            "constant": 1,
            "centering": 1,
            "linear": 2,
            "quadratic": 3,
            "qubic": 4,
        }
        d = mapping[d]

    if d == 0:
        return np.empty((L, 0))
    if d == 1:
        return np.full((L, 1), 1.0 / np.sqrt(L))

    x = np.arange(1, L + 1)
    # NumPy's ``poly`` is different from R's ``poly`` so we simply build a
    # Vandermonde matrix and orthonormalise it.
    M = np.column_stack([x**i for i in range(d)])
    Q, _ = qr(M)
    return Q


class SSA:

    """Simple SSA implementation mirroring ``rssa::ssa`` core features."""


    def __init__(self, x, L=None):
        self.x = np.asarray(x, dtype=float)
        N = self.x.size
        if L is None:
            L = (N + 1) // 2
        self.L = L
        self.K = N - L + 1

        # trajectory matrix
        self.X = np.column_stack([self.x[i : i + L] for i in range(self.K)])
        self.U, s, self.Vt = svd(self.X, full_matrices=False)
        self.s = s

    def reconstruct(self, indices):
        """Reconstruct series from selected eigentriples."""
        U = self.U[:, indices]
        s = self.s[indices]
        Vt = self.Vt[indices]
        Xr = (U * s) @ Vt
        # Averaging anti-diagonals
        N = self.x.size
        L = self.L
        K = self.K
        recon = np.zeros(N)
        counts = np.zeros(N)
        for i in range(L):
            for j in range(K):
                recon[i + j] += Xr[i, j]
                counts[i + j] += 1
        return recon / counts

    def decompose(self):
        """Return decomposition matrices (U, s, Vt)."""
        return self.U, self.s, self.Vt

    def wcor(self, groups=None):
        """Weighted correlation between reconstructed components."""
        if groups is None:
            groups = [[i] for i in range(len(self.s))]
        comps = [self.reconstruct(g) for g in groups]
        w = hankel_weights(self.L, self.K)
        res = np.empty((len(groups), len(groups)))
        for i in range(len(groups)):
            for j in range(i, len(groups)):
                num = np.sum(w * comps[i] * comps[j])
                den = np.sqrt(np.sum(w * comps[i] ** 2) * np.sum(w * comps[j] ** 2))
                c = num / den if den != 0 else 0.0
                res[i, j] = res[j, i] = c
        return res

    # ------------------------------------------------------------------
    # Helper methods replicating rssa functionality

    def wnorm(self):
        """Weighted norm of the original series."""
        w = hankel_weights(self.L, self.K)
        return np.sqrt(np.sum(w * self.x ** 2))

    def contributions(self, indices=None):
        """Return relative contribution of selected eigentriples."""
        if indices is None:
            indices = range(len(self.s))
        sigma = self.s[indices]
        return sigma ** 2 / (self.wnorm() ** 2)

    def residuals(self, indices=None):
        """Return residuals after reconstructing selected components."""
        if indices is None:
            indices = range(len(self.s))
        rec = self.reconstruct(indices)
        return self.x - rec

    # Simple accessors mirroring rssa helpers
    def nu(self):
        return self.U.shape[1]

    def nv(self):
        return self.Vt.shape[0]

    def nsigma(self):
        return self.s.size


class PartialSSA(SSA):
    """Projection SSA.

    This variant removes the subspaces specified by ``column_proj`` and
    ``row_proj`` from the trajectory matrix prior to decomposition.  The
    implementation is intentionally lightweight and supports only one
    dimensional series.
    """

    def __init__(self, x, L=None, *, column_proj=None, row_proj=None):
        self.x = np.asarray(x, dtype=float)
        N = self.x.size
        if L is None:
            L = (N + 1) // 2
        self.L = L
        self.K = N - L + 1

        # Build trajectory matrix of the original series
        X = np.column_stack([self.x[i : i + L] for i in range(self.K)])

        # Prepare projection matrices
        CP = None
        if column_proj not in (None, 0, "none"):
            CP = np.asarray(
                (
                    orthopoly(column_proj, L)
                    if np.isscalar(column_proj) or isinstance(column_proj, str)
                    else column_proj
                ),
                dtype=float,
            )
            CP, _ = qr(CP)

        RP = None
        if row_proj not in (None, 0, "none"):
            RP = np.asarray(
                (
                    orthopoly(row_proj, self.K)
                    if np.isscalar(row_proj) or isinstance(row_proj, str)
                    else row_proj
                ),
                dtype=float,
            )
            RP, _ = qr(RP)

        # Apply projections: (I - CP CP^T) X (I - RP RP^T)
        if RP is not None and RP.size > 0:
            X = X - (X @ RP) @ RP.T
        if CP is not None and CP.size > 0:
            X = X - CP @ (CP.T @ X)

        self.column_projector = CP
        self.row_projector = RP

        self.X = X
        self.U, s, self.Vt = svd(self.X, full_matrices=False)
        self.s = s


def ssa(x, L=None):
    """Convenience function returning :class:`SSA`."""
    return SSA(x, L=L)


def pssa(x, L=None, *, column_proj=None, row_proj=None):
    """Convenience function returning :class:`PartialSSA`."""
    return PartialSSA(x, L=L, column_proj=column_proj, row_proj=row_proj)


def hankel_weights(L, K):
    N = L + K - 1
    w = np.zeros(N, dtype=float)
    for i in range(N):
        w[i] = min(i + 1, L, K, N - i)
    return w

def wnorm(x, *, L=None, weights=None):
    """Weighted norm of a sequence.

    Parameters
    ----------
    x : array_like
        Input one-dimensional sequence.
    L : int, optional
        Window length used to derive Hankel weights when ``weights`` are not
        provided. Defaults to ``(N + 1) // 2`` where ``N`` is the length of
        ``x``.
    weights : array_like, optional
        Pre-computed Hankel weights of length ``N``.

    Returns
    -------
    float
        Weighted Euclidean norm of ``x``.
    """
    x = np.asarray(x)
    N = x.size
    if weights is None:
        if L is None:
            L = (N + 1) // 2
        weights = hankel_weights(L, N - L + 1)
    weights = np.asarray(weights)
    if weights.size != N:
        raise ValueError("weights length mismatch")
    return float(np.sqrt(np.sum(weights * np.abs(x) ** 2)))


def wcor(x, *, L=None, weights=None):
    """Weighted correlation matrix for columns of ``x``.

    Parameters
    ----------
    x : array_like, shape (N, M)
        Matrix whose columns correspond to series to correlate.
    L : int, optional
        Window length used to compute Hankel weights when ``weights`` are not
        supplied. Defaults to ``(N + 1) // 2``.
    weights : array_like, optional
        Pre-computed Hankel weights of length ``N``.

def igapfill(ssa_obj, groups, fill=None, tol=1e-6, maxiter=0,
             norm=None, base="original"):
    """Iterative gap filling using SSA reconstruction.

    Parameters
    ----------
    ssa_obj : :class:`SSA`
        Decomposition to use as a template. Only the window length from this
        object is utilised.
    groups : sequence of int
        Indices of eigentriples to use for the final reconstruction. 1-based
        indices are accepted for compatibility with the R implementation.
    fill : float or array-like, optional
        Initial values for the missing entries. If not provided the mean of the
        series (ignoring NaNs) is used.
    tol : float, optional
        Tolerance for the iteration stopping criterion.
    maxiter : int, optional
        Maximum number of iterations. ``0`` means no limit.
    norm : callable, optional
        Function used to compute the distance between successive
        approximations. Defaults to ``sqrt(max(x**2))``.
    base : {"original", "reconstructed"}
        Whether to return the reconstructed series itself or replace only the
        missing entries in the original series.

    Returns
    -------
    numpy.ndarray
        Series with filled gaps.
    """

    x = np.asarray(ssa_obj.x, dtype=float)
    groups = [g - 1 for g in groups]  # convert to zero based
    q = max(groups) + 1

    if norm is None:
        def norm(v):
            return np.sqrt(np.nanmax(v ** 2))

    F = x.copy()
    na_idx = np.isnan(F)

    if fill is None:
        fill_value = np.nanmean(F)
    else:
        fill_value = fill

    if np.isscalar(fill_value):
        F[na_idx] = fill_value
    else:
        fill_arr = np.asarray(fill_value)
        F[na_idx] = fill_arr[na_idx]

    it = 0
    while True:
        s = SSA(F, L=ssa_obj.L)
        r = s.reconstruct(list(range(q)))
        rF = F.copy()
        rF[na_idx] = r[na_idx]

        rss = norm(F - rF)
        it += 1
        if (maxiter > 0 and it >= maxiter) or rss < tol:
            F = rF
            break
        F = rF

    s_final = SSA(F, L=ssa_obj.L)
    rec = s_final.reconstruct(groups)

    if base == "reconstructed":
        return rec

    out = x.copy()
    out[na_idx] = rec[na_idx]
    return out

def _lrr(U, reverse=False, eps=np.sqrt(np.finfo(float).eps)):
    """Compute linear recurrence coefficients from column space ``U``.

    Parameters
    ----------
    U : ndarray
        Matrix of eigenvectors (``L x r``).
    reverse : bool, optional
        If ``True`` compute coefficients for reverse forecasting.
    eps : float, optional
        Numerical threshold for singularities.

    Returns
    -------
    ndarray
        Array of length ``L-1`` with linear recurrence coefficients.
    """
    if U.size == 0:
        # Zero subspace, return trivial recurrence
        return np.zeros(U.shape[0] - 1)

    n = U.shape[0]
    idx = n - 1 if not reverse else 0

    # Projection of rows on the last (or first) row of U
    lpf = np.sum(np.conj(U) * U[idx, :], axis=1)
    divider = 1.0 - lpf[idx]
    if abs(divider) < eps:
        raise ValueError("Verticality coefficient equals to 1")

    coeffs = np.delete(lpf, idx) / divider
    return coeffs


def _apply_lrr(F, lrr, steps, only_new=False, reverse=False):
    """Apply linear recurrence relation to extend the sequence ``F``."""
    F = np.asarray(F, dtype=float)
    r = len(lrr)
    if r > F.size:
        raise ValueError("LRR order is larger than the series length")

    ext = np.concatenate([F, np.zeros(steps)])

    if not reverse:
        for i in range(steps):
            ext[len(F) + i] = np.dot(ext[len(F) + i - r:len(F) + i], lrr)
        return ext[-steps:] if only_new else ext
    else:
        # reverse forecasting
        ext = np.concatenate([np.zeros(steps), F])
        for i in range(steps):
            ext[steps - i - 1] = np.dot(ext[steps - i:steps - i + r], lrr)
        return ext[:steps] if only_new else ext


def rforecast(ssa_obj, groups=None, steps=1, base="reconstructed", only_new=False):
    """Perform recurrent forecasting for :class:`SSA` object."""
    if groups is None:
        groups = list(range(len(ssa_obj.s)))

    U = ssa_obj.U[:, groups]

    lrr = _lrr(U)

    if base == "reconstructed":
        F = ssa_obj.reconstruct(groups)
    else:
        F = ssa_obj.x

    return _apply_lrr(F, lrr, steps, only_new=only_new)


def vforecast(ssa_obj, groups=None, steps=1, only_new=False):
    """Perform vector forecasting for :class:`SSA` object.

    This is a simplified implementation based on linear recurrence
    relations and should provide results similar to the R version for
    typical use cases.
    """

    # Vector forecasting in the R package relies on special shift
    # matrices.  Here we approximate it using the linear recurrence
    # derived from the selected eigentriples which usually gives
    # comparable results for short horizons.

    return rforecast(ssa_obj, groups=groups, steps=steps, base="reconstructed", only_new=only_new)


def bforecast(ssa_obj, groups=None, steps=1, R=100, level=0.95,
              method="recurrent", interval="prediction", only_new=True):
    """Bootstrap forecast with confidence intervals.

    Parameters
    ----------
    ssa_obj : :class:`SSA`
        Decomposed SSA object.
    groups : sequence, optional
        Indices of eigentriples to use.  All are used by default.
    steps : int, optional
        Number of forecast steps.
    R : int, optional
        Number of bootstrap replications.
    level : float, optional
        Confidence level for the intervals.
    method : {"recurrent", "vector"}
        Forecasting method used for each bootstrap sample.
    interval : {"prediction", "confidence"}
        Type of interval to compute.
    only_new : bool, optional
        Whether to return only the forecasted values.

    Returns
    -------
    dict
        Dictionary containing forecast mean and bounds.
    """

    if groups is None:
        groups = list(range(len(ssa_obj.s)))

    rec = ssa_obj.reconstruct(groups)
    resid = ssa_obj.x - rec

    forecasts = np.empty((steps, R))
    for i in range(R):
        noise = np.random.choice(resid, size=resid.size, replace=True)
        sample = rec + noise
        sample_ssa = SSA(sample, L=ssa_obj.L)
        if method == "vector":
            fc = vforecast(sample_ssa, groups=groups, steps=steps, only_new=True)
        else:
            fc = rforecast(sample_ssa, groups=groups, steps=steps, only_new=True)
        forecasts[:, i] = fc

    mean_fc = forecasts.mean(axis=1)
    lower = np.quantile(forecasts, (1 - level) / 2, axis=1)
    upper = np.quantile(forecasts, 1 - (1 - level) / 2, axis=1)

    if interval == "prediction":
        err = np.random.choice(resid, size=(steps, R), replace=True)
        lower = lower + np.quantile(err, (1 - level) / 2, axis=1)
        upper = upper + np.quantile(err, 1 - (1 - level) / 2, axis=1)

    result = {
        "mean": mean_fc if only_new else None,
        "lower": lower,
        "upper": upper,
    }
    if not only_new:
        base_series = rec if method == "recurrent" else rec
        result["series"] = np.concatenate([base_series, mean_fc])

    return result


def forecast(ssa_obj, groups=None, steps=1, method="recurrent", **kwargs):
    """Convenience wrapper for forecasting.

    Parameters
    ----------
    ssa_obj : :class:`SSA`
        Decomposed SSA object.
    groups : sequence, optional
        Indices of eigentriples to use.  All are used by default.
    steps : int, optional
        Number of forecast steps.
    method : {"recurrent", "vector"}
        Forecasting algorithm to use.


    Returns
    -------
    numpy.ndarray
        ``M x M`` weighted correlation matrix.
    """
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2:
        raise ValueError("input must be 1-D or 2-D array")
    N = x.shape[0]
    if weights is None:
        if L is None:
            L = (N + 1) // 2
        weights = hankel_weights(L, N - L + 1)
    weights = np.asarray(weights)
    if weights.size != N:
        raise ValueError("weights length mismatch")

    wx = weights[:, None] * np.conjugate(x)
    cov = wx.T @ x
    inv_norm = 1.0 / np.sqrt(np.abs(np.diag(cov)))
    cor = inv_norm[:, None] * cov * inv_norm[None, :]
    np.fill_diagonal(cor, 1.0)
    cor = np.clip(cor.real, -1.0, 1.0)
    return cor

        Forecasted series (original series extended by ``steps`` points).
    """

    if method == "vector":
        f = vforecast(ssa_obj, groups=groups, steps=steps, only_new=False)
    else:
        f = rforecast(ssa_obj, groups=groups, steps=steps, base="reconstructed", only_new=False)
    return f


