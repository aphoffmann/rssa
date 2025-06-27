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
    """Very small subset of the R `ssa` functionality."""

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
