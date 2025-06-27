import numpy as np
from numpy.linalg import svd

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
        self.X = np.column_stack([self.x[i:i+L] for i in range(self.K)])
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
                recon[i+j] += Xr[i, j]
                counts[i+j] += 1
        return recon / counts

    def decompose(self):
        """Return decomposition matrices (U, s, Vt)."""
        return self.U, self.s, self.Vt

    def wcor(self, groups=None):
        """Weighted correlation between reconstructed components."""
        if groups is None:
            groups = [[i] for i in range(len(self.s))]
        comps = [self.reconstruct(g) for g in groups]
        mx = np.column_stack(comps)
        weights = hankel_weights(self.L, self.K)
        return wcor(mx, weights=weights)


def ssa(x, L=None):
    """Convenience function returning :class:`SSA`."""
    return SSA(x, L=L)


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
