import numpy as np
from numpy.linalg import svd

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

        # full SVD decomposition
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
        w = hankel_weights(self.L, self.K)
        res = np.empty((len(groups), len(groups)))
        for i in range(len(groups)):
            for j in range(i, len(groups)):
                num = np.sum(w * comps[i] * comps[j])
                den = np.sqrt(np.sum(w * comps[i] ** 2) *
                              np.sum(w * comps[j] ** 2))
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


def ssa(x, L=None):
    """Convenience function returning :class:`SSA`."""
    return SSA(x, L=L)


def hankel_weights(L, K):
    N = L + K - 1
    w = np.zeros(N, dtype=float)
    for i in range(N):
        w[i] = min(i + 1, L, K, N - i)
    return w
