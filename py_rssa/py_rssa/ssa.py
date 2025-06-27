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


def ssa(x, L=None):
    """Convenience function returning :class:`SSA`."""
    return SSA(x, L=L)


def hankel_weights(L, K):
    N = L + K - 1
    w = np.zeros(N, dtype=float)
    for i in range(N):
        w[i] = min(i + 1, L, K, N - i)
    return w


class WOSSA(SSA):
    """Weighted (windowed) Oblique SSA for 1D series."""

    def __init__(self, x, L=None, column_oblique=None, row_oblique=None):
        x = np.asarray(x, dtype=float)
        self.column_oblique = None
        self.row_oblique = None

        super().__init__(x, L=L)

        # default oblique matrices
        if column_oblique is None:
            column_oblique = np.ones(self.L)
        if row_oblique is None:
            row_oblique = np.ones(self.K)

        self.column_oblique = np.asarray(column_oblique, dtype=float)
        self.row_oblique = np.asarray(row_oblique, dtype=float)

        if self.column_oblique.size != self.L:
            raise ValueError("Length of column_oblique must equal L")
        if self.row_oblique.size != self.K:
            raise ValueError("Length of row_oblique must equal K")

        # Build weighted trajectory matrix
        Xw = (self.column_oblique[:, None] * self.X) * self.row_oblique[None, :]
        Uo, s, Vto = svd(Xw, full_matrices=False)

        # Pseudo inverse with tolerance
        def pinv(v, eps=1e-6):
            res = np.zeros_like(v)
            mask = np.abs(v) >= eps
            res[mask] = 1.0 / v[mask]
            return res

        ic = pinv(self.column_oblique)
        ir = pinv(self.row_oblique)

        U = ic[:, None] * Uo
        V = ir[:, None] * Vto.T

        sU = np.sqrt(np.sum(U ** 2, axis=0))
        sV = np.sqrt(np.sum(V ** 2, axis=0))

        self.U = U / sU
        self.Vt = (V / sV).T
        self.s = s * sU * sV

    def _hankelize_one(self, U, V):
        L = U.size
        K = V.size
        N = L + K - 1
        res = np.zeros(N)
        counts = np.zeros(N)
        for i in range(L):
            for j in range(K):
                res[i + j] += U[i] * V[j]
                counts[i + j] += 1
        return res / counts

    def _oblique_weights(self):
        return self._hankelize_one(self.column_oblique ** 2, self.row_oblique ** 2)

    def reconstruct(self, indices):
        U = self.U[:, indices]
        s = self.s[indices]
        Vt = self.Vt[indices]
        weights = self._oblique_weights()
        recon = np.zeros(self.x.size)
        for col in range(len(indices)):
            uc = U[:, col] * self.column_oblique ** 2
            vc = Vt[col] * self.row_oblique ** 2
            recon += s[col] * self._hankelize_one(uc, vc)
        return recon / weights


def wossa(x, L=None, column_oblique=None, row_oblique=None):
    """Convenience function returning :class:`WOSSA`."""
    return WOSSA(x, L=L, column_oblique=column_oblique, row_oblique=row_oblique)

