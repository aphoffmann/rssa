import numpy as np
from numpy.linalg import svd
from .ssa import SSA

class MSSA:
    """Minimal multivariate SSA implementation."""
    def __init__(self, series, L=None):
        self.series = [np.asarray(s, dtype=float) for s in series]
        self.N = [len(s) for s in self.series]
        if L is None:
            L = (min(self.N) + 1) // 2
        self.L = L
        self.K = [n - L + 1 for n in self.N]
        # build joint trajectory matrix
        blocks = []
        for s, k in zip(self.series, self.K):
            block = np.column_stack([s[i:i+L] for i in range(k)])
            blocks.append(block)
        self.X = np.hstack(blocks)
        self.U, s, self.Vt = svd(self.X, full_matrices=False)
        self.s = s

    def reconstruct(self, indices):
        idx = [i-1 if i >= 1 else i for i in indices]
        U = self.U[:, idx]
        s = self.s[idx]
        Vt = self.Vt[idx, :]
        Xr = (U * s) @ Vt
        res = []
        offset = 0
        for n, k in zip(self.N, self.K):
            block = Xr[:, offset:offset+k]
            offset += k
            recon = np.zeros(n)
            counts = np.zeros(n)
            for i in range(self.L):
                for j in range(k):
                    recon[i+j] += block[i, j]
                    counts[i+j] += 1
            res.append(recon / counts)
        return res


def mssa(series, L=None):
    """Convenience function returning :class:`MSSA`."""
    return MSSA(series, L=L)


def reconstruct(obj, groups):
    """Reconstruct for :class:`MSSA` or :class:`SSA`."""
    idx = [g-1 if g >= 1 else g for g in groups]
    if isinstance(obj, MSSA):
        return obj.reconstruct(idx)
    elif isinstance(obj, SSA):
        return obj.reconstruct(idx)
    else:
        raise TypeError("Unsupported object")


def gapfill(obj, groups):
    """Fill missing values in a series using selected components."""
    if not isinstance(obj, SSA):
        raise TypeError("gapfill supports SSA objects only")
    rec = obj.reconstruct([g-1 if g >= 1 else g for g in groups])
    filled = obj.x.copy()
    mask = np.isnan(filled)
    filled[mask] = rec[mask]
    return filled


def forecast(obj, groups, steps):
    """Very naive forecasting using last difference."""
    if not isinstance(obj, SSA):
        raise TypeError("forecast supports SSA objects only")
    rec = obj.reconstruct([g-1 if g >= 1 else g for g in groups])
    diff = rec[-1] - rec[-2] if len(rec) > 1 else 0.0
    extra = rec[-1] + diff * np.arange(1, steps + 1)
    return np.concatenate([rec, extra])

