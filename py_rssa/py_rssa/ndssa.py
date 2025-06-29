import numpy as np
from numpy.linalg import svd
from numpy.lib.stride_tricks import sliding_window_view
from .hbhankel import factor_mask_2d, field_weights_2d

class NDSSA:
    """Basic N-Dimensional SSA with optional shaped window mask."""

    def __init__(self, x, L=None, wmask=None, mask=None, circular=False):
        x = np.asarray(x)
        self.is_complex = np.iscomplexobj(x)
        if mask is None:
            mask = np.isfinite(x)
        else:
            mask = np.asarray(mask, dtype=bool)
        fill = np.nanmean(x.real if self.is_complex else x, where=mask)
        x = np.where(mask, x, fill)
        self.x = x

        self.N = x.shape
        rank = x.ndim
        if L is None:
            L = tuple((n + 1) // 2 for n in self.N)
        L = tuple(int(l) for l in L)
        if wmask is None:
            wmask = np.ones(L, dtype=bool)
        else:
            wmask = np.asarray(wmask, dtype=bool)
            if wmask.shape != L:
                raise ValueError("wmask shape must match window length")
        self.L = L
        self.wmask = wmask

        if isinstance(circular, bool):
            circular = (circular,) * rank
        if len(circular) != rank:
            circular = tuple(circular[i % len(circular)] for i in range(rank))
        self.circular = tuple(bool(c) for c in circular)

        fmask = factor_mask_2d(mask, wmask, circular=self.circular)
        self.fmask = fmask
        self.weights = field_weights_2d(wmask, fmask, circular=self.circular)

        K = tuple(n if c else n - l + 1 for n, l, c in zip(self.N, L, self.circular))
        self.K = K

        sw = sliding_window_view(self.x, L)
        sw = sw.reshape(int(np.prod(K)), -1)
        cols = sw[fmask.ravel()]
        cols = cols[:, wmask.ravel()]
        self.X = cols.T

        self.U, s, self.Vt = svd(self.X, full_matrices=False)
        self.s = s

    def reconstruct(self, indices):
        U = self.U[:, indices]
        s = self.s[indices]
        Vt = self.Vt[indices]
        Xr = (U * s) @ Vt
        Ldim = self.wmask.sum()
        Kidx = np.argwhere(self.fmask.ravel()).ravel()
        recon = np.zeros(self.N, dtype=self.x.dtype)
        counts = np.zeros(self.N, dtype=int)
        Lslices = tuple(slice(0, l) for l in self.L)
        wmask_idx = np.where(self.wmask)
        for col_i, k in enumerate(Kidx):
            idx = []
            rem = k
            for n, l, c in zip(reversed(self.N), reversed(self.L), reversed(self.circular)):
                if c:
                    pos = rem % n
                    rem //= n
                else:
                    pos = rem % (n - l + 1)
                    rem //= (n - l + 1)
                idx.append(pos)
            idx = tuple(reversed(idx))
            slices = tuple(slice(i, i + l) for i, l in zip(idx, self.L))
            patch = np.zeros(self.L, dtype=self.x.dtype)
            patch[wmask_idx] = Xr[:, col_i]
            recon[slices][self.wmask] += patch[wmask_idx]
            counts[slices][self.wmask] += 1
        result = np.where(counts > 0, recon / counts, np.nan)
        return result

def ndssa(x, L=None, wmask=None, mask=None, circular=False):
    return NDSSA(x, L=L, wmask=wmask, mask=mask, circular=circular)

__all__ = ["NDSSA", "ndssa"]
