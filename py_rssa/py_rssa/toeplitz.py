import numpy as np


def Lcor(F, L, circular=False):
    """Compute lagged correlations of ``F`` up to lag ``L-1``.

    Parameters
    ----------
    F : array_like
        Input sequence.
    L : int
        Window length.
    circular : bool, optional
        Whether to use circular correlations.

    Returns
    -------
    numpy.ndarray
        Vector of length ``L`` with correlations for lags ``0`` to ``L-1``.
    """
    F = np.asarray(F, dtype=float)
    N = F.size
    if L <= 0 or L > N:
        raise ValueError("Invalid window length")
    R = np.empty(L, dtype=float)
    if circular:
        for i in range(L):
            R[i] = np.dot(F, np.roll(F, -i)) / N
    else:
        for i in range(L):
            R[i] = np.dot(F[:N - i], F[i:]) / (N - i)
    return R


class ToeplitzMatrix:
    """Toeplitz matrix represented via FFT based on its first row."""

    def __init__(self, R):
        self.R = np.asarray(R, dtype=float)
        self.L = self.R.size
        self.N = 2 * self.L - 1
        circ = np.concatenate((self.R, self.R[-2::-1]))
        self.freq = np.fft.rfft(circ, n=self.N)

    def matvec(self, v):
        v = np.asarray(v, dtype=float)
        if v.size != self.L:
            raise ValueError("Vector length mismatch")
        vec = np.concatenate((v, np.zeros(self.L - 1)))
        res = np.fft.irfft(np.fft.rfft(vec, n=self.N) * self.freq, n=self.N)
        return res[:self.L]

    def tmatvec(self, v):
        v = np.asarray(v, dtype=float)
        if v.size != self.L:
            raise ValueError("Vector length mismatch")
        vec = np.concatenate((np.zeros(self.L - 1), v))
        res = np.fft.irfft(np.fft.rfft(vec, n=self.N) * self.freq, n=self.N)
        return res[self.L - 1:]

    @property
    def shape(self):
        return (self.L, self.L)


def new_tmat(F, L=None, circular=False):
    F = np.asarray(F, dtype=float)
    N = F.size
    if L is None:
        L = (N + 1) // 2
    R = Lcor(F, L, circular=circular)
    return ToeplitzMatrix(R)


def tcols(t):
    return t.shape[1]


def trows(t):
    return t.shape[0]


def is_tmat(t):
    return isinstance(t, ToeplitzMatrix)


def tmatmul(tmat, v, transposed=False):
    if transposed:
        return tmat.tmatvec(v)
    else:
        return tmat.matvec(v)
