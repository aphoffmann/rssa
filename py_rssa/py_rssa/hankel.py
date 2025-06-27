import numpy as np


def _hankel_mv(F, v):
    """Internal FFT-based Hankel multiplication."""
    N = len(F)
    K = len(v)
    L = N - K + 1
    if L <= 0:
        raise ValueError("Invalid dimensions")
    M = 1 << ((N + K - 1).bit_length())
    F_freq = np.fft.rfft(F, n=M)
    V_freq = np.fft.rfft(np.r_[v[::-1], np.zeros(M - K)], n=M)
    res = np.fft.irfft(F_freq * V_freq, n=M)
    return res[K-1:K-1+L]


def hankel_mv(F, v):
    """Multiply Hankel matrix built from sequence ``F`` with vector ``v``.

    Parameters
    ----------
    F : array_like
        Input sequence of length ``L + K - 1``.
    v : array_like
        Vector to multiply with of length ``K``.

    Returns
    -------
    numpy.ndarray
        Resulting vector of length ``L``.
    """
    F = np.asarray(F, dtype=float)
    v = np.asarray(v, dtype=float)
    return _hankel_mv(F, v)


