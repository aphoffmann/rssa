"""Complex Hankel matrix helpers."""

import numpy as np

from .ssa import hankel_weights


def _chankel_mv(F, v):
    """Internal FFT-based complex Hankel multiplication."""
    N = len(F)
    K = len(v)
    L = N - K + 1
    if L <= 0:
        raise ValueError("Invalid dimensions")
    M = 1 << ((N + K - 1).bit_length())
    F_freq = np.fft.fft(F, n=M)
    V_freq = np.fft.fft(np.concatenate([v[::-1], np.zeros(M - K, dtype=complex)]), n=M)
    res = np.fft.ifft(F_freq * V_freq, n=M)
    return res[K-1:K-1+L]


def chankel_mv(F, v):
    """Multiply complex Hankel matrix built from sequence ``F`` with vector ``v``."""
    F = np.asarray(F, dtype=complex)
    v = np.asarray(v, dtype=complex)
    return _chankel_mv(F, v)


def chankelize(U, V):
    """Hankelize a single pair of complex vectors ``U`` and ``V``."""
    U = np.asarray(U, dtype=complex)
    V = np.asarray(V, dtype=complex)
    L = len(U)
    K = len(V)
    conv = np.convolve(U, V)
    w = hankel_weights(L, K)
    return conv / w


def chankelize_multi(U, V):
    """Hankelize multiple pairs of complex vectors.

    ``U`` and ``V`` should be 2-D arrays with shapes ``(L, r)`` and ``(K, r)``
    respectively. The result is a complex vector of length ``L + K - 1``.
    """
    U = np.asarray(U, dtype=complex)
    V = np.asarray(V, dtype=complex)
    if U.ndim != 2 or V.ndim != 2:
        raise ValueError("U and V must be two-dimensional")
    if U.shape[1] != V.shape[1]:
        raise ValueError("U and V must have the same number of columns")
    L, r = U.shape
    K = V.shape[0]
    res = np.zeros(L + K - 1, dtype=complex)
    for i in range(r):
        res += np.convolve(U[:, i], V[:, i])
    w = hankel_weights(L, K)
    return res / w

