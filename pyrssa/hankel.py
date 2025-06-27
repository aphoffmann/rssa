import numpy as np

try:
    from ._fast_hankel import hankel_mv as _hankel_mv
    HAVE_FAST = True
except Exception:  # pragma: no cover - fallback when extension missing
    HAVE_FAST = False
    def _hankel_mv(F, v):
        N = len(F)
        K = len(v)
        L = N - K + 1
        if L <= 0:
            raise ValueError("Invalid dimensions")
        F_freq = np.fft.rfft(F)
        v_freq = np.fft.rfft(np.r_[v, np.zeros(L-1)])
        res = np.fft.irfft(np.conj(v_freq) * F_freq)
        return res[:L]


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
    if HAVE_FAST:
        return _hankel_mv(F, v)
    else:
        return _hankel_mv(F, v)  # fallback defined above


