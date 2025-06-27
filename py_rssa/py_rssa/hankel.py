import numpy as np


def hankel_mv(c, r, v):
    """Multiply a Hankel matrix by a vector.

    Parameters
    ----------
    c : array_like
        First column of the Hankel matrix.
    r : array_like
        Last row of the Hankel matrix.
    v : array_like
        Vector to multiply.
    """
    c = np.asarray(c, dtype=float)
    r = np.asarray(r, dtype=float)
    v = np.asarray(v, dtype=float)
    m = c.size
    n = r.size
    if v.size != n:
        raise ValueError("vector length mismatch")
    out = np.zeros(m, dtype=float)
    for i in range(m):
        for j in range(n):
            if i + j < m:
                h = c[i + j]
            else:
                h = r[i + j - m]
            out[i] += h * v[j]
    return out
