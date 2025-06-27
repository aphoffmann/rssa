import numpy as np
from numpy.linalg import svd, qr

def cond(A):
    """Return condition number of matrix ``A`` using singular values."""
    s = svd(A, compute_uv=False)
    return s[0] / s[-1]


def pseudo_inverse(A):
    """Compute Moore-Penrose pseudoinverse using QR decomposition."""
    q, r = np.linalg.qr(A)
    return np.linalg.solve(r, q.T)


def orthogonalize(Y, Z, sigma=None, side="bi", normalize=True):
    """Orthogonalize column sets ``Y`` and ``Z``.

    Parameters
    ----------
    Y, Z : ndarray
        Matrices with the same number of columns.
    sigma : ndarray or None, optional
        Optional singular values for the column pairs.
    side : {'bi', 'left', 'right'}
        Which side should be orthogonalized.
    normalize : bool, optional
        Whether to normalize when ``side`` is ``'left'`` or ``'right'``.

    Returns
    -------
    dict with keys 'd', 'u', 'v'
    """
    side = side.lower()
    if sigma is None:
        sigma = np.ones(Y.shape[1])
    rank = len(sigma)
    if Y.shape[1] != rank or Z.shape[1] != rank:
        raise ValueError("`Y` and `Z` must have `rank` columns")
    if side == "bi":
        qy, ry = np.linalg.qr(Y)
        qz, rz = np.linalg.qr(Z)
        m = ry @ (np.diag(sigma) @ rz.T)
        u, d, vt = svd(m, full_matrices=False)
        return {"d": d, "u": qy @ u, "v": qz @ vt.T}
    elif side == "left":
        qy, ry = np.linalg.qr(Y)
        u = qy
        v = Z @ (np.diag(sigma) @ np.linalg.inv(ry.T))
        if normalize:
            d = np.sqrt(np.sum(v ** 2, axis=0))
            v = v / d
        else:
            d = np.ones(rank)
        return {"d": d, "u": u, "v": v}
    elif side == "right":
        dec = orthogonalize(Y=Z, Z=Y, sigma=sigma, side="left", normalize=normalize)
        return {"d": dec["d"], "u": dec["v"], "v": dec["u"]}
    else:
        raise ValueError("Invalid `side` value")


def svd_to_lrsvd(d, u, v, basis_L=None, basis_R=None, need_project=True, fast=True):
    """Low rank SVD in the provided left and right bases."""
    rank = len(d)
    if u.shape[1] != rank or v.shape[1] != rank:
        raise ValueError("`u` and `v` must have `rank` columns")

    if basis_L is None:
        basis_L = u
        ub_L = np.eye(rank)
    else:
        ub_L = u.T @ basis_L

    if basis_R is None:
        basis_R = v
        vb_R = np.eye(rank)
    else:
        vb_R = v.T @ basis_R

    if fast:
        dec_u, dec_s, dec_vt = svd(ub_L.T @ (vb_R / d), full_matrices=False)
        sigma = dec_s[::-1]
        u2 = dec_u[:, ::-1]
        v2 = dec_vt.T[:, ::-1]
    else:
        dec_u, dec_s, dec_vt = svd(np.linalg.solve(ub_L, (d[:, None] * np.linalg.inv(vb_R).T)), full_matrices=False)
        sigma = dec_s
        u2 = dec_u
        v2 = dec_vt.T

    if need_project:
        basis_L = u @ ub_L
        basis_R = v @ vb_R

    Y = basis_L @ u2
    Z = basis_R @ v2
    return {"sigma": sigma, "Y": Y, "Z": Z}


def _trajectory_matrix(x, L):
    K = len(x) - L + 1
    if K <= 0:
        raise ValueError("Invalid window length")
    return np.column_stack([x[i:i + L] for i in range(K)])


def owcor(Fs, LM, RM):
    """Oblique weighted correlations for reconstructed series."""
    mx_list = []
    for F in Fs:
        L = LM.shape[1]
        TF = _trajectory_matrix(F, L)
        m = LM @ (TF @ RM.T)
        mx_list.append(m.ravel())
    mx = np.column_stack(mx_list)
    cov = mx.T @ mx
    std = np.sqrt(np.diag(cov))
    cor = cov / std[:, None] / std[None, :]
    cor = np.clip(cor, -1.0, 1.0)
    return cor
