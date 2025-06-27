import numpy as np
from scipy.linalg import lstsq
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist

from .ssa import SSA


def _shift_matrix(U, circular=False):
    """Estimate the shift matrix for column subspace.

    Parameters
    ----------
    U : ndarray of shape (L, r)
        Left singular vectors (or any orthonormal basis).
    circular : bool, optional
        If ``True``, treat series as circular.
    Returns
    -------
    ndarray of shape (r, r)
        Estimated shift matrix.
    """
    if circular:
        left = U
        right = np.vstack([U[1:], U[:1]])
    else:
        left = U[:-1]
        right = U[1:]
    Z, _, _, _ = lstsq(left, right)
    return Z


def _clust_basis(evecs, roots, k=2):
    """Cluster ESPRIT eigenvectors to form real basis."""
    ord_idx = np.argsort(np.abs(np.angle(roots)))
    roots = roots[ord_idx]
    U = evecs[:, ord_idx]

    maxk = np.sum(np.imag(roots) >= -np.finfo(float).eps)
    if k > maxk:
        raise ValueError(
            "k exceeds the number of different ESPRIT roots with non-negative imaginary parts (%d)" % maxk
        )

    pts = np.column_stack([np.real(roots), np.abs(np.imag(roots))])
    d = pdist(pts, metric="euclidean")
    Z = linkage(d, method="complete")
    idx = fcluster(Z, t=k, criterion="maxclust")

    groups = [np.where(idx == i + 1)[0] for i in range(k)]

    for g in groups:
        if len(g) == 0:
            continue
        B = np.hstack([np.real(U[:, g]), np.imag(U[:, g])])
        u, _, _ = np.linalg.svd(B, full_matrices=False)
        U[:, g] = u[:, : len(g)]

    U = np.real(U)

    order_idx = np.concatenate(groups)
    U = U[:, order_idx]
    sizes = [len(g) for g in groups]
    cl = np.cumsum([0] + sizes)
    groups = [list(range(cl[i], cl[i + 1])) for i in range(k)]

    return U, groups


def eossa(ssa_obj, k=2):
    """Perform Exponential SSA on an :class:`SSA` object.

    Parameters
    ----------
    ssa_obj : :class:`SSA`
        Decomposition to process.
    k : int, optional
        Desired number of frequency groups.

    Returns
    -------
    :class:`SSA`
        New object with transformed decomposition.
    """
    if not isinstance(ssa_obj, SSA):
        raise TypeError("ssa_obj must be an SSA instance")

    U = ssa_obj.U
    Vt = ssa_obj.Vt

    Z = _shift_matrix(U)
    vals, vecs = np.linalg.eig(Z)

    C, groups = _clust_basis(vecs, vals, k=k)
    U_new = U @ C
    Vt_new = np.linalg.solve(C.T, Vt)
    s_new = np.ones(C.shape[1])

    res = SSA.__new__(SSA)
    res.x = ssa_obj.x
    res.L = ssa_obj.L
    res.K = ssa_obj.K
    res.X = ssa_obj.X
    res.U = U_new
    res.s = s_new
    res.Vt = Vt_new
    res.groups = groups
    return res
