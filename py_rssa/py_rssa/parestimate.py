import numpy as np


def roots2pars(roots):
    """Convert complex roots to parameter dictionary.

    Parameters
    ----------
    roots : array_like
        Complex roots of the characteristic polynomial.

    Returns
    -------
    dict
        Dictionary with fields ``roots``, ``periods``, ``frequencies``,
        ``moduli`` and ``rates``.
    """
    roots = np.asarray(roots, dtype=complex)
    angles = np.angle(roots)
    periods = np.empty_like(angles, dtype=float)
    periods[:] = np.inf
    nonzero = np.abs(angles) > np.finfo(float).eps ** 0.5
    periods[nonzero] = 2 * np.pi / angles[nonzero]
    frequencies = angles / (2 * np.pi)
    moduli = np.abs(roots)
    rates = np.log(moduli)
    return {
        "roots": roots,
        "periods": periods,
        "frequencies": frequencies,
        "moduli": moduli,
        "rates": rates,
    }


def parestimate_pairs(U, normalize=False):
    """Estimate parameters using pair method.

    Parameters
    ----------
    U : array_like
        Matrix with two eigenvectors as columns.
    normalize : bool, optional
        If ``True`` the estimated roots are projected to the unit circle.

    Returns
    -------
    dict
        Result of :func:`roots2pars`.
    """
    U = np.asarray(U, dtype=float)
    if U.shape[1] != 2:
        raise ValueError("exactly two eigenvectors are required")
    U1 = np.diff(U[1:, :], axis=0)
    U2 = np.diff(U[:-1, :], axis=0)
    num = np.sum(U1 * U2, axis=1)
    den = np.linalg.norm(U1, axis=1) * np.linalg.norm(U2, axis=1)
    scos = num / den
    r = np.exp(1j * np.arccos(np.median(scos)))
    if normalize:
        r = r / abs(r)
    return roots2pars(r)


def parestimate_esprit(U, circular=False, normalize=False):
    """Estimate parameters via 1D ESPRIT.

    Parameters
    ----------
    U : array_like
        Matrix of eigenvectors.
    circular : bool, optional
        If ``True`` circular topology is assumed when forming shift matrices.
    normalize : bool, optional
        If ``True`` the estimated roots are projected to the unit circle.

    Returns
    -------
    dict
        Result of :func:`roots2pars`.
    """
    U = np.asarray(U, dtype=float)
    if circular:
        U1 = U
        U2 = np.vstack([U[1:], U[:1]])
    else:
        U1 = U[:-1, :]
        U2 = U[1:, :]
    A, _, _, _ = np.linalg.lstsq(U1, U2, rcond=None)
    r = np.linalg.eigvals(A)
    if normalize:
        r = r / np.abs(r)
    return roots2pars(r)


def parestimate(ssa, groups=None, method="esprit", normalize=False, circular=None):
    """Estimate signal parameters from an :class:`py_rssa.SSA` object.

    Parameters
    ----------
    ssa : :class:`py_rssa.SSA`
        Decomposed SSA object.
    groups : list of sequences, optional
        Indices of eigenvectors used for estimation. By default all
        eigenvectors are used individually.
    method : {{"esprit", "pairs"}}, optional
        Estimation method.
    normalize : bool, optional
        If ``True`` estimated roots are projected to the unit circle.
    circular : bool, optional
        Treat series as circular for ESPRIT method. If ``None`` the value
        is taken from the SSA object when available.

    Returns
    -------
    list or dict
        List of parameter dictionaries. If ``groups`` contains a single
        element the dictionary is returned directly.
    """
    if groups is None:
        groups = [[i] for i in range(ssa.U.shape[1])]

    if circular is None:
        circular = getattr(ssa, "circular", False)

    res = []
    for g in groups:
        U = ssa.U[:, g]
        if method == "pairs":
            if len(g) != 2:
                raise ValueError("pairs method requires groups of length 2")
            r = parestimate_pairs(U, normalize=normalize)
        elif method == "esprit":
            r = parestimate_esprit(U, circular=circular, normalize=normalize)
        else:
            raise ValueError("unknown method")
        res.append(r)
    if len(res) == 1:
        return res[0]
    return res
