import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
from scipy import interpolate
import matplotlib.pyplot as plt


def grouping_auto(x, *args, grouping_method="pgram", **kwargs):
    """Automatic grouping wrapper.

    Parameters
    ----------
    x : SSA
        Decomposition object.
    grouping_method : {"pgram", "wcor"}
        Method used for grouping.
    """
    method = grouping_method.lower()
    if method == "pgram":
        return grouping_auto_pgram(x, *args, **kwargs)
    elif method == "wcor":
        return grouping_auto_wcor(x, *args, **kwargs)
    else:
        raise ValueError("Unknown grouping method: %s" % grouping_method)


def grouping_auto_wcor(x, groups=None, nclust=None, method="complete"):
    """Cluster components using w-correlation matrix."""
    if groups is None:
        groups = list(range(len(x.s)))
    groups = sorted(set(groups))
    w = x.wcor(groups=[[g] for g in groups])
    dist = squareform((1 - w) / 2, checks=False)
    Z = linkage(dist, method=method)
    if nclust is None:
        nclust = max(1, len(groups) // 2)
    labels = fcluster(Z, nclust, criterion="maxclust")
    res = [
        [g for g, lab in zip(groups, labels) if lab == i]
        for i in range(1, nclust + 1)
    ]
    return {"groups": res, "linkage": Z, "wcor": w}


def plot_grouping_auto_wcor(res, **kwargs):
    """Plot dendrogram for wcor-based grouping."""
    Z = res.get("linkage")
    if Z is None:
        raise ValueError("No linkage information in result")
    dendrogram(Z, **kwargs)
    plt.xlabel("Component")
    plt.ylabel("Distance")
    plt.show()


def pgram(x):
    """Compute periodogram for columns of ``x``."""
    x = np.asarray(x)
    if x.ndim == 1:
        x = x[:, None]
    if not np.all(np.isfinite(x)):
        raise ValueError("x must contain finite values only")
    n = x.shape[0]
    X = np.fft.fft(x, axis=0)
    N = n // 2 + 1
    spec = np.abs(X[:N]) ** 2
    if n % 2 == 0:
        if N > 2:
            spec[1 : N - 1] *= 2
    else:
        if N >= 2:
            spec[1:N] *= 2
    freq = np.linspace(0, 1, n + 1)[:N]
    cumspec = np.cumsum(spec, axis=0)
    return {"spec": spec, "freq": freq, "cumspec": cumspec}


def _integrate_linear(freq, cumspec, a, b):
    """Linear interpolation helper for periodogram integration."""
    v_a = np.interp(a, freq, cumspec)
    v_b = np.interp(b, freq, cumspec)
    return v_b - v_a


def grouping_auto_pgram(
    x,
    groups=None,
    base="series",
    freq_bins=2,
    threshold=0.0,
    method="constant",
    drop=True,
):
    """Group components using spectral contributions."""
    method = method.lower()
    if groups is None:
        groups = list(range(len(x.s)))
    groups = sorted(set(groups))

    if base == "eigen":
        Fs = x.U[:, groups]
    elif base == "factor":
        Fs = x.Vt.T[:, groups]
    else:
        Fs = np.column_stack([x.reconstruct([g]) for g in groups])

    pg = pgram(Fs)
    freq = pg["freq"]
    spec = pg["spec"]
    cumspec = pg["cumspec"]

    if not isinstance(freq_bins, list):
        fb = np.atleast_1d(freq_bins)
        if fb.size == 1 and fb[0] >= 2:
            fb = np.linspace(0, 0.5, int(fb[0]) + 1)[1:]
        freq_lower = np.r_[-np.inf, fb[:-1]]
        freq_upper = fb
    else:
        fl = []
        fu = []
        for b in freq_bins:
            b = np.atleast_1d(b)
            if b.size == 1:
                fl.append(-np.inf)
                fu.append(b[0])
            else:
                fl.append(b[0])
                fu.append(b[1])
        freq_lower = np.array(fl)
        freq_upper = np.array(fu)

    nres = max(len(freq_upper), np.atleast_1d(threshold).size)
    freq_lower = np.resize(freq_lower, nres)
    freq_upper = np.resize(freq_upper, nres)
    threshold = np.resize(np.atleast_1d(threshold), nres)

    norms = spec.sum(axis=0)
    contrib = np.zeros((len(groups), nres))
    for i in range(nres):
        if method == "constant":
            mask = (freq < freq_upper[i]) & (freq >= freq_lower[i])
            contrib[:, i] = spec[mask].sum(axis=0) / norms
        else:
            contrib[:, i] = np.array(
                [
                    _integrate_linear(freq, cumspec[:, j], freq_lower[i], freq_upper[i])
                    for j in range(spec.shape[1])
                ]
            ) / norms

    type_ = "splitting" if np.all(threshold <= 0) else "independent"
    if type_ == "splitting":
        gi = contrib.argmax(axis=1)
        result = [
            [g for g, idx in zip(groups, gi) if idx == i]
            for i in range(nres)
        ]
    else:
        result = [
            [g for g, val in zip(groups, contrib[:, i]) if val >= threshold[i]]
            for i in range(nres)
        ]
    if drop:
        result = [r for r in result if len(r) > 0]
    return {
        "groups": result,
        "contributions": contrib,
        "type": type_,
        "threshold": threshold,
    }


def plot_grouping_auto_pgram(res, superpose=None, order=None):
    """Plot contributions for pgram-based grouping."""
    contrib = np.asarray(res.get("contributions"))
    if contrib.ndim != 2:
        raise ValueError("Invalid contributions matrix")
    ncomp, ng = contrib.shape
    labels = [str(i + 1) for i in range(ncomp)]

    if order is None:
        order = res.get("type") == "independent"
    if superpose is None:
        superpose = res.get("type") == "independent"

    if order:
        idx = np.argsort(-contrib, axis=0)
        contrib = np.take_along_axis(contrib, idx, axis=0)
        labels = np.take_along_axis(np.array(labels)[:, None], idx, axis=0)

    if superpose:
        for i in range(ng):
            plt.plot(range(1, ncomp + 1), contrib[:, i], label=f"g{i + 1}")
        plt.xlabel("Component")
        plt.ylabel("Relative contribution")
        plt.legend()
    else:
        fig, axes = plt.subplots(ng, 1, sharex=True)
        if ng == 1:
            axes = [axes]
        for ax, i in zip(axes, range(ng)):
            ax.bar(range(1, ncomp + 1), contrib[:, i])
            ax.set_ylabel(f"g{i + 1}")
        axes[-1].set_xlabel("Component")
    plt.show()
