import numpy as np
import matplotlib.pyplot as plt
from typing import Mapping, Sequence, Any

__all__ = [
    "do_slice_array",
    "do_slice_reconstruction",
    "plot_nd_reconstruction",
    "plot_ssa_vectors_nd",
]

# Internal helper ------------------------------------------------------------

def _parse_dim_index(name: str) -> int:
    """Convert a dimension name like 'x' or 'd2' to an index."""
    try:
        if name.startswith("d") and name[1:].isdigit():
            return int(name[1:]) - 1
        if name.isdigit():
            return int(name) - 1
    except Exception:
        pass
    mapping = {"x": 0, "i": 0, "y": 1, "j": 1, "z": 2, "k": 2, "t": 3}
    if name not in mapping:
        raise ValueError(f"{name} is not a proper dimension name")
    return mapping[name]

# Public API -----------------------------------------------------------------

def do_slice_array(x: np.ndarray, slice_dict: Mapping[str, Any]) -> np.ndarray:
    """Slice array ``x`` according to ``slice_dict``.

    Parameters
    ----------
    x : np.ndarray
        Input array to slice.
    slice_dict : Mapping[str, Any]
        Mapping from dimension names (``"x"``, ``"y"``, ``"d1"``, etc.) to
        slices or indices usable in ``numpy`` indexing.
    """
    arr = np.asarray(x)
    rank = arr.ndim
    args = [slice(None)] * rank
    used = set()
    for name, slc in slice_dict.items():
        idx = _parse_dim_index(str(name))
        if idx >= rank:
            raise ValueError("slice dimension exceeds array rank")
        if idx in used:
            raise ValueError("duplicate slice for dimension")
        used.add(idx)
        args[idx] = slc
    return arr[tuple(args)]

def do_slice_reconstruction(recon: Mapping[str, Any], slice_dict: Mapping[str, Any]):
    """Slice a reconstruction dictionary returned by ``reconstruct``.

    The dictionary must contain ``components`` (a sequence of arrays), ``series``
    and ``residuals`` entries.
    """
    comps = [do_slice_array(c, slice_dict) for c in recon["components"]]
    series = do_slice_array(recon["series"], slice_dict)
    residuals = do_slice_array(recon["residuals"], slice_dict)
    return {"components": comps, "series": series, "residuals": residuals}

def _plot_1d_reconstruction(rec, *, add_original=True, add_residuals=True, ax=None):
    ax = ax or plt.gca()
    if add_original:
        ax.plot(rec["series"], label="Original")
    for i, comp in enumerate(rec["components"]):
        ax.plot(comp, label=f"F{i+1}")
    if add_residuals:
        ax.plot(rec["residuals"], label="Residuals")
    ax.legend()
    return ax

def _plot_2d_reconstruction(rec, *, add_original=True, add_residuals=True, cmap="viridis"):
    n = len(rec["components"])
    extra = int(add_original) + int(add_residuals)
    total = n + extra
    fig, axes = plt.subplots(1, total, figsize=(3 * total, 3))
    idx = 0
    if add_original:
        axes[idx].imshow(rec["series"], cmap=cmap)
        axes[idx].set_title("Original")
        idx += 1
    for i, comp in enumerate(rec["components"]):
        axes[idx].imshow(comp, cmap=cmap)
        axes[idx].set_title(f"F{i+1}")
        idx += 1
    if add_residuals:
        axes[idx].imshow(rec["residuals"], cmap=cmap)
        axes[idx].set_title("Residuals")
    return fig, axes

def plot_nd_reconstruction(recon: Mapping[str, Any], slice: Mapping[str, Any] | None = None, **kwargs):
    """Plot a (possibly sliced) reconstruction dictionary.

    Parameters
    ----------
    recon : mapping
        Reconstruction data with ``components``, ``series`` and ``residuals``.
    slice : mapping, optional
        Passed to :func:`do_slice_reconstruction`.
    """
    rec = do_slice_reconstruction(recon, slice) if slice else recon
    rank = np.asarray(rec["series"]).ndim
    if rank == 1:
        return _plot_1d_reconstruction(rec, **kwargs)
    if rank == 2:
        return _plot_2d_reconstruction(rec, **kwargs)
    raise ValueError("Cannot display array of rank higher than 2")

def plot_ssa_vectors_nd(ssa, slice: Mapping[str, Any] | None = None, *, what="eigen", idx: Sequence[int] = (0,)):
    """Visualize eigenvectors or factor vectors from an ``SSA`` object.

    This is a very small subset of the R ``plotn`` functionality and works for
    one-dimensional :class:`SSA` objects.
    """
    if what not in {"eigen", "factor"}:
        raise ValueError("`what` must be 'eigen' or 'factor'")
    if max(idx) >= len(ssa.s):
        raise ValueError("Too few eigentriples computed for this decomposition")
    if what == "eigen":
        vmatrix = ssa.U[:, idx]
        dimension = (ssa.L,)
    else:
        vmatrix = ssa.Vt[idx].T
        dimension = (ssa.K,)
    res = {
        "components": [],
        "series": np.zeros(dimension),
        "residuals": np.zeros(dimension),
    }
    for i in range(vmatrix.shape[1]):
        vec = np.empty(dimension)
        vec[:] = np.nan
        vec[...] = vmatrix[:, i]
        res["components"].append(vec)
    rec = do_slice_reconstruction(res, slice) if slice else res
    return plot_nd_reconstruction(rec, add_original=False, add_residuals=False)
