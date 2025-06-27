import numpy as np
import matplotlib.pyplot as plt


def _prep_recon_list(recon):
    if isinstance(recon, dict):
        names = list(recon.keys())
        arrays = [np.asarray(v) for v in recon.values()]
    else:
        arrays = [np.asarray(r) for r in recon]
        names = [f"Comp {i+1}" for i in range(len(arrays))]
    return names, arrays


def plot_2d_reconstructions(recon,
                            type="raw",
                            base_series=None,
                            add_original=True,
                            add_residuals=True,
                            cmap="gray",
                            vmin=None,
                            vmax=None,
                            fill_uncovered=np.nan):
    """Visualize 2D SSA reconstructions using Matplotlib.

    Parameters
    ----------
    recon : sequence or mapping
        Collection of 2D arrays representing reconstructed components.
    type : {"raw", "cumsum"}
        Show individual components or their cumulative sums.
    base_series : array_like, optional
        Optional base series to prepend before the components.
    add_original : bool, default True
        Whether to plot the original series as the first subplot when
        available through ``base_series``.
    add_residuals : bool, default True
        Whether to add residual plot as the last subplot when
        ``base_series`` is provided.
    cmap : str or matplotlib colormap, default "gray"
        Colormap used for imshow.
    vmin, vmax : float, optional
        Value range for the color scale.
    fill_uncovered : float, default NaN
        Fill value for missing data (``NaN``) in the reconstructions.

    Returns
    -------
    matplotlib.figure.Figure
        Figure object with subplots.
    """
    names, arrays = _prep_recon_list(recon)

    if type not in {"raw", "cumsum"}:
        raise ValueError("type must be 'raw' or 'cumsum'")

    arrays = [a.copy() for a in arrays]
    if type == "cumsum" and len(arrays) > 1:
        for i in range(1, len(arrays)):
            arrays[i] += arrays[i - 1]
            names[i] = f"{names[0]}:{names[i]}"

    if base_series is not None:
        base = np.asarray(base_series)
    else:
        base = None

    if add_original and base is not None:
        arrays = [base] + arrays
        names = ["Original"] + names
    if add_residuals and base is not None:
        residual = base - np.nansum(arrays[1:], axis=0)
        arrays = arrays + [residual]
        names = names + ["Residuals"]

    n = len(arrays)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, arr, title in zip(axes, arrays, names):
        arr = np.asarray(arr, dtype=float)
        if fill_uncovered is not np.nan:
            arr = np.where(np.isnan(arr), fill_uncovered, arr)
        im = ax.imshow(arr, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes[:n], shrink=0.6)
    return fig


def plot_2d_vectors(vectors,
                    shape,
                    cmap="gray",
                    vmin=None,
                    vmax=None,
                    titles=None,
                    symmetric=False):
    """Plot 2D eigenvectors or factor vectors using Matplotlib.

    Parameters
    ----------
    vectors : sequence of array_like
        Vectors to display. Each element is flattened and reshaped using
        ``shape``.
    shape : tuple of int
        Desired 2D shape ``(rows, cols)`` for the vectors.
    cmap : str or matplotlib colormap, default "gray"
        Colormap used for imshow.
    vmin, vmax : float, optional
        Value range for the color scale.
    titles : sequence of str, optional
        Titles for subplots.
    symmetric : bool, default False
        Force symmetric color limits around zero when True.

    Returns
    -------
    matplotlib.figure.Figure
        Figure with vector visualisations.
    """
    vectors = [np.asarray(v).reshape(shape) for v in vectors]
    if titles is None:
        titles = [f"Comp {i+1}" for i in range(len(vectors))]

    if symmetric:
        abs_max = max(np.nanmax(np.abs(v)) for v in vectors)
        vmin, vmax = -abs_max, abs_max

    n = len(vectors)
    cols = int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, vec, title in zip(axes, vectors, titles):
        im = ax.imshow(vec, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes[n:]:
        ax.axis("off")
    fig.colorbar(im, ax=axes[:n], shrink=0.6)
    return fig

__all__ = [
    "plot_2d_reconstructions",
    "plot_2d_vectors",
]
