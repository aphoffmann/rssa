import numpy as np
from numpy.fft import fftn, ifftn

__all__ = [
    "convolution_dims",
    "convolven",
    "factor_mask_2d",
    "field_weights_2d",
    "ball_mask",
    "simplex_mask",
]


def convolution_dims(x_dim, y_dim, type="circular"):
    """Compute input and output dimensions for N-D convolution."""
    x_dim = tuple(int(d) for d in x_dim)
    y_dim = tuple(int(d) for d in y_dim)
    if len(x_dim) != len(y_dim):
        raise ValueError("Dimension mismatch")
    if isinstance(type, str):
        type = [type] * len(x_dim)
    if len(type) != len(x_dim):
        if len(x_dim) % len(type) != 0:
            import warnings
            warnings.warn(
                "longer object length is not a multiple of shorter object length"
            )
        type = [type[i % len(type)] for i in range(len(x_dim))]
    input_dim = []
    output_dim = []
    for xd, yd, t in zip(x_dim, y_dim, type):
        if t == "circular":
            output_dim.append(xd)
            input_dim.append(xd)
        elif t == "open":
            output_dim.append(xd + yd - 1)
            input_dim.append(xd + yd - 1)
        elif t == "filter":
            if xd < yd:
                raise ValueError("x dimension should be >= y dimension for filter")
            output_dim.append(xd - yd + 1)
            input_dim.append(xd)
        else:
            raise ValueError(f"Unknown type: {t}")
    return {"input_dim": tuple(input_dim), "output_dim": tuple(output_dim)}


def convolven(x, y, conj=True, type="circular"):
    """N-dimensional convolution using FFT."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.ndim == 0:
        x = x.reshape(1)
    if y.ndim == 0:
        y = y.reshape(1)
    dims = convolution_dims(x.shape, y.shape, type)
    input_dim = dims["input_dim"]
    output_dim = dims["output_dim"]
    X = fftn(x, s=input_dim)
    Y = fftn(y, s=input_dim)
    if conj:
        Y = np.conj(Y)
    res = ifftn(X * Y).real
    slices = tuple(slice(0, n) for n in output_dim)
    res = res[slices]
    if res.ndim == 1:
        res = res.reshape(-1)
    return res


def factor_mask_2d(field_mask, window_mask, circular=False):
    field_mask = np.asarray(field_mask, dtype=float)
    window_mask = np.asarray(window_mask, dtype=float)
    conv = convolven(
        field_mask,
        window_mask,
        conj=True,
        type="circular" if circular else "filter",
    )
    return np.abs(conv - np.sum(window_mask)) < 0.5


def field_weights_2d(window_mask, factor_mask, circular=False):
    window_mask = np.asarray(window_mask, dtype=float)
    factor_mask = np.asarray(factor_mask, dtype=float)
    conv = convolven(
        factor_mask,
        window_mask,
        conj=False,
        type="circular" if circular else "open",
    )
    return np.round(conv).astype(int)


def ball_mask(R, rank):
    coords = [np.arange(2 * R - 1) for _ in range(rank)]
    grids = np.meshgrid(*coords, indexing="ij")
    dist2 = sum((g - (R - 1)) ** 2 for g in grids)
    return dist2 <= (R - 1) ** 2


def simplex_mask(side, rank):
    coords = [np.arange(1, side + 1) for _ in range(rank)]
    grids = np.meshgrid(*coords, indexing="ij")
    dist = sum(grids)
    return dist <= side + rank - 1
