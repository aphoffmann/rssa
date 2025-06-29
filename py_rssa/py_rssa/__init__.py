"""Lightweight Python implementation of selected rssa routines."""

from .ssa import (
    SSA,
    PartialSSA,
    WOSSA,
    cssa,
    ssa,
    pssa,
    wossa,
    orthopoly,
    hankel_weights,
    rforecast,
    vforecast,
    bforecast,
    forecast,
    igapfill,
    wcor,
    wnorm,
)

from .mssa import MSSA, mssa, reconstruct, gapfill as mssa_gapfill, forecast as mssa_forecast
from .ndssa import NDSSA, ndssa
from .gapfill import gapfill
from .ossa import cond, pseudo_inverse, orthogonalize, svd_to_lrsvd, owcor
from .eossa import eossa
from .hankel import hankel_mv
from .toeplitz import (
    Lcor,
    ToeplitzMatrix,
    new_tmat,
    tcols,
    trows,
    is_tmat,
    tmatmul,
)
from .plotn import (
    do_slice_array,
    do_slice_reconstruction,
    plot_nd_reconstruction,
    plot_ssa_vectors_nd,
)
from .plot2 import plot_2d_reconstructions, plot_2d_vectors
from .plot import (
    plot,
    plot_values,
    plot_vectors,
    plot_series,
    plot_wcor,
)
from .hbhankel import (
    convolution_dims,
    convolven,
    factor_mask_2d,
    field_weights_2d,
    ball_mask,
    simplex_mask,
)
from .hmatr import hmatr, plot_hmatr
from .chankel import chankel_mv, chankelize, chankelize_multi
from .cadzow import cadzow
from .capabilities import (
    register_capability,
    capable,
    object_capabilities,
    fftw_available,
)
from .autossa import (
    grouping_auto,
    grouping_auto_wcor,
    grouping_auto_pgram,
    pgram,
    plot_grouping_auto_wcor,
    plot_grouping_auto_pgram,
)
from . import datasets
from . import utils

__all__ = [
    "SSA",
    "PartialSSA",
    "WOSSA",
    "NDSSA",
    "cssa",
    "MSSA",
    "ssa",
    "ndssa",
    "pssa",
    "wossa",
    "orthopoly",
    "cond",
    "pseudo_inverse",
    "orthogonalize",
    "svd_to_lrsvd",
    "owcor",
    "hankel_mv",
    "hankel_weights",
    "chankel_mv",
    "chankelize",
    "chankelize_multi",
    "Lcor",
    "ToeplitzMatrix",
    "new_tmat",
    "tcols",
    "trows",
    "is_tmat",
    "tmatmul",
    "reconstruct",
    "gapfill",
    "mssa_gapfill",
    "wcor",
    "wnorm",
    "forecast",
    "mssa_forecast",
    "rforecast",
    "vforecast",
    "bforecast",
    "igapfill",
    "do_slice_array",
    "do_slice_reconstruction",
    "plot_nd_reconstruction",
    "plot_ssa_vectors_nd",
    "plot_2d_reconstructions",
    "plot_2d_vectors",
    "plot",
    "plot_values",
    "plot_vectors",
    "plot_series",
    "plot_wcor",
    "convolution_dims",
    "convolven",
    "factor_mask_2d",
    "field_weights_2d",
    "ball_mask",
    "simplex_mask",
    "hmatr",
    "plot_hmatr",
    "cadzow",
    "eossa",
    "datasets",
    "utils",
    "register_capability",
    "capable",
    "object_capabilities",
    "fftw_available",
    "grouping_auto",
    "grouping_auto_wcor",
    "grouping_auto_pgram",
    "pgram",
    "plot_grouping_auto_wcor",
    "plot_grouping_auto_pgram",
]
