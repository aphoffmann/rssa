"""Lightweight Python implementation of selected rssa routines."""

from .parestimate import (
    roots2pars,
    parestimate_pairs,
    parestimate_esprit,
    parestimate,

from .ossa import cond, pseudo_inverse, orthogonalize, svd_to_lrsvd, owcor
from .mssa import MSSA, mssa, reconstruct, gapfill, forecast
from .ssa import (
    SSA,
    ssa,
    rforecast,
    vforecast,
    bforecast,
    forecast,
    hankel_weights,
    igapfill
)

from .eossa import eossa
from .hankel import hankel_mv

from .plot import (
    plot,
    plot_values,
    plot_vectors,
    plot_series,
    plot_wcor,

from .hbhankel import (
    convolution_dims,
    convolven,
    factor_mask_2d,
    field_weights_2d,
    ball_mask,
    simplex_mask,
)


from .hmatr import hmatr, plot_hmatr
from .gapfill import gapfill
from .chankel import chankel_mv, chankelize, chankelize_multi
from .cadzow import cadzow
from . import datasets
from . import utils

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


__all__ = [
    "SSA",
    "ssa",
    "datasets",
    "hankel_mv",
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
    "eossa",
    "datasets",
    "hankel_mv",
    "gapfill",
    "utils",
    "rforecast",
    "vforecast",
    "bforecast",
    "forecast",
    "chankelize", "chankelize_multi", "hankel_weights"
    "cadzow",
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
    "MSSA", 
    "mssa","cond", "pseudo_inverse", "orthogonalize", "svd_to_lrsvd", "owcor",
]
