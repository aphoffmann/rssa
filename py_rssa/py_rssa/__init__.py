"""Lightweight Python implementation of selected rssa routines."""


from .ssa import (
    SSA,
    ssa,
    rforecast,
    vforecast,
    bforecast,
    forecast,
    hankel_weights,
)

from .eossa import eossa
from .hankel import hankel_mv
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

]
