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
