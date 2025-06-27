"""Lightweight Python implementation of selected rssa routines."""

from .ssa import SSA, ssa
from .hankel import hankel_mv
from .cadzow import cadzow
from . import datasets
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
