"""Lightweight Python implementation of selected rssa routines."""

from .ssa import SSA, ssa
from .hankel import hankel_mv
from .plot import (
    plot,
    plot_values,
    plot_vectors,
    plot_series,
    plot_wcor,
)
from . import datasets

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
]
