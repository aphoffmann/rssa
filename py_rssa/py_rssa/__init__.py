"""Lightweight Python implementation of selected rssa routines."""

from .ssa import SSA, ssa
from .hankel import hankel_mv
from .hmatr import hmatr, plot_hmatr
from . import datasets

__all__ = [
    "SSA",
    "ssa",
    "datasets",
    "hankel_mv",
    "hmatr",
    "plot_hmatr",
]
