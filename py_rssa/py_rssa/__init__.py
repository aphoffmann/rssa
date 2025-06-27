"""Lightweight Python implementation of selected rssa routines."""

from .ssa import SSA, ssa
from .hankel import hankel_mv
from .plot2 import plot_2d_reconstructions, plot_2d_vectors
from . import datasets

__all__ = [
    "SSA",
    "ssa",
    "datasets",
    "hankel_mv",
    "plot_2d_reconstructions",
    "plot_2d_vectors",
]
