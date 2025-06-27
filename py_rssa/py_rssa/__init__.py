"""Lightweight Python implementation of selected rssa routines."""

from .ssa import SSA, ssa
from .hankel import hankel_mv
from .plotn import (
    do_slice_array,
    do_slice_reconstruction,
    plot_nd_reconstruction,
    plot_ssa_vectors_nd,
)
from . import datasets

__all__ = [
    "SSA",
    "ssa",
    "datasets",
    "hankel_mv",
    "do_slice_array",
    "do_slice_reconstruction",
    "plot_nd_reconstruction",
    "plot_ssa_vectors_nd",
]
