"""Lightweight Python implementation of selected rssa routines."""

from .ssa import SSA, ssa
from .hankel import hankel_mv
from . import datasets
from . import utils

__all__ = [
    "SSA",
    "ssa",
    "datasets",
    "hankel_mv",
    "utils",
]
