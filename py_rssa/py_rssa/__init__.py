"""Lightweight Python implementation of selected rssa routines."""

from .ssa import SSA, PartialSSA, ssa, pssa, orthopoly
from .hankel import hankel_mv
from . import datasets

__all__ = [
    "SSA",
    "PartialSSA",
    "ssa",
    "pssa",
    "orthopoly",
    "datasets",
    "hankel_mv",
]
