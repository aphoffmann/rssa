"""Lightweight Python implementation of selected rssa routines."""

from .ssa import SSA, ssa, WOSSA, wossa
from .hankel import hankel_mv
from . import datasets

__all__ = ["SSA", "ssa", "WOSSA", "wossa", "datasets", "hankel_mv"]
