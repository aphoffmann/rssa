"""Lightweight Python implementation of selected rssa routines."""

from .ssa import SSA, ssa
from .hankel import hankel_mv, HAVE_FAST
from . import datasets

__all__ = ["SSA", "ssa", "datasets", "hankel_mv", "HAVE_FAST"]
