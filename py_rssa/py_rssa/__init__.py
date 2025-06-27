"""Lightweight Python implementation of selected rssa routines."""

from .ssa import SSA, ssa, wcor, wnorm
from .hankel import hankel_mv
from . import datasets

__all__ = ["SSA", "ssa", "wcor", "wnorm", "datasets", "hankel_mv"]
