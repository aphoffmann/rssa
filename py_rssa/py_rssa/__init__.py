"""Lightweight Python implementation of selected rssa routines."""

from .ssa import SSA, ssa
from .eossa import eossa
from .hankel import hankel_mv
from . import datasets

__all__ = ["SSA", "ssa", "eossa", "datasets", "hankel_mv"]
