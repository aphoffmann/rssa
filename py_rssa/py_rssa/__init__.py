"""Lightweight Python implementation of selected rssa routines."""

from .ssa import SSA, ssa
from .hankel import hankel_mv
from . import datasets
from .capabilities import (
    register_capability,
    capable,
    object_capabilities,
    fftw_available,
)

__all__ = [
    "SSA",
    "ssa",
    "datasets",
    "hankel_mv",
    "register_capability",
    "capable",
    "object_capabilities",
    "fftw_available",
]
