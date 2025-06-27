"""Lightweight Python implementation of selected rssa routines."""

from .ssa import SSA, ssa
from .hankel import hankel_mv
from .toeplitz import (
    Lcor,
    ToeplitzMatrix,
    new_tmat,
    tcols,
    trows,
    is_tmat,
    tmatmul,
)
from . import datasets

__all__ = [
    "SSA",
    "ssa",
    "datasets",
    "hankel_mv",
    "Lcor",
    "ToeplitzMatrix",
    "new_tmat",
    "tcols",
    "trows",
    "is_tmat",
    "tmatmul",
]
