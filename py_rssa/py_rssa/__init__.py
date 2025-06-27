"""Lightweight Python implementation of selected rssa routines."""

from .ssa import SSA, ssa
from .ossa import cond, pseudo_inverse, orthogonalize, svd_to_lrsvd, owcor
from .hankel import hankel_mv
from . import datasets

__all__ = ["SSA", "ssa", "datasets", "hankel_mv", "cond", "pseudo_inverse", "orthogonalize", "svd_to_lrsvd", "owcor"]
