"""Lightweight Python implementation of selected rssa routines."""

from .ssa import SSA, ssa
from .hankel import hankel_mv
from .hbhankel import (
    convolution_dims,
    convolven,
    factor_mask_2d,
    field_weights_2d,
    ball_mask,
    simplex_mask,
)
from . import datasets

__all__ = [
    "SSA",
    "ssa",
    "datasets",
    "hankel_mv",
    "convolution_dims",
    "convolven",
    "factor_mask_2d",
    "field_weights_2d",
    "ball_mask",
    "simplex_mask",
]
