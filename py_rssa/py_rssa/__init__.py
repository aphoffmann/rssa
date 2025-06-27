"""Lightweight Python implementation of selected rssa routines."""

from __future__ import annotations

import warnings

try:  # pragma: no cover - optional dependency
    import pyfftw  # type: ignore
    HAVE_FFTW = True
except Exception:  # pragma: no cover - pyfftw missing
    HAVE_FFTW = False

if not HAVE_FFTW:  # pragma: no cover - depends on environment
    warnings.warn(
        "\nWARNING: py_rssa was compiled without FFTW support.\n"
        "The speed of the routines will be slower as well.",
        RuntimeWarning,
    )

from .ssa import SSA, ssa
from .hankel import hankel_mv
from . import datasets

__all__ = ["SSA", "ssa", "datasets", "hankel_mv", "HAVE_FFTW"]
