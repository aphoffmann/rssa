"""Capability checking utilities similar to R `capabilities.R`."""

from __future__ import annotations

from typing import Callable, Dict, Any

_capabilities: Dict[str, Dict[str, Any]] = {}


def register_capability(name: str, func_name: str,
                        pred: Callable[[Any], bool] | None = None,
                        alias: str | None = None) -> None:
    """Register a capability for later checks.

    Parameters
    ----------
    name : str
        Human readable capability name.
    func_name : str
        Name of the method providing the capability.
    pred : callable, optional
        Additional predicate that should return ``True`` when the
        capability is available for the object. By default it always
        returns ``True``.
    alias : str, optional
        Alternative key under which the capability will be stored.
    """
    if alias is None:
        alias = func_name
    if pred is None:
        pred = lambda _obj: True
    _capabilities[alias] = {"name": name, "func": func_name, "pred": pred}


def capable(obj: Any, capname: str) -> bool:
    """Check whether ``obj`` supports the given capability."""
    cap = _capabilities.get(capname)
    if cap is None:
        raise KeyError(f"Unknown capability: {capname}")
    if not cap["pred"](obj):
        return False
    method = getattr(obj, cap["func"], None)
    return callable(method)


def object_capabilities(obj: Any) -> Dict[str, bool]:
    """Return available capabilities for ``obj`` as a dict."""
    res: Dict[str, bool] = {}
    for key, cap in _capabilities.items():
        res[cap["name"]] = capable(obj, key)
    return res


# FFTW availability ----------------------------------------------------------
try:  # pragma: no cover - depends on external library
    import pyfftw  # type: ignore

    _HAVE_FFTW = True
except Exception:  # pragma: no cover - pyfftw not installed
    _HAVE_FFTW = False


def fftw_available() -> bool:
    """Return ``True`` if ``pyfftw`` is importable."""
    return _HAVE_FFTW


# Register some default SSA capabilities ------------------------------------
from .ssa import SSA  # noqa: E402

register_capability("Decomposition", "decompose")
register_capability("Reconstruction", "reconstruct")
register_capability("Weighted correlations", "wcor", alias="wcor")

