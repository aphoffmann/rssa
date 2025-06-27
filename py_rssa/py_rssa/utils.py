"""Utility helpers loosely ported from ``R/common.R``."""

from __future__ import annotations

from typing import Any, Iterable, List, Optional

import numpy as np


# ---- Storage helpers -------------------------------------------------------

def create_storage(obj: Any) -> Any:
    """Attach storage dictionary to *obj* and return the object."""
    setattr(obj, "_env", {})
    return obj


def storage(obj: Any) -> dict:
    """Return internal storage for *obj* (creating if missing)."""
    if not hasattr(obj, "_env"):
        setattr(obj, "_env", {})
    return getattr(obj, "_env")


def get(obj: Any, name: str, default: Any = None, allow_null: bool = False) -> Any:
    env = storage(obj)
    if name in env:
        return env[name]
    if not allow_null or default is not None:
        return default
    return None


def get_or_create(obj: Any, name: str, default: Any) -> Any:
    env = storage(obj)
    if name not in env or env[name] is None:
        env[name] = default
    return env[name]


def set_value(obj: Any, name: str, value: Any) -> None:
    storage(obj)[name] = value


def exists(obj: Any, name: str) -> bool:
    return name in storage(obj)


def exists_non_null(obj: Any, name: str) -> bool:
    env = storage(obj)
    return name in env and env[name] is not None


def remove(obj: Any, name: str) -> None:
    env = storage(obj)
    if name in env:
        del env[name]


def deprecate(obj: Any, name: str, instead: Optional[str] = None) -> None:
    storage(obj)[name] = {"deprecated": True, "instead": instead}


def clone(obj: Any, copy_storage: bool = True) -> Any:
    """Shallow clone of *obj* optionally copying storage."""
    new_obj = obj.__class__(**obj.__dict__)
    create_storage(new_obj)
    if copy_storage and hasattr(obj, "_env"):
        new_obj._env.update(obj._env)
    return new_obj


# ---- Series helpers -------------------------------------------------------

def na_omit(x: Iterable[float]) -> np.ndarray:
    """Drop leading and trailing NaNs from *x*."""
    arr = np.asarray(list(x), dtype=float)
    mask = ~np.isnan(arr)
    if not mask.any():
        raise ValueError("all values are NaN")
    start = mask.argmax()
    end = len(arr) - mask[::-1].argmax()
    return arr[start:end]


def to_series_list(x: Any, na_rm: bool = True, template: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
    """Convert ``x`` to a list of 1D numpy arrays."""
    if isinstance(x, list):
        lst = x
    else:
        arr = np.asarray(x)
        if arr.ndim == 1:
            arr = arr[:, None]
        lst = [arr[:, i] for i in range(arr.shape[1])]

    if template is not None:
        result = []
        for serie, tmpl in zip(lst, template):
            remove = getattr(tmpl, "na_action", None)
            if remove is not None and len(remove):
                serie = np.delete(serie, remove)
            result.append(serie)
        lst = result
    elif na_rm:
        lst = [na_omit(serie) for serie in lst]

    return lst


def from_series_list(x: List[np.ndarray], pad: str = "none", simplify: bool = True) -> Any:
    """Combine series list back into an array."""
    pad = pad.lower()
    if pad == "none":
        return np.column_stack(x) if simplify else list(x)

    maxlen = max(len(v) for v in x)
    res = []
    for serie in x:
        diff = maxlen - len(serie)
        if pad == "left":
            padded = np.concatenate((np.full(diff, np.nan), serie))
        else:
            padded = np.concatenate((serie, np.full(diff, np.nan)))
        res.append(padded)
    return np.column_stack(res) if simplify else res


# ---- Arithmetic on series lists ------------------------------------------

def apply_series_op(op, e1: List[np.ndarray], e2: Optional[List[np.ndarray]] = None) -> List[np.ndarray]:
    """Apply binary operation ``op`` element-wise."""
    if e2 is not None:
        if len(e1) != len(e2):
            raise ValueError("series list should have equal number of elements")
        result = [op(a, b) for a, b in zip(e1, e2)]
    else:
        result = [op(a) for a in e1]
    return result


__all__ = [
    "create_storage",
    "storage",
    "get",
    "get_or_create",
    "set_value",
    "exists",
    "exists_non_null",
    "remove",
    "deprecate",
    "clone",
    "na_omit",
    "to_series_list",
    "from_series_list",
    "apply_series_op",
]

