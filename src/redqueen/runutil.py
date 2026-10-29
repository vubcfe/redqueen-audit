"""Helpers shared by the EuroSAT and qml-benchmarks drivers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

SEARCH_METHODS = ("random", "ga", "redqueen")


def search_setup(method: str, lam: float) -> tuple[float, str]:
    """(lambda, mode) for a search method. ``ga`` drops the surrogate term;
    ``random`` keeps RED-QUEEN's fitness and changes only the variation
    operator, so redqueen-vs-ga isolates the surrogate term and
    redqueen-vs-random isolates the search operator."""
    if method not in SEARCH_METHODS:
        raise ValueError(f"unknown search method {method!r}; choose from {SEARCH_METHODS}")
    return (0.0 if method == "ga" else lam), ("random" if method == "random" else "evolve")


def check_methods(methods, allowed) -> None:
    """Fail before any job runs, not hours into a shard."""
    bad = [m for m in methods if m not in allowed]
    if bad:
        raise ValueError(f"unknown methods {bad}; choose from {sorted(allowed)}")


def _default(o):
    if isinstance(o, (np.integer, np.floating, np.bool_)):
        return o.item()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def write_json(record: dict, path: Path) -> None:
    """Atomic write: ``--skip-existing`` never sees a half-written record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(record, indent=1, default=_default))
    tmp.replace(path)
