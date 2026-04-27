"""Content-addressed caching for expensive pure-function computations."""

from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path
from typing import Any, Callable, TypeVar

T = TypeVar("T")


def content_key(*parts: Any) -> str:
    """Stable hash of JSON-serialisable inputs."""
    blob = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


def memoize_to_disk(
    root: Path,
    key: str,
    fn: Callable[[], T],
) -> T:
    """Run ``fn`` once; cache its pickled return value under ``root / f"{key}.pkl"``."""
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{key}.pkl"
    if path.exists():
        with path.open("rb") as fh:
            return pickle.load(fh)
    value = fn()
    tmp = path.with_suffix(".pkl.tmp")
    with tmp.open("wb") as fh:
        pickle.dump(value, fh)
    tmp.rename(path)
    return value
