"""k-means clusterer for ERRANT profile vectors → ERRPROF bucket id."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ErrprofClusterer:
    """Thin wrapper around :class:`sklearn.cluster.KMeans`.

    Fits on an (N, D) matrix of normalised profile vectors. Predict returns
    an integer bucket id in ``[0, k)``. Persisted via pickle.
    """

    k: int = 16
    seed: int = 42
    _model: Any = None

    def fit(self, profiles: np.ndarray) -> "ErrprofClusterer":
        from sklearn.cluster import KMeans

        self._model = KMeans(
            n_clusters=self.k,
            n_init=10,
            random_state=self.seed,
            algorithm="lloyd",
        ).fit(profiles)
        return self

    def predict(self, profiles: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Clusterer is not fitted")
        return np.asarray(self._model.predict(profiles), dtype=np.int64)

    def save(self, path: str | Path) -> None:
        import pickle

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path: str | Path) -> "ErrprofClusterer":
        import pickle

        with Path(path).open("rb") as fh:
            obj = pickle.load(fh)
        if not isinstance(obj, cls):
            raise TypeError(f"Unpickled object is not {cls.__name__}: {type(obj)}")
        return obj
