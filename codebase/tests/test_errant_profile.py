"""Profile vectorisation shape/normalisation + clusterer round-trip."""

from __future__ import annotations

import numpy as np

from ilmcloze.errant_profile.cluster import ErrprofClusterer
from ilmcloze.errant_profile.profile import DEFAULT_TAGS, vectorise
from ilmcloze.errant_profile.tag import ErrantTag


def test_vectorise_shape_and_normalisation() -> None:
    tags = [
        ErrantTag("R:DET", 0, 1, 0, 1, "a", "the"),
        ErrantTag("R:DET", 3, 4, 3, 4, "a", "the"),
        ErrantTag("R:VERB:SVA", 5, 6, 5, 6, "have", "has"),
    ]
    v = vectorise(tags)
    assert v.shape == (len(DEFAULT_TAGS) + 1,)
    assert abs(v.sum() - 1.0) < 1e-6


def test_clusterer_roundtrip(tmp_path) -> None:
    rng = np.random.default_rng(0)
    X = rng.random((100, len(DEFAULT_TAGS) + 1)).astype(np.float32)
    X /= X.sum(axis=1, keepdims=True)
    cl = ErrprofClusterer(k=4, seed=0).fit(X)
    ids = cl.predict(X)
    assert ids.shape == (100,)
    path = tmp_path / "cl.pkl"
    cl.save(path)
    cl2 = ErrprofClusterer.load(path)
    ids2 = cl2.predict(X)
    assert (ids == ids2).all()
