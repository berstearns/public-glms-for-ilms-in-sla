"""Aggregate ERRANT tags into per-text profile vectors."""

from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

import numpy as np

from ilmcloze.errant_profile.tag import ErrantTag

# 40-dim ERRANT tag vocabulary (non-exhaustive; the rest fall in "OTHER").
DEFAULT_TAGS: tuple[str, ...] = (
    "M:DET", "U:DET", "R:DET",
    "M:PREP", "U:PREP", "R:PREP",
    "M:NOUN", "U:NOUN", "R:NOUN",
    "M:VERB", "U:VERB", "R:VERB",
    "R:VERB:FORM", "R:VERB:TENSE", "R:VERB:SVA", "R:VERB:INFL",
    "M:PUNCT", "U:PUNCT", "R:PUNCT",
    "R:ORTH", "R:SPELL",
    "M:ADJ", "U:ADJ", "R:ADJ",
    "M:ADV", "U:ADV", "R:ADV",
    "M:CONJ", "U:CONJ", "R:CONJ",
    "M:PRON", "U:PRON", "R:PRON",
    "R:WO", "R:MORPH",
    "M:OTHER", "U:OTHER", "R:OTHER",
)


def vectorise(tags: Sequence[ErrantTag], vocab: Sequence[str] = DEFAULT_TAGS) -> np.ndarray:
    """Normalised count vector over ``vocab`` (plus a trailing "OTHER" dim)."""
    counts = Counter(t.tag for t in tags)
    v = np.zeros(len(vocab) + 1, dtype=np.float32)
    for i, tag in enumerate(vocab):
        v[i] = counts.get(tag, 0)
    other = sum(c for t, c in counts.items() if t not in set(vocab))
    v[-1] = other
    total = v.sum()
    if total > 0:
        v /= total
    return v


def vectorise_many(
    batched_tags: Iterable[Sequence[ErrantTag]],
    vocab: Sequence[str] = DEFAULT_TAGS,
) -> np.ndarray:
    """Stack per-text profile vectors into an (N, D) matrix."""
    return np.stack([vectorise(t, vocab) for t in batched_tags], axis=0)
