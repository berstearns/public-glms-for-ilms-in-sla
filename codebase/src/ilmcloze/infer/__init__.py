"""Cloze inference.

Per-backbone inference routines. Each returns a :class:`PredictionRow` per
input cloze item containing:

* the model's top-k greedy fillers;
* the log-probability the model assigns to the learner filler;
* the log-probability the model assigns to the native reference filler
  (required for the learner-plausibility metric).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionRow:
    item_id: str
    corpus: str
    top_k: list[list[str]]  # list of tokens per rank
    logp_learner: float
    logp_native: float | None
    # Optional: per-empirical-filler logp, used for KL/JS.
    logp_empirical: list[float] | None = None
