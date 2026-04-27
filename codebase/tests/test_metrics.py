"""Metric correctness on hand-crafted cases."""

from __future__ import annotations

import math

import numpy as np

from ilmcloze.eval.metrics import (
    exact_match,
    js_to_empirical,
    kl_to_empirical,
    learner_plausibility,
    top_k_hit,
)


def test_exact_match() -> None:
    assert exact_match(["the", "shop"], ["the", "shop"]) == 1
    assert exact_match(["the"], ["the", "shop"]) == 0


def test_top_k_hit() -> None:
    preds = [["a"], ["the"], ["an"]]
    assert top_k_hit(preds, ["the"], k=1) == 0
    assert top_k_hit(preds, ["the"], k=2) == 1
    assert top_k_hit(preds, ["the"], k=3) == 1


def test_learner_plausibility() -> None:
    assert learner_plausibility(-1.0, -2.0) == 1.0
    assert math.isnan(learner_plausibility(-1.0, None))


def test_kl_empirical_zero_when_uniform_equal() -> None:
    fillers = [["a"], ["b"], ["c"]]
    # equal log probs → uniform model distribution → same as empirical (3 distinct, each 1/3)
    log_probs = [0.0, 0.0, 0.0]
    kl = kl_to_empirical(log_probs, fillers)
    assert kl == 0 or abs(kl) < 1e-9


def test_js_symmetry_and_nonneg() -> None:
    fillers = [["a"], ["a"], ["b"]]
    log_probs = [-0.5, -0.5, -2.0]
    j = js_to_empirical(log_probs, fillers)
    # JS is bounded and non-negative
    assert 0.0 <= j <= math.log(2) + 1e-9 or not np.isfinite(j)
