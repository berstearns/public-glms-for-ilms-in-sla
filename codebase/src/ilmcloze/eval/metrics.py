"""Scoring primitives: EM and its variants, top-k, LP, KL, JS.

EM variants
-----------
- :func:`exact_match`                     — strict string equality, list-wise.
- :func:`exact_match_case_insensitive`    — lower both sides before comparing.
- :func:`exact_match_normalized`          — lower + strip punctuation + collapse whitespace.
- :func:`edit_distance`                   — Levenshtein on the whitespace-joined strings.
- :func:`edit_distance_leq`               — 1 if ``edit_distance <= threshold``.

Each variant takes ``prediction, gold`` as ``Sequence[str]`` (word lists).
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Sequence

import numpy as np


def exact_match(prediction: Sequence[str], gold: Sequence[str]) -> int:
    """Strict string equality of the whitespace-token sequences."""
    return int(list(prediction) == list(gold))


def _lower(seq: Sequence[str]) -> list[str]:
    return [s.lower() for s in seq]


def exact_match_case_insensitive(prediction: Sequence[str], gold: Sequence[str]) -> int:
    """Lowercase both sides, then list-wise equality."""
    return int(_lower(prediction) == _lower(gold))


_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def _normalize(seq: Sequence[str]) -> list[str]:
    """Lower + NFKC unicode fold + drop punctuation + collapse whitespace.

    Applied token-by-token; empty tokens after stripping are dropped.
    """
    out: list[str] = []
    for tok in seq:
        s = unicodedata.normalize("NFKC", tok).lower()
        s = _PUNCT_RE.sub("", s)
        s = _WS_RE.sub(" ", s).strip()
        if s:
            out.append(s)
    return out


def exact_match_normalized(prediction: Sequence[str], gold: Sequence[str]) -> int:
    """Case- and punctuation-insensitive string equality after NFKC unicode fold."""
    return int(_normalize(prediction) == _normalize(gold))


def edit_distance(prediction: Sequence[str], gold: Sequence[str]) -> int:
    """Levenshtein distance on the whitespace-joined strings.

    We join with single spaces so that ``['party', ',']`` and ``['party,']``
    still differ only in the space/comma ordering rather than being
    artificially closer/farther.
    """
    a = " ".join(prediction)
    b = " ".join(gold)
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)
    # Classic DP, O(|a|·|b|) time, O(min(|a|,|b|)) space.
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[-1]


def edit_distance_leq(prediction: Sequence[str], gold: Sequence[str], threshold: int) -> int:
    """1 if the Levenshtein distance between the joined strings is ``<= threshold``."""
    return int(edit_distance(prediction, gold) <= threshold)


def token_f1(prediction: Sequence[str], gold: Sequence[str]) -> float:
    """SQuAD-style token-level F1 on whitespace tokens (strict / case-sensitive).

    Tokens are multiset-intersected; ``precision = common / |pred|``,
    ``recall = common / |gold|``, ``F1 = 2·P·R/(P+R)``. Both empty → 1.0,
    exactly one empty → 0.0.
    """
    pred_list, gold_list = list(prediction), list(gold)
    if not pred_list and not gold_list:
        return 1.0
    if not pred_list or not gold_list:
        return 0.0
    pred_counts = Counter(pred_list)
    gold_counts = Counter(gold_list)
    common = sum((pred_counts & gold_counts).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_list)
    recall    = common / len(gold_list)
    return 2 * precision * recall / (precision + recall)


def token_f1_normalized(prediction: Sequence[str], gold: Sequence[str]) -> float:
    """Token F1 after the same normalisation used by :func:`exact_match_normalized`."""
    return token_f1(_normalize(prediction), _normalize(gold))


def top_k_hit(predictions: Sequence[Sequence[str]], gold: Sequence[str], k: int) -> int:
    """1 if gold appears among the top-k predictions, else 0."""
    gold_list = list(gold)
    return int(any(list(p) == gold_list for p in predictions[:k]))


def learner_plausibility(logp_learner: float, logp_native: float | None) -> float:
    """LP = log p(learner filler) − log p(native filler).

    Positive values mean the model prefers the learner filler.
    """
    if logp_native is None:
        return float("nan")
    return float(logp_learner - logp_native)


def _to_probabilities(log_probs: Sequence[float], eps: float = 1e-9) -> np.ndarray:
    x = np.asarray(log_probs, dtype=np.float64)
    x = x - x.max()
    p = np.exp(x)
    p = p / (p.sum() + eps)
    return p


def _empirical_distribution(fillers: Sequence[Sequence[str]]) -> tuple[list[Sequence[str]], np.ndarray]:
    """Return unique fillers and their empirical probabilities."""
    key = lambda f: tuple(f)  # noqa: E731
    counts = Counter(key(f) for f in fillers)
    keys = list(counts)
    total = sum(counts.values())
    probs = np.array([counts[k] / total for k in keys], dtype=np.float64)
    return [list(k) for k in keys], probs


def kl_to_empirical(
    model_log_probs_per_filler: Sequence[float],
    fillers: Sequence[Sequence[str]],
) -> float:
    """KL(p_empirical || p_model) where p_model is normalised over the
    candidate fillers observed in the empirical distribution.
    """
    if not fillers:
        return float("nan")
    _, emp_probs = _empirical_distribution(fillers)
    model_probs = _to_probabilities(model_log_probs_per_filler)
    # Align lengths
    if len(model_probs) != len(emp_probs):
        return float("nan")
    eps = 1e-12
    return float(np.sum(emp_probs * (np.log(emp_probs + eps) - np.log(model_probs + eps))))


def js_to_empirical(
    model_log_probs_per_filler: Sequence[float],
    fillers: Sequence[Sequence[str]],
) -> float:
    if not fillers:
        return float("nan")
    _, emp_probs = _empirical_distribution(fillers)
    model_probs = _to_probabilities(model_log_probs_per_filler)
    if len(model_probs) != len(emp_probs):
        return float("nan")
    m = 0.5 * (emp_probs + model_probs)
    eps = 1e-12

    def _kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * (np.log(a + eps) - np.log(b + eps))))

    return 0.5 * _kl(emp_probs, m) + 0.5 * _kl(model_probs, m)
