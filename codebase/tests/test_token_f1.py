"""Token-F1 unit tests."""

from __future__ import annotations

import math

from ilmcloze.eval.metrics import token_f1, token_f1_normalized


def _close(a: float, b: float, tol: float = 1e-9) -> bool:
    return math.isclose(a, b, rel_tol=tol, abs_tol=tol)


def test_identical_is_one() -> None:
    assert token_f1(["a", "b", "c"], ["a", "b", "c"]) == 1.0


def test_both_empty_is_one() -> None:
    assert token_f1([], []) == 1.0


def test_one_empty_is_zero() -> None:
    assert token_f1([], ["a"]) == 0.0
    assert token_f1(["a"], []) == 0.0


def test_disjoint_is_zero() -> None:
    assert token_f1(["a", "b"], ["c", "d"]) == 0.0


def test_partial_overlap_harmonic_mean() -> None:
    # pred = [a,b], gold = [a,c]  → common=1, p=1/2, r=1/2, f1=0.5
    assert _close(token_f1(["a", "b"], ["a", "c"]), 0.5)


def test_multiset_counts_not_set() -> None:
    # pred = [a,a], gold = [a] → common=1, p=1/2, r=1, f1=2/3
    assert _close(token_f1(["a", "a"], ["a"]), 2 / 3)


def test_normalized_ignores_case_and_punct() -> None:
    assert token_f1_normalized(["Hi!"], ["hi"]) == 1.0
    assert token_f1_normalized(["Hello", "WORLD"], ["hello", "world"]) == 1.0
    # Case-only mismatch → full credit under normalisation.
    assert token_f1_normalized(["PARTY,"], ["party"]) == 1.0
    # Partial overlap under normalisation: pred=[party, was], gold=[party, loud]
    # common=1, p=1/2, r=1/2 → F1=0.5
    assert _close(token_f1_normalized(["Party,", "was"], ["party", "loud"]), 0.5)
