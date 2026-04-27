"""Unit tests for the new EM variants."""

from __future__ import annotations

from ilmcloze.eval.metrics import (
    edit_distance,
    edit_distance_leq,
    exact_match,
    exact_match_case_insensitive,
    exact_match_normalized,
)


def test_strict_em_is_case_and_punct_sensitive() -> None:
    assert exact_match(["Hi"], ["hi"]) == 0
    assert exact_match(["party,"], ["party"]) == 0
    assert exact_match(["hi"], ["hi"]) == 1


def test_case_insensitive_em_ignores_case_only() -> None:
    assert exact_match_case_insensitive(["Hi"], ["hi"]) == 1
    assert exact_match_case_insensitive(["Party,"], ["party,"]) == 1
    # punctuation still matters
    assert exact_match_case_insensitive(["party,"], ["party"]) == 0


def test_normalized_em_ignores_case_and_punct() -> None:
    assert exact_match_normalized(["Hi!"], ["hi"]) == 1
    assert exact_match_normalized(["party,"], ["PARTY"]) == 1
    assert exact_match_normalized(["This", "is", "!"], ["this", "is"]) == 1
    # unicode normalisation: full-width comma matches nothing
    assert exact_match_normalized(["ｈｉ"], ["hi"]) == 1


def test_edit_distance_basic() -> None:
    assert edit_distance(["hi"], ["hi"]) == 0
    assert edit_distance(["hi"], ["hii"]) == 1
    assert edit_distance(["hello"], ["hallo"]) == 1
    assert edit_distance(["kitten"], ["sitting"]) == 3


def test_edit_distance_threshold() -> None:
    assert edit_distance_leq(["cat"], ["cats"], threshold=1) == 1
    assert edit_distance_leq(["cat"], ["mice"], threshold=1) == 0
    assert edit_distance_leq(["cat"], ["mice"], threshold=4) == 1


def test_edit_distance_monotone_per_threshold() -> None:
    # threshold=1 implies threshold=2 implies threshold=3
    for p, g in [(["abc"], ["abd"]), (["hello"], ["helo"]), (["a b c"], ["a b"])]:
        d = edit_distance(p, g)
        for t in range(0, d + 5):
            assert edit_distance_leq(p, g, t) == int(d <= t)
