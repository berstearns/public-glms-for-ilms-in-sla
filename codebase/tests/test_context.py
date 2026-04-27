"""Context builders and synthetic corruption preserve the gap."""

from __future__ import annotations

import random

from ilmcloze.cloze.context import (
    build_clean_context,
    build_learner_context,
    corrupt_context,
)
from ilmcloze.cloze.gap_sampler import Gap


def test_learner_context_identity() -> None:
    tokens = "I have a dog".split()
    gap = Gap(start=2, end=3, tokens=("a",))
    ctx = build_learner_context(tokens, gap)
    assert ctx.left == ("I", "have")
    assert ctx.right == ("dog",)
    assert ctx.condition == "II"


def test_clean_context_substitutes_surroundings() -> None:
    learner = "I has an dog".split()
    clean = "I have a dog".split()
    gap = Gap(start=2, end=3, tokens=("an",))
    ctx = build_clean_context(learner, clean, gap)
    assert ctx.left == ("I", "have")
    assert ctx.right == ("dog",)
    assert ctx.condition == "I"


def test_corrupt_context_marks_condition() -> None:
    ctx = build_learner_context("the cat sat on the mat".split(),
                                Gap(start=3, end=4, tokens=("on",)))
    rng = random.Random(0)
    out = corrupt_context(ctx, rng, rate=1.0, target_condition="III")
    assert out.condition == "III"
    assert out.gap == ctx.gap
