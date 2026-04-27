"""Gap-sampler invariants: bounds, determinism, per-kind contracts."""

from __future__ import annotations

import random

from ilmcloze.cloze.gap_sampler import sample_multi_token, sample_single_token
from ilmcloze.config import ClozeConfig


def test_single_token_bounds(fixed_seed: int) -> None:
    rng = random.Random(fixed_seed)
    tokens = "this is a small test sentence".split()
    gaps = sample_single_token(tokens, n=3, rng=rng)
    assert len(gaps) <= 3
    for g in gaps:
        assert 0 <= g.start < len(tokens)
        assert g.end == g.start + 1
        assert g.tokens == (tokens[g.start],)


def test_multi_token_no_overlap(fixed_seed: int) -> None:
    rng = random.Random(fixed_seed)
    tokens = ["tok"] * 40
    gaps = sample_multi_token(tokens, n=5, rng=rng, lam=3.0, max_len=6)
    # sorted non-overlapping intervals
    sorted_gaps = sorted(gaps, key=lambda g: g.start)
    for a, b in zip(sorted_gaps, sorted_gaps[1:]):
        assert a.end <= b.start


def test_dispatch_multi_token() -> None:
    cfg = ClozeConfig(gap_type="multi_token", num_gaps_per_text=2, span_length_lambda=3.0)
    from ilmcloze.cloze.gap_sampler import sample

    rng = random.Random(0)
    gaps = sample("a " * 50, cfg, rng)
    assert len(gaps) <= 2
