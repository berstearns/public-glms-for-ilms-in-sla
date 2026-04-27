"""Sample gap positions and lengths within a tokenised text.

Implements the gap-type axis described in ``experiments-ideas/README.md``:

- ``single_token`` — length-1 gaps at uniform random positions.
- ``multi_token`` — length ~ Poisson(λ), clipped to ``[1, span_max_length]``.
- ``function`` / ``content`` — restrict gap positions to function-word or
  content-word tokens (POS-tag-based).
- ``l2_loci`` — gaps at tokens flagged as typical L2 error loci (determiners,
  prepositions, verb morphology, subject--verb agreement). Detection is
  POS/dependency-based via spaCy and matches the ERRANT locus set.

All samplers return a list of :class:`Gap` objects, one per sampled gap.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np

from ilmcloze.config import ClozeConfig


@dataclass(frozen=True)
class Gap:
    """A single gap sampled from a text."""

    start: int  # token index, inclusive
    end: int    # token index, exclusive
    tokens: tuple[str, ...]  # the (gold-learner) filler tokens
    locus: str = "generic"  # DET | PREP | VERB:FORM | VERB:SVA | function | content | generic


# ---------------------------------------------------------------------------
# Helpers


def _tokenise(text: str) -> list[str]:
    """Whitespace tokenisation. Model-specific subword tokenisation happens
    later in :mod:`ilmcloze.cloze.format`; gap sampling works on words."""
    return text.split()


def _sample_length(rng: random.Random, lam: float, lo: int, hi: int) -> int:
    k = max(lo, min(hi, int(np.random.default_rng(rng.randint(0, 2**31 - 1)).poisson(lam=lam))))
    return k


# ---------------------------------------------------------------------------
# Core samplers


def sample_single_token(
    tokens: Sequence[str], n: int, rng: random.Random, exclude_initial: bool = False
) -> list[Gap]:
    start_lo = 1 if exclude_initial else 0
    if len(tokens) <= start_lo:
        return []
    picks = rng.sample(range(start_lo, len(tokens)), k=min(n, len(tokens) - start_lo))
    return [Gap(start=i, end=i + 1, tokens=(tokens[i],), locus="generic") for i in picks]


def sample_multi_token(
    tokens: Sequence[str],
    n: int,
    rng: random.Random,
    lam: float,
    max_len: int,
    exclude_initial: bool = False,
) -> list[Gap]:
    gaps: list[Gap] = []
    if len(tokens) == 0:
        return []
    tries = 0
    used: list[tuple[int, int]] = []
    while len(gaps) < n and tries < 10 * n:
        tries += 1
        length = _sample_length(rng, lam=lam, lo=1, hi=max_len)
        start_lo = 1 if exclude_initial else 0
        if len(tokens) - length <= start_lo:
            continue
        start = rng.randint(start_lo, len(tokens) - length)
        end = start + length
        if any(not (end <= a or start >= b) for (a, b) in used):
            continue
        used.append((start, end))
        gaps.append(Gap(start=start, end=end, tokens=tuple(tokens[start:end])))
    return gaps


# POS helpers for function/content/L2 gaps use spaCy lazily to keep imports cheap.

_NLP = None


def _spacy():
    global _NLP
    if _NLP is None:
        import spacy

        _NLP = spacy.load("en_core_web_sm", disable=["lemmatizer", "ner"])
    return _NLP


_FUNCTION_POS = {"DET", "ADP", "AUX", "CCONJ", "SCONJ", "PART", "PRON"}
_CONTENT_POS = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}


def _pos_indices(text: str, target_pos: Iterable[str]) -> list[int]:
    nlp = _spacy()
    doc = nlp(text)
    target = set(target_pos)
    out: list[int] = []
    offset = 0
    whitespace_tokens = text.split()
    # Align spaCy tokens to whitespace tokens by position
    ws_idx = 0
    for tok in doc:
        if tok.is_space:
            continue
        # Find the ws_token index matching tok.text
        while ws_idx < len(whitespace_tokens) and tok.text not in whitespace_tokens[ws_idx]:
            ws_idx += 1
        if ws_idx < len(whitespace_tokens) and tok.pos_ in target:
            out.append(ws_idx)
        ws_idx += 1
        offset += 1
    return out


def sample_function_or_content(
    text: str,
    tokens: Sequence[str],
    n: int,
    rng: random.Random,
    kind: str,  # "function" | "content"
) -> list[Gap]:
    target = _FUNCTION_POS if kind == "function" else _CONTENT_POS
    idxs = _pos_indices(text, target)
    if not idxs:
        return []
    picks = rng.sample(idxs, k=min(n, len(idxs)))
    return [Gap(start=i, end=i + 1, tokens=(tokens[i],), locus=kind) for i in picks]


# L2 error loci: approximate via POS/morphology. The mapping below mirrors the
# ERRANT locus set used throughout the paper.
def _l2_locus_indices(text: str, loci: Iterable[str]) -> dict[str, list[int]]:
    nlp = _spacy()
    doc = nlp(text)
    loci_set = set(loci)
    out: dict[str, list[int]] = {loc: [] for loc in loci_set}
    ws_tokens = text.split()
    ws_idx = 0
    for tok in doc:
        if tok.is_space:
            continue
        while ws_idx < len(ws_tokens) and tok.text not in ws_tokens[ws_idx]:
            ws_idx += 1
        if ws_idx >= len(ws_tokens):
            break
        if "DET" in loci_set and tok.pos_ == "DET":
            out["DET"].append(ws_idx)
        if "PREP" in loci_set and tok.pos_ == "ADP":
            out["PREP"].append(ws_idx)
        if "VERB:FORM" in loci_set and tok.pos_ == "VERB" and tok.morph.get("VerbForm"):
            out["VERB:FORM"].append(ws_idx)
        if "VERB:SVA" in loci_set and tok.pos_ == "VERB" and "Person" in tok.morph:
            out["VERB:SVA"].append(ws_idx)
        ws_idx += 1
    return out


def sample_l2_loci(
    text: str,
    tokens: Sequence[str],
    n_per_locus: int,
    rng: random.Random,
    loci: Iterable[str],
) -> list[Gap]:
    buckets = _l2_locus_indices(text, loci)
    gaps: list[Gap] = []
    for locus, idxs in buckets.items():
        picks = rng.sample(idxs, k=min(n_per_locus, len(idxs))) if idxs else []
        gaps.extend(
            Gap(start=i, end=i + 1, tokens=(tokens[i],), locus=locus) for i in picks
        )
    return gaps


# ---------------------------------------------------------------------------
# Dispatch


def sample(
    text: str,
    cfg: ClozeConfig,
    rng: random.Random,
) -> list[Gap]:
    """Dispatch on :attr:`ClozeConfig.gap_type`."""
    tokens = _tokenise(text)
    gt = cfg.gap_type
    n = cfg.num_gaps_per_text
    if gt == "single_token":
        return sample_single_token(tokens, n, rng, exclude_initial=cfg.exclude_sentence_initial)
    if gt == "multi_token":
        return sample_multi_token(
            tokens, n, rng,
            lam=cfg.span_length_lambda,
            max_len=cfg.span_max_length,
            exclude_initial=cfg.exclude_sentence_initial,
        )
    if gt in ("function", "content"):
        return sample_function_or_content(text, tokens, n, rng, kind=gt)
    if gt == "l2_loci":
        return sample_l2_loci(text, tokens, n, rng, loci=cfg.l2_loci)
    raise ValueError(f"Unknown gap_type={gt!r}")
