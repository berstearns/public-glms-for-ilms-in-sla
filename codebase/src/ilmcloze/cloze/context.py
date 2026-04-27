"""Build the context surrounding each gap under condition I / II / III.

Condition I ("clean")
    Replace the learner text surrounding the gap with its GEC-corrected
    version (produced by :mod:`ilmcloze.gec`). The gap itself retains the
    learner filler as gold.

Condition II ("learner")
    The raw learner text is kept verbatim; the gap is the only modification.

Condition III ("synthetic corruption")
    Start from condition I and apply a calibrated synthetic error model to
    the surrounding tokens (token deletion, preposition swap, determiner
    drop, SVA mismatch) until the ERRANT error rate matches that of
    EFCAMDAT-B1.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from ilmcloze.cloze.gap_sampler import Gap


@dataclass(frozen=True)
class ContextualItem:
    """A gap + its left/right context under a specified condition."""

    left: tuple[str, ...]
    right: tuple[str, ...]
    gap: Gap
    condition: str  # "I" | "II" | "III"


# ---------------------------------------------------------------------------


def build_learner_context(tokens: Sequence[str], gap: Gap) -> ContextualItem:
    return ContextualItem(
        left=tuple(tokens[: gap.start]),
        right=tuple(tokens[gap.end :]),
        gap=gap,
        condition="II",
    )


def build_clean_context(
    learner_tokens: Sequence[str],
    clean_tokens: Sequence[str],
    gap: Gap,
) -> ContextualItem:
    """Take the GEC-cleaned surroundings and the learner gap.

    ``learner_tokens`` and ``clean_tokens`` must be whitespace-aligned at the
    gap boundary. The GEC backend is responsible for aligning them; see
    :mod:`ilmcloze.gec`.
    """
    left = tuple(clean_tokens[: gap.start])
    right = tuple(clean_tokens[gap.end :]) if gap.end <= len(clean_tokens) else tuple(clean_tokens[gap.start :])
    return ContextualItem(left=left, right=right, gap=gap, condition="I")


# --- synthetic corruption -------------------------------------------------


_PREPS = ("in", "on", "at", "to", "for", "of", "with", "by")


def _corrupt_token(tok: str, rng: random.Random) -> str | None:
    """Apply a small corruption; return the corrupted token, or ``None`` to delete."""
    low = tok.lower()
    if low in {"a", "an", "the"} and rng.random() < 0.5:
        return None
    if low in _PREPS and rng.random() < 0.5:
        candidates = [p for p in _PREPS if p != low]
        return rng.choice(candidates)
    if low.endswith("s") and len(low) > 3 and rng.random() < 0.3:
        return tok[:-1]  # strip trailing -s (crude SVA corruption)
    return tok  # keep


def corrupt_context(
    item: ContextualItem,
    rng: random.Random,
    rate: float = 0.05,
    target_condition: str = "III",
) -> ContextualItem:
    """Apply token-level corruptions to left/right contexts at rate ``rate``."""

    def _maybe(tokens: Sequence[str]) -> tuple[str, ...]:
        out: list[str] = []
        for t in tokens:
            if rng.random() < rate:
                ct = _corrupt_token(t, rng)
                if ct is not None:
                    out.append(ct)
            else:
                out.append(t)
        return tuple(out)

    return ContextualItem(
        left=_maybe(item.left),
        right=_maybe(item.right),
        gap=item.gap,
        condition=target_condition,
    )
