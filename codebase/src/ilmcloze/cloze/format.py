"""Format cloze items for each backbone family.

The formatters return **plain strings** which the tokenizer then turns into
model input. Keeping this layer model-agnostic is what lets a single cloze
dataset feed all of GLM / NWP / MLM evaluation.

Learner-conditioning prefix
---------------------------

The ILM learner-conditioning prefix has the shape::

    [L1=<lang>] [CEFR=<level>] [ERRPROF=<k>] <body>

The prefix is enabled/disabled per experiment via
:class:`~ilmcloze.config.ConditioningConfig`.
"""

from __future__ import annotations

from dataclasses import dataclass

from ilmcloze.cloze.context import ContextualItem
from ilmcloze.config import ConditioningConfig

# ---------------------------------------------------------------------------
# Metadata


@dataclass(frozen=True)
class LearnerMetadata:
    l1: str = "UNK"
    cefr: str = "UNK"
    errprof: int | str = "UNK"


def format_conditioning_prefix(m: LearnerMetadata, cfg: ConditioningConfig) -> str:
    if not cfg.enabled:
        return ""
    parts: list[str] = []
    if cfg.include_l1:
        parts.append(f"[L1={m.l1 or cfg.unknown_token}]")
    if cfg.include_cefr:
        parts.append(f"[CEFR={m.cefr or cfg.unknown_token}]")
    if cfg.include_errprof:
        parts.append(f"[ERRPROF={m.errprof if m.errprof != '' else cfg.unknown_token}]")
    return " ".join(parts) + (" " if parts else "")


# ---------------------------------------------------------------------------
# Backbone-specific formats


def format_glm_part_a(
    item: ContextualItem,
    meta: LearnerMetadata,
    cond: ConditioningConfig,
    mask_token: str = "[MASK]",
) -> str:
    """Part A for GLM: corrupted text with a single [MASK] standing for the span."""
    prefix = format_conditioning_prefix(meta, cond)
    left = " ".join(item.left)
    right = " ".join(item.right)
    return f"{prefix}{left} {mask_token} {right}".strip()


def format_glm_part_b_target(
    item: ContextualItem,
    start_token: str = "[START]",
    end_token: str = "[END]",
) -> str:
    """Teacher-forcing target for Part B during training."""
    return f"{start_token} {' '.join(item.gap.tokens)} {end_token}"


def format_nwp_prompt(
    item: ContextualItem,
    meta: LearnerMetadata,
    cond: ConditioningConfig,
) -> str:
    """Prompt-based fill-the-blank for decoder-only NWP models."""
    prefix = format_conditioning_prefix(meta, cond)
    left = " ".join(item.left)
    right = " ".join(item.right)
    instruction = (
        "Fill the blank in the sentence below with the most likely word(s). "
        "Return only the filler.\n"
    )
    return (
        f"{prefix}{instruction}Sentence: {left} ____ {right}\nFiller:"
    )


def format_nwp_lefttoright(item: ContextualItem, meta: LearnerMetadata, cond: ConditioningConfig) -> str:
    """Left-to-right continuation prompt — *discards* the right context."""
    prefix = format_conditioning_prefix(meta, cond)
    left = " ".join(item.left)
    return f"{prefix}{left}"


def format_mlm(
    item: ContextualItem,
    meta: LearnerMetadata,
    cond: ConditioningConfig,
    num_masks: int,
    mask_token: str = "[MASK]",
) -> str:
    """MLM input with ``num_masks`` copies of the mask token at the gap position."""
    prefix = format_conditioning_prefix(meta, cond)
    masks = " ".join([mask_token] * num_masks)
    return f"{prefix}{' '.join(item.left)} {masks} {' '.join(item.right)}".strip()
