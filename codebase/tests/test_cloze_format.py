"""Formatting: conditioning prefix, GLM Part A, NWP prompt, MLM template."""

from __future__ import annotations

from ilmcloze.cloze.context import ContextualItem
from ilmcloze.cloze.format import (
    LearnerMetadata,
    format_conditioning_prefix,
    format_glm_part_a,
    format_glm_part_b_target,
    format_mlm,
    format_nwp_lefttoright,
    format_nwp_prompt,
)
from ilmcloze.cloze.gap_sampler import Gap
from ilmcloze.config import ConditioningConfig


def _item() -> ContextualItem:
    return ContextualItem(
        left=("I", "went", "to"),
        right=("yesterday", "."),
        gap=Gap(start=3, end=5, tokens=("the", "shop")),
        condition="II",
    )


def test_conditioning_prefix_enabled() -> None:
    cond = ConditioningConfig(enabled=True)
    meta = LearnerMetadata(l1="es", cefr="B1", errprof=7)
    pfx = format_conditioning_prefix(meta, cond)
    assert "[L1=es]" in pfx
    assert "[CEFR=B1]" in pfx
    assert "[ERRPROF=7]" in pfx


def test_conditioning_prefix_disabled() -> None:
    assert format_conditioning_prefix(LearnerMetadata(), ConditioningConfig(enabled=False)) == ""


def test_glm_part_a_contains_mask() -> None:
    text = format_glm_part_a(_item(), LearnerMetadata(), ConditioningConfig(enabled=False))
    assert "[MASK]" in text
    assert "I went to" in text
    assert "yesterday" in text


def test_glm_part_b_target() -> None:
    assert format_glm_part_b_target(_item()) == "[START] the shop [END]"


def test_nwp_prompt_contains_underscore() -> None:
    text = format_nwp_prompt(_item(), LearnerMetadata(), ConditioningConfig(enabled=False))
    assert "____" in text
    assert "Filler:" in text


def test_nwp_lefttoright_omits_right() -> None:
    text = format_nwp_lefttoright(_item(), LearnerMetadata(), ConditioningConfig(enabled=False))
    assert "yesterday" not in text
    assert "I went to" in text


def test_mlm_template_count() -> None:
    text = format_mlm(_item(), LearnerMetadata(), ConditioningConfig(enabled=False),
                     num_masks=3, mask_token="[M]")
    assert text.count("[M]") == 3
