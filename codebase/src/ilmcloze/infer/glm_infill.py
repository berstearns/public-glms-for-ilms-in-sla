"""GLM blank-infilling inference.

The GLM generation step produces Part B autoregressively, conditioned on
Part A (which contains the conditioning prefix, left context, ``[MASK]``,
and right context).
"""

from __future__ import annotations

from typing import Iterable

from ilmcloze.cloze.context import ContextualItem
from ilmcloze.cloze.dataset import ClozeItem
from ilmcloze.cloze.format import LearnerMetadata, format_glm_part_a, format_glm_part_b_target
from ilmcloze.cloze.gap_sampler import Gap
from ilmcloze.config import ConditioningConfig, InferConfig
from ilmcloze.infer import PredictionRow
from ilmcloze.models.glm import GLMBackbone


def _item_to_context(it: ClozeItem) -> tuple[ContextualItem, LearnerMetadata]:
    ctx = ContextualItem(
        left=tuple(it.left),
        right=tuple(it.right),
        gap=Gap(start=it.gap_start, end=it.gap_end, tokens=tuple(it.gap_tokens), locus=it.locus),
        condition=it.condition,
    )
    meta = LearnerMetadata(
        l1=str(it.meta.get("l1", "UNK")),
        cefr=str(it.meta.get("cefr", "UNK")),
        errprof=it.meta.get("errprof", "UNK"),
    )
    return ctx, meta


def _target_logprob(
    backbone: GLMBackbone, part_a: str, target_span_tokens: list[str], device: str
) -> float:
    """Compute the log-probability of a target span under GLM given Part A."""
    import torch

    target = f"[START] {' '.join(target_span_tokens)} [END]"
    enc_a = backbone.tokenizer(part_a, return_tensors="pt", truncation=True).to(device)
    enc_b = backbone.tokenizer(target, return_tensors="pt", truncation=True).to(device)
    labels = enc_b["input_ids"]
    with torch.inference_mode():
        out = backbone.model(**enc_a, labels=labels)
    # HF returns mean CE over non-ignored tokens; scale back to sum log-prob.
    n = int(labels.ne(-100).sum().item())
    return float(-out.loss.item() * n)


def infer_batch(
    items: Iterable[ClozeItem],
    backbone: GLMBackbone,
    cond: ConditioningConfig,
    infer_cfg: InferConfig,
    device: str = "cuda",
) -> list[PredictionRow]:
    import torch

    backbone.load()
    out: list[PredictionRow] = []
    for it in items:
        ctx, meta = _item_to_context(it)
        part_a = format_glm_part_a(ctx, meta, cond)

        enc_a = backbone.tokenizer(part_a, return_tensors="pt", truncation=True).to(device)
        with torch.inference_mode():
            gen = backbone.model.generate(
                **enc_a,
                max_new_tokens=16,
                num_return_sequences=max(infer_cfg.top_k),
                num_beams=max(infer_cfg.top_k),
                do_sample=infer_cfg.sample,
                temperature=infer_cfg.temperature,
            )
        top_k: list[list[str]] = []
        for seq in gen:
            text = backbone.tokenizer.decode(seq, skip_special_tokens=True)
            top_k.append(text.split())
        logp_learner = _target_logprob(backbone, part_a, it.gap_tokens, device=device)
        logp_native = (
            _target_logprob(backbone, part_a, it.native_filler, device=device)
            if it.native_filler
            else None
        )
        logp_empirical = (
            [_target_logprob(backbone, part_a, f, device=device) for f in it.empirical_fillers]
            if it.empirical_fillers
            else None
        )
        out.append(
            PredictionRow(
                item_id=it.item_id,
                corpus=it.corpus,
                top_k=top_k[: max(infer_cfg.top_k)],
                logp_learner=logp_learner,
                logp_native=logp_native,
                logp_empirical=logp_empirical,
            )
        )
    return out
