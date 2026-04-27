"""Prompt-based fill-the-blank inference for decoder-only NWP models."""

from __future__ import annotations

from typing import Iterable

from ilmcloze.cloze.context import ContextualItem
from ilmcloze.cloze.dataset import ClozeItem
from ilmcloze.cloze.format import LearnerMetadata, format_nwp_prompt
from ilmcloze.cloze.gap_sampler import Gap
from ilmcloze.config import ConditioningConfig, InferConfig
from ilmcloze.infer import PredictionRow
from ilmcloze.models.nwp import NWPBackbone


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


def _target_logprob(backbone: NWPBackbone, prompt: str, target: str, device: str) -> float:
    import torch

    enc_full = backbone.tokenizer(prompt + " " + target, return_tensors="pt").to(device)
    enc_prompt = backbone.tokenizer(prompt, return_tensors="pt").to(device)
    n_prompt = enc_prompt["input_ids"].shape[1]
    labels = enc_full["input_ids"].clone()
    labels[:, :n_prompt] = -100  # only score target tokens
    with torch.inference_mode():
        out = backbone.model(**enc_full, labels=labels)
    n_target = int((labels != -100).sum().item())
    return float(-out.loss.item() * n_target)


def infer_batch(
    items: Iterable[ClozeItem],
    backbone: NWPBackbone,
    cond: ConditioningConfig,
    infer_cfg: InferConfig,
    device: str = "cuda",
) -> list[PredictionRow]:
    import torch

    backbone.load()
    out: list[PredictionRow] = []
    for it in items:
        ctx, meta = _item_to_context(it)
        prompt = format_nwp_prompt(ctx, meta, cond)

        enc = backbone.tokenizer(prompt, return_tensors="pt", truncation=True).to(device)
        with torch.inference_mode():
            gen = backbone.model.generate(
                **enc,
                max_new_tokens=16,
                num_return_sequences=max(infer_cfg.top_k),
                num_beams=max(infer_cfg.top_k),
                do_sample=infer_cfg.sample,
                temperature=infer_cfg.temperature,
                pad_token_id=backbone.tokenizer.pad_token_id,
            )
        top_k: list[list[str]] = []
        for seq in gen:
            text = backbone.tokenizer.decode(seq[enc["input_ids"].shape[1] :], skip_special_tokens=True)
            top_k.append(text.strip().split())
        logp_learner = _target_logprob(backbone, prompt, " ".join(it.gap_tokens), device=device)
        logp_native = (
            _target_logprob(backbone, prompt, " ".join(it.native_filler), device=device)
            if it.native_filler
            else None
        )
        out.append(
            PredictionRow(
                item_id=it.item_id,
                corpus=it.corpus,
                top_k=top_k,
                logp_learner=logp_learner,
                logp_native=logp_native,
                logp_empirical=(
                    [_target_logprob(backbone, prompt, " ".join(f), device=device) for f in it.empirical_fillers]
                    if it.empirical_fillers
                    else None
                ),
            )
        )
    return out
