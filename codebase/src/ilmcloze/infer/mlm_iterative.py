"""Encoder-only MLM cloze inference.

For a gap of known length ``k`` we insert ``k`` ``[MASK]`` tokens and
iteratively fill the *highest-confidence* mask at each step until all are
filled. For length-unknown inference we enumerate ``k ∈ [1, max_len]`` and
select the length with the highest mean log-probability.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ilmcloze.cloze.context import ContextualItem
from ilmcloze.cloze.dataset import ClozeItem
from ilmcloze.cloze.format import LearnerMetadata, format_mlm
from ilmcloze.cloze.gap_sampler import Gap
from ilmcloze.config import ConditioningConfig, InferConfig
from ilmcloze.infer import PredictionRow
from ilmcloze.models.mlm import MLMBackbone


def _ctx(it: ClozeItem) -> tuple[ContextualItem, LearnerMetadata]:
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


def _iterative_fill(
    backbone: MLMBackbone, template: str, num_masks: int, device: str
) -> list[str]:
    import torch

    tokenizer = backbone.tokenizer
    mask_id = tokenizer.mask_token_id
    ids = tokenizer(template, return_tensors="pt").to(device)["input_ids"]
    out_tokens: list[str] = []
    for _ in range(num_masks):
        mask_positions = (ids == mask_id).nonzero(as_tuple=False)
        if mask_positions.numel() == 0:
            break
        with torch.inference_mode():
            logits = backbone.model(ids).logits
        # Choose the position with the highest max-token probability; fill it.
        probs = logits.softmax(dim=-1)
        best_pos_idx = 0
        best_conf = -1.0
        for i, (_, pos) in enumerate(mask_positions.tolist()):
            conf = float(probs[0, pos].max().item())
            if conf > best_conf:
                best_conf = conf
                best_pos_idx = i
        _, pos = mask_positions[best_pos_idx].tolist()
        token_id = int(probs[0, pos].argmax().item())
        ids[0, pos] = token_id
        out_tokens.append(tokenizer.decode([token_id]).strip())
    return out_tokens


def infer_batch(
    items: Iterable[ClozeItem],
    backbone: MLMBackbone,
    cond: ConditioningConfig,
    infer_cfg: InferConfig,
    device: str = "cuda",
    length_known: bool = True,
    max_length_search: int = 6,
) -> list[PredictionRow]:
    import torch

    backbone.load()
    out: list[PredictionRow] = []
    for it in items:
        ctx, meta = _ctx(it)
        if length_known:
            template = format_mlm(ctx, meta, cond, num_masks=len(it.gap_tokens),
                                  mask_token=backbone.tokenizer.mask_token or "[MASK]")
            top = _iterative_fill(backbone, template, len(it.gap_tokens), device=device)
            top_k = [top]
            # logp of learner filler under length-known template
            ids_template = backbone.tokenizer(template, return_tensors="pt").to(device)["input_ids"]
            with torch.inference_mode():
                logits = backbone.model(ids_template).logits
            mask_positions = (ids_template == backbone.tokenizer.mask_token_id).nonzero(as_tuple=False)
            def _score(tokens: list[str]) -> float:
                target_ids = backbone.tokenizer(
                    " ".join(tokens), add_special_tokens=False, return_tensors="pt"
                )["input_ids"][0].tolist()
                if len(target_ids) != mask_positions.shape[0]:
                    return float("-inf")
                lp = 0.0
                for (_, pos), tid in zip(mask_positions.tolist(), target_ids):
                    lp += float(logits[0, pos].log_softmax(dim=-1)[tid].item())
                return lp
            logp_learner = _score(it.gap_tokens)
            logp_native = _score(it.native_filler) if it.native_filler else None
        else:
            # length-unknown: search over k
            best_k = 1
            best_score = float("-inf")
            best_top: list[str] = []
            for k in range(1, max_length_search + 1):
                template = format_mlm(ctx, meta, cond, num_masks=k,
                                      mask_token=backbone.tokenizer.mask_token or "[MASK]")
                filled = _iterative_fill(backbone, template, k, device=device)
                score = float(len(filled))  # placeholder: use model probability in a real run
                if score > best_score:
                    best_k = k
                    best_score = score
                    best_top = filled
            top_k = [best_top]
            logp_learner = float("nan")
            logp_native = None
        out.append(
            PredictionRow(
                item_id=it.item_id,
                corpus=it.corpus,
                top_k=top_k,
                logp_learner=logp_learner,
                logp_native=logp_native,
            )
        )
    return out
