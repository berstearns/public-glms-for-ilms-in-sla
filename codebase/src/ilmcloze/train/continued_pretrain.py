"""Continued pretraining of a GLM on learner corpora.

The training loop materialises GLM-style span-masked examples on-the-fly:
for each input text we sample multiple spans with Poisson(λ) lengths until
the masked budget is reached, shuffle their order in Part B, and compute
the standard cross-entropy over Part B tokens. The conditioning prefix is
prepended to Part A.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ilmcloze.cloze.format import LearnerMetadata, format_conditioning_prefix
from ilmcloze.config import ClozeConfig, ConditioningConfig, ModelConfig, TrainConfig
from ilmcloze.models.glm import GLMBackbone
from ilmcloze.utils.logging import get_logger
from ilmcloze.utils.seed import set_all

_LOG = get_logger(__name__)


# ---------------------------------------------------------------------------
# On-the-fly span masking


@dataclass(frozen=True)
class MaskedExample:
    part_a: str
    part_b: str


def _sample_spans(tokens: list[str], cloze: ClozeConfig, rng: random.Random) -> list[tuple[int, int]]:
    """Sample spans until ``cloze.mask_budget`` fraction of tokens is masked."""
    budget = int(len(tokens) * cloze.mask_budget)
    if budget <= 0:
        return []
    spans: list[tuple[int, int]] = []
    masked = 0
    tries = 0
    gen = np.random.default_rng(rng.randint(0, 2**31 - 1))
    while masked < budget and tries < 10 * budget:
        tries += 1
        length = max(1, min(cloze.span_max_length, int(gen.poisson(cloze.span_length_lambda))))
        if len(tokens) <= length:
            break
        start = rng.randint(0, len(tokens) - length)
        end = start + length
        if any(not (end <= a or start >= b) for (a, b) in spans):
            continue
        spans.append((start, end))
        masked += length
    return sorted(spans)


def build_masked_example(
    text: str,
    meta: LearnerMetadata,
    cond: ConditioningConfig,
    cloze: ClozeConfig,
    rng: random.Random,
    mask_token: str = "[MASK]",
    start_token: str = "[START]",
    end_token: str = "[END]",
) -> MaskedExample:
    tokens = text.split()
    spans = _sample_spans(tokens, cloze, rng)
    if not spans:
        return MaskedExample(part_a=text, part_b="")

    # Build Part A with a single [MASK] per span.
    part_a_tokens: list[str] = []
    last = 0
    for s, e in spans:
        part_a_tokens.extend(tokens[last:s])
        part_a_tokens.append(mask_token)
        last = e
    part_a_tokens.extend(tokens[last:])
    prefix = format_conditioning_prefix(meta, cond)
    part_a = prefix + " ".join(part_a_tokens)

    # Shuffle span order in Part B.
    perm = list(range(len(spans)))
    rng.shuffle(perm)
    part_b_pieces: list[str] = []
    for i in perm:
        s, e = spans[i]
        part_b_pieces.append(f"{start_token} {' '.join(tokens[s:e])} {end_token}")
    return MaskedExample(part_a=part_a, part_b=" ".join(part_b_pieces))


# ---------------------------------------------------------------------------
# Training loop (seq2seq teacher-forcing; PyTorch core)


def train(
    texts: Iterable[str],
    metas: Iterable[LearnerMetadata],
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    cond_cfg: ConditioningConfig,
    cloze_cfg: ClozeConfig,
    output_dir: str | Path,
    device: str = "cuda",
) -> Path:
    """Train a GLM with on-the-fly span masking on (texts, metas).

    Returns the path where the final checkpoint is saved.
    """
    import torch
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    set_all(train_cfg.seed)
    backbone = GLMBackbone(cfg=model_cfg, device=device)
    backbone.load()
    model = backbone.model
    tokenizer = backbone.tokenizer
    model.train()

    rng = random.Random(train_cfg.seed)
    texts_list = list(texts)
    metas_list = list(metas)
    if len(metas_list) != len(texts_list):
        raise ValueError("texts and metas must have the same length")

    steps_per_epoch = max(1, len(texts_list) // train_cfg.batch_size)
    total_steps = steps_per_epoch * train_cfg.epochs
    warmup_steps = int(total_steps * train_cfg.warmup_ratio)
    optim = AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    sched = get_linear_schedule_with_warmup(optim, warmup_steps, total_steps)

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _LOG.info("Training GLM for %d steps (epochs=%d)", total_steps, train_cfg.epochs)

    step = 0
    for epoch in range(train_cfg.epochs):
        order = list(range(len(texts_list)))
        rng.shuffle(order)
        for batch_start in range(0, len(order), train_cfg.batch_size):
            batch_idx = order[batch_start : batch_start + train_cfg.batch_size]
            pairs = [
                build_masked_example(
                    texts_list[i], metas_list[i], cond_cfg, cloze_cfg, rng
                )
                for i in batch_idx
            ]
            pairs = [p for p in pairs if p.part_b]
            if not pairs:
                continue
            inputs = tokenizer(
                [p.part_a for p in pairs],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=model_cfg.max_seq_length,
            ).to(device)
            labels = tokenizer(
                [p.part_b for p in pairs],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=model_cfg.max_seq_length,
            ).to(device)
            labels_ids = labels["input_ids"]
            labels_ids = labels_ids.masked_fill(labels["attention_mask"].eq(0), -100)

            outputs = model(**inputs, labels=labels_ids)
            loss = outputs.loss / train_cfg.grad_accum_steps
            loss.backward()

            if (step + 1) % train_cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)

            if step % 50 == 0:
                _LOG.info("epoch=%d step=%d loss=%.4f", epoch, step, loss.item() * train_cfg.grad_accum_steps)
            if train_cfg.save_every_steps and step > 0 and step % train_cfg.save_every_steps == 0:
                model.save_pretrained(out / f"step-{step}")
                tokenizer.save_pretrained(out / f"step-{step}")
            step += 1

    model.save_pretrained(out / "final")
    tokenizer.save_pretrained(out / "final")
    _LOG.info("Saved final checkpoint to %s", out / "final")
    return out / "final"
