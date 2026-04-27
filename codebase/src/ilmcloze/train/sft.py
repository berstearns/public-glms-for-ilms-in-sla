"""Supervised fine-tuning on pre-materialised cloze triples.

Input: a JSONL file of :class:`ilmcloze.cloze.dataset.ClozeItem`. Target:
the gap tokens. Loss: cross-entropy on Part B (teacher-forced).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from ilmcloze.cloze.context import ContextualItem
from ilmcloze.cloze.dataset import ClozeItem, read_items
from ilmcloze.cloze.format import LearnerMetadata, format_glm_part_a, format_glm_part_b_target
from ilmcloze.cloze.gap_sampler import Gap
from ilmcloze.config import ConditioningConfig, ModelConfig, TrainConfig
from ilmcloze.models.glm import GLMBackbone
from ilmcloze.utils.logging import get_logger
from ilmcloze.utils.seed import set_all

_LOG = get_logger(__name__)


def _pair_from_item(item: ClozeItem, cond: ConditioningConfig) -> tuple[str, str]:
    ctx = ContextualItem(
        left=tuple(item.left),
        right=tuple(item.right),
        gap=Gap(
            start=item.gap_start,
            end=item.gap_end,
            tokens=tuple(item.gap_tokens),
            locus=item.locus,
        ),
        condition=item.condition,
    )
    meta = LearnerMetadata(
        l1=str(item.meta.get("l1", "UNK")),
        cefr=str(item.meta.get("cefr", "UNK")),
        errprof=item.meta.get("errprof", "UNK"),
    )
    part_a = format_glm_part_a(ctx, meta, cond)
    part_b = format_glm_part_b_target(ctx)
    return part_a, part_b


def train_sft(
    items_path: str | Path,
    model_cfg: ModelConfig,
    train_cfg: TrainConfig,
    cond_cfg: ConditioningConfig,
    output_dir: str | Path,
    device: str = "cuda",
) -> Path:
    import torch
    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    set_all(train_cfg.seed)
    backbone = GLMBackbone(cfg=model_cfg, device=device)
    backbone.load()
    model = backbone.model
    tokenizer = backbone.tokenizer
    model.train()

    items: Iterable[ClozeItem] = list(read_items(items_path))
    pairs = [_pair_from_item(it, cond_cfg) for it in items]

    steps_per_epoch = max(1, len(pairs) // train_cfg.batch_size)
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
    step = 0
    for epoch in range(train_cfg.epochs):
        for batch_start in range(0, len(pairs), train_cfg.batch_size):
            batch = pairs[batch_start : batch_start + train_cfg.batch_size]
            a_texts = [a for a, _ in batch]
            b_texts = [b for _, b in batch]
            enc_a = tokenizer(
                a_texts, return_tensors="pt", padding=True, truncation=True,
                max_length=model_cfg.max_seq_length,
            ).to(device)
            enc_b = tokenizer(
                b_texts, return_tensors="pt", padding=True, truncation=True,
                max_length=model_cfg.max_seq_length,
            ).to(device)
            labels = enc_b["input_ids"].masked_fill(enc_b["attention_mask"].eq(0), -100)

            out_obj = model(**enc_a, labels=labels)
            loss = out_obj.loss / train_cfg.grad_accum_steps
            loss.backward()

            if (step + 1) % train_cfg.grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.max_grad_norm)
                optim.step()
                sched.step()
                optim.zero_grad(set_to_none=True)
            step += 1
    model.save_pretrained(out / "final")
    tokenizer.save_pretrained(out / "final")
    _LOG.info("Saved SFT checkpoint to %s", out / "final")
    return out / "final"
