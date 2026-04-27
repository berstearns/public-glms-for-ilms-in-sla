"""Stage 06 — materialise condition III by corrupting condition I."""
from __future__ import annotations

import logging
import random
from pathlib import Path

from ilmcloze.cloze.context import ContextualItem, corrupt_context
from ilmcloze.cloze.dataset import ClozeItem, read_items, write_items
from ilmcloze.cloze.gap_sampler import Gap
from pipeline._cli import stage_artifact_dir

from .config import CorruptContextConfig

log = logging.getLogger("pipeline.corrupt_context")


def run_corrupt_context(config: CorruptContextConfig, run_dir: Path) -> None:
    out = stage_artifact_dir(
        config.experiment.output_dir, config.experiment.name, "corrupt_context"
    )
    src = Path(config.stage.source_jsonl) if config.stage.source_jsonl else (
        stage_artifact_dir(
            config.experiment.output_dir, config.experiment.name, "build_cloze"
        )
        / "cloze.jsonl"
    )
    rate = config.cloze.synth_corruption_rate
    rng = random.Random(config.experiment.seed)

    def _corrupt(it: ClozeItem) -> ClozeItem:
        ctx = ContextualItem(
            left=tuple(it.left),
            right=tuple(it.right),
            gap=Gap(it.gap_start, it.gap_end, tuple(it.gap_tokens), it.locus),
            condition="I",
        )
        new_ctx = corrupt_context(ctx, rng, rate=rate, target_condition="III")
        return ClozeItem(
            corpus=it.corpus, item_id=it.item_id,
            gap_start=it.gap_start, gap_end=it.gap_end,
            gap_tokens=it.gap_tokens, locus=it.locus,
            condition="III",
            left=list(new_ctx.left), right=list(new_ctx.right),
            meta=it.meta, native_filler=it.native_filler,
            empirical_fillers=it.empirical_fillers,
        )

    out_path = out / "cloze.jsonl"
    write_items(out_path, (_corrupt(it) for it in read_items(src)))
    log.info("wrote %s", out_path)
