"""Stage 08 — supervised fine-tuning on cloze triples."""
from __future__ import annotations

import logging
from pathlib import Path

from ilmcloze.train.sft import train_sft
from pipeline._cli import stage_artifact_dir

from .config import TrainSftConfig

log = logging.getLogger("pipeline.train_sft")


def run_train_sft(config: TrainSftConfig, run_dir: Path) -> None:
    out = stage_artifact_dir(
        config.experiment.output_dir, config.experiment.name, "train_sft"
    )
    items_path = (
        stage_artifact_dir(
            config.experiment.output_dir, config.experiment.name, "build_cloze"
        )
        / "cloze.jsonl"
    )
    ckpt = train_sft(
        items_path=items_path,
        model_cfg=config.model,
        train_cfg=config.train,
        cond_cfg=config.conditioning,
        output_dir=out,
        device=config.experiment.device,
    )
    (out / "checkpoint.txt").write_text(str(ckpt))
    log.info("wrote checkpoint pointer %s", out / "checkpoint.txt")
