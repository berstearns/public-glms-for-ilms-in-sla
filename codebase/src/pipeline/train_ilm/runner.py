"""Stage 07 — continued-pretrain a GLM on EFCAMDAT with GLM objective."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ilmcloze.cloze.format import LearnerMetadata
from ilmcloze.train.continued_pretrain import train
from pipeline._cli import stage_artifact_dir

from .config import TrainIlmConfig

log = logging.getLogger("pipeline.train_ilm")


def run_train_ilm(config: TrainIlmConfig, run_dir: Path) -> None:
    out = stage_artifact_dir(
        config.experiment.output_dir, config.experiment.name, "train_ilm"
    )
    src = (
        stage_artifact_dir(
            config.experiment.output_dir, config.experiment.name, "prepare_splits"
        )
        / "efcamdat_train.csv"
    )
    df = pd.read_csv(src)

    assignments_path = (
        stage_artifact_dir(
            config.experiment.output_dir, config.experiment.name, "cluster_errprof"
        )
        / "errprof_assignments.parquet"
    )
    errprof: dict[str, int] = {}
    if assignments_path.exists():
        ap = pd.read_parquet(assignments_path)
        errprof = dict(zip(ap["item_id"].astype(str), ap["errprof"].astype(int)))

    texts = df["text"].astype(str).tolist()
    metas = [
        LearnerMetadata(
            l1=str(r.get("l1", "UNK")),
            cefr=str(r.get("cefr", "UNK")),
            errprof=errprof.get(str(r["item_id"]), "UNK"),
        )
        for _, r in df.iterrows()
    ]

    ckpt = train(
        texts=texts,
        metas=metas,
        model_cfg=config.model,
        train_cfg=config.train,
        cond_cfg=config.conditioning,
        cloze_cfg=config.cloze,
        output_dir=out,
        device=config.experiment.device,
    )
    (out / "checkpoint.txt").write_text(str(ckpt))
    log.info("wrote checkpoint pointer %s", out / "checkpoint.txt")
