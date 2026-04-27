"""Stage 01 — CEFR-stratified sampling of EFCAMDAT train/test."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ilmcloze.io.splits import load_efcamdat
from pipeline._cli import stage_artifact_dir

from .config import PrepareSplitsConfig

log = logging.getLogger("pipeline.prepare_splits")


def _stratify(df: pd.DataFrame, n_per: int, seed: int) -> pd.DataFrame:
    if n_per <= 0:
        return df
    parts = [
        g.sample(n=min(n_per, len(g)), random_state=seed) for _, g in df.groupby("cefr")
    ]
    return pd.concat(parts, ignore_index=True)


def run_prepare_splits(config: PrepareSplitsConfig, run_dir: Path) -> None:
    out = stage_artifact_dir(
        config.experiment.output_dir, config.experiment.name, "prepare_splits"
    )
    train = load_efcamdat(config.data, split="train")
    test = load_efcamdat(config.data, split="test")

    _stratify(train, config.stage.train_sample, config.experiment.seed).to_csv(
        out / "efcamdat_train.csv", index=False
    )
    _stratify(test, config.stage.test_sample, config.experiment.seed).to_csv(
        out / "efcamdat_test.csv", index=False
    )
    log.info("wrote train+test to %s", out)
