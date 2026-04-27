"""Stage 02 — produce GEC-corrected texts via CoEdit.

For every row in the input CSV, persists ``{item_id}.json`` with the shape
``{"learner": str, "clean": str}``. Re-runs skip rows whose file exists
unless ``stage.overwrite`` is true.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ilmcloze.gec.coedit import CoEditBackend
from pipeline._cli import stage_artifact_dir

from .config import GecCleanConfig

log = logging.getLogger("pipeline.gec_clean")


def run_gec_clean(config: GecCleanConfig, run_dir: Path) -> None:
    out = stage_artifact_dir(
        config.experiment.output_dir, config.experiment.name, "gec_clean"
    )
    src = Path(config.stage.input_csv) if config.stage.input_csv else (
        stage_artifact_dir(
            config.experiment.output_dir, config.experiment.name, "prepare_splits"
        )
        / "efcamdat_train.csv"
    )
    df = pd.read_csv(src)
    backend = CoEditBackend(repo=config.stage.coedit_repo, device=config.experiment.device)

    for _, row in tqdm(df.iterrows(), total=len(df)):
        path = out / f"{row['item_id']}.json"
        if path.exists() and not config.stage.overwrite:
            continue
        clean = backend.correct(row["text"])
        path.write_text(json.dumps({"learner": row["text"], "clean": clean}))
    log.info("wrote pairs to %s", out)
