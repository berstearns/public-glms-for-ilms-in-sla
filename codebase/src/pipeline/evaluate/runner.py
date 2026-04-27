"""Stage 10 — score predictions and emit summary + stratified reports."""
from __future__ import annotations

import json
import logging
from pathlib import Path

from ilmcloze.cloze.dataset import read_items
from ilmcloze.eval.report import emit_report, score_rows
from ilmcloze.infer import PredictionRow
from pipeline._cli import stage_artifact_dir

from .config import EvaluateConfig

log = logging.getLogger("pipeline.evaluate")


def _load_predictions(path: Path) -> list[PredictionRow]:
    rows: list[PredictionRow] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(PredictionRow(**json.loads(line)))
    return rows


def run_evaluate(config: EvaluateConfig, run_dir: Path) -> None:
    out = stage_artifact_dir(
        config.experiment.output_dir, config.experiment.name, "evaluate"
    )
    items = list(
        read_items(
            stage_artifact_dir(
                config.experiment.output_dir, config.experiment.name, "build_cloze"
            )
            / "cloze.jsonl"
        )
    )
    preds = _load_predictions(
        stage_artifact_dir(
            config.experiment.output_dir, config.experiment.name, "infer"
        )
        / "predictions.jsonl"
    )
    rows = score_rows(preds, items, top_k=config.infer.top_k)
    emit_report(rows, out)
    log.info("wrote report to %s", out)
