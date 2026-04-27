"""Stage 09 — run the selected backbone on the cloze JSONL.

Dispatches on ``model.kind``:
  - ``glm`` → ``ilmcloze.infer.glm_infill``
  - ``nwp`` → ``ilmcloze.infer.nwp_prompt`` (or ``nwp_l2r`` if stage.nwp_l2r)
  - ``mlm`` → ``ilmcloze.infer.mlm_iterative``
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from ilmcloze.cloze.dataset import read_items
from ilmcloze.models.registry import build
from pipeline._cli import stage_artifact_dir

from .config import InferStageConfig

log = logging.getLogger("pipeline.infer")


def run_infer(config: InferStageConfig, run_dir: Path) -> None:
    out = stage_artifact_dir(
        config.experiment.output_dir, config.experiment.name, "infer"
    )
    items_path = Path(config.stage.items_jsonl) if config.stage.items_jsonl else (
        stage_artifact_dir(
            config.experiment.output_dir, config.experiment.name, "build_cloze"
        )
        / "cloze.jsonl"
    )
    items = list(read_items(items_path))
    backbone = build(config.model, device=config.experiment.device)

    if config.model.kind == "glm":
        from ilmcloze.infer.glm_infill import infer_batch
        rows = infer_batch(items, backbone, config.conditioning, config.infer,
                           device=config.experiment.device)
    elif config.model.kind == "nwp":
        if config.stage.nwp_l2r:
            from ilmcloze.infer.nwp_l2r import infer_batch
        else:
            from ilmcloze.infer.nwp_prompt import infer_batch
        rows = infer_batch(items, backbone, config.conditioning, config.infer,
                           device=config.experiment.device)
    elif config.model.kind == "mlm":
        from ilmcloze.infer.mlm_iterative import infer_batch
        rows = infer_batch(
            items, backbone, config.conditioning, config.infer,
            device=config.experiment.device,
            length_known=not config.stage.mlm_length_unknown,
        )
    else:
        raise ValueError(f"Unknown model kind {config.model.kind!r}")

    out_path = out / "predictions.jsonl"
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(asdict(r)) + "\n")
    log.info("wrote %d predictions to %s", len(rows), out_path)
