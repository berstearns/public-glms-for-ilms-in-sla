"""Stage 11 — apply the trained checkpoint to transfer corpora."""
from __future__ import annotations

import json
import logging
import random
from dataclasses import asdict
from pathlib import Path

from ilmcloze.cloze.context import build_learner_context
from ilmcloze.cloze.dataset import make_item, write_items
from ilmcloze.cloze.format import LearnerMetadata
from ilmcloze.cloze.gap_sampler import sample as sample_gaps
from ilmcloze.eval.report import emit_report, score_rows
from ilmcloze.io.splits import load_corpus
from ilmcloze.models.registry import build
from pipeline._cli import stage_artifact_dir

from .config import TransferEvalConfig

log = logging.getLogger("pipeline.transfer_eval")


def run_transfer_eval(config: TransferEvalConfig, run_dir: Path) -> None:
    out = stage_artifact_dir(
        config.experiment.output_dir, config.experiment.name, "transfer_eval"
    )
    if not config.transfer.corpora:
        raise SystemExit("transfer.corpora is empty — nothing to evaluate")

    backbone = build(config.model, device=config.experiment.device)
    if config.model.kind == "glm":
        from ilmcloze.infer.glm_infill import infer_batch
    elif config.model.kind == "nwp":
        from ilmcloze.infer.nwp_prompt import infer_batch
    elif config.model.kind == "mlm":
        from ilmcloze.infer.mlm_iterative import infer_batch
    else:
        raise ValueError(f"Unknown model kind {config.model.kind!r}")

    rng = random.Random(config.experiment.seed)
    for corpus_name in config.transfer.corpora:
        corpus_dir = out / corpus_name
        corpus_dir.mkdir(parents=True, exist_ok=True)
        df = load_corpus(config.data, corpus_name)
        items = []
        for _, row in df.iterrows():
            text = str(row["text"])
            gaps = sample_gaps(text, config.cloze, rng)
            tokens = text.split()
            meta = LearnerMetadata(
                l1=str(row.get("l1", "UNK")),
                cefr=str(row.get("cefr", "UNK")),
                errprof="UNK",
            )
            for gap in gaps:
                ctx = build_learner_context(tokens, gap)
                items.append(
                    make_item(
                        corpus_name,
                        f"{row['item_id']}:{gap.start}-{gap.end}",
                        ctx,
                        meta,
                    )
                )
        write_items(corpus_dir / "cloze.jsonl", items)

        preds = infer_batch(
            items, backbone, config.conditioning, config.infer,
            device=config.experiment.device,
        )
        with (corpus_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
            for r in preds:
                fh.write(json.dumps(asdict(r)) + "\n")

        rows = score_rows(preds, items, top_k=config.infer.top_k)
        emit_report(rows, corpus_dir)
        log.info("transfer eval done for corpus=%s", corpus_name)
