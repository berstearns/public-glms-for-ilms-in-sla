"""Stage 05 — sample gaps and materialise ClozeItems (cond I or II)."""
from __future__ import annotations

import json
import logging
import random
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ilmcloze.cloze.context import build_clean_context, build_learner_context
from ilmcloze.cloze.dataset import ClozeItem, make_item
from ilmcloze.cloze.format import LearnerMetadata
from ilmcloze.cloze.gap_sampler import sample as sample_gaps
from pipeline._cli import stage_artifact_dir

from .config import BuildClozeConfig

log = logging.getLogger("pipeline.build_cloze")


def run_build_cloze(config: BuildClozeConfig, run_dir: Path) -> None:
    out = stage_artifact_dir(
        config.experiment.output_dir, config.experiment.name, "build_cloze"
    )
    src = Path(config.stage.source_csv) if config.stage.source_csv else (
        stage_artifact_dir(
            config.experiment.output_dir, config.experiment.name, "prepare_splits"
        )
        / "efcamdat_test.csv"
    )
    df = pd.read_csv(src)
    gec_dir = stage_artifact_dir(
        config.experiment.output_dir, config.experiment.name, "gec_clean"
    )
    assignments_path = (
        stage_artifact_dir(
            config.experiment.output_dir, config.experiment.name, "cluster_errprof"
        )
        / "errprof_assignments.parquet"
    )
    assignments: dict[str, int] = {}
    if assignments_path.exists():
        ap = pd.read_parquet(assignments_path)
        assignments = dict(zip(ap["item_id"].astype(str), ap["errprof"].astype(int)))

    rng = random.Random(config.experiment.seed)
    condition = config.cloze.context_condition
    out_path = out / "cloze.jsonl"
    items: list[ClozeItem] = []

    with out_path.open("w", encoding="utf-8") as fh:
        for _, row in tqdm(df.iterrows(), total=len(df)):
            text = str(row["text"])
            gaps = sample_gaps(text, config.cloze, rng)
            tokens = text.split()
            pair_path = gec_dir / f"{row['item_id']}.json"
            clean_tokens: list[str] = []
            if pair_path.exists():
                clean_tokens = json.loads(pair_path.read_text())["clean"].split()
            meta = LearnerMetadata(
                l1=str(row.get("l1", "UNK")),
                cefr=str(row.get("cefr", "UNK")),
                errprof=assignments.get(str(row["item_id"]), "UNK"),
            )
            for gap in gaps:
                ctx = (
                    build_learner_context(tokens, gap)
                    if condition == "II"
                    else build_clean_context(tokens, clean_tokens or tokens, gap)
                )
                native_filler = None
                if clean_tokens and gap.end <= len(clean_tokens):
                    native_filler = clean_tokens[gap.start : gap.end]
                item = make_item(
                    corpus=str(row.get("corpus", "EFCAMDAT")),
                    item_id=f"{row['item_id']}:{gap.start}-{gap.end}",
                    ctx=ctx,
                    meta=meta,
                    native_filler=native_filler,
                )
                fh.write(item.to_json() + "\n")
                items.append(item)
    log.info("wrote %d items to %s", len(items), out_path)
