"""Stage 03 — ERRANT-tag each (learner, clean) pair."""
from __future__ import annotations

import json
import logging
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from ilmcloze.errant_profile.profile import DEFAULT_TAGS, vectorise
from ilmcloze.errant_profile.tag import tag_pair
from pipeline._cli import stage_artifact_dir

from .config import ErrantProfileConfig

log = logging.getLogger("pipeline.errant_profile")


def run_errant_profile(config: ErrantProfileConfig, run_dir: Path) -> None:
    out = stage_artifact_dir(
        config.experiment.output_dir, config.experiment.name, "errant_profile"
    )
    pairs_dir = Path(config.stage.pairs_dir) if config.stage.pairs_dir else (
        stage_artifact_dir(
            config.experiment.output_dir, config.experiment.name, "gec_clean"
        )
    )
    columns = [*DEFAULT_TAGS, "OTHER"]

    rows: list[dict] = []
    for path in tqdm(sorted(pairs_dir.glob("*.json"))):
        payload = json.loads(path.read_text())
        tags = tag_pair(payload["learner"], payload["clean"])
        vec = vectorise(tags)
        rows.append(
            {"item_id": path.stem, **{c: float(v) for c, v in zip(columns, vec)}}
        )
    pd.DataFrame(rows).to_parquet(out / "errant_profiles.parquet")
    log.info("wrote %s", out / "errant_profiles.parquet")
