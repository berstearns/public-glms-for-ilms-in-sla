"""Stage 04 — k-means over ERRANT profile vectors (ERRPROF)."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from ilmcloze.errant_profile.cluster import ErrprofClusterer
from ilmcloze.errant_profile.profile import DEFAULT_TAGS
from pipeline._cli import stage_artifact_dir

from .config import ClusterErrprofConfig

log = logging.getLogger("pipeline.cluster_errprof")


def run_cluster_errprof(config: ClusterErrprofConfig, run_dir: Path) -> None:
    out = stage_artifact_dir(
        config.experiment.output_dir, config.experiment.name, "cluster_errprof"
    )
    profiles_path = Path(config.stage.profiles_path) if config.stage.profiles_path else (
        stage_artifact_dir(
            config.experiment.output_dir, config.experiment.name, "errant_profile"
        )
        / "errant_profiles.parquet"
    )
    df = pd.read_parquet(profiles_path)

    k = config.stage.k_override or config.conditioning.errprof_num_clusters
    cols = [*DEFAULT_TAGS, "OTHER"]
    clusterer = ErrprofClusterer(k=k, seed=config.experiment.seed).fit(df[cols].values)
    clusterer.save(out / "errprof_clusterer.pkl")

    assignments = pd.DataFrame(
        {
            "item_id": df["item_id"],
            "errprof": clusterer.predict(df[cols].values),
        }
    )
    assignments.to_parquet(out / "errprof_assignments.parquet")
    log.info("wrote clusterer + assignments to %s", out)
