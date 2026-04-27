"""Stage 12 — produce the figures used in the paper."""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from pipeline._cli import stage_artifact_dir

from .config import PlotResultsConfig

log = logging.getLogger("pipeline.plot_results")


def run_plot_results(config: PlotResultsConfig, run_dir: Path) -> None:
    import matplotlib.pyplot as plt

    out = stage_artifact_dir(
        config.experiment.output_dir, config.experiment.name, "plot_results"
    )
    eval_dir = stage_artifact_dir(
        config.experiment.output_dir, config.experiment.name, "evaluate"
    )
    which = config.stage.which

    if which in ("all", "ranking") and (eval_dir / "summary.csv").exists():
        df = pd.read_csv(eval_dir / "summary.csv")
        ax = df.T.reset_index().rename(
            columns={"index": "metric", 0: "value"}
        ).plot.bar(x="metric", y="value", legend=False)
        ax.set_title("Main metrics")
        plt.tight_layout()
        plt.savefig(out / "fig_ranking_bar.png", dpi=200)
        plt.close()

    if which in ("all", "heatmap") and (eval_dir / "by_cefr.csv").exists():
        df = pd.read_csv(eval_dir / "by_cefr.csv")
        if "lp" in df.columns and df["lp"].notna().any():
            import seaborn as sns

            pivot = df.pivot_table(values="lp", index="stratum").fillna(0)
            if not pivot.empty:
                plt.figure()
                sns.heatmap(pivot, annot=True)
                plt.title("Learner-plausibility by CEFR")
                plt.tight_layout()
                plt.savefig(out / "fig_per_locus_heatmap.png", dpi=200)
                plt.close()
            else:
                log.info("skipping heatmap: by_cefr.csv pivot is empty")
        else:
            log.info("skipping heatmap: no lp values in by_cefr.csv")
    log.info("wrote figures to %s", out)
