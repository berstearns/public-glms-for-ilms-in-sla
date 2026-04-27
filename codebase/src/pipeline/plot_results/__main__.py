from pipeline._cli import run_stage

from .config import SECTION_MAP, PlotResultsConfig
from .runner import run_plot_results


if __name__ == "__main__":
    run_stage(
        description="Produce the figures used in the paper.",
        config_cls=PlotResultsConfig,
        section_map=SECTION_MAP,
        runner=run_plot_results,
        use_run_dir=False,
    )
