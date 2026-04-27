"""Config for pipeline/plot_results."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import yaml

from pipeline._config_common import ExperimentConfig


@dataclass
class StageConfig:
    """``which`` selects figure set: all | ranking | calibration | heatmap | transfer."""
    which: str = "all"


SECTION_MAP: dict[str, type] = {
    "stage": StageConfig,
    "experiment": ExperimentConfig,
}


@dataclass
class PlotResultsConfig:
    stage: StageConfig = field(default_factory=StageConfig)
    experiment: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(name="plot_results")
    )

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)
