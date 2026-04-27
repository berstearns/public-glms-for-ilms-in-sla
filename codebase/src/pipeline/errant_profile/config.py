"""Config for pipeline/errant_profile."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import yaml

from pipeline._config_common import ExperimentConfig


@dataclass
class StageConfig:
    """Empty ``pairs_dir`` = "use gec_clean output for this experiment"."""
    pairs_dir: str = ""


SECTION_MAP: dict[str, type] = {
    "stage": StageConfig,
    "experiment": ExperimentConfig,
}


@dataclass
class ErrantProfileConfig:
    stage: StageConfig = field(default_factory=StageConfig)
    experiment: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(name="errant_profile")
    )

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)
