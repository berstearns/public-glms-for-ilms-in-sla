"""Config for pipeline/gec_clean."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import yaml

from pipeline._config_common import ExperimentConfig


@dataclass
class StageConfig:
    """Empty string for ``input_csv`` means "use prepare_splits' train CSV"."""
    coedit_repo: str = "grammarly/coedit-large"
    input_csv: str = ""
    overwrite: bool = False


SECTION_MAP: dict[str, type] = {
    "stage": StageConfig,
    "experiment": ExperimentConfig,
}


@dataclass
class GecCleanConfig:
    stage: StageConfig = field(default_factory=StageConfig)
    experiment: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(name="gec_clean")
    )

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)
