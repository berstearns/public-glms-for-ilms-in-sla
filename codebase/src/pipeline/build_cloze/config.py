"""Config for pipeline/build_cloze."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import yaml

from ilmcloze.config import ClozeConfig, DataConfig
from pipeline._config_common import ExperimentConfig


@dataclass
class StageConfig:
    """Empty ``source_csv`` = "use prepare_splits' test CSV"."""
    source_csv: str = ""


SECTION_MAP: dict[str, type] = {
    "data": DataConfig,
    "cloze": ClozeConfig,
    "stage": StageConfig,
    "experiment": ExperimentConfig,
}


@dataclass
class BuildClozeConfig:
    data: DataConfig = field(default_factory=lambda: DataConfig(splits_dir=""))
    cloze: ClozeConfig = field(default_factory=ClozeConfig)
    stage: StageConfig = field(default_factory=StageConfig)
    experiment: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(name="build_cloze")
    )

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)
