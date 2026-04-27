"""Config for pipeline/prepare_splits."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import yaml

from ilmcloze.config import DataConfig
from pipeline._config_common import ExperimentConfig


@dataclass
class StageConfig:
    """Stage-specific knobs. 0 means "no subsampling — keep the full split"."""
    train_sample: int = 0
    test_sample: int = 0


SECTION_MAP: dict[str, type] = {
    "data": DataConfig,
    "stage": StageConfig,
    "experiment": ExperimentConfig,
}


@dataclass
class PrepareSplitsConfig:
    data: DataConfig = field(default_factory=lambda: DataConfig(splits_dir=""))
    stage: StageConfig = field(default_factory=StageConfig)
    experiment: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(name="prepare_splits")
    )

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)
