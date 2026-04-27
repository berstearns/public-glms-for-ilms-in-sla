"""Config for pipeline/train_ilm."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import yaml

from ilmcloze.config import (
    ClozeConfig,
    ConditioningConfig,
    ModelConfig,
    TrainConfig,
)
from pipeline._config_common import ExperimentConfig


SECTION_MAP: dict[str, type] = {
    "model": ModelConfig,
    "train": TrainConfig,
    "conditioning": ConditioningConfig,
    "cloze": ClozeConfig,
    "experiment": ExperimentConfig,
}


@dataclass
class TrainIlmConfig:
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(name="glm-base", kind="glm", hf_repo="THUDM/glm-335M")
    )
    train: TrainConfig = field(default_factory=TrainConfig)
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    cloze: ClozeConfig = field(default_factory=ClozeConfig)
    experiment: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(name="train_ilm")
    )

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)
