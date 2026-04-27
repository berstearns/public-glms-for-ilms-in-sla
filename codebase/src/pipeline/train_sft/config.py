"""Config for pipeline/train_sft."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import yaml

from ilmcloze.config import ConditioningConfig, ModelConfig, TrainConfig
from pipeline._config_common import ExperimentConfig


SECTION_MAP: dict[str, type] = {
    "model": ModelConfig,
    "train": TrainConfig,
    "conditioning": ConditioningConfig,
    "experiment": ExperimentConfig,
}


@dataclass
class TrainSftConfig:
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(name="glm-base", kind="glm", hf_repo="THUDM/glm-335M")
    )
    train: TrainConfig = field(default_factory=lambda: TrainConfig(regime="sft"))
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    experiment: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(name="train_sft")
    )

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)
