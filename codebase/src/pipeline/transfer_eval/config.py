"""Config for pipeline/transfer_eval."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import yaml

from ilmcloze.config import (
    ClozeConfig,
    ConditioningConfig,
    DataConfig,
    InferConfig,
    ModelConfig,
)
from pipeline._config_common import ExperimentConfig


@dataclass
class TransferConfig:
    """``corpora`` is the set of cross-corpus targets for this evaluation run."""
    corpora: list[str] = field(default_factory=list)


SECTION_MAP: dict[str, type] = {
    "data": DataConfig,
    "cloze": ClozeConfig,
    "model": ModelConfig,
    "conditioning": ConditioningConfig,
    "infer": InferConfig,
    "transfer": TransferConfig,
    "experiment": ExperimentConfig,
}


@dataclass
class TransferEvalConfig:
    data: DataConfig = field(default_factory=lambda: DataConfig(splits_dir=""))
    cloze: ClozeConfig = field(default_factory=ClozeConfig)
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(name="glm-base", kind="glm", hf_repo="THUDM/glm-335M")
    )
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    transfer: TransferConfig = field(default_factory=TransferConfig)
    experiment: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(name="transfer_eval")
    )

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)
