"""Config for pipeline/infer."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import yaml

from ilmcloze.config import ConditioningConfig, InferConfig, ModelConfig
from pipeline._config_common import ExperimentConfig


@dataclass
class StageConfig:
    """Empty ``items_jsonl`` = "use build_cloze output".

    ``nwp_l2r``: when model.kind == "nwp", use left-to-right instead of prompt.
    ``mlm_length_unknown``: when model.kind == "mlm", enumerate span lengths.
    """
    items_jsonl: str = ""
    nwp_l2r: bool = False
    mlm_length_unknown: bool = False


SECTION_MAP: dict[str, type] = {
    "model": ModelConfig,
    "conditioning": ConditioningConfig,
    "infer": InferConfig,
    "stage": StageConfig,
    "experiment": ExperimentConfig,
}


@dataclass
class InferStageConfig:
    model: ModelConfig = field(
        default_factory=lambda: ModelConfig(name="glm-base", kind="glm", hf_repo="THUDM/glm-335M")
    )
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    stage: StageConfig = field(default_factory=StageConfig)
    experiment: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(name="infer")
    )

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)
