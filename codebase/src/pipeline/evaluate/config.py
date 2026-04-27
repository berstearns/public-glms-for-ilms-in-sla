"""Config for pipeline/evaluate."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import yaml

from ilmcloze.config import EvalConfig, InferConfig
from pipeline._config_common import ExperimentConfig


SECTION_MAP: dict[str, type] = {
    "eval": EvalConfig,
    "infer": InferConfig,
    "experiment": ExperimentConfig,
}


@dataclass
class EvaluateConfig:
    eval: EvalConfig = field(default_factory=EvalConfig)
    infer: InferConfig = field(default_factory=InferConfig)
    experiment: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(name="evaluate")
    )

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)
