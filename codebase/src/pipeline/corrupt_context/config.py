"""Config for pipeline/corrupt_context."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import yaml

from ilmcloze.config import ClozeConfig
from pipeline._config_common import ExperimentConfig


@dataclass
class StageConfig:
    """Empty ``source_jsonl`` = "use build_cloze output"."""
    source_jsonl: str = ""


SECTION_MAP: dict[str, type] = {
    "cloze": ClozeConfig,
    "stage": StageConfig,
    "experiment": ExperimentConfig,
}


@dataclass
class CorruptContextConfig:
    cloze: ClozeConfig = field(default_factory=ClozeConfig)
    stage: StageConfig = field(default_factory=StageConfig)
    experiment: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(name="corrupt_context")
    )

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)
