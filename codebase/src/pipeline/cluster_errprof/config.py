"""Config for pipeline/cluster_errprof."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import yaml

from ilmcloze.config import ConditioningConfig
from pipeline._config_common import ExperimentConfig


@dataclass
class StageConfig:
    """Empty ``profiles_path`` = "use errant_profile output for this experiment".

    ``k_override`` = 0 means "use ``conditioning.errprof_num_clusters``".
    """
    profiles_path: str = ""
    k_override: int = 0


SECTION_MAP: dict[str, type] = {
    "conditioning": ConditioningConfig,
    "stage": StageConfig,
    "experiment": ExperimentConfig,
}


@dataclass
class ClusterErrprofConfig:
    conditioning: ConditioningConfig = field(default_factory=ConditioningConfig)
    stage: StageConfig = field(default_factory=StageConfig)
    experiment: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(name="cluster_errprof")
    )

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)
