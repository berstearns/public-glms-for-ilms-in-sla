"""Config for pipeline/download_data."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import yaml

from ilmcloze.config import DataConfig
from pipeline._config_common import ExperimentConfig


SECTION_MAP: dict[str, type] = {
    "data": DataConfig,
    "experiment": ExperimentConfig,
}


@dataclass
class DownloadDataConfig:
    data: DataConfig = field(
        default_factory=lambda: DataConfig(splits_dir="")
    )
    experiment: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(name="download_data")
    )

    def to_yaml(self) -> str:
        return yaml.dump(asdict(self), default_flow_style=False, sort_keys=False)
