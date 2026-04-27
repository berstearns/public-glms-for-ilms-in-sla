"""Config for pipeline/e2e — orchestration knobs only.

The per-stage sub-configs live alongside in the composite YAML under their
stage keys; the orchestrator slices them out and hydrates each stage's own
config dataclass from the STAGE_REGISTRY.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import yaml

from pipeline._config_common import ExperimentConfig


@dataclass
class PipelineConfig:
    order: list[str] = field(
        default_factory=lambda: [
            "download-data",
            "prepare-splits",
            "gec-clean",
            "errant-profile",
            "cluster-errprof",
            "build-cloze",
            "corrupt-context",
            "train-ilm",
            "train-sft",
            "infer",
            "evaluate",
            "transfer-eval",
            "plot-results",
        ]
    )
    stop_on_error: bool = True
    # comma-separated list of stages to run; if non-empty, filters ``order``.
    # Set via ``--pipeline.only download-data,prepare-splits``.
    only: str = ""


SECTION_MAP: dict[str, type] = {
    "pipeline": PipelineConfig,
    "experiment": ExperimentConfig,
}


@dataclass
class E2EConfig:
    pipeline: PipelineConfig = field(default_factory=PipelineConfig)
    experiment: ExperimentConfig = field(
        default_factory=lambda: ExperimentConfig(name="e2e_full")
    )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_yaml(self) -> str:
        return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
