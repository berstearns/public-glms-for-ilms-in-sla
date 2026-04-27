"""Registry of pipeline stages for the e2e orchestrator.

Each entry points at the same building blocks the per-stage ``__main__.py``
shims use: a config dataclass + its SECTION_MAP + a runner callable.
Adding a new stage to the e2e pipeline is a single registry entry here.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pipeline.build_cloze.config import SECTION_MAP as BUILD_CLOZE_SECTIONS
from pipeline.build_cloze.config import BuildClozeConfig
from pipeline.build_cloze.runner import run_build_cloze
from pipeline.cluster_errprof.config import SECTION_MAP as CLUSTER_SECTIONS
from pipeline.cluster_errprof.config import ClusterErrprofConfig
from pipeline.cluster_errprof.runner import run_cluster_errprof
from pipeline.corrupt_context.config import SECTION_MAP as CORRUPT_SECTIONS
from pipeline.corrupt_context.config import CorruptContextConfig
from pipeline.corrupt_context.runner import run_corrupt_context
from pipeline.download_data.config import SECTION_MAP as DOWNLOAD_SECTIONS
from pipeline.download_data.config import DownloadDataConfig
from pipeline.download_data.runner import run_download_data
from pipeline.errant_profile.config import SECTION_MAP as ERRANT_SECTIONS
from pipeline.errant_profile.config import ErrantProfileConfig
from pipeline.errant_profile.runner import run_errant_profile
from pipeline.evaluate.config import SECTION_MAP as EVALUATE_SECTIONS
from pipeline.evaluate.config import EvaluateConfig
from pipeline.evaluate.runner import run_evaluate
from pipeline.gec_clean.config import SECTION_MAP as GEC_SECTIONS
from pipeline.gec_clean.config import GecCleanConfig
from pipeline.gec_clean.runner import run_gec_clean
from pipeline.infer.config import SECTION_MAP as INFER_SECTIONS
from pipeline.infer.config import InferStageConfig
from pipeline.infer.runner import run_infer
from pipeline.plot_results.config import SECTION_MAP as PLOT_SECTIONS
from pipeline.plot_results.config import PlotResultsConfig
from pipeline.plot_results.runner import run_plot_results
from pipeline.prepare_splits.config import SECTION_MAP as PREPARE_SECTIONS
from pipeline.prepare_splits.config import PrepareSplitsConfig
from pipeline.prepare_splits.runner import run_prepare_splits
from pipeline.train_ilm.config import SECTION_MAP as TRAIN_ILM_SECTIONS
from pipeline.train_ilm.config import TrainIlmConfig
from pipeline.train_ilm.runner import run_train_ilm
from pipeline.train_sft.config import SECTION_MAP as TRAIN_SFT_SECTIONS
from pipeline.train_sft.config import TrainSftConfig
from pipeline.train_sft.runner import run_train_sft
from pipeline.transfer_eval.config import SECTION_MAP as TRANSFER_SECTIONS
from pipeline.transfer_eval.config import TransferEvalConfig
from pipeline.transfer_eval.runner import run_transfer_eval


@dataclass
class StageSpec:
    key: str
    config_cls: type
    section_map: dict[str, type]
    runner: Callable[[Any, Any], None]


STAGE_REGISTRY: dict[str, StageSpec] = {
    "download-data": StageSpec(
        key="download-data", config_cls=DownloadDataConfig,
        section_map=DOWNLOAD_SECTIONS, runner=run_download_data,
    ),
    "prepare-splits": StageSpec(
        key="prepare-splits", config_cls=PrepareSplitsConfig,
        section_map=PREPARE_SECTIONS, runner=run_prepare_splits,
    ),
    "gec-clean": StageSpec(
        key="gec-clean", config_cls=GecCleanConfig,
        section_map=GEC_SECTIONS, runner=run_gec_clean,
    ),
    "errant-profile": StageSpec(
        key="errant-profile", config_cls=ErrantProfileConfig,
        section_map=ERRANT_SECTIONS, runner=run_errant_profile,
    ),
    "cluster-errprof": StageSpec(
        key="cluster-errprof", config_cls=ClusterErrprofConfig,
        section_map=CLUSTER_SECTIONS, runner=run_cluster_errprof,
    ),
    "build-cloze": StageSpec(
        key="build-cloze", config_cls=BuildClozeConfig,
        section_map=BUILD_CLOZE_SECTIONS, runner=run_build_cloze,
    ),
    "corrupt-context": StageSpec(
        key="corrupt-context", config_cls=CorruptContextConfig,
        section_map=CORRUPT_SECTIONS, runner=run_corrupt_context,
    ),
    "train-ilm": StageSpec(
        key="train-ilm", config_cls=TrainIlmConfig,
        section_map=TRAIN_ILM_SECTIONS, runner=run_train_ilm,
    ),
    "train-sft": StageSpec(
        key="train-sft", config_cls=TrainSftConfig,
        section_map=TRAIN_SFT_SECTIONS, runner=run_train_sft,
    ),
    "infer": StageSpec(
        key="infer", config_cls=InferStageConfig,
        section_map=INFER_SECTIONS, runner=run_infer,
    ),
    "evaluate": StageSpec(
        key="evaluate", config_cls=EvaluateConfig,
        section_map=EVALUATE_SECTIONS, runner=run_evaluate,
    ),
    "transfer-eval": StageSpec(
        key="transfer-eval", config_cls=TransferEvalConfig,
        section_map=TRANSFER_SECTIONS, runner=run_transfer_eval,
    ),
    "plot-results": StageSpec(
        key="plot-results", config_cls=PlotResultsConfig,
        section_map=PLOT_SECTIONS, runner=run_plot_results,
    ),
}
