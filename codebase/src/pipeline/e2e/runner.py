"""End-to-end orchestrator.

Reads a composite YAML (typically ``configs/e2e/full.yaml``) whose top-level
keys are stage names. For each stage in ``pipeline.order``:

  1. look up the stage in STAGE_REGISTRY;
  2. slice the composite dict's ``{stage_key: …}`` section;
  3. hydrate that into the stage's config dataclass;
  4. snapshot the effective per-stage config under the run_dir;
  5. call the stage's runner.

The composite YAML is loaded with the ``!include`` loader, so each section
is the full per-stage config inlined at load time — the per-stage YAMLs
remain the source of truth.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import yaml

from pipeline._cli import hydrate_config, load_yaml_with_includes
from pipeline._config_common import ExperimentConfig

from .config import E2EConfig
from .stages import STAGE_REGISTRY, StageSpec

log = logging.getLogger("e2e")


def _inherit_experiment(cfg_obj: Any, top_exp: ExperimentConfig) -> None:
    """Propagate the composite-level experiment section into each stage.

    Stages share the artifact dir convention ``{output_dir}/{name}/{stage}/``,
    so their per-stage ``experiment`` sections are placeholders — the single
    source of truth is the composite's top-level ``experiment:``.
    """
    stage_exp = getattr(cfg_obj, "experiment", None)
    if stage_exp is None:
        return
    for f in ("name", "output_dir", "save_config_snapshot", "verbose", "seed", "device"):
        if hasattr(stage_exp, f) and hasattr(top_exp, f):
            setattr(stage_exp, f, getattr(top_exp, f))


def _resolve_order(config: E2EConfig) -> list[str]:
    order = list(config.pipeline.order)
    if config.pipeline.only:
        keep = {s.strip() for s in config.pipeline.only.split(",") if s.strip()}
        order = [s for s in order if s in keep]
    return order


def _snapshot_stage_config(
    run_dir: Path, stage_key: str, sub_raw: dict, spec: StageSpec, cfg_obj: Any
) -> None:
    snapshot_dir = run_dir / "stages" / stage_key
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    if hasattr(cfg_obj, "to_yaml"):
        (snapshot_dir / "effective.yaml").write_text(cfg_obj.to_yaml())
    else:
        (snapshot_dir / "effective.yaml").write_text(
            yaml.dump(sub_raw, default_flow_style=False, sort_keys=False)
        )


def run_e2e(config: E2EConfig, run_dir: Path) -> None:
    composite_path_str = getattr(config, "_composite_path", None)
    if composite_path_str is None:
        raise RuntimeError("run_e2e: composite path not threaded through; see __main__")
    composite_path = Path(composite_path_str)
    raw = load_yaml_with_includes(composite_path)

    order = _resolve_order(config)
    log.info("e2e pipeline: %s", " → ".join(order) or "(empty)")

    for stage_key in order:
        if stage_key not in STAGE_REGISTRY:
            log.warning("unknown stage %s — skipping (not in STAGE_REGISTRY)", stage_key)
            continue
        spec = STAGE_REGISTRY[stage_key]
        sub_raw = raw.get(stage_key)
        if sub_raw is None:
            log.warning("stage %s has no section in composite — skipping", stage_key)
            continue

        log.info("── %s ───────────────────────────────────", stage_key)
        cfg_obj = hydrate_config(sub_raw, spec.config_cls, spec.section_map)
        _inherit_experiment(cfg_obj, config.experiment)

        stage_run_dir = run_dir / "stages" / stage_key
        stage_run_dir.mkdir(parents=True, exist_ok=True)
        _snapshot_stage_config(run_dir, stage_key, sub_raw, spec, cfg_obj)

        try:
            spec.runner(cfg_obj, stage_run_dir)
        except Exception as e:  # noqa: BLE001
            log.error("stage %s failed: %s", stage_key, e)
            if config.pipeline.stop_on_error:
                raise
            log.warning("stop_on_error=false — continuing")

    log.info("✓ e2e done.")
