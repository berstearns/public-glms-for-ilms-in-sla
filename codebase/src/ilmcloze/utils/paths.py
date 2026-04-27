"""Filesystem conventions for artifacts.

All pipeline outputs land under ``artifacts_dir`` with a stable naming scheme
so downstream stages can locate their inputs without coordination.
"""

from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ilmcloze.config import ExperimentConfig


def artifacts_root(cfg: ExperimentConfig) -> Path:
    """Return the root directory for this experiment's artifacts.

    Controlled by the ``ILMCLOZE_CACHE`` env var if set, else
    ``cfg.artifacts_dir / cfg.name``.
    """
    env = os.environ.get("ILMCLOZE_CACHE")
    root = Path(env) if env else Path(cfg.artifacts_dir)
    out = root / cfg.name
    out.mkdir(parents=True, exist_ok=True)
    return out


def stage_dir(cfg: ExperimentConfig, stage: str) -> Path:
    """Return the per-stage artifact directory; creates it if missing."""
    out = artifacts_root(cfg) / stage
    out.mkdir(parents=True, exist_ok=True)
    return out


def _git_sha() -> str | None:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return sha.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def write_run_manifest(cfg: ExperimentConfig, stage: str, extra: dict[str, Any] | None = None) -> Path:
    """Write a ``run.json`` sidecar capturing config + git + timestamp."""
    out = stage_dir(cfg, stage) / "run.json"
    payload: dict[str, Any] = {
        "stage": stage,
        "experiment": cfg.name,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "config": asdict(cfg),
    }
    if extra:
        payload.update(extra)
    out.write_text(json.dumps(payload, indent=2, default=str))
    return out
