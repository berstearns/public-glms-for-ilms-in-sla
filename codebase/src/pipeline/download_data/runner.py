"""Stage 00 — verify splits directory and emit an inventory.

Does NOT redistribute EFCAMDAT / CELVA-SP / KUPA-KEYS / andrew100k. Checks
that the expected CSVs exist under ``data.splits_dir`` and writes
``inventory.json`` + ``data_config.json`` under the stage artifact dir.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

from ilmcloze.io.splits import CORPUS_LOADERS
from pipeline._cli import stage_artifact_dir

from .config import DownloadDataConfig

log = logging.getLogger("pipeline.download_data")


def run_download_data(config: DownloadDataConfig, run_dir: Path) -> None:
    out = stage_artifact_dir(
        config.experiment.output_dir, config.experiment.name, "download_data"
    )
    inventory: dict[str, dict] = {}
    for name, loader in CORPUS_LOADERS.items():
        try:
            df = loader(config.data)
            inventory[name] = {"rows": int(len(df)), "status": "ok"}
        except Exception as exc:  # noqa: BLE001
            inventory[name] = {"rows": 0, "status": f"error: {exc}"}
    (out / "inventory.json").write_text(json.dumps(inventory, indent=2))
    (out / "data_config.json").write_text(json.dumps(asdict(config.data), indent=2))
    log.info("wrote %s", out / "inventory.json")
