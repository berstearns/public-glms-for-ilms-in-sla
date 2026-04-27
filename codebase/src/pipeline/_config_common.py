"""Shared config sections.

Per-stage ``config.py`` files compose these with their own sections. Every
stage has an ``experiment`` section, and the science-side sections
(``data``, ``cloze``, ``conditioning``, ``model``, ``train``, ``infer``,
``eval``) are re-used directly from :mod:`ilmcloze.config` — those frozen
dataclasses are the source of truth for what the business-logic functions
accept, and the CLI override machinery handles frozen sections via
``dataclasses.replace``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    """Identity + output settings.

    ``output_dir`` + ``name`` determine where artifacts land for this
    experiment: ``{output_dir}/{name}/{stage_key}/``. ``seed`` is global
    (numpy/torch/python); ``device`` is forwarded to torch-side builders.
    """
    name: str = "run"
    output_dir: str = "./artifacts"
    save_config_snapshot: bool = True
    verbose: bool = False
    seed: int = 42
    device: str = "cpu"
