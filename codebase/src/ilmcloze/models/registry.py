"""Model registry — maps ``ModelConfig.kind`` to a backbone class."""

from __future__ import annotations

from typing import Protocol

from ilmcloze.config import ModelConfig


class Backbone(Protocol):
    """Minimal interface every model backbone satisfies.

    Concrete classes are defined in :mod:`ilmcloze.models.glm`,
    :mod:`ilmcloze.models.nwp`, :mod:`ilmcloze.models.mlm`.
    """

    def load(self) -> None: ...
    @property
    def device(self) -> str: ...


def build(cfg: ModelConfig, device: str = "cpu") -> Backbone:
    """Factory: instantiate a backbone from a :class:`ModelConfig`."""
    if cfg.kind == "glm":
        from ilmcloze.models.glm import GLMBackbone

        return GLMBackbone(cfg=cfg, device=device)
    if cfg.kind == "nwp":
        from ilmcloze.models.nwp import NWPBackbone

        return NWPBackbone(cfg=cfg, device=device)
    if cfg.kind == "mlm":
        from ilmcloze.models.mlm import MLMBackbone

        return MLMBackbone(cfg=cfg, device=device)
    raise ValueError(f"Unknown model kind {cfg.kind!r}")
