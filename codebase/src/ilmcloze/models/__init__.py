"""Model backbones.

The registry in :mod:`ilmcloze.models.registry` dispatches on
:class:`~ilmcloze.config.ModelConfig.kind` ∈ {``glm``, ``nwp``, ``mlm``} to
yield a concrete backbone object.
"""

from __future__ import annotations
