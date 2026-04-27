"""Cloze primitives.

The cloze pipeline maps raw texts to cloze evaluation items:

1. :mod:`ilmcloze.cloze.gap_sampler` — choose gap positions and lengths.
2. :mod:`ilmcloze.cloze.context` — materialise the surrounding context
   under condition I (clean), II (learner), or III (synthetic corruption).
3. :mod:`ilmcloze.cloze.format` — produce Part A / Part B inputs for GLM and
   equivalent formats for NWP prompt / MLM baselines, with the learner-
   conditioning prefix.
4. :mod:`ilmcloze.cloze.dataset` — :class:`torch.utils.data.Dataset` wrappers
   for training and evaluation.
"""

from __future__ import annotations
