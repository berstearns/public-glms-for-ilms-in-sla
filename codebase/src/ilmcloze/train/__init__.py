"""Training loops.

Two regimes are supported:

* :mod:`ilmcloze.train.continued_pretrain` — continued-pretraining a GLM
  checkpoint on learner text with the native autoregressive-blank-infilling
  objective (λ=3, 15% mask budget, span shuffle).
* :mod:`ilmcloze.train.sft` — supervised fine-tuning on cloze triples
  ``(context, gap, learner_filler)`` teacher-forced on Part B.
"""

from __future__ import annotations
