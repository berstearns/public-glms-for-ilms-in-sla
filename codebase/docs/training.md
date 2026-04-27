# Training

## Continued pretraining (variant 01, variant 02)

`scripts/07_train_ilm.py` + `ilmcloze.train.continued_pretrain.train`.

The GLM objective is applied on-the-fly:

1. Whitespace-tokenise the text.
2. Sample spans until ``mask_budget`` (default 15%) of tokens are masked,
   with span lengths drawn from Poisson(λ=3, default).
3. Replace each span with a single ``[MASK]`` to form Part A, prepend the
   learner-conditioning prefix.
4. Shuffle the span order in Part B; each span is wrapped with ``[START]
   ... [END]``.
5. Compute cross-entropy loss over Part B tokens (teacher-forced via
   ``model(**inputs, labels=...)``).

## Supervised fine-tuning (variant 03, variant 05, variant 09-SFT)

`scripts/08_train_sft_cloze.py` + `ilmcloze.train.sft.train_sft`.

Reads the pre-materialised cloze JSONL (`05_build_cloze_dataset`) and
fine-tunes on (Part A, Part B) pairs without on-the-fly span masking.

## Determinism

- Seeds set globally via `utils.seed.set_all(train_cfg.seed)`.
- `torch.use_deterministic_algorithms(True, warn_only=True)` and
  `torch.backends.cudnn.deterministic = True`.
- Data order shuffles use a Python `random.Random(seed)` instance passed
  explicitly through the call graph.

## Checkpointing

The final checkpoint lands under `artifacts/<exp>/07_train_ilm/final/`.
Intermediate checkpoints are at `artifacts/<exp>/07_train_ilm/step-<N>/`.
The path is written to `artifacts/<exp>/07_train_ilm/checkpoint.txt` for
downstream stages to pick up.
