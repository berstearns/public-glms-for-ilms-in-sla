# Overview

The `ilmcloze` codebase implements the research programme described in the
paper *From Next-Word Prediction to Interlanguage Infilling: Adapting GLMs for
Learner Cloze Simulation*.

## Design principles

1. **Standalone scripts.** Each research step is a top-level script in
   `scripts/` with a clear role and `--config` argument. Pipelines compose by
   invoking scripts in order.
2. **Config-driven.** Every run is fully specified by a YAML under
   `configs/experiments/`. Configs resolve to frozen dataclasses — downstream
   code never touches raw dicts.
3. **Typed, testable library.** Shared logic lives under `src/ilmcloze/`
   (importable, `mypy --strict`-compliant). Scripts are thin CLI adapters.
4. **Artifact caching.** Stage outputs land under `artifacts/<experiment>/
   <stage>/`. A `run.json` sidecar records git SHA, resolved config, and
   wall-clock time. Re-runs only recompute invalidated stages.
5. **Reproducibility.** Global seeds via `ilmcloze.utils.seed.set_all`.
   Pinned HF model revisions. Deterministic torch flags. Split CSVs are
   addressed by path and never resampled.
6. **Separation of backbones.** GLM / NWP / MLM all implement the same
   minimal `Backbone` protocol; the cloze formatting layer is
   model-agnostic so a single cloze JSONL feeds all three.

## The research pipeline

```
00  verify splits directory
01  stratified sampling (EFCAMDAT)
02  GEC correction (CoEdit)                    → native fillers + cond I context
03  ERRANT tagging of (learner, clean) pairs   → profile vectors
04  k-means over profile vectors               → ERRPROF bucket ids
05  cloze dataset construction                 → JSONL of ClozeItems
06  synthetic context corruption               → condition III counterpart
07  continued pretraining of GLM (on-the-fly span masking + cond prefix)
08  supervised fine-tuning (alt regime)
09  inference (GLM / NWP / MLM)
10  scoring (EM, top-k, LP, KL, JS) + stratification
11  transfer to CELVA-SP / KUPA-KEYS / andrew100k
12  plots
```

See [`reproducing-variants.md`](reproducing-variants.md) for per-variant
commands.
