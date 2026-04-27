# Variant 10 — Cross-corpus transfer

**GLM + learner-conditioning · multi-token spans · learner context (II) · EFCAMDAT-trained → CELVA-SP, KUPA-KEYS, andrew100k**

## Axes

| Axis | Setting |
|---|---|
| Backbone | (e) GLM + learner-conditioning (variant 01 checkpoint) |
| Gap type | (ii) multi-token spans |
| Context condition | (II) learner |
| Training regime | continued-PT on EFCAMDAT only; **no adaptation** to target corpus |

## Role

Generalisation evidence: does the ILM transfer to out-of-distribution
learner corpora across L1 compositions, tasks, and proficiency
distributions?

## Target corpora

From `./data/splits/`:

- `norm-CELVA-SP.csv` — academic-context learner writing.
- `norm-KUPA-KEYS.csv` — process-logged writing.
- `norm-andrew100k-{train,test}-label.csv` — large mixed learner set.
- `norm-universal-cefr-label.csv` — meta-corpus with multiple sources.

For each, construct the same multi-token cloze test set used in-domain on
EFCAMDAT.

## Conditioning prefix for transfer

- `CEFR` — taken from the target corpus gold CEFR label.
- `L1` — taken from target corpus metadata where present; else `UNK`.
- `ERRPROF` — imputed from the first 300 tokens of the learner's text,
  using a frozen ERRANT-profile clusterer fit on EFCAMDAT.

## Protocol

1. Evaluate variant-01 checkpoint directly on each transfer corpus (no
   weight updates).
2. Evaluate variant-03 NWP-SFT baseline identically.
3. Report EM, LP, KL, per-CEFR scores.
4. Degradation bars: `metric(target) − metric(EFCAMDAT-test)`.

## Hypothesis

- ILM degrades less than NWP-SFT on LP and KL — the distribution-level
  metrics transfer better than the mode-level metric.
- Largest degradation on CELVA-SP (register shift: academic) — consistent
  with the finding in the IRAISE-2026 sketch that academic register is a
  known blind spot of EFCAMDAT-trained artificial learners.
- KUPA-KEYS degrades least — general learner writing, close to EFCAMDAT.
- ERRPROF-imputed conditioning recovers some of the degradation vs an
  ablation where ERRPROF is set to `UNK` — concrete evidence that learner
  conditioning is doing work, not just decoration.

## Paper role

Section 6.4 + Table 6. Completes the paper's evaluation story. Same narrative
move as the CMCL-2026 → IRAISE-2026 cross-corpus bridge, but for infilling
instead of CEFR classification.
