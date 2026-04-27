# Variant 01 — Main experiment

**GLM + learner-conditioning · multi-token span gaps · learner-context (II) · continued-pretraining on EFCAMDAT**

## Axes

| Axis | Setting |
|---|---|
| Backbone | (e) GLM-style autoregressive blank infilling + **learner-conditioning prefix** (L1, CEFR, error profile) |
| Gap type | (ii) contiguous multi-token span gaps (length ~ Poisson(λ=3), as in the original GLM) |
| Context condition | (II) authentic learner context — surrounding tokens may contain learner errors |
| Training regime | Continued pretraining on EFCAMDAT with the GLM objective, starting from a public GLM (or GLM-Base) checkpoint |

## Research question

*Does a GLM-style infiller with learner-conditioning more faithfully model the
distribution of fillers a real learner would produce in a cloze exercise,
compared to the strongest NWP infilling baseline?*

More precisely, on multi-token cloze items drawn from authentic learner text
(condition II), does the conditioned ILM beat a prompt-based NWP baseline
(variant 03) on:

1. Learner-plausibility (LP) against real learner fillers;
2. KL to the empirical filler distribution at matched CEFR;
3. Robustness from clean (I) to learner (II) context;
4. Cross-corpus transfer to CELVA-SP / KUPA-KEYS / andrew100k.

## Motivation

- The NWP backbone used in most artificial-learner work cannot see the right
  context of a gap, so it must *guess the future from the past* — which on
  learner texts is precisely the wrong inductive bias, since the signal that
  a learner will produce error $e$ is often *downstream* of the gap (a later
  agreement target, a later disambiguating noun, a later L1-transferred
  calque).
- GLM's bidirectional attention over the corrupted Part A (Du et al. 2022,
  §2.1) removes this limitation for free.
- Learner-conditioning turns the infiller from "what does English look like
  here" to "what does *this* learner's English look like here", which is the
  object CEFR assessment and interlanguage modelling actually care about.

## Model

- **Base checkpoint.** GLM-Base (110M) or GLM-RoBERTa (335M) from
  `THUDM/GLM`. Report both sizes if compute allows; default is GLM-Base.
- **Conditioning prefix.** Prepended to Part A at training and inference:
  ```
  [L1=<lang>] [CEFR=<A1|A2|B1|B2|C1|C2>] [ERRPROF=<k-bucket>] <text with [MASK] spans>
  ```
  where `ERRPROF` is a coarse bucket of the learner's ERRANT error-type
  profile, clustered to 16 buckets following the CMCL-2026 artificial
  learners pipeline. At inference, conditioning values are taken from
  EFCAMDAT metadata; for transfer corpora we use their own CEFR labels and
  impute the error-profile bucket from a held-out fragment.
- **GLM hyperparameters.** Poisson span length λ=3, 15% masking rate (as in
  Du et al., 2022 §2.1.1). Span shuffling retained.

## Data

- **Training.** `norm-EFCAMDAT-train+remainder.csv` — CEFR-stratified.
  Continued pretraining on the native GLM objective (span masking + AR
  generation) applied to learner text. ~80k texts.
- **Held-out test.** `norm-EFCAMDAT-test.csv` — CEFR-stratified sample of
  5k cloze items built as follows: pick spans of length 1–6 at positions
  uniformly sampled over the text (not restricted to sentence-initial).
- **Gold learner fillers.** For each held-out gap, the *original* learner
  token(s) constitute the gold filler. For items where EFCAMDAT contains
  multiple learner responses to a prompt (when prompts repeat across
  learners, as they do in the Englishtown task pool), we aggregate to
  produce an *empirical filler distribution* at matched CEFR.
- **Native reference filler.** A strong NWP infiller (GPT-4-class) or a
  small committee of native models is used to produce the "native target
  filler", which is also used to construct clean (I) context counterfactuals.

## Protocol

1. Build the cloze test set from EFCAMDAT-test.
2. Continued-pretrain GLM on EFCAMDAT-train with the learner-conditioning
   prefix. 3 epochs, effective batch 256, lr 1e-5, linear warm-up 5%, AdamW.
   Mask budget 15% / λ=3 / span shuffle as in the GLM paper.
3. Evaluate the resulting ILM on the held-out cloze test set under (II), and
   on a parallel (I) version where surrounding tokens are replaced with the
   native filler for every downstream gap.
4. Evaluate the same checkpoint on transfer corpora (variant 10).

## Metrics

- Exact-match and top-k accuracy.
- **Learner-plausibility LP** (primary).
- **KL / JS** to empirical filler distribution at matched CEFR.
- Robustness gap: metric(II) − metric(I).
- Per-CEFR-level stratified scores (A1/A2/B1/B2/C1/C2).
- Per-L1 stratified scores (top-5 L1s in EFCAMDAT).

## Baselines

- Variant 03 (NWP prompt fill-the-blank), same data, same conditioning
  prefix format — the **primary baseline**.
- Variant 02 (vanilla GLM, no learner conditioning) — isolates the
  contribution of conditioning.
- Variant 05 (BERT MLM iterative) — bidirectional but non-autoregressive.

## Expected outcome

- ILM ≥ NWP prompt on every metric on (II); the gap is largest on
  learner-plausibility and calibration — the metrics that matter for
  learner simulation.
- Robustness gap (II−I) is smaller for ILM than for NWP.
- Transfer to CELVA-SP / KUPA-KEYS degrades for all systems; ILM degrades
  less, with feature-level (conditioning) transfer explaining most of the
  retained performance.

## Paper role

**This variant is the main experiment in the draft.** Tables 1–3 in §5 and
Figure 1 (calibration plot) come from this variant. Variants 02–10 are
presented as ablations and analysis in §6 and the appendix.
