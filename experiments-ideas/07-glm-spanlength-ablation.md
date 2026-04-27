# Variant 07 — GLM span-length (Poisson λ) ablation

**GLM + learner-conditioning · multi-token spans of varying λ · learner context (II) · continued-PT**

## Axes

| Axis | Setting |
|---|---|
| Backbone | (e) GLM + learner-conditioning |
| Gap type | (ii) multi-token spans, length ~ Poisson(λ), λ ∈ {1.5, 3, 5, 7} |
| Context condition | (II) learner |
| Training regime | continued-PT on EFCAMDAT with the chosen λ |

## Role

Hyperparameter ablation over GLM's core pretraining knob. The original GLM
paper (Du et al., 2022, §2.2.1) finds that a 15% mask budget is critical
for NLU downstream tasks; they do not ablate λ for infilling on learner
data. We do.

## Protocol

Four separately continued-pretrained checkpoints, one per λ. Evaluate each
on the same EFCAMDAT-test cloze set, measuring EM/LP/KL as a function of
held-out gap length.

## Hypothesis

- λ=1.5 → better at single-token gaps, worse at longer spans.
- λ=3 → balanced; matches original GLM setting.
- λ=5, 7 → better at long gaps (paragraph-level cloze, multi-clause
  deletions) but worse at short gaps.
- Best **overall LP** at λ=3 or λ=5 on learner texts — the empirical
  question the ablation answers.

## Paper role

Appendix table. The main-experiment variant 01 uses λ=3 as the headline
setting; this variant justifies that choice.
