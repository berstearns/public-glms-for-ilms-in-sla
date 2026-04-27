# Variant 08 — Context-condition ablation

**GLM + learner-conditioning · multi-token spans · clean (I) vs learner (II) vs synthetic-corruption (III) · continued-PT**

## Axes

| Axis | Setting |
|---|---|
| Backbone | (e) GLM + learner-conditioning |
| Gap type | (ii) multi-token spans |
| Context condition | (I), (II), (III) — the full context axis |
| Training regime | continued-PT (same checkpoint as variant 01) |

## Role

**Directly quantifies the core claim.** NWP collapses when surrounding
tokens are learner-error-laden; GLM should be more robust; the ablation
says how much, and whether it is because of bidirectionality, learner-
conditioning, or both.

## Context construction

- **(I) Clean.** Replace learner tokens with GEC-corrected native tokens
  (CoEdit-Large or equivalent) everywhere except the gap.
- **(II) Learner.** Use the raw learner text surrounding the gap.
- **(III) Synthetic corruption.** Apply a controlled corruption model to
  the *clean* baseline: token deletion (p=0.05), preposition swap
  (p=0.05), determiner drop (p=0.05), subject–verb disagreement injection
  (p=0.03). Calibrated to match the mean ERRANT error rate of EFCAMDAT-B1.

## Protocol

Evaluate three systems on each of (I), (II), (III):
- Variant 01 (ILM)
- Variant 03 (NWP-SFT prompt)
- Variant 05 (MLM-SFT)

Report: EM, LP, KL, robustness gap (II−I), (III−I).

## Hypothesis

- ILM degrades ~10% relative from (I) → (II); NWP degrades ~30%+; MLM
  degrades ~20%. Synthetic corruption (III) reproduces the ranking,
  confirming that the degradation is *noise-driven* and not
  domain-artifactual.

## Paper role

Main body, Table 4. This is the table that demonstrates NWP's cloze accuracy
collapses when the surrounding context contains learner errors.
