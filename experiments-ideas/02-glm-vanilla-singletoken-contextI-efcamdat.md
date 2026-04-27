# Variant 02 — Vanilla GLM baseline

**GLM (no conditioning) · single-token gaps · clean context (I) · continued-pretraining on EFCAMDAT**

## Axes

| Axis | Setting |
|---|---|
| Backbone | (d) GLM — no learner-conditioning prefix |
| Gap type | (i) single-token |
| Context condition | (I) clean target-language context (surrounding tokens from a native reference) |
| Training regime | Continued pretraining on EFCAMDAT with GLM objective |

## Role

Isolates the contribution of **learner-conditioning** (variant 01 vs 02) and
the contribution of **multi-token spans** (variant 01 vs 02) and **learner
context** (variant 01 vs 02).

## Protocol

Same as variant 01 but:
- No conditioning prefix. Part A is the raw masked text.
- Gaps are single-token only.
- Surrounding context is the native-referenced "cleaned" version.

## Metric of interest

- Absolute exact-match and top-k on single-token gaps (expected to be high
  for all systems; the point is that this is the *ceiling* regime for NWP-
  style infillers).
- Degradation from 02 → 01 setting changes, reported as an attribution table.

## Expected outcome

On single-token clean-context gaps, vanilla GLM ≈ strong NWP prompt baseline.
This is the expected null result for the main claim — the bidirectional
advantage is minimal when both context and gap are small and clean. The
*gap opens up* under learner context (II) and multi-token spans (ii), which
is the headline of the paper.
