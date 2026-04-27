# Variant 04 — NWP left-to-right continuation baseline

**Decoder-only NWP · left-to-right continuation up to the gap · single-token · clean context · zero-shot**

## Axes

| Axis | Setting |
|---|---|
| Backbone | (a) decoder-only NWP with left-to-right continuation — the gap is treated as "what comes next" after the left context, the right context is **discarded** |
| Gap type | (i) single-token |
| Context condition | (I) clean |
| Training regime | zero-shot |

## Role

The **weakest** baseline, included to demonstrate the quantitative gain
from having access to the right context of the gap at all. This is the
configuration implicit in pure language-model-as-learner work that does not
explicitly model infilling.

## Protocol

- Input: tokens left of the gap only.
- Prediction: greedy top-1 next token = model's filler.
- Evaluation: exact-match on the learner filler; LP against learner vs
  native fillers.

## Expected outcome

Severe degradation relative to (b), (c), (d), (e) whenever the gap is
non-initial and the right context is informative (which it almost always
is in natural text). Useful mainly to anchor the "bidirectionality matters"
story.

## Paper role

Table 1 row; also referenced in §1 as the motivating strawman ("NWP as
literally next-word prediction cannot see the right context, so the problem
is trivially mis-specified; (b) recovers some of this but with well-known
pathologies").
