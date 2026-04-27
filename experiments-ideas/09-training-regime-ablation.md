# Variant 09 — Training-regime ablation

**GLM + learner-conditioning · multi-token spans · learner context (II) · zero-shot vs continued-PT vs SFT**

## Axes

| Axis | Setting |
|---|---|
| Backbone | (e) GLM + learner-conditioning |
| Gap type | (ii) multi-token spans |
| Context condition | (II) learner |
| Training regime | three cells: (a) zero-shot public GLM; (b) continued-PT on EFCAMDAT GLM objective (= variant 01); (c) (b) + SFT on EFCAMDAT cloze triples |

## Role

Isolates the contribution of each adaptation stage. Answers: is continued
pretraining enough, or do we need explicit cloze-format SFT on top?

## Protocol

Three checkpoints, same GLM-Base backbone:
- (a) Public `THUDM/glm-roberta-large` or `THUDM/glm-335M` zero-shot. Apply
  the conditioning prefix at inference only.
- (b) Continued-PT on EFCAMDAT with conditioning prefix. 3 epochs.
- (c) (b) followed by SFT on (context, learner-filler) cloze triples,
  teacher-forced on the Part B spans.

Evaluate on EFCAMDAT-test under (II) with multi-token gaps.

## Hypothesis

- (a) zero-shot already beats NWP-SFT on LP because GLM has the right
  *shape* of objective, even without learner adaptation.
- (b) continued-PT gains another margin on LP and calibration.
- (c) SFT gains on exact-match but **hurts LP and KL** — the usual
  over-confidence artefact of teacher-forced SFT. This is important
  evidence for the paper's "density matters more than mode" argument.

## Paper role

Table 5 + §6.3. Supports the recommendation that for *learner simulation*,
continued-PT alone is the right adaptation — SFT improves surface accuracy
at the cost of the distribution-level objective this paper is selling.
