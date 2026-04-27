# Variant 06 — L2-error-loci cloze evaluation

**GLM + learner-conditioning · gaps at L2 error loci (articles, prepositions, verb morphology, agreement) · learner context (II) · continued-PT**

## Axes

| Axis | Setting |
|---|---|
| Backbone | (e) GLM + learner-conditioning (same checkpoint as variant 01) |
| Gap type | (iv) gaps placed at L2-error-loci: determiners, prepositions, verb morphology, subject–verb agreement targets |
| Context condition | (II) learner |
| Training regime | inherits variant-01 checkpoint; no extra training |

## Role

**The SLA-content evaluation.** Construct validity for learner simulation:
does the ILM behave differently at gaps that SLA research identifies as
acquisition-sensitive versus at generic gaps? The predictions are specific
(Goldschneider & DeKeyser 2001; Pienemann 1998) and fall out of the same
framework used in the IRAISE-2026 sketch and the CMCL-2026 artificial
learners paper.

## Gap construction

We use ERRANT-style tagging over EFCAMDAT gold-corrected text to identify
token positions corresponding to:

- `DET` — determiner (a/an/the/∅).
- `PREP` — preposition.
- `VERB:FORM` / `VERB:TENSE` — verb morphology.
- `VERB:SVA` — subject–verb agreement target.

For each locus type, we sample ~1k gaps stratified by CEFR, place a
`[MASK]` at the locus, and run all systems.

## Metric

- Per-locus exact-match, LP, KL.
- **Per-locus-per-CEFR** heatmap: the empirical claim is that ILM's
  learner-plausibility at `DET` and `VERB:SVA` is highest at A2/B1 and
  decreases toward C2 — matching the natural-order prediction that these
  loci stop being a production challenge at higher proficiency.

## Expected outcome

- ILM produces a plausibly SLA-shaped response surface: high LP at
  acquisition-sensitive loci for lower CEFR, low LP at the same loci for
  C1/C2. NWP baselines produce a *uniform* LP surface — they do not
  differentiate.
- This result is the **construct-validation half** of the paper, mirroring
  the CMCL-2026 validation of artificial learners as ERRANT-distribution
  simulators.

## Paper role

Section 6.2 ("Does the ILM behave like a learner?"). Figure 2 (heatmap).
