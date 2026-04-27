# Variant 03 — NWP prompt fill-the-blank baseline

**Decoder-only NWP · prompt-based fill-the-blank · both gap lengths · both contexts · zero-shot + SFT**

## Axes

| Axis | Setting |
|---|---|
| Backbone | (b) decoder-only NWP (GPT-2, Llama-3-8B, or equivalent) with prompt-based "fill the blank" formatting |
| Gap type | (i) single-token and (ii) multi-token spans |
| Context condition | (I) clean and (II) learner |
| Training regime | zero-shot and supervised fine-tuning (SFT) on cloze triples |

## Role

**The primary baseline the paper measures itself against.** This is the
strongest off-the-shelf way to do cloze with a decoder-only model, and it is
the one most likely to be used in practice by someone building an artificial
learner for fill-the-gap exercises today.

## Prompt template (zero-shot)

```
Fill the blank in the sentence below with the most likely word(s).
Return only the filler.
Sentence: <left context> ____ <right context>
Filler:
```

For learner-conditioned zero-shot, the prompt is prepended with:
```
The writer is a learner of English with L1=<lang>, CEFR level <level>.
Produce the word(s) this learner is most likely to write.
```

## SFT protocol

- Training data: cloze triples (context, gap span, learner filler) built
  from `norm-EFCAMDAT-train+remainder.csv`.
- Loss: CE over filler tokens, teacher-forced.
- Same total optimisation budget as variant 01 for fair comparison.

## Protocol

1. Zero-shot evaluation with and without learner-conditioning in the prompt.
2. SFT on EFCAMDAT cloze triples; same held-out EFCAMDAT-test set as
   variant 01.
3. Report the same metrics as variant 01.

## Expected outcome

- Zero-shot NWP: brittle under condition (II) and multi-token gaps.
  Calibration is especially poor — NWP concentrates mass on the modal
  native answer regardless of CEFR.
- SFT NWP: closes the exact-match gap on (I) but not on (II);
  learner-plausibility and calibration remain below the ILM of variant 01.

## Paper role

Baseline line in Tables 1–3. Headline contrast in the abstract:
`ILM − NWP-SFT` on LP and KL is the number the paper is selling.
