# Variant 05 — BERT / MLM baseline

**Encoder-only masked LM · single-token + iterative multi-token · both contexts · zero-shot + SFT**

## Axes

| Axis | Setting |
|---|---|
| Backbone | (c) encoder-only MLM (BERT-Large, RoBERTa-Large, or DeBERTa-v3-large). Multi-token handled via **iterative unmasking** — predict each `[MASK]` in turn, conditioning on previously filled predictions |
| Gap type | (i) single-token and (ii) multi-token (iterative) |
| Context condition | (I) and (II) |
| Training regime | zero-shot and continued MLM pretraining + SFT on cloze triples |

## Role

Bidirectional **non-autoregressive** baseline. The GLM paper argues MLM is
worse for multi-token blanks because BERT must either know the length
(enumerate) or use independence assumptions across masked positions
(Du et al., 2022, §2.4, comparison with BERT). This variant tests that
argument empirically on *learner* multi-token gaps.

## Protocol

- Length-known variant: feed the correct number of `[MASK]` tokens.
- Length-unknown variant: enumerate span lengths 1–8 and pick the length
  that maximises mean log-probability.
- Metrics identical to variant 01.

## Expected outcome

- Length-known MLM is competitive on single-token gaps.
- Multi-token iterative MLM underperforms GLM because it cannot model
  dependencies between filled tokens without autoregressive decoding.
- The gap widens on learner context (II) because MLM, trained only on
  clean text, is even more sensitive to upstream/downstream noise than GLM.

## Paper role

Second strongest baseline. Used to argue that the win over NWP is **not
just "bidirectional attention"** — MLM also has that — but **"bidirectional
attention + autoregressive span generation"**, which is GLM's specific
contribution.
