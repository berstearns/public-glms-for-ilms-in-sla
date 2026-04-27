# Experiment variants

One `.md` per variant. The design is combinatorial along four axes:

- **Backbone / objective.** (a) decoder-only NWP left-to-right
  continuation up to the gap; (b) decoder-only NWP with prompt-based "fill the
  blank"; (c) encoder-only MLM (BERT-style, single-token and iterative
  multi-token); (d) GLM-style autoregressive blank infilling with span
  masking; (e) GLM + learner-conditioning (L1, CEFR, error-profile prefix).
- **Gap type.** (i) single-token; (ii) contiguous multi-token spans;
  (iii) function-word vs content-word; (iv) L2-error-loci (articles,
  prepositions, verb morphology, agreement).
- **Context condition.** (I) clean target-language context; (II) authentic
  learner context with surrounding errors; (III) synthetically corrupted
  context.
- **Training regime.** zero-shot pretrained; continued pretraining on learner
  corpora with the GLM objective; supervised fine-tuning on cloze items.

The full cross-product has $5 \times 4 \times 3 \times 3 = 180$ cells. We
instantiate the **10 cells below** that are either load-bearing for the main
claim or informative ablations.

## Index

| # | File | Backbone | Gap | Context | Regime | Role |
|---|---|---|---|---|---|---|
| 01 | [main experiment](01-glm-learnercond-multitoken-contextII-efcamdat.md) | (e) GLM + L1/CEFR cond | (ii) multi-token span | (II) learner | continued-PT on EFCAMDAT | **main** |
| 02 | [vanilla GLM baseline](02-glm-vanilla-singletoken-contextI-efcamdat.md) | (d) GLM | (i) single-token | (I) clean | continued-PT | isolates blank-infilling vs learner-conditioning |
| 03 | [NWP prompt fill-blank](03-nwp-prompt-fillblank-baseline.md) | (b) NWP prompt | (i) + (ii) | (I) + (II) | zero-shot + SFT | **primary baseline** |
| 04 | [NWP left-to-right](04-nwp-lefttoright-baseline.md) | (a) NWP L→R | (i) | (I) | zero-shot | weakest baseline; isolates bidirectional gain |
| 05 | [BERT MLM baseline](05-bert-mlm-baseline.md) | (c) MLM | (i) + (ii) iterative | (I) + (II) | zero-shot + SFT | bidirectional non-autoregressive baseline |
| 06 | [L2-error-loci gaps](06-glm-learnercond-l2errorloci-contextII.md) | (e) GLM + cond | (iv) L2 loci | (II) learner | continued-PT | tests whether ILM captures SLA-known difficulty |
| 07 | [span-length ablation](07-glm-spanlength-ablation.md) | (e) GLM + cond | (ii) varying λ | (II) learner | continued-PT | Poisson λ sweep (GLM hyperparameter) |
| 08 | [context-condition ablation](08-context-condition-ablation.md) | (e) GLM + cond | (ii) | (I) vs (II) vs (III) | continued-PT | quantifies the bidirectional-context gain under learner noise |
| 09 | [training-regime ablation](09-training-regime-ablation.md) | (e) GLM + cond | (ii) | (II) | zero-shot vs cont-PT vs SFT | isolates contribution of each adaptation stage |
| 10 | [cross-corpus transfer](10-crosscorpus-transfer.md) | (e) GLM + cond | (ii) | (II) | continued-PT on EFCAMDAT | transfer to CELVA-SP / KUPA-KEYS / andrew100k |

## Shared evaluation protocol

All variants report:

1. **Exact-match** and **top-k** accuracy on the held-out gap (k ∈ {1, 5}).
2. **Learner-plausibility**: $\mathrm{LP} = \log p_\theta(\text{filler}_\text{learner} \mid x_\text{corrupt}) - \log p_\theta(\text{filler}_\text{native} \mid x_\text{corrupt})$,
   i.e. the log-odds the model assigns the learner-attested filler relative to
   the native filler. Positive LP = model prefers learner filler.
3. **Filler-distribution calibration**: KL and Jensen–Shannon divergence
   between the model's predictive distribution and the empirical distribution
   of fillers produced by real learners at matched CEFR level, on items where
   multiple learner fillers exist.
4. **Robustness under context noise**: metric degradation from (I) → (II).
5. **Cross-corpus degradation**: EFCAMDAT-test vs CELVA-SP / KUPA-KEYS /
   andrew100k.

## Shared data

- **Primary.** EFCAMDAT; CEFR-stratified train/test split from
  `./data/splits/norm-EFCAMDAT-{train,test,remainder}.csv`.
- **Transfer.** CELVA-SP, KUPA-KEYS, andrew100k (same splits directory).
- **Not a priority.** C4-200M — auxiliary synthetic large-scale source.
  Mention only as a scaling control if wall-clock allows.
