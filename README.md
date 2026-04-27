# GLMs for Interlanguage Models (ILMs) in SLA

Sketch for an EMNLP submission proposing **GLM-style autoregressive blank
infilling** (Du et al., 2022) as a principled backbone for modelling second
language learner text in **fill-the-gap / cloze** tasks.

## Thesis

Next-word-prediction (NWP) decoder-only models are the de-facto backbone for
*artificial learners* in SLA, but they are poorly suited to the tasks teachers
and assessment systems actually use to probe learner knowledge: **cloze** and
other fill-the-gap items. NWP cannot condition on the right context of the
gap, it is biased toward fluent target-language completions rather than
learner-plausible completions, and its cloze accuracy collapses when the gap
is non-initial or when the surrounding context contains learner errors. The
GLM objective — autoregressive generation of span-masked blanks with
bidirectional attention over the corrupted context — directly addresses all
three pathologies, and with lightweight *learner-conditioning* (L1, CEFR
level, error profile as a prefix) it becomes an **Interlanguage Language
Model** (ILM): a model that predicts not the native filler, but the
distribution of fillers a learner at a specified proficiency would plausibly
produce.

## Three-part narrative

1. **Diagnosis.** NWP fails as an infilling backbone in learner context.
   Quantify across gap position, gap length, and context noise.
2. **Proposal.** Adapt the original GLM autoregressive blank-infilling
   objective to learner text. Add learner-conditioning prefixes (L1, CEFR,
   error profile) to turn a GLM into an ILM.
3. **Evaluation.** EFCAMDAT as in-domain training and test; cross-corpus
   transfer to CELVA-SP, KUPA-KEYS, andrew100k. Metrics beyond token accuracy:
   **learner-plausibility** and **calibration against the empirical filler
   distribution** at matched CEFR level.

## Structure

```
.
├── main.tex                       # EMNLP draft assembling sections/*.tex
├── acl.sty, acl_natbib.bst        # ACL/EMNLP style files
├── 2103.10360v2.pdf               # GLM paper (Du et al., 2022)
├── sections/                      # Markdown drafts of each section
├── experiments-ideas/             # One .md per combinatorial experiment variant
├── bibs/                          # Bibliography
├── tables/, figures/              # Assets
├── metadata/                      # Title, authors, submission meta
├── tasks/                         # Claude Code task files
└── docs/                          # Writing guidelines
```

## Source projects

| Source | Role here |
|---|---|
| `<source-project-1>/` | NWP baselines, metadata-conditioning framing |
| `<source-project-2>/` | Artificial-learner validation and evaluation philosophy |
| `<source-project-3>/` | SLA-grounded error analysis; construct validation protocol |
| `<source-writings>/` | Overarching framing of learner modelling |
| `./data/splits/` | Cross-corpus SLA datasets (EFCAMDAT, CELVA-SP, KUPA-KEYS, andrew100k) |

## Experiment variants

See `experiments-ideas/README.md`. The **main experiment** in the draft is
variant `01-glm-learnercond-multitoken-contextII-efcamdat.md`. The remaining
variants are planned ablations.

## Venue

- Target: **EMNLP 2026** (long paper, anonymous)
- Template: `acl-style-files` (ACL / EMNLP shared LaTeX style)
