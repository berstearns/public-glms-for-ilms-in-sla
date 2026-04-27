# Running non-training variant benchmarks

Of the ten variants under `configs/e2e/`, only **04** has no training stage at
all. For **03 / 05 / 09** the benchmark is "zero-shot" = run the pipeline
with `--pipeline.only` filtering out the `train-*` stages. In zero-shot
mode, `infer` builds the HF base checkpoint directly (no fine-tuned
weights), so skipping the train stage still yields a meaningful inference
+ evaluation run.

## Intrinsically non-training

### Variant 04 — NWP left-to-right (GPT-2, single-token, cond I)

```bash
just variant 04-nwp-lefttoright
```

Stages: `download-data → prepare-splits → build-cloze → infer → evaluate →
plot-results`. Artifacts under `./artifacts/04-nwp-lefttoright/`.

## Training variants run as zero-shot benchmarks

Use `--pipeline.only` to drop the `train-*` stages. `infer` then runs the
untrained HF base. Override `--experiment.name` to keep the zero-shot
results in a separate `./artifacts/<name>/` directory so they don't
clobber later cont-PT / SFT arms.

### Variant 03 — NWP prompt fill-blank (GPT-2, zero-shot arm)

```bash
just variant 03-nwp-prompt \
  --pipeline.only download-data,prepare-splits,gec-clean,build-cloze,infer,evaluate,plot-results \
  --experiment.name 03-nwp-prompt-zeroshot
```

### Variant 05 — BERT MLM (zero-shot arm, iterative multi-token)

```bash
just variant 05-mlm \
  --pipeline.only download-data,prepare-splits,gec-clean,build-cloze,infer,evaluate,plot-results \
  --experiment.name 05-mlm-zeroshot
```

### Variant 09 — training-regime ablation, zero-shot arm

```bash
just variant 09-training-regime \
  --pipeline.only download-data,prepare-splits,gec-clean,errant-profile,cluster-errprof,build-cloze,infer,evaluate,plot-results \
  --experiment.name 09-regime-zeroshot
```

## Inference + evaluation only (on any existing artifacts)

If `build_cloze/cloze.jsonl` already exists under some `experiment.name`
(e.g. you've previously run data-prep for that variant), you can replay
just `infer → evaluate → plot-results` against any variant's model config:

```bash
just variant 01-ilm-main \
  --pipeline.only infer,evaluate,plot-results \
  --experiment.name 01-ilm-main
```

## Overview — which variants involve training

| # | Variant | Has training? | Train stage(s) |
|---|---|---|---|
| 01 | ilm-main                | yes | train-ilm |
| 02 | glm-vanilla             | yes | train-ilm |
| 03 | nwp-prompt              | yes | train-sft (can be skipped for zero-shot) |
| 04 | nwp-lefttoright         | **no** | — |
| 05 | mlm                     | yes | train-sft (can be skipped for zero-shot) |
| 06 | l2errorloci             | yes | train-ilm |
| 07 | span-length             | yes | train-ilm |
| 08 | context-condition       | yes | train-ilm |
| 09 | training-regime         | yes | train-ilm, train-sft (both skippable for zero-shot arm) |
| 10 | crosscorpus-transfer    | yes | train-ilm |

## Peek at results

```bash
just peek
ls ./artifacts/04-nwp-lefttoright/evaluate/      # summary.csv, by_cefr.csv, …
ls ./artifacts/04-nwp-lefttoright/plot_results/  # fig_ranking_bar.png, fig_per_locus_heatmap.png, …
```

Under every experiment root you also get an e2e run-dir:
`./artifacts/{hash}_{timestamp}_<experiment.name>/` with
`config_used.yaml`, `composite_expanded.yaml`, and per-stage
`stages/<key>/effective.yaml` snapshots — the exact config that produced
the numbers, for audit.
