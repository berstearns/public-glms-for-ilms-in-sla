# Evaluation

`ilmcloze.eval.metrics` implements the paper's metrics; `ilmcloze.eval.report`
aggregates per-item scores into the tables referenced in the draft.

## Metrics

| Metric | Implementation | Role |
|---|---|---|
| Exact match | `metrics.exact_match` | mode-level accuracy on the gold learner filler |
| Top-k | `metrics.top_k_hit` | relaxed accuracy |
| Learner-plausibility (LP) | `metrics.learner_plausibility` | log-odds learner filler vs native filler under the model; positive = model prefers learner |
| KL-to-empirical | `metrics.kl_to_empirical` | distribution-level calibration against the empirical filler distribution at matched CEFR |
| JS-to-empirical | `metrics.js_to_empirical` | symmetric variant of the above |

## Stratification

`ilmcloze.eval.stratify.by` produces per-CEFR, per-L1, per-gap-length, and
per-gap-position breakdowns. Gap-position is binned into
``initial / medial / final`` thirds.

## Reports emitted

`report.emit_report` writes, under `artifacts/<exp>/10_evaluate/`:

- `summary.csv` — overall averages.
- `per_item.jsonl` — per-item scores (long form).
- `by_cefr.csv`, `by_l1.csv`, `by_gap_length.csv`, `by_gap_position.csv`.

## Cross-corpus transfer

`11_transfer_eval.py` runs the same inference + scoring against each corpus
listed in `cfg.transfer_corpora`, emitting per-corpus reports under
`artifacts/<exp>/11_transfer_eval/<corpus>/`.
