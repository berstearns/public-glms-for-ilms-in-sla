# runs/

Per-run output artifacts: input cloze items, `predictions.jsonl`, plus
the CSV tables emitted by `eval_scripts/`. One subdirectory per
(model, dataset, regime) tuple.

Convention:

    runs/<model_label>-<dataset_label>[-<note>]/

Each subdirectory contains:

    cloze_items.csv         # input items used for the run
    predictions.jsonl       # one record per (model, item)
    tables/
      cloze_accuracy.csv
      learner_plausibility.csv
      cefr_breakdown.csv

Real input data (EFCAMDAT, CELVA-SP, KUPA-KEYS, andrew100k) is not
committed; full-corpus runs pull from the private rclone remote.

## Existing runs

| Subdirectory                         | Model         | Data                                  | n  | What it shows |
|--------------------------------------|---------------|---------------------------------------|----|---------------|
| `nwp-baseline-gpt2-smoke/`           | `gpt2` (124M) | hand-crafted 5-item cloze sample      | 5  | end-to-end harness on real model + ground truth; gpt2-native scores 0/5 on cloze and learner plausibility, illustrating the NWP-baseline failure mode the paper diagnoses |
