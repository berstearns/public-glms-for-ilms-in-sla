# eval_scripts/

Standalone evaluation scripts for the glms-for-ilms paper. Each script
consumes one `predictions.jsonl` and emits one CSV table corresponding
to one paper-ready figure.

These scripts depend only on the Python standard library, so they can
be run anywhere the JSONL is available, without installing the
`ilmcloze` package or any ML dependencies.

## Predictions JSONL schema

One record per (model, item):

```jsonl
{"model": "<name>", "item_id": <int>, "dataset": "<corpus>",
 "cefr": "<A1|A2|B1|B2|C1|C2|null>", "l1": "<L1|null>",
 "predicted_filler": "<str>", "predicted_logprob": <float|null>,
 "native_gold_filler": "<str|null>", "learner_gold_filler": "<str|null>"}
```

Required: `model`, `item_id`, `predicted_filler`. Other fields are
optional; eval scripts skip records gracefully when a needed field is
missing.

## Tables

| Script                                  | Output CSV               | Purpose |
|-----------------------------------------|--------------------------|---------|
| `eval_cloze_accuracy_table.py`          | `cloze_accuracy.csv`     | top-1 match against native gold filler, per (model, dataset) |
| `eval_learner_plausibility_table.py`    | `learner_plausibility.csv` | top-1 match against learner gold filler + mean logprob, per (model, dataset) |
| `eval_cefr_breakdown_table.py`          | `cefr_breakdown.csv`     | learner-plausibility stratified by CEFR level |

Cross-corpus transfer is read off `cloze_accuracy.csv` /
`learner_plausibility.csv` by aggregating across `dataset` rows for the
same `model`.

## Running one table

```bash
python -m eval_scripts.eval_cloze_accuracy_table \
    --input runs/<id>/predictions.jsonl \
    --out tables/cloze_accuracy.csv
```

## Running every table at once

```bash
python -m eval_scripts.run_all_tables \
    --input runs/<id>/predictions.jsonl \
    --out_dir tables/
```

## Adding a new table

1. Create `eval_scripts/eval_<name>_table.py`.
2. Implement `build_rows(records)` and a `main(argv=None)` entry point.
3. Use `from eval_scripts._io import load_records, group_by, write_csv`.
4. Add a smoke test under `tests/test_eval_scripts/`.
5. Add the script to the orchestrator in `run_all_tables.py`.
6. Update the table in this README.

## Emitting predictions

Prediction scripts (in `src/ilmcloze/infer/`, downstream tools, etc.)
should produce JSONL using the canonical schema. Two helpers are
provided:

```python
from eval_scripts.emit import build_record, write_records
from pathlib import Path

records = [
    build_record(
        model="glm-learnercond-ft",
        item_id=item.id,
        predicted_filler=hypothesis.text,
        predicted_logprob=hypothesis.logprob,
        dataset="EFCAMDAT",
        cefr=item.cefr,                  # case-normalised internally
        l1=item.l1,
        native_gold_filler=item.native_gold,
        learner_gold_filler=item.learner_gold,
    )
    for item, hypothesis in run(...)
]
write_records(Path("predictions.jsonl"), records)
```

Both `build_record` and `write_records` raise on schema-invalid output —
fail at the producer rather than ship bad JSONL.

## Validating an existing JSONL

```bash
python -m eval_scripts.schema --input predictions.jsonl
```

Prints `OK: <path>` on a clean file, or per-line issues + non-zero exit
otherwise. Use this in CI or as a `pytest` fixture before running any
eval table.
