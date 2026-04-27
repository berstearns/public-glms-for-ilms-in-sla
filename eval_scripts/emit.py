"""Reference emitter: turn an `ilmcloze` inference result into a JSONL record.

Use this in any prediction script that wants to feed `eval_scripts/`:

    from eval_scripts.emit import build_record, write_records

    records = (
        build_record(
            model="glm-learnercond-ft",
            item_id=item.id,
            predicted_filler=hyp.text,
            predicted_logprob=hyp.logprob,
            dataset="EFCAMDAT",
            cefr=item.cefr,
            l1=item.l1,
            native_gold_filler=item.native_gold,
            learner_gold_filler=item.learner_gold,
        )
        for item, hyp in pipeline_run(...)
    )
    write_records(Path("predictions.jsonl"), records)

`build_record` validates against `eval_scripts.schema` and raises if the
result would be non-conformant — fail loudly at the producer rather
than ship bad JSONL downstream.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Iterator

from eval_scripts.schema import validate_record


def build_record(
    *,
    model: str,
    item_id: int,
    predicted_filler: str,
    dataset: str | None = None,
    cefr: str | None = None,
    l1: str | None = None,
    predicted_logprob: float | None = None,
    native_gold_filler: str | None = None,
    learner_gold_filler: str | None = None,
    **extra: object,
) -> dict:
    """Build a schema-valid record. Raises ValueError on validation failure.

    `extra` is allowed (forward-compat); unknown keys pass through.
    """
    rec: dict[str, object] = {
        "model": model,
        "item_id": item_id,
        "predicted_filler": predicted_filler,
    }
    if dataset is not None:
        rec["dataset"] = dataset
    if cefr is not None:
        rec["cefr"] = cefr.upper() if isinstance(cefr, str) else cefr
    if l1 is not None:
        rec["l1"] = l1
    if predicted_logprob is not None:
        rec["predicted_logprob"] = float(predicted_logprob)
    if native_gold_filler is not None:
        rec["native_gold_filler"] = native_gold_filler
    if learner_gold_filler is not None:
        rec["learner_gold_filler"] = learner_gold_filler
    rec.update(extra)

    issues = validate_record(rec)
    if issues:
        raise ValueError(f"build_record produced an invalid record: {issues}")
    return rec


def write_records(out: Path, records: Iterable[dict]) -> int:
    """Write JSONL, validating each record. Returns line count."""
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out.open("w") as f:
        for rec in records:
            issues = validate_record(rec)
            if issues:
                raise ValueError(f"refusing to write invalid record at line {n + 1}: {issues}")
            f.write(json.dumps(rec) + "\n")
            n += 1
    return n
