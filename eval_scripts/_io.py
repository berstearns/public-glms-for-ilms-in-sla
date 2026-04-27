"""Shared I/O for standalone glms-for-ilms eval scripts.

Predictions JSONL schema (one record per cloze item, per model):

    {
      "model": "<name>",
      "item_id": <int>,
      "dataset": "<corpus_label>",
      "cefr": "<A1|A2|B1|B2|C1|C2|null>",
      "l1": "<L1_label|null>",
      "predicted_filler": "<str>",
      "predicted_logprob": <float|null>,
      "native_gold_filler": "<str|null>",
      "learner_gold_filler": "<str|null>"
    }

Required: `model`, `item_id`, `predicted_filler`. Everything else is
optional — eval scripts that need a missing field skip records gracefully.

Eval scripts only depend on stdlib. They consume `predictions.jsonl`
via `load_records(path)` and emit one CSV table per script.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping


def iter_jsonl(path: Path) -> Iterator[dict]:
    """Yield each non-blank line of a JSONL file as a parsed dict."""
    if not path.exists():
        raise SystemExit(f"jsonl not found at {path}")
    with path.open() as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{path}:{lineno}: invalid JSON ({exc})") from exc


def load_records(path: Path) -> list[dict]:
    """Load all JSONL records into a list. Validates `model` is present."""
    records = list(iter_jsonl(path))
    if not records:
        raise SystemExit(f"{path}: no records")
    bad = [i for i, r in enumerate(records) if not isinstance(r.get("model"), str) or not r["model"]]
    if bad:
        raise SystemExit(f"{path}: {len(bad)} records missing 'model' field (e.g., line {bad[0]+1})")
    return records


def group_by(records: Iterable[Mapping[str, Any]], *keys: str) -> dict[tuple, list[dict]]:
    """Group records by a tuple of fields. Missing fields are stored as None."""
    out: dict[tuple, list[dict]] = {}
    for r in records:
        k = tuple(r.get(field) for field in keys)
        out.setdefault(k, []).append(dict(r))
    return out


def write_csv(out: Path, fieldnames: list[str], rows: Iterable[Mapping[str, Any]]) -> int:
    """Write rows to CSV, creating parent dirs. Returns row count."""
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    with out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)
    return len(rows)
