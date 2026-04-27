#!/usr/bin/env python
"""backfill_token_counts — retrofit tokenizer-explicit length columns onto
existing ``artifacts/smoke/sample-benchmark-*/`` runs.

Walks each sample-benchmark directory, recomputes gap-length under every
registered tokenizer (see :mod:`ilmcloze.cloze.token_counters`), and writes:

* ``per_item_v2.csv``     — per_item.csv + explicit n_{tokenizer}_tokens columns
* ``by_{field}.csv``      — one file per tokenizer, stratifying EM by length
* ``run_v2.json``         — original run.json + active counter manifest

The original files are left untouched.
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ilmcloze.cloze.dataset import read_items  # noqa: E402
from ilmcloze.cloze.token_counters import (  # noqa: E402
    count_all,
    counter_manifest,
    default_counters,
    reference_field,
)


def backfill(run_dir: Path) -> None:
    per_item_path = run_dir / "per_item.csv"
    cloze_path = run_dir / "cloze.jsonl"
    if not per_item_path.exists() or not cloze_path.exists():
        print(f"  skip {run_dir.name}: missing per_item.csv or cloze.jsonl")
        return

    items_by_id: dict[str, list[str]] = {}
    for it in read_items(cloze_path):
        items_by_id[it.item_id] = it.gap_tokens

    df = pd.read_csv(per_item_path)
    counters = default_counters()
    new_cols: dict[str, list[int]] = {c.field: [] for c in counters}
    for _, row in df.iterrows():
        gap_tokens = items_by_id.get(str(row["item_id"])) or str(row.get("gold", "")).split()
        gap_text = " ".join(gap_tokens)
        for c in counters:
            new_cols[c.field].append(c.count(gap_text))
    for field_name, values in new_cols.items():
        df[field_name] = values

    df.to_csv(run_dir / "per_item_v2.csv", index=False)

    for c in counters:
        (df.groupby(c.field)
           .agg(n=("em", "size"), em_mean=("em", "mean"))
           .reset_index()
           .to_csv(run_dir / f"by_{c.field}.csv", index=False))

    # Augmented run.json
    orig_run = {}
    if (run_dir / "run.json").exists():
        orig_run = json.loads((run_dir / "run.json").read_text())
    orig_run["reference_length_field"] = reference_field()
    orig_run["token_counters"] = counter_manifest()
    (run_dir / "run_v2.json").write_text(json.dumps(orig_run, indent=2, default=str))

    print(f"  ✓ {run_dir.name}: added {len(counters)} length columns; reference={reference_field()}")


def main() -> None:
    roots = list((REPO_ROOT / "artifacts" / "smoke").glob("sample-benchmark-*"))
    roots = [r for r in roots if r.is_dir()]
    print(f"Backfilling {len(roots)} runs with reference={reference_field()}")
    for r in sorted(roots):
        backfill(r)


if __name__ == "__main__":
    main()
