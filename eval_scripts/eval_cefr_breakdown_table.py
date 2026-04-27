"""Standalone eval: per-(model, dataset, CEFR) learner-plausibility breakdown.

Columns: model | dataset | cefr | n | learner_top1 | learner_top1_acc

This is the CEFR-stratified view of `eval_learner_plausibility_table`.
Used for the cross-corpus transfer table when paired with multiple
dataset labels.

Usage:
    python -m eval_scripts.eval_cefr_breakdown_table \\
        --input runs/<id>/predictions.jsonl \\
        --out tables/cefr_breakdown.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_scripts._io import group_by, load_records, write_csv


_FIELDS = ["model", "dataset", "cefr", "n", "learner_top1", "learner_top1_acc"]
_CEFR_ORDER = {lvl: i for i, lvl in enumerate(["A1", "A2", "B1", "B2", "C1", "C2"])}


def _eq(a: object, b: object) -> bool:
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    return a.strip() == b.strip()


def _sort_key(key: tuple) -> tuple:
    model, dataset, cefr = key
    return (model or "", dataset or "", _CEFR_ORDER.get(cefr or "", 99))


def build_rows(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    grouped = group_by(records, "model", "dataset", "cefr")
    for (model, dataset, cefr), items in sorted(grouped.items(), key=lambda kv: _sort_key(kv[0])):
        scored = [r for r in items if r.get("learner_gold_filler") is not None]
        n = len(scored)
        matched = sum(1 for r in scored if _eq(r.get("predicted_filler"), r.get("learner_gold_filler")))
        rows.append({
            "model": model,
            "dataset": dataset or "",
            "cefr": cefr or "",
            "n": n,
            "learner_top1": matched,
            "learner_top1_acc": (matched / n) if n else 0.0,
        })
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="predictions.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="output CSV path")
    args = ap.parse_args(argv)
    rows = build_rows(load_records(args.input))
    n = write_csv(args.out, _FIELDS, rows)
    print(f"CEFR breakdown table: wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
