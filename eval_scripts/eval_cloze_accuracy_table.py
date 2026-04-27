"""Standalone eval: per-(model, dataset) cloze accuracy table.

Columns: model | dataset | n | correct | accuracy

`correct` = number of records where `predicted_filler == native_gold_filler`
(case-sensitive, exact match after stripping). Records lacking
`native_gold_filler` are skipped.

Usage:
    python -m eval_scripts.eval_cloze_accuracy_table \\
        --input runs/<id>/predictions.jsonl \\
        --out tables/cloze_accuracy.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_scripts._io import group_by, load_records, write_csv


_FIELDS = ["model", "dataset", "n", "correct", "accuracy"]


def _eq(a: object, b: object) -> bool:
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    return a.strip() == b.strip()


def build_rows(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    by_md = group_by(records, "model", "dataset")
    for (model, dataset), items in sorted(by_md.items(), key=lambda kv: (kv[0][0] or "", kv[0][1] or "")):
        scored = [r for r in items if r.get("native_gold_filler") is not None]
        n = len(scored)
        correct = sum(1 for r in scored if _eq(r.get("predicted_filler"), r.get("native_gold_filler")))
        rows.append({
            "model": model,
            "dataset": dataset or "",
            "n": n,
            "correct": correct,
            "accuracy": (correct / n) if n else 0.0,
        })
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="predictions.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="output CSV path")
    args = ap.parse_args(argv)
    rows = build_rows(load_records(args.input))
    n = write_csv(args.out, _FIELDS, rows)
    print(f"cloze accuracy table: wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
