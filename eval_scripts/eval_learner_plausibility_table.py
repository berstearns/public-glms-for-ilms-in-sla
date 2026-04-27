"""Standalone eval: per-(model, dataset) learner-plausibility table.

Columns:
    model | dataset | n | learner_top1 | learner_top1_acc | mean_logprob_learner_gold

`learner_top1` = number of records where `predicted_filler == learner_gold_filler`.
`learner_top1_acc` = fraction.
`mean_logprob_learner_gold` = mean of `predicted_logprob` across the same
records (i.e., the model's confidence on the cases where it matched the
learner's actual filler). NaN if no records have a logprob.

Records lacking `learner_gold_filler` are skipped.

Usage:
    python -m eval_scripts.eval_learner_plausibility_table \\
        --input runs/<id>/predictions.jsonl \\
        --out tables/learner_plausibility.csv
"""

from __future__ import annotations

import argparse
import math
import statistics
from pathlib import Path

from eval_scripts._io import group_by, load_records, write_csv


_FIELDS = ["model", "dataset", "n", "learner_top1", "learner_top1_acc", "mean_logprob_learner_gold"]


def _eq(a: object, b: object) -> bool:
    if not isinstance(a, str) or not isinstance(b, str):
        return False
    return a.strip() == b.strip()


def build_rows(records: list[dict]) -> list[dict]:
    rows: list[dict] = []
    by_md = group_by(records, "model", "dataset")
    for (model, dataset), items in sorted(by_md.items(), key=lambda kv: (kv[0][0] or "", kv[0][1] or "")):
        scored = [r for r in items if r.get("learner_gold_filler") is not None]
        n = len(scored)
        matched = [r for r in scored if _eq(r.get("predicted_filler"), r.get("learner_gold_filler"))]
        logprobs = [
            float(r["predicted_logprob"])
            for r in matched
            if isinstance(r.get("predicted_logprob"), (int, float))
        ]
        mean_lp = statistics.fmean(logprobs) if logprobs else math.nan
        rows.append({
            "model": model,
            "dataset": dataset or "",
            "n": n,
            "learner_top1": len(matched),
            "learner_top1_acc": (len(matched) / n) if n else 0.0,
            "mean_logprob_learner_gold": mean_lp,
        })
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="predictions.jsonl")
    ap.add_argument("--out", type=Path, required=True, help="output CSV path")
    args = ap.parse_args(argv)
    rows = build_rows(load_records(args.input))
    n = write_csv(args.out, _FIELDS, rows)
    print(f"learner plausibility table: wrote {n} rows to {args.out}")


if __name__ == "__main__":
    main()
