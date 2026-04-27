"""Convenience wrapper: produce every standalone glms-for-ilms eval table.

Usage:
    python -m eval_scripts.run_all_tables \\
        --input runs/<id>/predictions.jsonl \\
        --out_dir tables/
"""

from __future__ import annotations

import argparse
from pathlib import Path

from eval_scripts import (
    eval_cefr_breakdown_table,
    eval_cloze_accuracy_table,
    eval_learner_plausibility_table,
)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="predictions.jsonl")
    ap.add_argument("--out_dir", type=Path, required=True, help="directory to write CSVs into")
    args = ap.parse_args(argv)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    eval_cloze_accuracy_table.main(["--input", str(args.input), "--out", str(args.out_dir / "cloze_accuracy.csv")])
    eval_learner_plausibility_table.main(["--input", str(args.input), "--out", str(args.out_dir / "learner_plausibility.csv")])
    eval_cefr_breakdown_table.main(["--input", str(args.input), "--out", str(args.out_dir / "cefr_breakdown.csv")])


if __name__ == "__main__":
    main()
