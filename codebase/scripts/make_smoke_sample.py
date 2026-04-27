#!/usr/bin/env python
"""make_smoke_sample — deterministically sample ~100 EFCAMDAT rows.

CEFR-stratified: 20/level for A1..B2, + whatever C1/C2 can offer, capped at
~100 total. Writes ``codebase/samples/efcamdat-smoke-100.csv``.

This file is an explicit, version-controllable sample — regenerating it
produces the same rows by seed, and downstream smoke-test scripts read from
it without touching the full splits directory.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

DEFAULT_SOURCE = Path(
    "./data/splits/"
    "norm-EFCAMDAT-test.csv"
)


def build_sample(source: Path, out: Path, per_level: int, seed: int) -> pd.DataFrame:
    df = pd.read_csv(source)
    parts: list[pd.DataFrame] = []
    for lvl, g in df.groupby("cefr_level"):
        n = min(per_level, len(g))
        parts.append(g.sample(n=n, random_state=seed))
    sample = pd.concat(parts, ignore_index=True).sort_values(
        ["cefr_level", "writing_id"]
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    sample.to_csv(out, index=False)
    return sample


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "samples" / "efcamdat-smoke-100.csv",
    )
    p.add_argument("--per-level", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    sample = build_sample(args.source, args.out, args.per_level, args.seed)
    print(f"Wrote {len(sample)} rows to {args.out}")
    print(sample["cefr_level"].value_counts().to_string())


if __name__ == "__main__":
    main()
