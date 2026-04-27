#!/usr/bin/env python
"""rescore_em_variants — backfill EM variants onto existing per_item.csv files.

Reads each run's saved predictions (``gold`` + ``pred`` columns), recomputes:

* ``em_strict``  — original case/punct-sensitive match.
* ``em_ci``      — case-insensitive.
* ``em_norm``    — case+punct-insensitive + NFKC fold.
* ``edit_dist``  — Levenshtein on joined strings.
* ``em_ed<=1``, ``em_ed<=2``, ``em_ed<=3`` — edit distance thresholds.

Writes ``per_item_rescored.csv`` per run + a summary CSV to
``artifacts/smoke/final/em_variants_summary.csv`` + prints the table.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ilmcloze.eval.metrics import (  # noqa: E402
    edit_distance,
    edit_distance_leq,
    exact_match,
    exact_match_case_insensitive,
    exact_match_normalized,
)


def _as_list(s: str | float) -> list[str]:
    if pd.isna(s) or s is None:
        return []
    return str(s).split()


def rescore_run(run_dir: Path) -> pd.DataFrame | None:
    for name in ("per_item_v2.csv", "per_item.csv"):
        path = run_dir / name
        if path.exists():
            df = pd.read_csv(path).copy()
            break
    else:
        return None

    em_strict, em_ci, em_norm, ed = [], [], [], []
    ed1, ed2, ed3 = [], [], []
    for _, row in df.iterrows():
        pred = _as_list(row.get("pred", ""))
        gold = _as_list(row.get("gold", ""))
        em_strict.append(exact_match(pred, gold))
        em_ci.append(exact_match_case_insensitive(pred, gold))
        em_norm.append(exact_match_normalized(pred, gold))
        d = edit_distance(pred, gold)
        ed.append(d)
        ed1.append(edit_distance_leq(pred, gold, 1))
        ed2.append(edit_distance_leq(pred, gold, 2))
        ed3.append(edit_distance_leq(pred, gold, 3))

    df["em_strict"]   = em_strict
    df["em_ci"]       = em_ci
    df["em_norm"]     = em_norm
    df["edit_dist"]   = ed
    df["em_ed_le_1"]  = ed1
    df["em_ed_le_2"]  = ed2
    df["em_ed_le_3"]  = ed3
    df.to_csv(run_dir / "per_item_rescored.csv", index=False)
    return df


# Map legacy runs (no strategy suffix in folder name) to (model, strategy).
_LEGACY: dict[str, tuple[str, str]] = {
    "4e533e18": ("distilbert-base-uncased", "mlm_iterative_confident"),
    "5b7156ae": ("gpt2",                    "nwp_greedy_l2r"),
    "ebc8a91b": ("distilgpt2",              "nwp_greedy_prompt"),
    "908cbd1b": ("glm-roberta-large",       "glm_ar_span"),
}


def parse_run(name: str) -> tuple[str, str] | None:
    parts = name.split("-")
    # sample benchmark <hash> <model...> <strategy> <ts>
    # find strategy token if present
    for cand in ("mlm_none", "mlm_iterative_confident", "nwp_greedy_prompt",
                 "nwp_greedy_l2r", "glm_ar_span"):
        if f"-{cand}-" in name:
            hash_ = parts[2]
            # model = between hash and strategy
            model_part = name.split(f"-{cand}-")[0].split(f"{hash_}-", 1)[1]
            return (model_part, cand)
    # Legacy (no strategy)
    if len(parts) >= 4:
        h = parts[2]
        if h in _LEGACY:
            return _LEGACY[h]
    return None


def main() -> None:
    smoke = REPO_ROOT / "artifacts" / "smoke"
    out_dir = smoke / "final"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for run in sorted(smoke.glob("sample-benchmark-*")):
        if not run.is_dir():
            continue
        meta = parse_run(run.name)
        if meta is None:
            continue
        model, strat = meta
        df = rescore_run(run)
        if df is None or df.empty:
            continue
        row = {
            "model": model,
            "decoding": strat,
            "n": int(len(df)),
            "em_strict_%":  round(df["em_strict"].mean()  * 100, 1),
            "em_ci_%":      round(df["em_ci"].mean()      * 100, 1),
            "em_norm_%":    round(df["em_norm"].mean()    * 100, 1),
            "em_ed<=1_%":   round(df["em_ed_le_1"].mean() * 100, 1),
            "em_ed<=2_%":   round(df["em_ed_le_2"].mean() * 100, 1),
            "em_ed<=3_%":   round(df["em_ed_le_3"].mean() * 100, 1),
            "median_ed":    float(df["edit_dist"].median()),
            "mean_ed":      round(float(df["edit_dist"].mean()), 1),
        }
        rows.append(row)

    t = pd.DataFrame(rows).sort_values(["decoding", "model"]).reset_index(drop=True)
    t.to_csv(out_dir / "em_variants_summary.csv", index=False)

    print("=" * 100)
    print("EM variants — all runs on the 100-item EFCAMDAT smoke")
    print("=" * 100)
    print(t.to_string(index=False))


if __name__ == "__main__":
    main()
