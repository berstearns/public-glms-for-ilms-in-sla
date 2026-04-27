"""Aggregate :class:`PredictionRow`s + gold items into score tables.

Emits:
* ``summary.csv`` — overall metrics.
* ``per_item.jsonl`` — per-item scores (for later stratification).
* ``by_cefr.csv``, ``by_l1.csv``, ``by_gap_length.csv``,
  ``by_gap_position.csv`` — stratified reports.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ilmcloze.cloze.dataset import ClozeItem
from ilmcloze.eval.metrics import (
    exact_match,
    js_to_empirical,
    kl_to_empirical,
    learner_plausibility,
    top_k_hit,
)
from ilmcloze.eval.stratify import by, gap_position_bin
from ilmcloze.infer import PredictionRow


def score_rows(
    predictions: Iterable[PredictionRow],
    items: Iterable[ClozeItem],
    top_k: tuple[int, ...] = (1, 5),
) -> list[dict]:
    items_by_id = {it.item_id: it for it in items}
    out: list[dict] = []
    for p in predictions:
        it = items_by_id.get(p.item_id)
        if it is None:
            continue
        em = exact_match(p.top_k[0] if p.top_k else [], it.gap_tokens)
        tops = {f"top_{k}": top_k_hit(p.top_k, it.gap_tokens, k=k) for k in top_k}
        lp = learner_plausibility(p.logp_learner, p.logp_native)
        kl = (
            kl_to_empirical(p.logp_empirical, it.empirical_fillers)
            if (p.logp_empirical and it.empirical_fillers)
            else float("nan")
        )
        js = (
            js_to_empirical(p.logp_empirical, it.empirical_fillers)
            if (p.logp_empirical and it.empirical_fillers)
            else float("nan")
        )
        row = {
            "item_id": it.item_id,
            "corpus": it.corpus,
            "cefr": it.meta.get("cefr", "UNK"),
            "l1": it.meta.get("l1", "UNK"),
            "gap_length": len(it.gap_tokens),
            "gap_position": gap_position_bin(it.gap_start, len(it.left) + len(it.right) + len(it.gap_tokens)),
            "condition": it.condition,
            "em": em,
            **tops,
            "lp": lp,
            "kl": kl,
            "js": js,
        }
        out.append(row)
    return out


def _mean(xs: list[float]) -> float:
    vals = [x for x in xs if not (isinstance(x, float) and (np.isnan(x) or np.isinf(x)))]
    return float(np.mean(vals)) if vals else float("nan")


def summarise(rows: list[dict]) -> dict:
    keys = [k for k in rows[0] if k not in {"item_id", "corpus", "cefr", "l1", "gap_length", "gap_position", "condition"}]
    return {k: _mean([r[k] for r in rows]) for k in keys}


def _agg(items: list[dict]) -> dict:
    keys = [k for k in items[0] if k not in {"item_id", "corpus", "cefr", "l1", "gap_length", "gap_position", "condition"}]
    return {k: _mean([r[k] for r in items]) for k in keys}


def emit_report(rows: list[dict], out_dir: str | Path) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pd.DataFrame([summarise(rows)]).to_csv(out / "summary.csv", index=False)

    with (out / "per_item.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")

    by(rows, key_fn=lambda r: str(r["cefr"]), agg_fn=_agg).to_csv(out / "by_cefr.csv", index=False)
    by(rows, key_fn=lambda r: str(r["l1"]), agg_fn=_agg).to_csv(out / "by_l1.csv", index=False)
    by(rows, key_fn=lambda r: str(r["gap_length"]), agg_fn=_agg).to_csv(out / "by_gap_length.csv", index=False)
    by(rows, key_fn=lambda r: str(r["gap_position"]), agg_fn=_agg).to_csv(out / "by_gap_position.csv", index=False)
