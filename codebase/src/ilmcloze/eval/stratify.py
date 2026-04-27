"""Per-CEFR, per-L1, per-gap-length, per-gap-position stratification."""

from __future__ import annotations

from collections import defaultdict
from typing import Callable, Iterable

import pandas as pd


def by(
    rows: Iterable[dict],
    key_fn: Callable[[dict], str],
    agg_fn: Callable[[list[dict]], dict],
) -> pd.DataFrame:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        groups[key_fn(r)].append(r)
    data: list[dict] = []
    for k, items in sorted(groups.items()):
        summary = {"stratum": k, "n": len(items), **agg_fn(items)}
        data.append(summary)
    return pd.DataFrame(data)


def gap_position_bin(gap_start: int, total_tokens: int) -> str:
    if total_tokens == 0:
        return "initial"
    frac = gap_start / total_tokens
    if frac < 1 / 3:
        return "initial"
    if frac < 2 / 3:
        return "medial"
    return "final"
