#!/usr/bin/env python
"""build_final_tables — aggregate all smoke runs into the paper's tables.

Walks every ``sample-benchmark-*`` directory, reads ``per_item_v2.csv`` (or
``per_item.csv`` fallback), and produces:

* ``artifacts/smoke/final/table_1_by_decoding.csv``
    Model × decoding-strategy × overall EM.
* ``artifacts/smoke/final/table_2_mlm_none_by_subtok.csv``
    MLM-None systems stratified by **their own** subtoken count of the gold;
    demonstrates EM=0 on any gap with >1 subtoken in that model's tokenizer.
* ``artifacts/smoke/final/table_3_by_nltk_length.csv``
    All systems × NLTK Treebank gold-length bin.

Each table is also printed to stdout as a Markdown-friendly block so it can
be pasted into the paper.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ilmcloze.cloze.token_counters import reference_field  # noqa: E402

SMOKE = REPO_ROOT / "artifacts" / "smoke"
OUT = SMOKE / "final"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Map legacy (pre-strategy-aware) run folders to explicit (model, strategy)
# labels. New runs already encode the strategy in the folder name.

_LEGACY_STRATEGY: dict[str, tuple[str, str]] = {
    # model_hash prefix -> (pretty model, decoding strategy)
    "4e533e18": ("distilbert-base-uncased", "mlm_iterative_confident"),
    "5b7156ae": ("gpt2", "nwp_greedy_l2r"),
    "ebc8a91b": ("distilgpt2", "nwp_greedy_prompt"),
    "908cbd1b": ("glm-roberta-large", "glm_ar_span"),  # (buggy decode; flagged below)
}


_RUN_DIR_RE = re.compile(
    r"sample-benchmark-(?P<hash>[0-9a-f]{8})-(?P<model>[^-]+(?:-[^-]+)*?)-"
    r"(?P<strategy>(?:mlm_none|mlm_iterative_confident|mlm_enumerate_lengths|"
    r"nwp_greedy_prompt|nwp_greedy_l2r|glm_ar_span))-"
    r"(?P<ts>\d{8}T\d{6}Z)$"
)


def parse_run(run_dir: Path) -> dict[str, str] | None:
    m = _RUN_DIR_RE.match(run_dir.name)
    if m:
        return {
            "model": m.group("model"),
            "model_hash": m.group("hash"),
            "decoding_strategy": m.group("strategy"),
            "timestamp": m.group("ts"),
            "legacy": "0",
        }
    # Legacy pattern: sample-benchmark-<hash>-<model>-<ts>
    legacy_re = re.compile(
        r"sample-benchmark-(?P<hash>[0-9a-f]{8})-(?P<model>.+?)-(?P<ts>\d{8}T\d{6}Z)$"
    )
    m = legacy_re.match(run_dir.name)
    if m:
        h = m.group("hash")
        if h not in _LEGACY_STRATEGY:
            return None
        pretty_model, strategy = _LEGACY_STRATEGY[h]
        return {
            "model": pretty_model,
            "model_hash": h,
            "decoding_strategy": strategy,
            "timestamp": m.group("ts"),
            "legacy": "1",
        }
    return None


def load_per_item(run_dir: Path) -> pd.DataFrame | None:
    for name in ("per_item_v2.csv", "per_item.csv"):
        p = run_dir / name
        if p.exists():
            return pd.read_csv(p)
    return None


def _tokenizer_field_for_model(model_name: str, cols: list[str]) -> str | None:
    """Find the per-model subtoken length column in a per_item row, if present."""
    mapping = {
        "distilbert-base-uncased": "n_distilbert_wordpiece_uncased_",
        "bert-base-cased":         "n_bert_wordpiece_cased_",
        "bert-base-uncased":       "n_bert_wordpiece_uncased_",
        "roberta-base":            "n_roberta_bpe_",
        "distilgpt2":              "n_distilgpt2_bpe_",
        "gpt2":                    "n_gpt2_bpe_",
    }
    prefix = mapping.get(model_name)
    if not prefix:
        return None
    for c in cols:
        if c.startswith(prefix) and c.endswith("_tokens"):
            return c
    return None


def main() -> None:
    runs = []
    for d in sorted(SMOKE.glob("sample-benchmark-*")):
        if not d.is_dir():
            continue
        meta = parse_run(d)
        if meta is None:
            continue
        df = load_per_item(d)
        if df is None or df.empty:
            continue
        runs.append((meta, df, d))

    # --- Table 1: model x decoding x overall EM -----------------------------
    rows_t1 = []
    for meta, df, d in runs:
        rows_t1.append({
            "model":             meta["model"],
            "decoding_strategy": meta["decoding_strategy"],
            "n":                 int(len(df)),
            "em_mean_pct":       round(df["em"].mean() * 100, 1),
            "run_dir":           d.name,
            "legacy":            meta["legacy"],
        })
    t1 = pd.DataFrame(rows_t1).sort_values(["decoding_strategy", "model"]).reset_index(drop=True)
    t1.to_csv(OUT / "table_1_by_decoding.csv", index=False)

    print("=" * 88)
    print("TABLE 1 — Overall EM (%) by (model × decoding strategy)")
    print("=" * 88)
    print(t1.to_string(index=False))
    print()

    # --- Table 2: MLM-None by per-model subtoken length --------------------
    rows_t2 = []
    for meta, df, d in runs:
        if meta["decoding_strategy"] != "mlm_none":
            continue
        subtok_col = _tokenizer_field_for_model(meta["model"], df.columns.tolist())
        if subtok_col is None:
            continue
        by_len = df.groupby(subtok_col).agg(n=("em", "size"), em=("em", "mean"))
        total_multi = int((df[subtok_col] > 1).sum())
        em_multi = float(df.loc[df[subtok_col] > 1, "em"].mean()) if total_multi else float("nan")
        em_single = float(df.loc[df[subtok_col] == 1, "em"].mean()) if (df[subtok_col] == 1).sum() else float("nan")
        row = {
            "model":               meta["model"],
            "subtoken_field":      subtok_col,
            "em_single_subtok_pct": round(em_single * 100, 1),
            "n_single_subtok":     int((df[subtok_col] == 1).sum()),
            "em_multi_subtok_pct": round(em_multi * 100, 1),
            "n_multi_subtok":      total_multi,
        }
        for L in sorted(by_len.index):
            row[f"L={int(L)}(n={int(by_len.loc[L,'n'])})"] = round(float(by_len.loc[L, "em"]) * 100, 1)
        rows_t2.append(row)
    t2 = pd.DataFrame(rows_t2)
    t2.to_csv(OUT / "table_2_mlm_none_by_subtok.csv", index=False)

    print("=" * 88)
    print("TABLE 2 — MLM-None (single [MASK], top-1 subtoken) stratified by")
    print("the model's OWN subtoken count of the gold gap.")
    print("em_multi_subtok_pct MUST be 0.0 by construction (1 predicted subtoken < gold).")
    print("=" * 88)
    if len(t2):
        print(t2.to_string(index=False))
    else:
        print("(no MLM-None runs found)")
    print()

    # --- Table 3: every system, stratified by NLTK Treebank length ---------
    ref = reference_field()
    rows_t3 = []
    for meta, df, d in runs:
        if ref not in df.columns:
            continue
        agg = df.groupby(ref).agg(n=("em", "size"), em=("em", "mean"))
        row = {
            "model":             meta["model"],
            "decoding_strategy": meta["decoding_strategy"],
            "overall_em_pct":    round(df["em"].mean() * 100, 1),
        }
        for L in [1, 2, 3, 4, 5]:
            if L in agg.index:
                row[f"L={L}(n={int(agg.loc[L,'n'])})"] = round(float(agg.loc[L, "em"]) * 100, 1)
        rows_t3.append(row)
    t3 = pd.DataFrame(rows_t3).sort_values(
        ["decoding_strategy", "model"]
    ).reset_index(drop=True)
    t3.to_csv(OUT / "table_3_by_nltk_length.csv", index=False)

    print("=" * 88)
    print(f"TABLE 3 — EM (%) by NLTK TreebankWordTokenizer length ({ref})")
    print("=" * 88)
    print(t3.to_string(index=False))


if __name__ == "__main__":
    main()
