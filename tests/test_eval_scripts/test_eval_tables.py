"""Smoke tests for the three glms-for-ilms eval-table scripts."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import pytest

from eval_scripts import (
    eval_cefr_breakdown_table,
    eval_cloze_accuracy_table,
    eval_learner_plausibility_table,
    run_all_tables,
)


def _read(path: Path) -> list[dict]:
    with path.open() as f:
        return list(csv.DictReader(f))


def test_cloze_accuracy(predictions_jsonl_path: Path, tmp_path: Path):
    out = tmp_path / "cloze.csv"
    eval_cloze_accuracy_table.main(["--input", str(predictions_jsonl_path), "--out", str(out)])
    rows = _read(out)
    by_md = {(r["model"], r["dataset"]): r for r in rows}
    # glm-ft on EFCAMDAT: 3 items, native gold matched on item 0 only
    r = by_md[("glm-ft", "EFCAMDAT")]
    assert int(r["n"]) == 3
    assert int(r["correct"]) == 1
    assert float(r["accuracy"]) == pytest.approx(1 / 3)
    # glm-ft on CELVA-SP: 1 item, matched
    r = by_md[("glm-ft", "CELVA-SP")]
    assert int(r["n"]) == 1
    assert int(r["correct"]) == 1


def test_learner_plausibility(predictions_jsonl_path: Path, tmp_path: Path):
    out = tmp_path / "lp.csv"
    eval_learner_plausibility_table.main(["--input", str(predictions_jsonl_path), "--out", str(out)])
    rows = _read(out)
    by_md = {(r["model"], r["dataset"]): r for r in rows}
    # glm-ft on EFCAMDAT: 3 items, learner gold matched on items 0 and 1 only
    # (item 2: predicted "very", learner gold "much" — no match)
    r = by_md[("glm-ft", "EFCAMDAT")]
    assert int(r["n"]) == 3
    assert int(r["learner_top1"]) == 2
    assert float(r["learner_top1_acc"]) == pytest.approx(2 / 3)
    # mean_logprob_learner_gold = mean over matched items: -0.2, -0.5
    assert float(r["mean_logprob_learner_gold"]) == pytest.approx((-0.2 - 0.5) / 2)


def test_learner_plausibility_nan_when_no_matches(tmp_path: Path):
    """If no learner-gold matches, mean_logprob is NaN (serialized as 'nan')."""
    p = tmp_path / "x.jsonl"
    p.write_text(
        '{"model": "m", "item_id": 0, "dataset": "d", "predicted_filler": "wrong", "learner_gold_filler": "right"}\n'
    )
    out = tmp_path / "lp.csv"
    eval_learner_plausibility_table.main(["--input", str(p), "--out", str(out)])
    rows = _read(out)
    assert int(rows[0]["learner_top1"]) == 0
    assert rows[0]["mean_logprob_learner_gold"] in {"nan", "NaN"}


def test_cefr_breakdown(predictions_jsonl_path: Path, tmp_path: Path):
    out = tmp_path / "cefr.csv"
    eval_cefr_breakdown_table.main(["--input", str(predictions_jsonl_path), "--out", str(out)])
    rows = _read(out)
    by_mdc = {(r["model"], r["dataset"], r["cefr"]): r for r in rows}
    r = by_mdc[("glm-ft", "EFCAMDAT", "A2")]
    assert int(r["n"]) == 1
    assert int(r["learner_top1"]) == 1
    r = by_mdc[("glm-ft", "EFCAMDAT", "B1")]
    assert int(r["n"]) == 2
    # item 1 matches (make=make), item 2 doesn't (very≠much)
    assert int(r["learner_top1"]) == 1


def test_run_all_tables_emits_three_csvs(predictions_jsonl_path: Path, tmp_path: Path):
    out_dir = tmp_path / "tables"
    run_all_tables.main(["--input", str(predictions_jsonl_path), "--out_dir", str(out_dir)])
    expected = {"cloze_accuracy.csv", "learner_plausibility.csv", "cefr_breakdown.csv"}
    assert {p.name for p in out_dir.iterdir()} == expected
    for name in expected:
        assert (out_dir / name).read_text().count("\n") > 1


def test_load_records_rejects_missing_model(tmp_path: Path):
    from eval_scripts._io import load_records
    p = tmp_path / "bad.jsonl"
    p.write_text('{"item_id": 0, "predicted_filler": "x"}\n')
    with pytest.raises(SystemExit):
        load_records(p)
