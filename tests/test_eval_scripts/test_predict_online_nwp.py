"""Tests for predict_online_nwp: pure-Python parts via dependency injection."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import pytest

from eval_scripts import predict_online_nwp


def _write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def test_read_items_minimal(tmp_path: Path):
    p = tmp_path / "items.csv"
    _write_csv(p, [{"item_id": "0", "prefix": "Hello"}, {"item_id": "1", "prefix": "World"}], ["item_id", "prefix"])
    items = predict_online_nwp.read_items(p)
    assert len(items) == 2
    assert items[0] == {"item_id": "0", "prefix": "Hello"}


def test_read_items_missing_required_column(tmp_path: Path):
    p = tmp_path / "bad.csv"
    _write_csv(p, [{"id": "0"}], ["id"])
    with pytest.raises(SystemExit):
        predict_online_nwp.read_items(p)


def test_predict_records_uses_injected_predictor():
    items = [
        {"item_id": "0", "prefix": "He went to", "native_gold_filler": "school", "learner_gold_filler": "school", "cefr": "B1", "l1": "French"},
        {"item_id": "1", "prefix": "She wants to", "native_gold_filler": "go", "learner_gold_filler": "make"},
        {"item_id": "2", "prefix": ""},  # empty prefix → predictor returns None
    ]

    def predict(prefix: str):
        if not prefix.strip():
            return None
        return ("school", -0.5) if "went" in prefix else ("go", -0.3)

    records = list(predict_online_nwp.predict_records(
        items, predict_next_token=predict, model_label="fake", dataset_label="EFCAMDAT",
    ))
    # item 2 (empty prefix → predictor None) is skipped to keep denominators honest
    assert len(records) == 2
    assert records[0]["model"] == "fake"
    assert records[0]["item_id"] == 0
    assert records[0]["predicted_filler"] == "school"
    assert records[0]["dataset"] == "EFCAMDAT"
    assert records[0]["cefr"] == "B1"
    assert records[1]["predicted_filler"] == "go"


def test_full_cli_with_monkeypatched_hf(tmp_path: Path, monkeypatch):
    csv_path = tmp_path / "items.csv"
    _write_csv(
        csv_path,
        [
            {"item_id": "0", "prefix": "He went to", "native_gold_filler": "school", "learner_gold_filler": "school"},
            {"item_id": "1", "prefix": "She wants to", "native_gold_filler": "go", "learner_gold_filler": "make"},
        ],
        ["item_id", "prefix", "native_gold_filler", "learner_gold_filler"],
    )
    out = tmp_path / "predictions.jsonl"

    fake_predictions = {"He went to": ("school", -0.5), "She wants to": ("make", -0.4)}
    def fake_predict_next(model, tokenizer, device):
        return lambda prefix: fake_predictions.get(prefix.strip())

    monkeypatch.setattr(predict_online_nwp, "_load_hf", lambda m, d: (None, None))
    monkeypatch.setattr(predict_online_nwp, "_hf_predict_next_token_fn", fake_predict_next)

    predict_online_nwp.main([
        "--model", "test/dummy",
        "--data", str(csv_path),
        "--out", str(out),
        "--model_name_label", "nwp-test",
        "--dataset", "EFCAMDAT",
    ])

    records = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert len(records) == 2
    assert records[0] == {
        "model": "nwp-test", "item_id": 0, "predicted_filler": "school",
        "predicted_logprob": pytest.approx(-0.5), "dataset": "EFCAMDAT",
        "native_gold_filler": "school", "learner_gold_filler": "school",
    }
    assert records[1]["predicted_filler"] == "make"


def test_predict_then_eval_full_round_trip(tmp_path: Path, monkeypatch):
    """predict_online_nwp → eval_cloze_accuracy_table → CSV with real numbers."""
    from eval_scripts import eval_cloze_accuracy_table, eval_learner_plausibility_table

    csv_path = tmp_path / "items.csv"
    _write_csv(
        csv_path,
        [
            {"item_id": "0", "prefix": "He went to", "native_gold_filler": "school", "learner_gold_filler": "school"},
            {"item_id": "1", "prefix": "She wants to", "native_gold_filler": "go", "learner_gold_filler": "make"},
            {"item_id": "2", "prefix": "I will", "native_gold_filler": "try", "learner_gold_filler": "trying"},
        ],
        ["item_id", "prefix", "native_gold_filler", "learner_gold_filler"],
    )
    jsonl = tmp_path / "predictions.jsonl"

    # native gold matched on items 0 only; learner gold matched on items 0 and 1
    fake = {
        "He went to": ("school", -0.2),
        "She wants to": ("make", -0.4),
        "I will": ("try not", -0.9),
    }
    monkeypatch.setattr(predict_online_nwp, "_load_hf", lambda m, d: (None, None))
    monkeypatch.setattr(
        predict_online_nwp, "_hf_predict_next_token_fn",
        lambda m, t, d: (lambda p: fake.get(p.strip())),
    )
    predict_online_nwp.main([
        "--model", "x", "--data", str(csv_path), "--out", str(jsonl),
        "--model_name_label", "nwp-fake", "--dataset", "EFCAMDAT",
    ])

    cloze_csv = tmp_path / "cloze.csv"
    eval_cloze_accuracy_table.main(["--input", str(jsonl), "--out", str(cloze_csv)])
    rows = list(csv.DictReader(cloze_csv.open()))
    assert len(rows) == 1
    assert rows[0]["model"] == "nwp-fake"
    assert int(rows[0]["correct"]) == 1
    assert float(rows[0]["accuracy"]) == pytest.approx(1 / 3)

    lp_csv = tmp_path / "lp.csv"
    eval_learner_plausibility_table.main(["--input", str(jsonl), "--out", str(lp_csv)])
    rows = list(csv.DictReader(lp_csv.open()))
    assert int(rows[0]["learner_top1"]) == 2
    assert float(rows[0]["learner_top1_acc"]) == pytest.approx(2 / 3)
