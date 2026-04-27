"""Tests for the JSONL schema validator + emit helper."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from eval_scripts import schema
from eval_scripts.emit import build_record, write_records


# ---- validate_record -------------------------------------------------------

def test_minimal_valid_record():
    assert schema.validate_record({"model": "m", "item_id": 0, "predicted_filler": "x"}) == []


def test_missing_required_fields():
    issues = schema.validate_record({"item_id": 0})
    assert any("missing required field 'model'" in i for i in issues)
    assert any("missing required field 'predicted_filler'" in i for i in issues)


def test_wrong_required_types():
    issues = schema.validate_record({"model": "", "item_id": "0", "predicted_filler": 123})
    assert any("model" in i for i in issues)
    assert any("item_id" in i for i in issues)
    assert any("predicted_filler" in i for i in issues)


def test_invalid_cefr_rejected():
    issues = schema.validate_record({
        "model": "m", "item_id": 0, "predicted_filler": "x", "cefr": "ZZ"
    })
    assert any("cefr" in i for i in issues)


@pytest.mark.parametrize("level", ["A1", "A2", "B1", "B2", "C1", "C2"])
def test_valid_cefr_levels(level):
    assert schema.validate_record({
        "model": "m", "item_id": 0, "predicted_filler": "x", "cefr": level,
    }) == []


def test_positive_logprob_rejected():
    issues = schema.validate_record({
        "model": "m", "item_id": 0, "predicted_filler": "x", "predicted_logprob": 0.5,
    })
    assert any("predicted_logprob" in i and "> 0" in i for i in issues)


def test_extra_fields_allowed():
    """Forward-compat: unknown keys do not fail validation."""
    assert schema.validate_record({
        "model": "m", "item_id": 0, "predicted_filler": "x",
        "future_field": {"nested": [1, 2, 3]},
    }) == []


# ---- validate_file ---------------------------------------------------------

def test_validate_file_passes_clean(tmp_path: Path):
    p = tmp_path / "ok.jsonl"
    p.write_text(
        '{"model": "m", "item_id": 0, "predicted_filler": "x"}\n'
        '{"model": "m", "item_id": 1, "predicted_filler": "y", "cefr": "B1"}\n'
    )
    assert schema.validate_file(p) == {}


def test_validate_file_flags_per_line(tmp_path: Path):
    p = tmp_path / "bad.jsonl"
    p.write_text(
        '{"model": "m", "item_id": 0, "predicted_filler": "x"}\n'
        '{"item_id": 0, "predicted_filler": "y"}\n'
        'not json\n'
    )
    issues = schema.validate_file(p)
    assert "line:1" not in issues
    assert "line:2" in issues
    assert "line:3" in issues
    assert any("invalid JSON" in s for s in issues["line:3"])


# ---- build_record / write_records -----------------------------------------

def test_build_record_valid_passthrough():
    r = build_record(
        model="glm-ft",
        item_id=42,
        predicted_filler="school",
        dataset="EFCAMDAT",
        cefr="b1",  # case-insensitive in
        l1="French",
        predicted_logprob=-0.4,
        native_gold_filler="school",
        learner_gold_filler="school",
    )
    assert r["cefr"] == "B1"  # normalized
    assert r["model"] == "glm-ft"
    assert schema.validate_record(r) == []


def test_build_record_raises_on_invalid_cefr():
    with pytest.raises(ValueError):
        build_record(model="m", item_id=0, predicted_filler="x", cefr="XX")


def test_write_records_round_trip(tmp_path: Path):
    out = tmp_path / "p.jsonl"
    rs = [
        build_record(model="m", item_id=i, predicted_filler=f"w{i}", cefr="B1")
        for i in range(3)
    ]
    n = write_records(out, rs)
    assert n == 3
    lines = [json.loads(l) for l in out.read_text().splitlines() if l.strip()]
    assert [r["item_id"] for r in lines] == [0, 1, 2]
    assert all(r["cefr"] == "B1" for r in lines)


def test_write_records_refuses_invalid(tmp_path: Path):
    out = tmp_path / "p.jsonl"
    bad_record = {"model": "", "item_id": 0, "predicted_filler": "x"}  # empty model
    with pytest.raises(ValueError):
        write_records(out, [bad_record])
