"""Synthetic fixture for glms-for-ilms eval_scripts smoke tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def predictions_jsonl_path(tmp_path: Path) -> Path:
    """Two-model, two-dataset, mixed-CEFR synthetic predictions."""
    records = [
        # ----- glm-ft on EFCAMDAT -----
        {"model": "glm-ft", "item_id": 0, "dataset": "EFCAMDAT", "cefr": "A2", "l1": "French",
         "predicted_filler": "school", "predicted_logprob": -0.2,
         "native_gold_filler": "school", "learner_gold_filler": "school"},
        {"model": "glm-ft", "item_id": 1, "dataset": "EFCAMDAT", "cefr": "B1", "l1": "Spanish",
         "predicted_filler": "make", "predicted_logprob": -0.5,
         "native_gold_filler": "do", "learner_gold_filler": "make"},
        {"model": "glm-ft", "item_id": 2, "dataset": "EFCAMDAT", "cefr": "B1", "l1": "Spanish",
         "predicted_filler": "very", "predicted_logprob": -1.2,
         "native_gold_filler": "much", "learner_gold_filler": "much"},
        # ----- glm-ft on CELVA-SP -----
        {"model": "glm-ft", "item_id": 3, "dataset": "CELVA-SP", "cefr": "A2", "l1": "French",
         "predicted_filler": "go", "predicted_logprob": -0.4,
         "native_gold_filler": "go", "learner_gold_filler": "go"},
        # ----- nwp-baseline on EFCAMDAT -----
        {"model": "nwp-baseline", "item_id": 0, "dataset": "EFCAMDAT", "cefr": "A2", "l1": "French",
         "predicted_filler": "house", "predicted_logprob": -0.7,
         "native_gold_filler": "school", "learner_gold_filler": "school"},
        {"model": "nwp-baseline", "item_id": 1, "dataset": "EFCAMDAT", "cefr": "B1", "l1": "Spanish",
         "predicted_filler": "do", "predicted_logprob": -0.9,
         "native_gold_filler": "do", "learner_gold_filler": "make"},
    ]
    p = tmp_path / "predictions.jsonl"
    with p.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return p
