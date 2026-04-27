"""Pytest configuration: isolate artifacts to a temp dir, fix seeds."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _isolate_artifacts(tmp_path, monkeypatch):
    monkeypatch.setenv("ILMCLOZE_CACHE", str(tmp_path))
    yield


@pytest.fixture()
def fixed_seed() -> int:
    os.environ["PYTHONHASHSEED"] = "0"
    return 0
