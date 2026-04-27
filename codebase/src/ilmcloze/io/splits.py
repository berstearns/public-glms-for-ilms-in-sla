"""Unified loaders over the splits directory.

The splits directory is the one referenced by
:class:`ilmcloze.config.DataConfig`. The files there have per-corpus columns;
we map them to the shared schema documented in :mod:`ilmcloze.io`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import pandas as pd

from ilmcloze.config import DataConfig

CANONICAL_COLUMNS: tuple[str, ...] = ("text", "cefr", "l1", "corpus", "item_id")


def _coerce(df: pd.DataFrame, corpus: str) -> pd.DataFrame:
    """Normalise arbitrary corpus columns to the canonical schema."""
    out = pd.DataFrame()
    # Try a set of common column spellings.
    text_col = next((c for c in ("text", "sentence", "essay", "learner_text") if c in df.columns), None)
    cefr_col = next((c for c in ("cefr", "CEFR", "cefr_level", "level", "label") if c in df.columns), None)
    l1_col = next((c for c in ("l1", "L1", "native_language", "mother_tongue") if c in df.columns), None)
    id_col = next((c for c in ("item_id", "id", "doc_id") if c in df.columns), None)

    if text_col is None:
        raise KeyError(f"No text-like column found for corpus={corpus}; saw {list(df.columns)}")

    out["text"] = df[text_col].astype(str)
    out["cefr"] = df[cefr_col].astype(str) if cefr_col else pd.NA
    out["l1"] = df[l1_col].astype(str) if l1_col else "UNK"
    out["corpus"] = corpus
    out["item_id"] = df[id_col].astype(str) if id_col else df.index.astype(str)
    return out[list(CANONICAL_COLUMNS)]


def _resolve(splits_dir: str | Path, filename: str) -> Path:
    path = Path(splits_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return path


def load_efcamdat(cfg: DataConfig, split: str) -> pd.DataFrame:
    """Load an EFCAMDAT split. ``split`` ∈ {train, test, remainder}."""
    mapping: dict[str, str] = {
        "train": cfg.efcamdat_train,
        "test": cfg.efcamdat_test,
        "remainder": cfg.efcamdat_remainder,
    }
    if split not in mapping:
        raise ValueError(f"Unknown EFCAMDAT split {split!r}; expected one of {list(mapping)}")
    return _coerce(pd.read_csv(_resolve(cfg.splits_dir, mapping[split])), "EFCAMDAT")


def load_celva_sp(cfg: DataConfig) -> pd.DataFrame:
    return _coerce(pd.read_csv(_resolve(cfg.splits_dir, cfg.celva_sp)), "CELVA-SP")


def load_kupa_keys(cfg: DataConfig) -> pd.DataFrame:
    return _coerce(pd.read_csv(_resolve(cfg.splits_dir, cfg.kupa_keys)), "KUPA-KEYS")


def load_andrew(cfg: DataConfig, split: str = "test") -> pd.DataFrame:
    filename = cfg.andrew_train if split == "train" else cfg.andrew_test
    return _coerce(pd.read_csv(_resolve(cfg.splits_dir, filename)), "andrew100k")


def load_universal(cfg: DataConfig) -> pd.DataFrame:
    return _coerce(pd.read_csv(_resolve(cfg.splits_dir, cfg.universal_label)), "universal-cefr")


CORPUS_LOADERS: dict[str, Callable[[DataConfig], pd.DataFrame]] = {
    "efcamdat-train": lambda c: load_efcamdat(c, "train"),
    "efcamdat-test": lambda c: load_efcamdat(c, "test"),
    "celva-sp": load_celva_sp,
    "kupa-keys": load_kupa_keys,
    "andrew-test": lambda c: load_andrew(c, "test"),
    "andrew-train": lambda c: load_andrew(c, "train"),
    "universal": load_universal,
}


def load_corpus(cfg: DataConfig, name: str) -> pd.DataFrame:
    """Generic dispatch on a corpus name."""
    if name not in CORPUS_LOADERS:
        raise KeyError(f"Unknown corpus {name!r}; available: {list(CORPUS_LOADERS)}")
    return CORPUS_LOADERS[name](cfg)
