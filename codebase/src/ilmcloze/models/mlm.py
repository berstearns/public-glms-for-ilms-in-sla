"""Encoder-only MLM backbone (BERT / RoBERTa / DeBERTa)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ilmcloze.config import ModelConfig


@dataclass
class MLMBackbone:
    cfg: ModelConfig
    device: str = "cpu"
    _tokenizer: Any = None
    _model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.cfg.torch_dtype]
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.hf_repo, revision=self.cfg.hf_revision
        )
        self._model = AutoModelForMaskedLM.from_pretrained(
            self.cfg.hf_repo, revision=self.cfg.hf_revision, torch_dtype=dtype
        )
        self._model.to(self.device).eval()

    @property
    def tokenizer(self) -> Any:
        if self._tokenizer is None:
            self.load()
        return self._tokenizer

    @property
    def model(self) -> Any:
        if self._model is None:
            self.load()
        return self._model
