"""Decoder-only NWP backbone (GPT-2 / Llama / Pythia / etc.)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ilmcloze.config import ModelConfig


@dataclass
class NWPBackbone:
    cfg: ModelConfig
    device: str = "cpu"
    _tokenizer: Any = None
    _model: Any = None

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.cfg.torch_dtype]
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.hf_repo, revision=self.cfg.hf_revision
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model = AutoModelForCausalLM.from_pretrained(
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
