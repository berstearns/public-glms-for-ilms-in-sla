"""CoEdit GEC backend.

Uses ``grammarly/coedit-large`` on HuggingFace. Output is obtained by
prepending the ``"Fix grammatical errors in this sentence: "`` instruction and
taking the generated correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from transformers import PreTrainedModel, PreTrainedTokenizerBase


_INSTRUCTION = "Fix grammatical errors in this sentence: "


@dataclass
class CoEditBackend:
    """Lazy CoEdit backend. Load the heavy models only on first ``.correct``."""

    repo: str = "grammarly/coedit-large"
    revision: str = "main"
    device: str = "cpu"
    max_new_tokens: int = 256
    _tokenizer: "PreTrainedTokenizerBase | None" = None
    _model: "PreTrainedModel | None" = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self.repo, revision=self.revision)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.repo, revision=self.revision)
        self._model.to(self.device).eval()

    def correct(self, text: str) -> str:
        self._load()
        assert self._tokenizer is not None and self._model is not None
        inputs = self._tokenizer(
            _INSTRUCTION + text,
            return_tensors="pt",
            truncation=True,
            max_length=self._tokenizer.model_max_length,
        ).to(self.device)

        import torch

        with torch.inference_mode():
            out = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                num_beams=4,
                do_sample=False,
            )
        return self._tokenizer.decode(out[0], skip_special_tokens=True)
