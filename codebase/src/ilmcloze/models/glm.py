"""GLM backbone wrapper.

GLM uses a custom architecture (:class:`GLMForConditionalGeneration` in
``THUDM/GLM``). We load via :func:`transformers.AutoModel.from_pretrained`
with ``trust_remote_code=True``; the official HF-compatible checkpoints
(e.g. ``THUDM/glm-335M``) work out of the box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from ilmcloze.config import ModelConfig

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizerBase


@dataclass
class GLMBackbone:
    cfg: ModelConfig
    device: str = "cpu"
    _tokenizer: Any = None
    _model: Any = None
    _extras: dict[str, Any] = field(default_factory=dict)

    def load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        dtype = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[self.cfg.torch_dtype]
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.cfg.hf_repo, revision=self.cfg.hf_revision, trust_remote_code=True
        )
        try:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                self.cfg.hf_repo,
                revision=self.cfg.hf_revision,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
        except ValueError:
            # Fall back: some GLM checkpoints register a custom auto-class.
            from transformers import AutoModel

            self._model = AutoModel.from_pretrained(
                self.cfg.hf_repo,
                revision=self.cfg.hf_revision,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
        self._model.to(self.device).eval()

    @property
    def tokenizer(self) -> "PreTrainedTokenizerBase":
        if self._tokenizer is None:
            self.load()
        return self._tokenizer  # type: ignore[return-value]

    @property
    def model(self) -> Any:
        if self._model is None:
            self.load()
        return self._model
