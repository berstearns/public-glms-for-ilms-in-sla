"""Explicit, tokenizer-named, version-hashed token counting.

Every gap (and any other span we care about) reports its length under
multiple tokenization schemes. The field name encodes the scheme exactly:

    n_{succinct_name}_{version_hash}_tokens

where

* ``succinct_name`` identifies the tokenizer family precisely — e.g.
  ``whitespace``, ``nltk_treebank``, ``nltk_word_tokenize``,
  ``spacy_en_core_web_sm``, ``gpt2_bpe``, ``distilbert_wordpiece_uncased``.
* ``version_hash`` is an 8-char sha256 prefix of a ``"{scheme}@{version}"``
  string that pins the exact tokenizer version / HF revision. For
  parameter-free schemes (pure ``str.split()``) the hash is empty and the
  field reduces to ``n_{succinct_name}_tokens``.

The **default reference** for stratification and reporting is the NLTK
Treebank word tokenizer (``nltk.tokenize.TreebankWordTokenizer``). It is
deterministic, has no neural dependencies, and matches the "word" granularity
used in SLA error-tagging tools (ERRANT/spaCy) and most linguistic
annotation conventions. Pick whatever granularity the downstream analysis
calls for, but document it by *field name*, not by convention.

Usage::

    from ilmcloze.cloze.token_counters import count_all, reference_field

    counts = count_all("with your friends.")
    # {'n_whitespace_tokens': 3,
    #  'n_nltk_treebank_<hash>_tokens': 4,
    #  'n_gpt2_bpe_<hash>_tokens': 4, ...}

    print(reference_field())  # "n_nltk_treebank_<hash>_tokens"
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Callable


def _hash(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:8]


@dataclass(frozen=True)
class TokenCounter:
    """A named, version-hashed token-count function."""

    succinct_name: str
    """Family identifier: ``whitespace``, ``nltk_treebank``, ``gpt2_bpe``, …"""

    version_hash: str
    """8-char sha256 of ``"{scheme}@{version}"``. Empty for parameter-free schemes."""

    count: Callable[[str], int]

    @property
    def field(self) -> str:
        """Stable column / JSON key, including version hash when non-empty."""
        if not self.version_hash:
            return f"n_{self.succinct_name}_tokens"
        return f"n_{self.succinct_name}_{self.version_hash}_tokens"


# ---------------------------------------------------------------------------
# Individual counter builders. Each is lazy; any that fails to construct
# (missing optional dependency, missing model) is silently skipped by
# :func:`default_counters`.
# ---------------------------------------------------------------------------


def whitespace_counter() -> TokenCounter:
    """``str.split()`` — no versioning needed."""
    return TokenCounter("whitespace", "", lambda s: len(s.split()))


def nltk_treebank_counter() -> TokenCounter:
    """NLTK's :class:`~nltk.tokenize.TreebankWordTokenizer`.

    This is the **default reference** for length-stratified reporting.
    Deterministic and model-independent.
    """
    import nltk
    from nltk.tokenize import TreebankWordTokenizer

    tokenizer = TreebankWordTokenizer()
    ver = getattr(nltk, "__version__", "unknown")
    return TokenCounter(
        succinct_name="nltk_treebank",
        version_hash=_hash(f"nltk_treebank@{ver}"),
        count=lambda s, _t=tokenizer: len(_t.tokenize(s)),
    )


def nltk_word_tokenize_counter() -> TokenCounter:
    """NLTK :func:`~nltk.tokenize.word_tokenize` (sentence split + Treebank).

    Requires the ``punkt_tab`` resource; downloaded on demand.
    """
    import nltk

    try:
        nltk.data.find("tokenizers/punkt_tab")
    except LookupError:
        nltk.download("punkt_tab", quiet=True)
    from nltk.tokenize import word_tokenize

    ver = getattr(nltk, "__version__", "unknown")
    return TokenCounter(
        succinct_name="nltk_word_tokenize",
        version_hash=_hash(f"nltk_word_tokenize@{ver}"),
        count=lambda s: len(word_tokenize(s)),
    )


def spacy_counter(model: str = "en_core_web_sm") -> TokenCounter:
    """spaCy pipeline token count (whitespace tokens excluded)."""
    import spacy

    nlp = spacy.load(model, disable=["lemmatizer", "ner", "parser"])
    ver = nlp.meta.get("version", "unknown")
    name = f"spacy_{model}".replace("-", "_")
    return TokenCounter(
        succinct_name=name,
        version_hash=_hash(f"{model}@{ver}"),
        count=lambda s, _n=nlp: sum(1 for t in _n(s) if not t.is_space),
    )


def hf_counter(succinct_name: str, repo: str, revision: str = "main") -> TokenCounter:
    """HuggingFace ``AutoTokenizer`` subword count (no special tokens).

    ``trust_remote_code=True`` is passed so custom-code checkpoints (GLM,
    some ChatGLM variants) do not trigger an interactive ``[y/N]`` prompt
    that stalls non-interactive runs.
    """
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        repo, revision=revision, trust_remote_code=True
    )
    return TokenCounter(
        succinct_name=succinct_name,
        version_hash=_hash(f"{repo}@{revision}"),
        count=lambda s, _t=tokenizer: len(_t.tokenize(s)),
    )


# ---------------------------------------------------------------------------
# Default battery + reference choice
# ---------------------------------------------------------------------------

REFERENCE_COUNTER_NAME = "nltk_treebank"
"""Succinct name of the default reference tokenizer for stratification."""

_HF_BATTERY: tuple[tuple[str, str], ...] = (
    ("gpt2_bpe", "gpt2"),
    ("distilgpt2_bpe", "distilgpt2"),
    ("distilbert_wordpiece_uncased", "distilbert-base-uncased"),
    ("bert_wordpiece_cased", "bert-base-cased"),
    ("bert_wordpiece_uncased", "bert-base-uncased"),
    ("roberta_bpe", "roberta-base"),
    ("glm_roberta_large", "THUDM/glm-roberta-large"),
)

_CACHE: list[TokenCounter] | None = None


def default_counters() -> list[TokenCounter]:
    """Return a cached list of counters: whitespace + NLTK (reference + word_tokenize)
    + spaCy + a battery of HF subword tokenizers.

    Any counter that cannot be constructed (missing resource, missing model
    checkpoint on disk) is silently skipped so the pipeline degrades
    gracefully on machines that don't have every tokenizer cached.
    """
    global _CACHE
    if _CACHE is not None:
        return _CACHE

    counters: list[TokenCounter] = [whitespace_counter()]
    for builder in (nltk_treebank_counter, nltk_word_tokenize_counter):
        try:
            counters.append(builder())
        except Exception:  # noqa: BLE001
            pass
    try:
        counters.append(spacy_counter())
    except Exception:  # noqa: BLE001
        pass
    for succinct, repo in _HF_BATTERY:
        try:
            counters.append(hf_counter(succinct, repo))
        except Exception:  # noqa: BLE001
            pass
    _CACHE = counters
    return counters


def reference_field() -> str:
    """Return the column/key name of the default reference counter."""
    for c in default_counters():
        if c.succinct_name == REFERENCE_COUNTER_NAME:
            return c.field
    # Fall back to whitespace if NLTK isn't installed — still named explicitly.
    return whitespace_counter().field


def count_all(text: str, counters: list[TokenCounter] | None = None) -> dict[str, int]:
    """Compute all counters' field → count for ``text``."""
    counters = counters if counters is not None else default_counters()
    return {c.field: c.count(text) for c in counters}


def counter_manifest(counters: list[TokenCounter] | None = None) -> list[dict[str, str]]:
    """Machine-readable description of the active counters (for ``run.json``)."""
    counters = counters if counters is not None else default_counters()
    return [
        {"succinct_name": c.succinct_name, "version_hash": c.version_hash, "field": c.field}
        for c in counters
    ]
