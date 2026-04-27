"""Tokenizer-explicit token counters: field-naming, hashing, counts."""

from __future__ import annotations

from ilmcloze.cloze.token_counters import (
    TokenCounter,
    count_all,
    nltk_treebank_counter,
    reference_field,
    whitespace_counter,
)


def test_whitespace_field_has_no_hash() -> None:
    c = whitespace_counter()
    assert c.field == "n_whitespace_tokens"
    assert c.count("a b c") == 3
    assert c.count("party,") == 1


def test_nltk_treebank_splits_punctuation() -> None:
    c = nltk_treebank_counter()
    # Hash is 8 hex chars.
    assert c.field.startswith("n_nltk_treebank_")
    assert c.field.endswith("_tokens")
    hash_part = c.field[len("n_nltk_treebank_") : -len("_tokens")]
    assert len(hash_part) == 8 and all(ch in "0123456789abcdef" for ch in hash_part)
    # "party," becomes ["party", ","] under Treebank.
    assert c.count("party,") == 2
    # Plain text behaves as expected.
    assert c.count("I am happy") == 3


def test_reference_field_is_nltk_treebank() -> None:
    assert reference_field().startswith("n_nltk_treebank_")


def test_count_all_covers_multiple_tokenizers() -> None:
    counts = count_all("with your friends.")
    # At minimum: whitespace + at least one subword or linguistic counter.
    assert counts["n_whitespace_tokens"] == 3
    assert len(counts) >= 2


def test_custom_counter_field() -> None:
    c = TokenCounter("my_scheme", "deadbeef", lambda s: len(s))
    assert c.field == "n_my_scheme_deadbeef_tokens"
