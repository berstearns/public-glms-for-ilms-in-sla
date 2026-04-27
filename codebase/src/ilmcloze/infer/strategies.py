"""Explicit decoding strategies.

A **baseline is a (model, decoding-strategy) pair**, not a model alone. A
RoBERTa with iterative-confident unmasking is a different experimental unit
from a RoBERTa with no decoding strategy — the latter emits a single
subtoken regardless of gap length and, by construction, scores 0% EM on
any gap whose gold text tokenizes to more than one subtoken.

Every strategy is enumerated here with a stable short name used in
output folder names, CSV columns, and paper tables.

Strategy taxonomy
-----------------

MLM family
~~~~~~~~~~

``mlm_none``
    Insert a **single** ``[MASK]`` for the whole gap. Forward-pass once.
    Predict the argmax single subtoken. Always returns exactly one
    subtoken. This is the strategy-free baseline for encoder-only MLMs:
    without an explicit multi-token decoding wrapper around the MLM, this
    is what the model can do on its own.

``mlm_iterative_confident``
    Length-known. Insert ``k`` ``[MASK]`` tokens where ``k`` matches the
    gold gap length under the model's own tokenizer. Iteratively fill the
    most-confident mask, conditioning remaining masks on already-filled
    predictions (Mask-Predict / CMLM, Ghazvininejad et al. 2019).

``mlm_enumerate_lengths``
    Length-unknown. Enumerate ``k ∈ {1..max_length}``, run
    ``mlm_iterative_confident`` for each ``k``, pick the ``k`` that
    maximises mean log-probability of the filled tokens.

NWP family
~~~~~~~~~~

``nwp_greedy_prompt``
    Decoder-only LM with a fill-the-blank prompt template, greedy
    continuation, decoder stops at ``max_new_tokens``.

``nwp_greedy_l2r``
    Decoder-only LM, left context only, greedy continuation — the right
    context is explicitly discarded.

GLM family
~~~~~~~~~~

``glm_ar_span``
    GLM native autoregressive blank-infilling on Part A containing a
    single mask; Part B generated autoregressively until ``[END]``.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DecodingStrategy:
    """A named decoding strategy applicable to a model family."""

    short_name: str
    family: str  # "mlm" | "nwp" | "glm"
    length_handling: str  # "none" | "length_known" | "length_unknown" | "eos"
    description: str

    @property
    def field(self) -> str:
        return self.short_name


REGISTRY: dict[str, DecodingStrategy] = {
    s.short_name: s
    for s in [
        DecodingStrategy(
            short_name="mlm_none",
            family="mlm",
            length_handling="none",
            description="Single [MASK]; top-1 subtoken; always emits 1 subtoken.",
        ),
        DecodingStrategy(
            short_name="mlm_iterative_confident",
            family="mlm",
            length_handling="length_known",
            description="k masks where k = gap length in model subtokens; fill most-confident first.",
        ),
        DecodingStrategy(
            short_name="mlm_enumerate_lengths",
            family="mlm",
            length_handling="length_unknown",
            description="Enumerate k in [1, L_max]; pick k maximising mean log-prob of filled tokens.",
        ),
        DecodingStrategy(
            short_name="nwp_greedy_prompt",
            family="nwp",
            length_handling="eos",
            description="Fill-the-blank prompt + greedy decoding.",
        ),
        DecodingStrategy(
            short_name="nwp_greedy_l2r",
            family="nwp",
            length_handling="eos",
            description="Left context only + greedy continuation (discards right context).",
        ),
        DecodingStrategy(
            short_name="glm_ar_span",
            family="glm",
            length_handling="eos",
            description="GLM autoregressive blank infilling; generates Part B until [END].",
        ),
    ]
}


def get(short_name: str) -> DecodingStrategy:
    if short_name not in REGISTRY:
        raise KeyError(f"Unknown decoding strategy {short_name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[short_name]
