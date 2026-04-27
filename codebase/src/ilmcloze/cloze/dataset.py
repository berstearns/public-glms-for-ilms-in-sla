"""torch Datasets for training and evaluation.

Two responsibilities:

* :class:`ClozeItem` — a single serialised evaluation item (JSONL row).
* :class:`ClozeDataset` — on-disk JSONL backed Dataset with lazy parsing.
* :class:`ContinuedPretrainDataset` — yields GLM-style Part A / Part B pairs
  built on-the-fly from raw text (no pre-materialised cloze items).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from ilmcloze.cloze.context import ContextualItem
from ilmcloze.cloze.format import LearnerMetadata
from ilmcloze.cloze.gap_sampler import Gap
from ilmcloze.cloze.token_counters import count_all


@dataclass
class ClozeItem:
    """A single cloze evaluation item, JSONL-serialisable.

    Token-count fields (``gap_token_counts``) are keyed by tokenizer-named,
    version-hashed column names — see
    :mod:`ilmcloze.cloze.token_counters`. There is **no implicit length
    field**: every length reported by this dataclass is explicitly tied to
    a named tokenizer.
    """

    corpus: str
    item_id: str
    gap_start: int  # gap position in *whitespace* tokens (the sampling grid)
    gap_end: int
    gap_tokens: list[str]  # whitespace-split gap text
    locus: str
    condition: str
    left: list[str]   # whitespace-split left context
    right: list[str]  # whitespace-split right context
    meta: dict[str, str | int]
    native_filler: list[str] | None = None
    empirical_fillers: list[list[str]] | None = None
    gap_token_counts: dict[str, int] = field(default_factory=dict)
    """``{n_{succinct_name}_{hash}_tokens: int}`` for every active counter.

    Populated by :func:`make_item`. Never use "gap length" without naming
    its tokenizer — consult this dict by field name.
    """

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, line: str) -> ClozeItem:
        data = json.loads(line)
        # Backward compatibility: items written before token_counters existed.
        data.setdefault("gap_token_counts", {})
        return cls(**data)


def write_items(path: str | Path, items: Iterable[ClozeItem]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(it.to_json() + "\n")
            n += 1
    return n


def read_items(path: str | Path) -> Iterator[ClozeItem]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield ClozeItem.from_json(line)


# ---------------------------------------------------------------------------
# Helpers to construct a ClozeItem from upstream building blocks.


def make_item(
    corpus: str,
    item_id: str,
    ctx: ContextualItem,
    meta: LearnerMetadata,
    native_filler: list[str] | None = None,
    empirical_fillers: list[list[str]] | None = None,
) -> ClozeItem:
    """Construct a cloze item, computing gap token counts under every
    registered tokenizer (see :mod:`ilmcloze.cloze.token_counters`)."""
    gap: Gap = ctx.gap
    gap_text = " ".join(gap.tokens)
    return ClozeItem(
        corpus=corpus,
        item_id=item_id,
        gap_start=gap.start,
        gap_end=gap.end,
        gap_tokens=list(gap.tokens),
        locus=gap.locus,
        condition=ctx.condition,
        left=list(ctx.left),
        right=list(ctx.right),
        meta={"l1": meta.l1, "cefr": meta.cefr, "errprof": meta.errprof},
        native_filler=native_filler,
        empirical_fillers=empirical_fillers,
        gap_token_counts=count_all(gap_text),
    )
