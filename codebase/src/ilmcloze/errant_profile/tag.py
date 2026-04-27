"""Thin wrapper around ERRANT.

Produces :class:`ErrantTag` rows for a ``(source, target)`` pair, where the
source is the learner text and the target is its GEC correction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

if TYPE_CHECKING:
    import errant


@dataclass(frozen=True)
class ErrantTag:
    tag: str  # e.g. "R:VERB:SVA"
    source_start: int
    source_end: int
    target_start: int
    target_end: int
    source_str: str
    target_str: str


_ANNOTATOR = None


def _get_annotator():
    global _ANNOTATOR
    if _ANNOTATOR is None:
        import errant

        _ANNOTATOR = errant.load("en")
    return _ANNOTATOR


def tag_pair(source: str, target: str) -> list[ErrantTag]:
    """Return ERRANT edits converting ``source`` → ``target``."""
    ann = _get_annotator()
    src = ann.parse(source)
    tgt = ann.parse(target)
    edits = ann.annotate(src, tgt)
    out: list[ErrantTag] = []
    for e in edits:
        out.append(
            ErrantTag(
                tag=e.type,
                source_start=e.o_start,
                source_end=e.o_end,
                target_start=e.c_start,
                target_end=e.c_end,
                source_str=e.o_str,
                target_str=e.c_str,
            )
        )
    return out


def tag_many(pairs: Iterable[tuple[str, str]]) -> list[list[ErrantTag]]:
    return [tag_pair(s, t) for (s, t) in pairs]
