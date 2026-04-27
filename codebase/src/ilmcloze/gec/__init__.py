"""Grammatical Error Correction wrappers used to build condition I.

Default backend is CoEdit-Large via HuggingFace. The backend is swappable;
any class that implements :meth:`correct(text) -> str` satisfies the
protocol.
"""

from __future__ import annotations

from typing import Protocol


class GECBackend(Protocol):
    def correct(self, text: str) -> str:
        ...


from ilmcloze.gec.coedit import CoEditBackend  # noqa: E402,F401  (re-export)
