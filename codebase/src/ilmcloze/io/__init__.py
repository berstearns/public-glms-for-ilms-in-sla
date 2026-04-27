"""Corpus loaders.

All corpora are read from CSVs under ``data.splits_dir`` and normalised to a
shared schema:

==============  ================================================
Column          Meaning
==============  ================================================
``text``        the raw learner (or reference) text
``cefr``        CEFR label in {A1, A2, B1, B2, C1, C2}; else NaN
``l1``          ISO-639-1 L1 code; else ``UNK``
``corpus``      source corpus name
``item_id``     stable identifier within source corpus
==============  ================================================

Loaders return :class:`pandas.DataFrame` instances with this schema.
"""

from __future__ import annotations
