"""Per-learner ERRANT error-type profiles and 16-bucket clustering.

The profile for a text is the normalised distribution of ERRANT error-type
counts obtained by tagging the GEC-corrected text against the original
learner text. A batch of profiles is clustered with k-means to yield the
``ERRPROF`` categorical feature used in the learner-conditioning prefix.
"""

from __future__ import annotations
