# Data

## Canonical splits directory

All corpora are read from a single directory — the user's
`phd-experimental-data/cefr-classification/data/splits/`. The directory
contains normalised CSVs:

| File | Corpus | Role |
|---|---|---|
| `norm-EFCAMDAT-{train,test,remainder}.csv` | EFCAMDAT | primary training / in-domain test |
| `norm-CELVA-SP.csv` + `norm-CELVA-SP-label.csv` | CELVA-SP | transfer (academic register) |
| `norm-KUPA-KEYS.csv` + `norm-KUPA-KEYS-label.csv` | KUPA-KEYS | transfer (process-logged writing) |
| `norm-andrew100k-{train,test}-label.csv` | andrew100k | transfer (mixed) |
| `norm-universal-cefr-label.csv` | universal | transfer (meta-corpus) |

We do not redistribute these CSVs from this repository. `scripts/00_download_data.py`
verifies their presence and emits an inventory.

## Canonical schema

After loading, every corpus has columns `text, cefr, l1, corpus, item_id`.
`ilmcloze/io/splits.py` handles the mapping from per-corpus column names to
this schema.

## Cloze items

`scripts/05_build_cloze_dataset.py` emits JSONL of
`ilmcloze.cloze.dataset.ClozeItem`. Each row contains:

- the gap span (`gap_start`, `gap_end`, `gap_tokens`, `locus`);
- the surrounding context under the selected condition (`left`, `right`,
  `condition`);
- learner metadata (`meta.l1`, `meta.cefr`, `meta.errprof`);
- the native reference filler (`native_filler`) when GEC-cleaned text is
  available, which enables the learner-plausibility metric;
- optional `empirical_fillers` (multiple learner fillers at matched CEFR),
  which enables the KL / JS calibration metrics.

## Condition III (synthetic corruption)

`scripts/06_corrupt_context.py` starts from condition I and applies
rate-controlled token-level corruptions (determiner drop, preposition swap,
SVA stripping) with the goal of matching EFCAMDAT-B1's average ERRANT error
rate. The actual target rate is set by `cloze.synth_corruption_rate`.
