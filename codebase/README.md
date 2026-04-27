# ilmcloze — Interlanguage Language Models for Learner Cloze Simulation

Reproducible codebase for the EMNLP 2026 submission
**From Next-Word Prediction to Interlanguage Infilling: Adapting GLMs for
Learner Cloze Simulation**.

The codebase is a **config-driven DAG**. Every stage is
`python -m pipeline.{stage} --config configs/pipeline/{stage}.yaml`. Any
field in any config is overridable at the call site via `--section.field
value`. End-to-end runs (and per-variant experiments) are composite YAMLs
under `configs/e2e/` that `!include` the per-stage configs and apply
variant-specific overrides.

`just` is the orchestrator. `just --list` to browse.

## Install

```bash
just setup                # uv sync (falls back to pip install -e ".[dev]")
python -m spacy download en_core_web_sm
python -m ilmcloze.errant_profile.tag --install-errant
```

Python 3.10+. PyTorch 2.1+. HuggingFace `transformers` 4.40+. Optional:
CUDA GPU for any `train-*` stage.

## Layout

```
codebase/
├── Justfile                         # orchestrator — `just --list`
├── pyproject.toml                   # uv + hatchling; two packages: ilmcloze, pipeline
├── configs/
│   ├── common/                      # reusable section fragments (!include'd)
│   │   ├── data.yaml
│   │   ├── cloze.yaml
│   │   ├── conditioning.yaml
│   │   ├── infer.yaml
│   │   ├── eval.yaml
│   │   ├── train/
│   │   │   ├── continued-pretrain.yaml
│   │   │   └── sft.yaml
│   │   └── model/
│   │       ├── glm-base.yaml
│   │       ├── glm-roberta-large.yaml
│   │       ├── gpt2.yaml
│   │       └── bert-large-cased.yaml
│   ├── pipeline/                    # per-stage configs (single-stage invocation)
│   │   ├── download-data.yaml
│   │   ├── prepare-splits.yaml
│   │   ├── gec-clean.yaml
│   │   ├── errant-profile.yaml
│   │   ├── cluster-errprof.yaml
│   │   ├── build-cloze.yaml
│   │   ├── corrupt-context.yaml
│   │   ├── train-ilm.yaml
│   │   ├── train-sft.yaml
│   │   ├── infer.yaml
│   │   ├── evaluate.yaml
│   │   ├── transfer-eval.yaml
│   │   └── plot-results.yaml
│   └── e2e/                         # composites — full.yaml + one per variant
│       ├── full.yaml
│       ├── 01-ilm-main.yaml ... 10-crosscorpus-transfer.yaml
├── src/
│   ├── pipeline/                    # config-driven stage wrappers + e2e orchestrator
│   │   ├── _cli.py                  # run_stage, IncludeLoader, dotted overrides
│   │   ├── _config_common.py        # shared ExperimentConfig
│   │   ├── {stage}/                 # one subdir per stage: {__main__,config,runner}.py
│   │   └── e2e/                     # composite-YAML orchestrator + STAGE_REGISTRY
│   └── ilmcloze/                    # business logic (unchanged): cloze/, eval/, train/…
├── scripts/                         # ad-hoc utilities (NOT pipeline stages)
└── tests/
```

## Pipeline DAG

```
  raw splits ──► [00] download-data ─► [01] prepare-splits ──┐
                                                             │
                                             ┌───────────────┤
                                             ▼               ▼
                                      [02] gec-clean    [05] build-cloze
                                             │               │
                                             ▼               │
                                   [03] errant-profile       │
                                             │               │
                                             ▼               ▼
                                   [04] cluster-errprof   [06] corrupt-context  (optional, cond III)
                                             │               │
                                             └──────┬────────┘
                                                    ▼
                                          [07] train-ilm    OR   [08] train-sft
                                                    │               │
                                                    └──────┬────────┘
                                                           ▼
                                                      [09] infer
                                                           │
                                                           ▼
                                                    [10] evaluate ──► [12] plot-results
                                                           │
                                                           └─► [11] transfer-eval (variants 01, 10)
```

## Quickstart — variant 01 (main experiment)

```bash
just variant 01-ilm-main
# equivalent to:
uv run python -m pipeline.e2e --config configs/e2e/01-ilm-main.yaml
```

Artifacts land under `./artifacts/01-ilm-main/{stage}/`; the composite run
also saves a `composite_expanded.yaml` + `config_used.yaml` snapshot under
a run-dir `./artifacts/{hash}_{timestamp}_01-ilm-main/`.

## Everyday flows

```bash
# Single stage with defaults:
just prepare-splits

# Single stage with CLI overrides (dotted section.field):
just prepare-splits --stage.train_sample 2000 --experiment.name smoke

# Different config file for one stage:
ILM_CFG_BUILD_CLOZE=configs/pipeline/build-cloze-cond-I.yaml just build-cloze

# Full end-to-end pipeline:
just e2e

# Run a subset of e2e:
just e2e --pipeline.only prepare-splits,gec-clean,errant-profile

# Run a variant composite (configs/e2e/{NAME}.yaml):
just variant 05-mlm

# Variant with inline override (e.g. sweep span-length λ):
just variant 07-span-length --build-cloze.cloze.span_length_lambda 5.0
```

## Config mechanics

* **Per-stage configs** (`configs/pipeline/*.yaml`) are flat dataclass dumps —
  each top-level key is a section (`data`, `cloze`, `model`, `experiment`, …).
  Sections can be inlined or `!include`d from `configs/common/`.
* **Composite configs** (`configs/e2e/*.yaml`) have one top-level key per
  stage (matching the stage's key in `STAGE_REGISTRY`) plus a `pipeline:`
  orchestration section and a top-level `experiment:`. The top-level
  `experiment:` is propagated onto every stage, so `name`/`output_dir`/
  `seed`/`device` need to be set in exactly one place.
* **CLI overrides** work on every stage and on the composite. Example:
  `just train-ilm --train.learning_rate 5e-6 --train.epochs 5`.
  List/dict/tuple-typed fields are NOT CLI-overridable — edit the YAML.
* **Validation**: the argparse layer only accepts `--section.field` flags
  that actually exist on a dataclass field. Unknown overrides log a
  warning and are ignored.

## Adding a new stage

1. `src/pipeline/{new_stage}/{__init__,__main__,config,runner}.py`. The
   `config.py` defines a dataclass + `SECTION_MAP`; `runner.py` exposes
   `run_{new_stage}(config, run_dir) -> None`; `__main__.py` calls
   `run_stage(...)` from `pipeline._cli`.
2. `configs/pipeline/{new-stage}.yaml` with defaults.
3. Register in `src/pipeline/e2e/stages.py::STAGE_REGISTRY`.
4. Add a recipe in `Justfile` and one line to `configs/e2e/full.yaml`.

## Reproducibility

* **Deterministic seeds**. `experiment.seed` is applied globally (Python,
  NumPy, PyTorch CPU+CUDA) by `pipeline._cli.run_stage`.
* **Config pinning**. Every e2e run saves `config_used.yaml` +
  `composite_expanded.yaml` + per-stage `stages/{key}/effective.yaml`.
* **Data splits**. Canonical CSVs under `data.splits_dir` are referenced
  by path; we never resample without an explicit `--stage.train_sample`.
* **Model checkpoints**. HuggingFace revisions are pinned per config
  (`model.hf_revision`).

## Engineering

```bash
just test       # pytest -v
just fmt        # ruff format + fix
just lint       # ruff check
just peek       # ls over artifacts/
```

## License

MIT. See `LICENSE`.

## Citation

See `CITATION.cff`.
