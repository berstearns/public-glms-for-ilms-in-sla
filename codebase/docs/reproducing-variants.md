# Reproducing experiment variants

Every variant in `experiments-ideas/` maps to a single YAML under
`configs/experiments/` and a single one-liner.

## Environment

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
export ILMCLOZE_CACHE=/path/to/scratch/ilmcloze
```

## Variant 01 — main experiment (GLM + learner-conditioning)

```bash
python scripts/run_experiment.py --config configs/experiments/01-ilm-main.yaml --device cuda
# or
make variant VARIANT=01-ilm-main
```

Artifacts land under `$ILMCLOZE_CACHE/01-ilm-main/`.

## Variant 02 — vanilla GLM (no conditioning)

```bash
python scripts/run_experiment.py --config configs/experiments/02-glm-vanilla.yaml
```

## Variant 03 — NWP prompt baseline

```bash
# zero-shot: skip training stages
python scripts/run_experiment.py \
    --config configs/experiments/03-nwp-prompt.yaml \
    --skip-training

# SFT
python scripts/run_experiment.py --config configs/experiments/03-nwp-prompt.yaml
```

## Variant 04 — NWP left-to-right

```bash
# 09_infer takes --l2r to switch the decoder strategy
python scripts/run_experiment.py \
    --config configs/experiments/04-nwp-lefttoright.yaml --skip-training

# or manually, after 05_build_cloze_dataset:
python scripts/09_infer_cloze.py \
    --config configs/experiments/04-nwp-lefttoright.yaml --l2r
```

## Variant 05 — MLM baseline

```bash
python scripts/run_experiment.py --config configs/experiments/05-mlm.yaml

# length-unknown variant (enumerate spans at inference)
python scripts/09_infer_cloze.py \
    --config configs/experiments/05-mlm.yaml --mlm-length-unknown
```

## Variant 06 — L2-error-loci cloze

Uses the variant-01 checkpoint — `train.regime: zero_shot` in the YAML
means run_experiment will skip the training stages and reuse the cached
checkpoint from `01-ilm-main`.

```bash
python scripts/run_experiment.py \
    --config configs/experiments/06-l2errorloci.yaml --skip-training
```

## Variant 07 — span-length ablation

```bash
for lam in 1.5 3 5 7; do
    # `run_experiment` uses the YAML as-is; duplicate and override
    sed "s/span_length_lambda: .*/span_length_lambda: $lam/" \
        configs/experiments/07-span-length.yaml > /tmp/07-lam$lam.yaml
    python scripts/run_experiment.py --config /tmp/07-lam$lam.yaml
done
```

## Variant 08 — context-condition ablation

```bash
for cond in I II III; do
    sed "s/context_condition: .*/context_condition: $cond/" \
        configs/experiments/08-context-condition.yaml > /tmp/08-$cond.yaml
    python scripts/run_experiment.py --config /tmp/08-$cond.yaml
done
```

Condition III requires running `06_corrupt_context.py` explicitly because
the driver expects the cloze JSONL to exist:

```bash
python scripts/06_corrupt_context.py --config /tmp/08-III.yaml --rate 0.05
```

## Variant 09 — training-regime ablation

```bash
for regime in zero_shot continued_pretrain sft; do
    sed "s/regime: .*/regime: $regime/" \
        configs/experiments/09-training-regime.yaml > /tmp/09-$regime.yaml
    python scripts/run_experiment.py --config /tmp/09-$regime.yaml
done
```

## Variant 10 — cross-corpus transfer

```bash
# First run variant 01 (for the checkpoint)
python scripts/run_experiment.py --config configs/experiments/01-ilm-main.yaml

# Then
python scripts/11_transfer_eval.py \
    --config configs/experiments/10-crosscorpus-transfer.yaml --device cuda
```
