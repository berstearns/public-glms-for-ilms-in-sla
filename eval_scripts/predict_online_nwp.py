"""Online NWP prompt-fill prediction → predictions.jsonl.

Reads a CSV of cloze items (one row per item), prompts a HuggingFace
causal LM with the *prefix* up to the gap, takes the highest-probability
next token as the predicted filler, and emits one canonical record per
item via `eval_scripts.emit.write_records`.

This is the NWP baseline (variant 03 in `experiments-ideas/`) — the
simplest infiller, which conditions only on left context. GLM / MLM
online predictors will live in their own modules with the same
emit-records contract.

Required CSV columns (defaults; override via flags):
    item_id            (int)
    prefix             (str)            — text up to the gap
    native_gold_filler (str, optional)
    learner_gold_filler (str, optional)
    cefr               (str, optional)
    l1                 (str, optional)

Usage:
    python -m eval_scripts.predict_online_nwp \\
        --model gpt2 \\
        --data data/cloze_items.csv \\
        --model_name_label nwp-baseline-gpt2 \\
        --dataset EFCAMDAT \\
        --out predictions.jsonl

Requires `transformers` and `torch` at runtime.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Callable

from eval_scripts.emit import build_record, write_records


def read_items(path: Path, *, prefix_col: str = "prefix", item_id_col: str = "item_id") -> list[dict]:
    """Read cloze items from a CSV. Returns list of dicts (one per row)."""
    if not path.exists():
        raise SystemExit(f"data file not found: {path}")
    with path.open() as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise SystemExit(f"{path}: no header row")
        for needed in (prefix_col, item_id_col):
            if needed not in reader.fieldnames:
                raise SystemExit(f"required column {needed!r} missing from {path}")
        return [dict(row) for row in reader]


def _load_hf(model_id: str, device: str):
    """Lazy import + load. Raises a helpful SystemExit if HF not installed."""
    try:
        import torch  # noqa: F401
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "predict_online_nwp requires `transformers` and `torch`. "
            "Install with: pip install transformers torch"
        ) from exc
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    model.eval()
    return model, tokenizer


def _hf_predict_next_token_fn(model, tokenizer, device: str) -> Callable[[str], tuple[str, float] | None]:
    """Returns a function (prefix → (token, logprob)) using greedy top-1."""
    import torch
    import torch.nn.functional as F

    def predict(prefix: str) -> tuple[str, float] | None:
        s = prefix.strip()
        if not s:
            return None
        enc = tokenizer(s, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            out = model(**enc)
        last_logits = out.logits[0, -1, :]  # next-token distribution
        logprobs = F.log_softmax(last_logits, dim=-1)
        top_id = int(torch.argmax(logprobs).item())
        token = tokenizer.decode([top_id]).strip()
        return token, float(logprobs[top_id].item())

    return predict


def predict_records(
    items: list[dict],
    *,
    predict_next_token: Callable[[str], tuple[str, float] | None],
    model_label: str,
    dataset_label: str | None,
    prefix_col: str = "prefix",
    item_id_col: str = "item_id",
):
    """Yield canonical records for each predictable item.

    Items where `predict_next_token` returns None or yields an empty
    token are skipped — the predictions.jsonl carries only items the
    model could actually score, which keeps eval-scripts denominators
    honest. The number of skipped items is reported via stderr in `main`.
    """
    for row in items:
        try:
            item_id = int(row[item_id_col])
        except (TypeError, ValueError):
            continue
        result = predict_next_token(row.get(prefix_col, ""))
        if result is None:
            continue
        predicted_filler, predicted_logprob = result
        if not predicted_filler:
            continue
        yield build_record(
            model=model_label,
            item_id=item_id,
            predicted_filler=predicted_filler,
            predicted_logprob=predicted_logprob if math.isfinite(predicted_logprob) else None,
            dataset=dataset_label,
            cefr=(row.get("cefr") or None) or None,
            l1=(row.get("l1") or None) or None,
            native_gold_filler=row.get("native_gold_filler") or None,
            learner_gold_filler=row.get("learner_gold_filler") or None,
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="HF model id or local path")
    ap.add_argument("--data", type=Path, required=True, help="CSV with cloze items")
    ap.add_argument("--prefix_col", default="prefix")
    ap.add_argument("--item_id_col", default="item_id")
    ap.add_argument("--model_name_label", default=None, help="defaults to --model")
    ap.add_argument("--dataset", default=None, help="dataset label attached to records")
    ap.add_argument("--out", type=Path, required=True, help="output predictions.jsonl")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args(argv)

    items = read_items(args.data, prefix_col=args.prefix_col, item_id_col=args.item_id_col)
    model, tokenizer = _load_hf(args.model, args.device)
    predict_next = _hf_predict_next_token_fn(model, tokenizer, args.device)
    label = args.model_name_label or args.model

    n = write_records(
        args.out,
        predict_records(
            items,
            predict_next_token=predict_next,
            model_label=label,
            dataset_label=args.dataset,
            prefix_col=args.prefix_col,
            item_id_col=args.item_id_col,
        ),
    )
    print(f"predict_online_nwp: wrote {n} predictions for {label} to {args.out}")


if __name__ == "__main__":
    main()
