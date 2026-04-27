#!/usr/bin/env python
"""smoke_test — zero-shot baselines on a tiny EFCAMDAT sample.

No training. Loads ``samples/efcamdat-smoke-100.csv``, builds 100 cloze
items (one multi-token gap per text, condition II = learner context), and
runs the three baselines that require no fine-tuning:

* ``distilgpt2``               — NWP prompt fill-the-blank
* ``gpt2``                     — NWP left-to-right continuation
* ``distilbert-base-uncased``  — MLM iterative (length-known)

Each run's artifacts land under::

    artifacts/smoke/sample-benchmark-{model-hash}-{model-name}-{timestamp}/
        sample.csv          # copy of the input sample
        cloze.jsonl         # cloze items evaluated
        predictions.jsonl
        summary.csv         # overall EM
        by_cefr.csv         # EM stratified by CEFR
        run.json            # config + git SHA + timestamp
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ilmcloze.cloze.context import build_learner_context  # noqa: E402
from ilmcloze.cloze.dataset import make_item  # noqa: E402
from ilmcloze.cloze.format import LearnerMetadata  # noqa: E402
from ilmcloze.cloze.gap_sampler import sample_multi_token  # noqa: E402
from ilmcloze.config import (  # noqa: E402
    ConditioningConfig,
    InferConfig,
    ModelConfig,
)
from ilmcloze.cloze.token_counters import (  # noqa: E402
    counter_manifest,
    reference_field,
)
from ilmcloze.eval.metrics import exact_match, top_k_hit  # noqa: E402

def _mcfg(name: str, kind: str, repo: str, dtype: str = "float32", rev: str = "main") -> ModelConfig:
    return ModelConfig(
        name=name, kind=kind, hf_repo=repo, hf_revision=rev,
        max_seq_length=512, torch_dtype=dtype,
    )


# Baselines = (label, model_config, decoding_strategy_short_name).
# Labels embed BOTH the model and the decoding strategy so the downstream
# output folder, CSVs, and paper tables can never silently collapse the two.
BASELINES: list[tuple[str, ModelConfig, str]] = [
    # --- NWP family, greedy (decoder-only; the "decoding strategy" is just
    #     greedy continuation up to max_new_tokens) ---------------------------
    ("distilgpt2-nwp-prompt",
     _mcfg("distilgpt2", "nwp", "distilgpt2"),
     "nwp_greedy_prompt"),
    ("gpt2-nwp-l2r",
     _mcfg("gpt2", "nwp", "gpt2"),
     "nwp_greedy_l2r"),

    # --- MLM family with NO DECODING STRATEGY (single [MASK], top-1) --------
    #   Expected: EM = 0 on any gap with >1 subtoken in that model's tokenizer.
    ("distilbert-base-uncased-mlm-none",
     _mcfg("distilbert-base-uncased", "mlm", "distilbert-base-uncased"),
     "mlm_none"),
    ("bert-base-cased-mlm-none",
     _mcfg("bert-base-cased", "mlm", "bert-base-cased"),
     "mlm_none"),
    ("bert-base-uncased-mlm-none",
     _mcfg("bert-base-uncased", "mlm", "bert-base-uncased"),
     "mlm_none"),
    ("roberta-base-mlm-none",
     _mcfg("roberta-base", "mlm", "roberta-base"),
     "mlm_none"),

    # --- MLM family with iterative-confident decoding (length-known) --------
    ("distilbert-base-uncased-mlm-iter-conf",
     _mcfg("distilbert-base-uncased", "mlm", "distilbert-base-uncased"),
     "mlm_iterative_confident"),
    ("bert-base-cased-mlm-iter-conf",
     _mcfg("bert-base-cased", "mlm", "bert-base-cased"),
     "mlm_iterative_confident"),

    # --- GLM family (run when GPU available; pulls ~1.3 GB on first use) ---
    ("glm-roberta-large-ar-span",
     _mcfg("glm-roberta-large", "glm", "THUDM/glm-roberta-large"),
     "glm_ar_span"),
]


def model_hash(cfg: ModelConfig) -> str:
    key = f"{cfg.hf_repo}@{cfg.hf_revision}".encode()
    return hashlib.sha256(key).hexdigest()[:8]


def timestamp() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:  # noqa: BLE001
        return None


def build_cloze(sample_csv: Path, seed: int, max_gap_len: int) -> list:
    df = pd.read_csv(sample_csv)
    rng = random.Random(seed)
    items = []
    for _, row in df.iterrows():
        text = str(row["text"])
        tokens = text.split()
        if len(tokens) < 6:
            continue
        # one multi-token span, length 1..max_gap_len
        gaps = sample_multi_token(tokens, n=1, rng=rng, lam=2.0, max_len=max_gap_len)
        if not gaps:
            continue
        gap = gaps[0]
        ctx = build_learner_context(tokens, gap)
        meta = LearnerMetadata(
            l1=str(row.get("l1", "UNK")),
            cefr=str(row.get("cefr_level", "UNK")),
            errprof="UNK",
        )
        items.append(
            make_item(
                corpus="EFCAMDAT-smoke",
                item_id=f"{row['writing_id']}:{gap.start}-{gap.end}",
                ctx=ctx,
                meta=meta,
                native_filler=None,
                empirical_fillers=None,
            )
        )
    return items


def _run_inference(strategy_name: str, cfg: ModelConfig, items: list,
                   cond: ConditioningConfig, infer_cfg: InferConfig, device: str) -> list:
    """Fast in-lined inference path for the smoke test, dispatched on the
    **decoding strategy short name** (see :mod:`ilmcloze.infer.strategies`).

    The strategy is a first-class axis — different strategies paired with
    the same model are different experimental units.
    """
    import torch

    from ilmcloze.cloze.format import (
        format_glm_part_a,
        format_mlm,
        format_nwp_lefttoright,
        format_nwp_prompt,
    )
    from ilmcloze.infer import PredictionRow
    from ilmcloze.infer.strategies import get as get_strategy

    strategy = get_strategy(strategy_name)

    if strategy.family == "nwp":
        from ilmcloze.models.nwp import NWPBackbone
        backbone: object = NWPBackbone(cfg=cfg, device=device)
    elif strategy.family == "mlm":
        from ilmcloze.models.mlm import MLMBackbone
        backbone = MLMBackbone(cfg=cfg, device=device)
    elif strategy.family == "glm":
        from ilmcloze.models.glm import GLMBackbone
        backbone = GLMBackbone(cfg=cfg, device=device)
    else:
        raise ValueError(f"Unknown strategy family: {strategy.family}")
    backbone.load()  # type: ignore[attr-defined]

    tok = backbone.tokenizer  # type: ignore[attr-defined]
    mdl = backbone.model  # type: ignore[attr-defined]

    def _n_subtokens(text: str) -> int:
        return len(tok.tokenize(text))

    preds: list[PredictionRow] = []
    for it in items:
        from ilmcloze.cloze.context import ContextualItem
        from ilmcloze.cloze.format import LearnerMetadata
        from ilmcloze.cloze.gap_sampler import Gap

        ctx = ContextualItem(
            left=tuple(it.left), right=tuple(it.right),
            gap=Gap(it.gap_start, it.gap_end, tuple(it.gap_tokens), it.locus),
            condition=it.condition,
        )
        meta = LearnerMetadata(
            l1=str(it.meta.get("l1", "UNK")),
            cefr=str(it.meta.get("cefr", "UNK")),
            errprof=it.meta.get("errprof", "UNK"),
        )
        gold_text = " ".join(it.gap_tokens)
        gap_len_subtok = max(1, _n_subtokens(gold_text))  # in THIS model's tokenizer
        gap_len_words = max(1, len(it.gap_tokens))        # whitespace words
        max_new = max(4, gap_len_subtok * 2)

        # Dispatch on the decoding strategy short name, not on the model kind.
        if strategy_name == "nwp_greedy_prompt":
            prompt = format_nwp_prompt(ctx, meta, cond)
            enc = tok(prompt, return_tensors="pt", truncation=True,
                      max_length=cfg.max_seq_length).to(device)
            with torch.inference_mode():
                gen = mdl.generate(**enc, max_new_tokens=max_new,
                                   num_beams=1, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            completion = tok.decode(gen[0][enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()
            predicted = completion.split()[:gap_len_words]

        elif strategy_name == "nwp_greedy_l2r":
            prompt = format_nwp_lefttoright(ctx, meta, cond)
            if not prompt.strip():
                prompt = tok.bos_token or tok.eos_token or " "
            enc = tok(prompt, return_tensors="pt", truncation=True,
                      max_length=cfg.max_seq_length).to(device)
            if enc["input_ids"].shape[1] == 0:
                bos = tok.bos_token_id or tok.eos_token_id or 0
                enc = {"input_ids": torch.tensor([[bos]], device=device),
                       "attention_mask": torch.tensor([[1]], device=device)}
            with torch.inference_mode():
                gen = mdl.generate(**enc, max_new_tokens=max_new,
                                   num_beams=1, do_sample=False,
                                   pad_token_id=tok.pad_token_id)
            completion = tok.decode(gen[0][enc["input_ids"].shape[1]:],
                                    skip_special_tokens=True).strip()
            predicted = completion.split()[:gap_len_words]

        elif strategy_name == "mlm_none":
            # NO DECODING STRATEGY: a single [MASK] regardless of gap length;
            # model emits exactly one subtoken. EM == 0 whenever the gold
            # text requires more than one subtoken under this tokenizer.
            mask_tok = tok.mask_token or "[MASK]"
            template = format_mlm(ctx, meta, cond, num_masks=1, mask_token=mask_tok)
            ids = tok(template, return_tensors="pt", truncation=True,
                      max_length=cfg.max_seq_length)["input_ids"].to(device)
            mask_positions = (ids == tok.mask_token_id).nonzero(as_tuple=False)
            if mask_positions.numel() == 0:
                predicted = []
            else:
                _, pos = mask_positions[0].tolist()
                with torch.inference_mode():
                    logits = mdl(ids).logits
                tid = int(logits[0, pos].argmax().item())
                single_subtoken = tok.decode([tid]).strip()
                # Keep as one whitespace-token prediction; compared to the
                # whitespace-split gold directly.
                predicted = [single_subtoken] if single_subtoken else []

        elif strategy_name == "mlm_iterative_confident":
            mask_tok = tok.mask_token or "[MASK]"
            # Length-known: insert exactly `gap_len_subtok` masks for THIS tokenizer.
            template = format_mlm(ctx, meta, cond, num_masks=gap_len_subtok,
                                  mask_token=mask_tok)
            ids = tok(template, return_tensors="pt", truncation=True,
                      max_length=cfg.max_seq_length)["input_ids"].to(device)
            predicted_ids: list[int] = []
            mask_id = tok.mask_token_id
            for _ in range(gap_len_subtok):
                mask_positions = (ids == mask_id).nonzero(as_tuple=False)
                if mask_positions.numel() == 0:
                    break
                with torch.inference_mode():
                    logits = mdl(ids).logits
                best_pos_idx, best_conf = 0, -1.0
                for i, (_, pos) in enumerate(mask_positions.tolist()):
                    conf = float(logits[0, pos].softmax(dim=-1).max().item())
                    if conf > best_conf:
                        best_conf, best_pos_idx = conf, i
                _, pos = mask_positions[best_pos_idx].tolist()
                tid = int(logits[0, pos].argmax().item())
                ids[0, pos] = tid
                predicted_ids.append(tid)
            # Decode the full subtoken string then re-split on whitespace to
            # match the gold word-level representation.
            full = tok.decode(predicted_ids, skip_special_tokens=True).strip()
            predicted = full.split() if full else []

        elif strategy_name == "glm_ar_span":
            # GLM requires its tokenizer's build_inputs_for_generation helper
            # (which sets up the 2D positional encodings + span-generation
            # attention mask) and decoding ended by eop_token_id. Without
            # these, vanilla .generate() produces garbage — the mask token
            # is tokenised correctly but the model lacks the structural
            # signals needed to fill the blank autoregressively.
            part_a = format_glm_part_a(ctx, meta, cond)
            enc = tok(part_a, return_tensors="pt", truncation=True,
                      max_length=cfg.max_seq_length - max_new - 4)
            enc = tok.build_inputs_for_generation(enc, max_gen_length=max_new)
            enc = {k: v.to(device) for k, v in enc.items()}
            prefix_len = enc["input_ids"].shape[1]
            with torch.inference_mode():
                gen = mdl.generate(
                    **enc,
                    max_length=prefix_len + max_new,
                    eos_token_id=getattr(tok, "eop_token_id", tok.eos_token_id),
                    num_beams=1,
                    do_sample=False,
                )
            # Slice off the input prefix; decode only the generated span.
            completion = tok.decode(
                gen[0][prefix_len:], skip_special_tokens=True
            ).strip()
            predicted = completion.split()[:gap_len_words]

        else:
            raise ValueError(f"No inline handler for strategy {strategy_name!r}")

        preds.append(
            PredictionRow(
                item_id=it.item_id,
                corpus=it.corpus,
                top_k=[predicted],
                logp_learner=float("nan"),
                logp_native=None,
            )
        )
    return preds


def run_one(name: str, cfg: ModelConfig, strategy_name: str, items: list,
            cond: ConditioningConfig, infer_cfg: InferConfig,
            device: str, out_root: Path, sample_csv: Path) -> Path:
    # Folder name carries BOTH the model hash AND the decoding strategy so
    # (model, strategy) pairs are never conflated in the artifact tree.
    run_dir = out_root / (
        f"sample-benchmark-{model_hash(cfg)}-{cfg.name}-"
        f"{strategy_name}-{timestamp()}"
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(sample_csv, run_dir / "sample.csv")

    # write cloze items
    with (run_dir / "cloze.jsonl").open("w", encoding="utf-8") as fh:
        for it in items:
            fh.write(it.to_json() + "\n")

    t0 = time.time()
    preds = _run_inference(strategy_name, cfg, items, cond, infer_cfg, device)

    dur = time.time() - t0

    # dump predictions
    with (run_dir / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for p in preds:
            fh.write(json.dumps(asdict(p)) + "\n")

    # Per-item scoring with EXPLICIT tokenizer-named length columns.
    # `gap_length` (implicit, whitespace-based) is deliberately NOT emitted —
    # every length column carries its tokenizer identity in its name.
    items_by_id = {it.item_id: it for it in items}
    ref_field = reference_field()
    rows = []
    for p in preds:
        it = items_by_id.get(p.item_id)
        if it is None:
            continue
        first = p.top_k[0] if p.top_k else []
        row = {
            "item_id": it.item_id,
            "cefr": str(it.meta.get("cefr", "UNK")),
            "gold": " ".join(it.gap_tokens),
            "pred": " ".join(first),
            "em": exact_match(first, it.gap_tokens),
            "top_1": top_k_hit(p.top_k, it.gap_tokens, k=1),
            "decoding_strategy": strategy_name,
            "model": cfg.name,
        }
        # Expand gap_token_counts dict into explicit named columns.
        row.update(it.gap_token_counts)
        rows.append(row)

    df_rows = pd.DataFrame(rows)
    if len(df_rows):
        df_rows.to_csv(run_dir / "per_item.csv", index=False)
        summary = {
            "n": int(len(df_rows)),
            "em_mean": float(df_rows["em"].mean()),
            "top_1_mean": float(df_rows["top_1"].mean()),
            "duration_seconds": round(dur, 2),
            "reference_length_field": ref_field,
        }
        pd.DataFrame([summary]).to_csv(run_dir / "summary.csv", index=False)
        df_rows.groupby("cefr").agg(
            n=("em", "size"), em_mean=("em", "mean")
        ).reset_index().to_csv(run_dir / "by_cefr.csv", index=False)
        # Length-stratified EM for EVERY tokenizer we know about, not just
        # the reference — lets downstream analysis pick any granularity.
        length_cols = [c for c in df_rows.columns if c.startswith("n_") and c.endswith("_tokens")]
        for col in length_cols:
            df_rows.groupby(col).agg(
                n=("em", "size"), em_mean=("em", "mean")
            ).reset_index().to_csv(run_dir / f"by_{col}.csv", index=False)
    else:
        pd.DataFrame([{"n": 0}]).to_csv(run_dir / "summary.csv", index=False)

    from ilmcloze.infer.strategies import get as _get_strategy

    (run_dir / "run.json").write_text(
        json.dumps(
            {
                "name": name,
                "timestamp_utc": timestamp(),
                "git_sha": git_sha(),
                "model": asdict(cfg),
                "decoding_strategy": asdict(_get_strategy(strategy_name)),
                "conditioning": asdict(cond),
                "infer": asdict(infer_cfg),
                "duration_seconds": round(dur, 2),
                "n_items": len(items),
                "reference_length_field": ref_field,
                "token_counters": counter_manifest(),
            },
            indent=2,
            default=str,
        )
    )
    return run_dir


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--sample",
        type=Path,
        default=REPO_ROOT / "samples" / "efcamdat-smoke-100.csv",
        help="CSV built by make_smoke_sample.py",
    )
    p.add_argument("--out-root", type=Path,
                   default=REPO_ROOT / "artifacts" / "smoke")
    p.add_argument("--device", default="cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-gap-len", type=int, default=3)
    p.add_argument(
        "--only",
        default=None,
        help="Comma-separated baseline names to run (e.g. 'distilbert-mlm').",
    )
    args = p.parse_args()

    random.seed(args.seed)
    if not args.sample.exists():
        raise SystemExit(f"Sample file not found: {args.sample} — run make_smoke_sample.py first")

    items = build_cloze(args.sample, args.seed, args.max_gap_len)
    print(f"Built {len(items)} cloze items from {args.sample.name}")

    cond = ConditioningConfig(enabled=False)  # zero-shot baselines, no cond prefix
    infer_cfg = InferConfig(top_k=(1,), sample=False)

    selected = set(s.strip() for s in args.only.split(",")) if args.only else None

    args.out_root.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    for name, cfg, strategy_name in BASELINES:
        if selected and name not in selected:
            continue
        print(f"\n=== {name} ({cfg.hf_repo}, decoding={strategy_name}) ===")
        run_dir = run_one(name, cfg, strategy_name, items, cond, infer_cfg,
                          args.device, args.out_root, args.sample)
        summary = pd.read_csv(run_dir / "summary.csv").to_dict(orient="records")[0]
        results.append({
            "name": name,
            "model": cfg.name,
            "decoding_strategy": strategy_name,
            "dir": str(run_dir.relative_to(REPO_ROOT)),
            **summary,
        })
        print(f"  → {run_dir.relative_to(REPO_ROOT)}")
        print(f"    EM={summary.get('em_mean', 'n/a')}  n={summary.get('n', 0)}  t={summary.get('duration_seconds', 'n/a')}s")

    roll = args.out_root / f"rollup-{timestamp()}.csv"
    pd.DataFrame(results).to_csv(roll, index=False)
    print(f"\nRollup: {roll.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
