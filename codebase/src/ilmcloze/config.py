"""Typed configuration loading.

Every pipeline stage is driven by an experiment YAML. The YAML may ``extend``
a base config (simple single-inheritance) and override any leaf field.
Values resolve to :class:`ExperimentConfig`, a frozen dataclass, so downstream
code never sees stringly-typed config dicts.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, TypeVar

import yaml

T = TypeVar("T")


# ---------------------------------------------------------------------------
# Nested config dataclasses


@dataclass(frozen=True)
class DataConfig:
    """Where corpora live on disk."""

    splits_dir: str
    efcamdat_train: str = "norm-EFCAMDAT-train.csv"
    efcamdat_test: str = "norm-EFCAMDAT-test.csv"
    efcamdat_remainder: str = "norm-EFCAMDAT-remainder.csv"
    celva_sp: str = "norm-CELVA-SP.csv"
    celva_sp_label: str = "norm-CELVA-SP-label.csv"
    kupa_keys: str = "norm-KUPA-KEYS.csv"
    kupa_keys_label: str = "norm-KUPA-KEYS-label.csv"
    andrew_test: str = "norm-andrew100k-test-label.csv"
    andrew_train: str = "norm-andrew100k-train-label.csv"
    universal_label: str = "norm-universal-cefr-label.csv"


@dataclass(frozen=True)
class ClozeConfig:
    """How cloze items are built from raw text."""

    gap_type: str = "multi_token"  # single_token | multi_token | function | content | l2_loci
    span_length_lambda: float = 3.0
    span_max_length: int = 8
    mask_budget: float = 0.15
    num_gaps_per_text: int = 1
    context_condition: str = "II"  # I | II | III
    synth_corruption_rate: float = 0.05
    l2_loci: tuple[str, ...] = ("DET", "PREP", "VERB:FORM", "VERB:SVA")
    exclude_sentence_initial: bool = False


@dataclass(frozen=True)
class ConditioningConfig:
    """Learner-conditioning prefix format."""

    enabled: bool = True
    include_l1: bool = True
    include_cefr: bool = True
    include_errprof: bool = True
    errprof_num_clusters: int = 16
    unknown_token: str = "UNK"


@dataclass(frozen=True)
class ModelConfig:
    """Model backbone specification."""

    name: str  # e.g. "glm-base", "glm-roberta-large", "gpt2", "bert-large-cased"
    kind: str  # glm | nwp | mlm
    hf_repo: str
    hf_revision: str = "main"
    max_seq_length: int = 512
    torch_dtype: str = "float32"  # float16 | bfloat16 | float32


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters."""

    regime: str = "continued_pretrain"  # zero_shot | continued_pretrain | sft
    epochs: int = 3
    batch_size: int = 8
    grad_accum_steps: int = 32
    learning_rate: float = 1e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    seed: int = 42
    save_every_steps: int = 500
    eval_every_steps: int = 500
    resume_from: str | None = None


@dataclass(frozen=True)
class InferConfig:
    """Cloze inference settings."""

    batch_size: int = 16
    top_k: tuple[int, ...] = (1, 5)
    sample: bool = False
    temperature: float = 1.0
    num_samples: int = 1


@dataclass(frozen=True)
class EvalConfig:
    """What to report."""

    metrics: tuple[str, ...] = ("em", "top_k", "learner_plausibility", "kl", "js")
    stratify_by: tuple[str, ...] = ("cefr", "l1", "gap_length", "gap_position")
    emit_per_item_scores: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level config, one per experiment variant."""

    name: str
    description: str
    data: DataConfig
    cloze: ClozeConfig
    conditioning: ConditioningConfig
    model: ModelConfig
    train: TrainConfig
    infer: InferConfig
    eval: EvalConfig
    artifacts_dir: str = "./artifacts"
    seed: int = 42
    transfer_corpora: tuple[str, ...] = field(default_factory=tuple)


# ---------------------------------------------------------------------------
# Loading


def _merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``override`` into ``base``; override wins on scalar leaves."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise TypeError(f"{path} did not parse to a mapping")
    return data


def _resolve_extends(path: Path, visited: set[Path] | None = None) -> dict[str, Any]:
    visited = visited or set()
    resolved = path.resolve()
    if resolved in visited:
        raise ValueError(f"Cycle detected while resolving `extends` at {resolved}")
    visited.add(resolved)

    data = _load_yaml(path)
    extends = data.pop("extends", None)
    if extends is None:
        return data
    parent_path = (path.parent / extends).resolve()
    parent = _resolve_extends(parent_path, visited=visited)
    return _merge(parent, data)


def _instantiate(cls: type[T], data: dict[str, Any]) -> T:
    """Instantiate a frozen dataclass from a dict, recursing on nested ones."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    kwargs: dict[str, Any] = {}
    for f in fields(cls):
        if f.name not in data:
            continue
        v = data[f.name]
        if is_dataclass(f.type) and isinstance(v, dict):
            kwargs[f.name] = _instantiate(f.type, v)
        elif isinstance(v, list) and f.type in (tuple, "tuple"):
            kwargs[f.name] = tuple(v)
        else:
            kwargs[f.name] = v
    # Also coerce any list-typed field to tuple where the annotation is tuple[...]
    for f in fields(cls):
        if f.name in kwargs and isinstance(kwargs[f.name], list):
            ann = str(f.type)
            if ann.startswith("tuple"):
                kwargs[f.name] = tuple(kwargs[f.name])
    return cls(**kwargs)  # type: ignore[return-value]


def load_experiment(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML (with ``extends`` support) into a typed config."""
    path = Path(path)
    raw = _resolve_extends(path)
    # Instantiate nested dataclasses explicitly
    nested: dict[str, Any] = {}
    for key, cls in (
        ("data", DataConfig),
        ("cloze", ClozeConfig),
        ("conditioning", ConditioningConfig),
        ("model", ModelConfig),
        ("train", TrainConfig),
        ("infer", InferConfig),
        ("eval", EvalConfig),
    ):
        section = raw.get(key, {})
        nested[key] = _instantiate(cls, section) if isinstance(section, dict) else section
    top = {k: v for k, v in raw.items() if k not in nested}
    top.update(nested)
    if "transfer_corpora" in top and isinstance(top["transfer_corpora"], list):
        top["transfer_corpora"] = tuple(top["transfer_corpora"])
    return ExperimentConfig(**top)  # type: ignore[arg-type]


def dump_experiment(cfg: ExperimentConfig, path: str | Path) -> None:
    """Write a resolved config back to YAML, for artifact sidecars."""
    from dataclasses import asdict

    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(asdict(cfg), fh, sort_keys=False)
