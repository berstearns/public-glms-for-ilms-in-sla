"""Shared CLI machinery for every stage module.

Every stage's ``__main__.py`` is a one-call shim:

    from pipeline._cli import run_stage
    from .config import MyConfig, SECTION_MAP
    from .runner import run_me

    if __name__ == "__main__":
        run_stage(
            description="what this stage does",
            config_cls=MyConfig,
            section_map=SECTION_MAP,
            runner=run_me,
        )

The config dataclass must:
  - have one field per SECTION_MAP key, each defaulting via ``field(default_factory=...)``;
  - NOT be frozen itself (the top-level instance must be mutable so we can
    replace sections in-place during CLI override merge) — its *sub*-sections
    may be frozen and will be replaced via ``dataclasses.replace``;
  - optionally define ``validate(self)`` (raise from there on invalid configs);
  - optionally define ``to_yaml(self) -> str`` (used for the run-dir snapshot).

The runner signature is ``runner(config, run_dir) -> None``. ``run_dir`` is
either a fresh ``{hash}_{timestamp}_{name}/`` folder (when ``use_run_dir=True``)
or ``Path(".")`` (deterministic stages; they write to ``{output_dir}/{name}/{stage}/``).
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import logging
from dataclasses import asdict, fields, is_dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

import yaml

T = TypeVar("T")
log = logging.getLogger("cli")


# ── !include loader ────────────────────────────────────────────────
# Composite YAMLs use `!include path/to/other.yaml` to inline another
# file. Paths are relative to the including file; nested includes follow
# their own parent.

class IncludeLoader(yaml.SafeLoader):
    """SafeLoader + an ``!include <path>`` tag for YAML composition."""

    def __init__(self, stream) -> None:
        name = getattr(stream, "name", None)
        self._root = Path(name).parent if name else Path.cwd()
        super().__init__(stream)


def _include_constructor(loader: IncludeLoader, node: yaml.Node) -> Any:
    raw = loader.construct_scalar(node)
    inc_path = Path(raw)
    full = inc_path if inc_path.is_absolute() else loader._root / inc_path
    with full.open() as f:
        return yaml.load(f, IncludeLoader)


IncludeLoader.add_constructor("!include", _include_constructor)


def load_yaml_with_includes(path: Path) -> Any:
    with path.open() as f:
        return yaml.load(f, IncludeLoader) or {}


# ── type coercion for argparse from dataclass fields ───────────────

def _python_type_for_field(f: dataclasses.Field) -> type | None:
    """Dataclass field type → argparse ``type=`` callable, or ``None`` to skip.

    Returns ``None`` for bool (handled separately) and for list/tuple/dict
    (not CLI-overridable — edit the YAML).
    """
    origin = getattr(f.type, "__origin__", None)
    if origin in (list, dict, tuple):
        return None
    if f.type is bool or f.type == "bool":
        return None
    if f.type in (int, float, str):
        return f.type
    type_str = str(f.type)
    if "list" in type_str or "tuple" in type_str or "dict" in type_str:
        return None
    return {"int": int, "float": float, "str": str}.get(type_str, str)


def build_parser(description: str, section_map: dict[str, type]) -> argparse.ArgumentParser:
    """Build a parser with ``--config`` and one ``--section.field`` flag per field."""
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "-c", "--config", type=Path, required=True,
        help="Path to the stage's YAML config file.",
    )
    for section_name, cls in section_map.items():
        group = parser.add_argument_group(f"{section_name} overrides")
        for f in fields(cls):
            key = f"--{section_name}.{f.name}"
            if f.type is bool or f.type == "bool":
                group.add_argument(
                    key, type=str, default=None, metavar="BOOL",
                    help=f"{section_name}.{f.name} (true/false)",
                )
                continue
            py_type = _python_type_for_field(f)
            if py_type is not None:
                group.add_argument(key, type=py_type, default=None)
    return parser


# ── YAML → dataclass tree ──────────────────────────────────────────

def _build_sub_config(cls: type, raw: dict[str, Any]) -> Any:
    if not isinstance(raw, dict):
        raw = {}
    valid = {f.name for f in fields(cls)}
    filtered = {k: v for k, v in raw.items() if k in valid}
    for f in fields(cls):
        if f.name in filtered and isinstance(filtered[f.name], list):
            ann = str(f.type)
            if ann.startswith("tuple"):
                filtered[f.name] = tuple(filtered[f.name])
    return cls(**filtered)


def hydrate_config(
    raw: dict[str, Any],
    config_cls: type[T],
    section_map: dict[str, type],
) -> T:
    subs = {
        section_key: _build_sub_config(cls, raw.get(section_key, {}) or {})
        for section_key, cls in section_map.items()
    }
    return config_cls(**subs)


def load_config_from_yaml(
    yaml_path: Path,
    config_cls: type[T],
    section_map: dict[str, type],
) -> T:
    if not yaml_path.exists():
        raise FileNotFoundError(f"config not found: {yaml_path}")
    raw = load_yaml_with_includes(yaml_path)
    return hydrate_config(raw, config_cls, section_map)


# ── dotted-key CLI overrides ───────────────────────────────────────

def apply_cli_overrides(config: T, overrides: dict[str, Any]) -> T:
    """Walk ``section.field``-keyed overrides onto the config.

    Works whether the section dataclass is frozen (uses ``dataclasses.replace``)
    or not (uses ``setattr``). The top-level ``config`` itself must NOT be
    frozen — we rebind its section attributes.
    """
    for dotted_key, value in overrides.items():
        if value is None:
            continue
        section, _, param = dotted_key.partition(".")
        if not section or not param:
            continue
        sub = getattr(config, section, None)
        if sub is None or not is_dataclass(sub):
            log.warning("ignoring unknown override: %s", dotted_key)
            continue
        if not any(f.name == param for f in fields(sub)):
            log.warning("ignoring unknown override: %s", dotted_key)
            continue
        current = getattr(sub, param)
        if isinstance(current, bool):
            value = str(value).lower() in ("true", "1", "yes", "y")
        elif isinstance(current, int) and not isinstance(current, bool):
            value = int(value)
        elif isinstance(current, float):
            value = float(value)
        try:
            setattr(sub, param, value)
        except dataclasses.FrozenInstanceError:
            setattr(config, section, replace(sub, **{param: value}))
    if hasattr(config, "validate"):
        config.validate()
    return config


# ── run directory + config snapshot ────────────────────────────────

def _to_yaml_text(config: Any) -> str:
    if hasattr(config, "to_yaml"):
        return config.to_yaml()
    return yaml.dump(asdict(config), default_flow_style=False, sort_keys=False)


def make_run_dir(experiment_name: str, output_root: Path, config_yaml_text: str) -> Path:
    config_hash = hashlib.sha256(config_yaml_text.encode()).hexdigest()[:8]
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{config_hash}_{timestamp}_{experiment_name}"
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def stage_artifact_dir(output_dir: str, name: str, stage_key: str) -> Path:
    """``{output_dir}/{name}/{stage_key}/`` — deterministic stages' output dir.

    Upstream stages write here; downstream stages read from it using the
    same convention. Created on access.
    """
    out = Path(output_dir) / name / stage_key
    out.mkdir(parents=True, exist_ok=True)
    return out


# ── the one-call bootstrapper ──────────────────────────────────────

def run_stage(
    description: str,
    config_cls: type[T],
    section_map: dict[str, type],
    runner: Callable[[T, Path], Any],
    *,
    use_run_dir: bool = False,
) -> None:
    """Parse → load → override → seed+log → (maybe) make run_dir → runner."""
    parser = build_parser(description, section_map)
    args = parser.parse_args()

    config = load_config_from_yaml(args.config, config_cls, section_map)
    overrides = {k: v for k, v in vars(args).items() if "." in k and v is not None}
    config = apply_cli_overrides(config, overrides)

    exp = getattr(config, "experiment", None)
    verbose = bool(getattr(exp, "verbose", False))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    seed = int(getattr(exp, "seed", 0) or 0)
    if seed:
        _seed_everything(seed)

    config_yaml_text = _to_yaml_text(config)

    if use_run_dir and exp is not None:
        output_root = Path(getattr(exp, "output_dir", "."))
        run_name = getattr(exp, "name", "run")
        run_dir = make_run_dir(run_name, output_root, config_yaml_text)
        if getattr(exp, "save_config_snapshot", True):
            (run_dir / "config_used.yaml").write_text(config_yaml_text)
    else:
        run_dir = Path(".")

    runner(config, run_dir)

    if use_run_dir:
        log.info("✓ done. run_dir: %s", run_dir)
    else:
        log.info("✓ done.")


def _seed_everything(seed: int) -> None:
    """Best-effort deterministic seeding. No-ops cleanly if deps are absent."""
    import os
    import random

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass
