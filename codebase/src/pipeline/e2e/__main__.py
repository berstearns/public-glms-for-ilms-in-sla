"""e2e entry point.

Wraps the standard ``run_stage`` flow but also threads the composite YAML
path through to the runner so it can resolve per-stage sub-configs.
"""
from __future__ import annotations

import logging
from pathlib import Path

import yaml

from pipeline._cli import (
    apply_cli_overrides,
    build_parser,
    load_config_from_yaml,
    load_yaml_with_includes,
    make_run_dir,
)

from .config import SECTION_MAP, E2EConfig
from .runner import run_e2e


def main() -> None:
    parser = build_parser(
        description="Run the whole pipeline from a composite YAML.",
        section_map=SECTION_MAP,
    )
    args = parser.parse_args()
    composite_path: Path = args.config

    config = load_config_from_yaml(composite_path, E2EConfig, SECTION_MAP)
    overrides = {k: v for k, v in vars(args).items() if "." in k and v is not None}
    config = apply_cli_overrides(config, overrides)
    config._composite_path = str(composite_path)  # type: ignore[attr-defined]

    verbose = bool(config.experiment.verbose)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    run_dir = make_run_dir(
        config.experiment.name,
        Path(config.experiment.output_dir),
        config.to_yaml(),
    )
    if config.experiment.save_config_snapshot:
        (run_dir / "config_used.yaml").write_text(config.to_yaml())
        expanded = load_yaml_with_includes(composite_path)
        (run_dir / "composite_expanded.yaml").write_text(
            yaml.dump(expanded, default_flow_style=False, sort_keys=False)
        )

    run_e2e(config, run_dir)


if __name__ == "__main__":
    main()
