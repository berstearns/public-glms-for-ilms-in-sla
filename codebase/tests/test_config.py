"""Config loader: ``extends`` merging and dataclass instantiation."""

from __future__ import annotations

from pathlib import Path

from ilmcloze.config import load_experiment


def test_load_base(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        "name: base\n"
        "description: ''\n"
        "data: {splits_dir: /tmp}\n"
        "cloze: {gap_type: single_token}\n"
        "conditioning: {enabled: true}\n"
        "model: {name: m, kind: glm, hf_repo: r}\n"
        "train: {regime: zero_shot}\n"
        "infer: {top_k: [1, 5]}\n"
        "eval: {}\n"
    )
    cfg = load_experiment(base)
    assert cfg.name == "base"
    assert cfg.cloze.gap_type == "single_token"
    assert cfg.model.kind == "glm"


def test_extends(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    base.write_text(
        "name: base\n"
        "description: ''\n"
        "data: {splits_dir: /tmp}\n"
        "cloze: {gap_type: multi_token, span_length_lambda: 3.0}\n"
        "conditioning: {enabled: true}\n"
        "model: {name: m, kind: glm, hf_repo: r}\n"
        "train: {}\n"
        "infer: {}\n"
        "eval: {}\n"
    )
    child = tmp_path / "child.yaml"
    child.write_text(
        "extends: base.yaml\n"
        "name: child\n"
        "description: ''\n"
        "cloze: {gap_type: single_token}\n"
    )
    cfg = load_experiment(child)
    assert cfg.name == "child"
    assert cfg.cloze.gap_type == "single_token"
    # inherited
    assert cfg.cloze.span_length_lambda == 3.0
