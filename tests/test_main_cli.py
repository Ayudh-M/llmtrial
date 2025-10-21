from __future__ import annotations

import argparse
from pathlib import Path

import pytest

from src.main import _build_parser, _resolve_log_paths, _resolve_scenario_argument


def _parse(argv: list[str]) -> argparse.Namespace:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _resolve_scenario_argument(parser, args)
    return args


def test_explicit_scenario_preserved() -> None:
    args = _parse(["--scenario", "demo-task"])
    assert args.scenario == "demo-task"


def test_matrix_arguments_form_scenario_id() -> None:
    args = _parse(
        [
            "--dataset",
            "boolean_eval_small",
            "--language",
            "DSL",
            "--pair",
            "Boolean-ProposeCheck",
            "--rep",
            "2",
        ]
    )
    assert args.scenario == "boolean_eval_small:DSL:Boolean-ProposeCheck:rep=2"


def test_matrix_arguments_missing_field_errors(capsys: pytest.CaptureFixture[str]) -> None:
    parser = _build_parser()
    with pytest.raises(SystemExit):
        args = parser.parse_args(["--dataset", "boolean_eval_small", "--language", "DSL"])
        _resolve_scenario_argument(parser, args)
    stderr = capsys.readouterr().err
    assert "--pair" in stderr


def test_logdir_adjusts_default_paths(tmp_path: Path) -> None:
    parser = _build_parser()
    logdir = tmp_path / "logs"
    args = parser.parse_args(
        [
            "--scenario",
            "demo",
            "--logdir",
            str(logdir),
        ]
    )
    csv_path, jsonl_path = _resolve_log_paths(args)
    assert csv_path == logdir / "runs.csv"
    assert jsonl_path == logdir / "runs.jsonl"
    assert csv_path.name == "runs.csv"
    assert jsonl_path.name == "runs.jsonl"

