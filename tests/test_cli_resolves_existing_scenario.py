from src.main import _build_parser, _resolve_scenario_argument
from src.template_loader import get_scenario


def _parse(argv: list[str]):
    parser = _build_parser()
    args = parser.parse_args(argv)
    _resolve_scenario_argument(parser, args)
    return args


def test_cli_args_resolve_existing_scenario():
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
            "--mock",
        ]
    )
    scenario = get_scenario(args.scenario)
    assert scenario["strategy"] == "DSL"
    assert str(scenario["roleset"]).endswith("boolean_proposecheck.json")
