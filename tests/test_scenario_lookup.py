from src.template_loader import get_scenario


def test_matrix_scenario_lookup_strips_rep() -> None:
    scenario = get_scenario("boolean_eval_small:DSL:Boolean-ProposeCheck:rep=3")
    assert scenario["strategy"] == "DSL"
    assert scenario["roleset"] == "rolesets/boolean_proposecheck.json"
