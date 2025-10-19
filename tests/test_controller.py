import pytest

from src.agents_mock import MockAgent
from src.controller import run_controller
from src.strategies import build_strategy


def test_json_schema_consensus():
    a = MockAgent("Agent A", "42", strategy=build_strategy("json_schema"))
    b = MockAgent("Agent B", "42", strategy=build_strategy("json_schema"))
    result = run_controller("Return the answer", a, b, max_rounds=2)
    assert result["status"] == "CONSENSUS"
    assert result["canonical_text"] == "42"
    assert result["analytics"]["a"] == {}


def test_pseudocode_strategy_normalises_output():
    messy = "- STEP 1: add numbers\n- RETURN 5"
    strategy = build_strategy("pseudocode")
    a = MockAgent("A", messy, strategy=strategy)
    b = MockAgent("B", messy, strategy=strategy)
    result = run_controller("Return pseudocode", a, b, max_rounds=2)
    final = result["final_messages"]["a"]["envelope"]["final_solution"]
    assert final["canonical_text"].startswith("- STEP 1: add numbers")
    assert final["return_value"] == "5"


def test_symbolic_acl_invalid_message_raises():
    class BadACLAgent:
        def __init__(self):
            self.strategy = build_strategy("symbolic_acl")
            self.name = "BadACL"

        def step(self, task, transcript):  # type: ignore[override]
            return {
                "status": "PROPOSED",
                "tag": "[CONTACT]",
                "content": {"acl": "not valid"},
            }, "raw"

    good = MockAgent("Good", "OK", strategy=build_strategy("symbolic_acl"))
    bad = BadACLAgent()
    with pytest.raises(ValueError):
        run_controller("Test", bad, good, max_rounds=1)


def test_emergent_dsl_transcript_contains_parse():
    strategy = build_strategy("emergent_dsl")
    a = MockAgent("A", "Define component", strategy=strategy)
    b = MockAgent("B", "Define component", strategy=strategy)
    result = run_controller("Use DSL", a, b, max_rounds=1)
    entry = result["transcript"][0]
    assert entry["strategy"]["dsl"]["intent"] == "PLAN"
