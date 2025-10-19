from src.controller import run_controller
from src.agents_mock import MockAgent
from src.dsl import default_dsl_spec

def test_mock_consensus():
    spec = default_dsl_spec()
    validator = spec.create_validator()
    a = MockAgent("A", "TRUE")
    b = MockAgent("B", "TRUE")
    out = run_controller("Return TRUE", a, b, max_rounds=4, kind=None, dsl_validator=validator)
    assert out["status"] == "CONSENSUS"
    assert out["canonical_text"] == "TRUE"
    assert out["final_message"] is not None
    assert out["final_message"]["dsl"]["canonical_text"] == "TRUE"
    assert out["final_message"]["actor"] == "b"
    assert out["dsl_trace"] and out["dsl_trace"][-1]["status"] == "SOLVED"
