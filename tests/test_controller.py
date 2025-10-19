from src.controller import run_controller
from src.agents_mock import MockAgent

PSEUDOCODE_TRUE = """
- STEP 1: confirm request
- RETURN TRUE
"""


def test_mock_consensus():
    a = MockAgent("A", PSEUDOCODE_TRUE)
    b = MockAgent("B", PSEUDOCODE_TRUE)
    out = run_controller("Return TRUE", a, b, max_rounds=4, kind=None)
    assert out["status"] == "CONSENSUS"
    assert out["canonical_text"] == "TRUE"
