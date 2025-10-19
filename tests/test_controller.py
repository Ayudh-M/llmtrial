import pytest

from src.agents_mock import MockAgent
from src.controller import run_controller


class BadACLAgent(MockAgent):
    def step(self, task, transcript):  # type: ignore[override]
        env = {"tag": "[CONTACT]", "status": "PROPOSED", "content": {"acl": "bad message"}}
        return env, str(env)


def test_mock_consensus():
    a = MockAgent("A", "TRUE")
    b = MockAgent("B", "TRUE")
    out = run_controller("Return TRUE", a, b, max_rounds=4, kind=None)
    assert out["status"] == "CONSENSUS"
    assert out["canonical_text"] == "TRUE"
    intents = out["analytics"]["intent_counts"]
    assert intents["a"].get("PROPOSE", 0) == 1
    assert intents["a"].get("SOLVED", 0) >= 1
    assert intents["b"].get("SOLVED", 0) >= 1


def test_invalid_acl_message_raises():
    a = BadACLAgent("A", "TRUE")
    b = MockAgent("B", "TRUE")
    with pytest.raises(ValueError) as exc:
        run_controller("Return TRUE", a, b, max_rounds=2, kind=None)
    assert "Invalid ACL message" in str(exc.value)


def test_no_consensus_when_solutions_differ():
    a = MockAgent("A", "TRUE")
    b = MockAgent("B", "FALSE")
    out = run_controller("Return TRUE", a, b, max_rounds=3, kind=None)
    assert out["status"] == "NO_CONSENSUS"
    intents = out["analytics"]["intent_counts"]
    assert intents["a"].get("SOLVED", 0) >= 1
    assert intents["b"].get("SOLVED", 0) >= 1
