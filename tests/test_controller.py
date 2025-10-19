import pytest

from src.controller import run_controller
from src.agents_mock import MockAgent
from src.schemas import get_envelope_validator

def test_mock_consensus():
    a = MockAgent("A", "TRUE")
    b = MockAgent("B", "TRUE")
    out = run_controller("Return TRUE", a, b, max_rounds=4, kind=None)
    assert out["status"] == "CONSENSUS"
    assert out["canonical_text"] == "TRUE"


class InvalidNumberAgent:
    def __init__(self, name: str):
        self.name = name
        self._turn = 0

    def step(self, task, transcript):
        if self._turn == 0:
            self._turn += 1
            env = {"tag": "[CONTACT]", "status": "PROPOSED", "content": {"proposal": "working"}}
            return env, str(env)
        env = {
            "tag": "[SOLVED]",
            "status": "SOLVED",
            "content": {"note": "not numeric"},
            "final_solution": {"canonical_text": "NOT_A_NUMBER"},
        }
        return env, str(env)


def test_controller_schema_rejects_invalid_message():
    validator = get_envelope_validator("prompts/schemas/envelope.number.schema.json")
    a = InvalidNumberAgent("BadNumber")
    b = MockAgent("Good", "42")
    with pytest.raises(ValueError) as exc:
        run_controller("Return a number", a, b, max_rounds=2, kind="number", schema_validator=validator)
    assert "BadNumber" in str(exc.value)


def test_controller_schema_accepts_valid_messages():
    validator = get_envelope_validator("prompts/schemas/envelope.number.schema.json")
    a = MockAgent("A", "42")
    b = MockAgent("B", "42")
    out = run_controller(
        "Return 42",
        a,
        b,
        max_rounds=2,
        kind="number",
        schema_validator=validator,
    )
    assert out["status"] == "CONSENSUS"
    assert out["canonical_text"] == "42"
