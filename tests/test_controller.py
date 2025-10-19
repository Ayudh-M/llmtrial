from src.controller import run_controller
from src.agents_mock import MockAgent, ConciseTextAgent
from src.strategies import build_strategy
import pytest

from src.agents_mock import MockAgent
from src.strategies import Strategy


class TrackingStrategy(Strategy):
    def __init__(self):
        super().__init__(name="tracking", decoding={"max_new_tokens": 8})
        self.calls = []

    def prepare_prompt(self, task, transcript, *, actor, agent_name):
        self.calls.append(("prepare", actor, len(transcript)))
        hints = super().prepare_prompt(task, transcript, actor=actor, agent_name=agent_name)
        hints.update({"format_hint": "Use DSL envelope"})
        return hints

    def validate_message(self, envelope, *, raw, original, transcript, actor, agent_name):
        self.calls.append(("validate", actor, envelope.tag))
        meta = super().validate_message(
            envelope,
            raw=raw,
            original=original,
            transcript=transcript,
            actor=actor,
            agent_name=agent_name,
        )
        meta.update({"tokens": len(raw or "")})
        return meta

    def postprocess(self, envelope, *, raw, validation, transcript, actor, agent_name):
        self.calls.append(("postprocess", actor, envelope.tag))
        return envelope, {"parsed_tag": envelope.tag}

    def should_stop(self, envelope, *, validation, transcript, actor, agent_name):
        self.calls.append(("should_stop", actor, envelope.tag))
        return False, None
from src.controller import run_controller


class BadACLAgent(MockAgent):
    def step(self, task, transcript):  # type: ignore[override]
        env = {"tag": "[CONTACT]", "status": "PROPOSED", "content": {"acl": "bad message"}}
        return env, str(env)


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


def test_concise_text_strategy_transcript():
    strat = build_strategy(
        {
            "id": "S_concise",
            "name": "concise_text",
            "json_only": False,
            "validator": "concise_text",
            "validator_params": {"max_sentences": 1, "max_tokens": 6},
            "envelope_required": False,
            "decoding": {"do_sample": False, "temperature": 0.3, "max_new_tokens": 32},
            "prompt_snippet": "Keep answers compact.",
        }
    )
    a = ConciseTextAgent("A", ["  Ready."], "42")
    b = ConciseTextAgent("B", [" Standing by.  "], "42")
    out = run_controller("Return 42", a, b, max_rounds=3, kind=None, strategy=strat)

    assert out["status"] == "CONSENSUS"
    assert out["canonical_text"] == "42"
    texts = [entry["envelope"]["text"] for entry in out["transcript"]]
    for text in texts:
        assert text == text.strip()
        assert len(text.split()) <= 6
def test_controller_tracks_strategy_hooks():
    strat_a = TrackingStrategy()
    strat_b = TrackingStrategy()
    a = MockAgent("A", "TRUE", strategy=strat_a)
    b = MockAgent("B", "TRUE", strategy=strat_b)
    out = run_controller("Return TRUE", a, b, max_rounds=4, kind=None)

    assert out["status"] == "CONSENSUS"
    first_entry = out["transcript"][0]
    assert first_entry["strategy"]["name"] == "tracking"
    assert first_entry["strategy"]["hooks"]["prepare"]["format_hint"] == "Use DSL envelope"
    assert first_entry["strategy"]["hooks"]["validation"]["ok"] is True
    assert first_entry["strategy"]["hooks"]["postprocess"]["parsed_tag"] == "[CONTACT]"
    assert first_entry["strategy"]["hooks"]["should_stop"]["stop"] is False
    assert isinstance(first_entry["raw"], str)

    # All hooks should have been invoked for both agents.
    hook_names = [c[0] for c in strat_a.calls]
    for expected in ("prepare", "validate", "postprocess", "should_stop"):
        assert expected in hook_names
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
