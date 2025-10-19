from src.controller import run_controller
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

def test_mock_consensus():
    a = MockAgent("A", "TRUE")
    b = MockAgent("B", "TRUE")
    out = run_controller("Return TRUE", a, b, max_rounds=4, kind=None)
    assert out["status"] == "CONSENSUS"
    assert out["canonical_text"] == "TRUE"


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
