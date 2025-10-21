import pytest

pytestmark = pytest.mark.skip(
    reason="Legacy control-trailer/consensus disabled in simplified fixed-turn runner"
)

from src.controller import (
    HandshakeTracker,
    _control_summary,
    _handle_handshake_event,
    _update_control_stats,
    run_controller,
)
from src.agents_mock import MockAgent, ConciseTextAgent
from src.dsl import default_dsl_spec
from src.schemas import Envelope, get_envelope_validator
from src.strategies import Strategy, build_strategy

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


def _make_envelope(tag: str, status: str, canonical: str) -> Envelope:
    return Envelope(tag=tag, status=status, content={}, final_solution={"canonical_text": canonical})


def test_handshake_accepts_normalised_canonical_match():
    proposer = _make_envelope("[SOLVER]", "READY_TO_SOLVE", "ANSWER: 60")
    acceptor = _make_envelope("[SOLVED]", "SOLVED", "ANSWER: 60.0")
    tracker = HandshakeTracker()
    proposal_event = tracker.observe("a", proposer, 1)
    assert proposal_event and proposal_event["kind"] == "proposal"
    acceptance = tracker.observe("b", acceptor, 1)
    assert acceptance and acceptance["kind"] == "accepted"
    assert acceptance["canonical"] == "ANSWER: 60"


def test_handshake_rejects_mismatched_canonical():
    proposer = _make_envelope("[SOLVER]", "READY_TO_SOLVE", "ANSWER: 60")
    acceptor = _make_envelope("[SOLVED]", "SOLVED", "ANSWER: 61")
    tracker = HandshakeTracker()
    tracker.observe("a", proposer, 1)
    acceptance = tracker.observe("b", acceptor, 1)
    assert acceptance and acceptance["kind"] == "error"
    assert acceptance["error"] == "illegal_transition"


def test_handshake_requires_valid_proposer_status():
    proposer = _make_envelope("[SOLVER]", "NEED_PEER", "ANSWER: 60")
    acceptor = _make_envelope("[SOLVED]", "SOLVED", "ANSWER: 60")
    tracker = HandshakeTracker()
    proposal = tracker.observe("a", proposer, 1)
    assert proposal and proposal["kind"] == "error"
    assert proposal["error"] == "illegal_transition"


def test_handle_handshake_event_records_errors():
    stats = {"trailer_missing_ct": 0, "invalid_trailer_ct": 0, "retry_count": 0, "first_error": None, "error_log": []}
    error_event = {"kind": "error", "error": "missing_trailer", "round": 2, "actor": "a"}
    accepted = _handle_handshake_event(stats, error_event)
    assert accepted is None
    assert stats["trailer_missing_ct"] == 1
    assert stats["handshake_error_ct"] == 1
    assert "missing_trailer" in stats["error_log"]

    accepted_event = {"kind": "accepted", "canonical": "ANSWER: 5", "actor": "b", "round": 2}
    result = _handle_handshake_event(stats, accepted_event)
    assert result is accepted_event


def test_control_stats_summary_tracks_overflow_and_stop_reasons():
    stats = {"trailer_missing_ct": 0, "invalid_trailer_ct": 0, "retry_count": 0, "first_error": None, "error_log": []}
    telemetry = {
        "retry_count": 0,
        "body_len": 10,
        "trailer_len": 42,
        "stopped_on": "ctrl_suffix",
        "stopped_on_ctrl": True,
        "tokens_reserved": 16,
        "tokens_used_trailer": 20,
        "tokens_used_body": 12,
        "tokens_overflow": 4,
        "tokens_used_total": 32,
        "has_tail": False,
        "trailer_start": 5,
        "trailer_end": 47,
        "has_ctrl": True,
        "closed_ctrl": True,
        "tokens_body_budget": 12,
        "tokens_trailer_budget": 20,
        "tokens_body_overflow": 0,
        "tokens_trailer_overflow": 0,
        "suffix_triggered": True,
    }
    env = Envelope(
        tag="[SOLVER]",
        status="READY_TO_SOLVE",
        content={"control": {"telemetry": telemetry}},
        final_solution={"canonical_text": "ANSWER: 5"},
    )

    _update_control_stats(stats, env, 1)

    telemetry_max = {
        "retry_count": 0,
        "body_len": 5,
        "trailer_len": 30,
        "stopped_on": "max_new_tokens",
        "stopped_on_ctrl": False,
        "tokens_reserved": 16,
        "tokens_used_trailer": 10,
        "tokens_used_body": 22,
        "tokens_overflow": 0,
        "tokens_used_total": 32,
        "has_tail": False,
        "trailer_start": 3,
        "trailer_end": 33,
        "has_ctrl": True,
        "closed_ctrl": False,
        "tokens_body_budget": 22,
        "tokens_trailer_budget": 10,
        "tokens_body_overflow": 0,
        "tokens_trailer_overflow": 0,
        "suffix_triggered": False,
    }
    env_max = Envelope(
        tag="[PLAN]",
        status="PROPOSED",
        content={"control": {"telemetry": telemetry_max}},
    )

    _update_control_stats(stats, env_max, 2)

    summary = _control_summary(stats)
    assert summary["overflow_turns"] == 1
    assert summary["max_overflow"] == 4
    assert summary["stopped_on_ctrl"] == 1
    assert summary["stopped_on_max_new_tokens"] == 1
    assert summary["needs_higher_reserve"] is True
    assert summary["tokens_used_trailer_total"] == 30


def test_update_control_stats_registers_incomplete_trailer():
    stats = {"trailer_missing_ct": 0, "invalid_trailer_ct": 0, "retry_count": 0, "first_error": None, "error_log": []}
    telemetry = {
        "retry_count": 0,
        "body_len": 5,
        "trailer_len": 12,
        "stopped_on": "max_new_tokens",
        "stopped_on_ctrl": False,
        "tokens_reserved": 24,
        "tokens_used_trailer": 8,
        "tokens_used_body": 16,
        "tokens_overflow": 0,
        "tokens_used_total": 24,
        "has_tail": True,
        "trailer_start": 4,
        "trailer_end": 20,
        "has_ctrl": True,
        "closed_ctrl": False,
    }
    env = Envelope(
        tag="[PLAN]",
        status="PROPOSED",
        content={"control": {"telemetry": telemetry}},
    )

    _update_control_stats(stats, env, 1)

    assert "ERR_TRAILER_INCOMPLETE" in stats["error_log"]
    assert stats["invalid_trailer_ct"] >= 1
