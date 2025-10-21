import pytest
from dataclasses import replace

pytestmark = pytest.mark.skip(
    reason="Legacy control-trailer/consensus disabled in simplified fixed-turn runner"
)

from src.agents_hf import (
    HFChatAgent,
    _extract_last_json,
    _retry_instructions,
    _validate_envelope_candidate,
)
from src.control_trailer import CTRL_SUFFIX, extract_control_trailer
from src.model_loader import GenerationResult
from src.strategies import Strategy


def test_extract_last_json_returns_last_object():
    text = "noise {\"first\": 1} trailing {\"second\": {\"inner\": \"{brace}\"}} extra"
    assert _extract_last_json(text) == '{"second": {"inner": "{brace}"}}'


def test_extract_control_trailer_reports_unclosed():
    text = "Body\n<<<CTRL{\"tag\":\"[PLAN]\""
    result = extract_control_trailer(text)
    assert result["ok"] is False
    assert result["error"] == "ERR_TRAILER_UNCLOSED"
    assert result["body"] == "Body\n"


def test_validate_envelope_candidate_accepts_valid_payload():
    envelope = {
        "tag": "[CONTACT]",
        "status": "PROPOSED",
        "content": {"acl": "PROPOSE: outline => WAIT_FOR_PEER"},
        "final_solution": {"canonical_text": "ANSWER: 42"},
    }
    assert _validate_envelope_candidate(envelope) == []


def test_validate_envelope_candidate_reports_issues():
    envelope = {
        "tag": "",
        "status": "UNKNOWN",
        "content": {"acl": "bad format"},
    }
    errors = _validate_envelope_candidate(envelope)
    assert any("tag" in err for err in errors)
    assert any("status" in err for err in errors)
    assert any("ACL" in err for err in errors)


def test_retry_instructions_mentions_errors():
    message = _retry_instructions(["Missing or empty 'tag' field."])
    assert "Missing or empty 'tag' field." in message
    assert message.count("\n") >= 1


def test_trailer_only_retry_completes_truncated_trailer(monkeypatch):
    class DummyTokenizer:
        def encode(self, text, add_special_tokens=False):
            length = max(len(text) // 8, 1)
            return list(range(length))

    strategy = Strategy(
        id="ctrl",
        name="ctrl",
        json_only=False,
        decoding={"max_new_tokens": 6},
        metadata={"body_style": "control"},
    )
    agent = HFChatAgent(
        name="Agent",
        system_prompt="System",
        tokenizer=DummyTokenizer(),
        model=object(),
        strategy=strategy,
    )

    body = "Plan body\n"
    partial = (
        body
        + '<<<CTRL{"tag":"[PLAN]","status":"PROPOSED","content":{"acl":"PLAN: test => WAIT"}'
    )
    completed = (
        '<<<CTRL{"tag":"[PLAN]","status":"PROPOSED","content":{"acl":"PLAN: test => WAIT"},'
        '"final_solution":{}}CTRL>>>'
    )

    first = GenerationResult(
        text=partial,
        stop_reason="max_new_tokens",
        tokens_used=12,
        overflow_tokens=0,
        has_tail=True,
        trailer_offset=-1,
        input_tokens=5,
        max_new_tokens=16,
        body_budget=6,
        trailer_budget=10,
        tokens_reserved=16,
        body_tokens=4,
        trailer_tokens=0,
        tokens_body_overflow=0,
        tokens_trailer_overflow=0,
        suffix_triggered=False,
    )

    trailer_offset = completed.find("{")
    second = GenerationResult(
        text=completed,
        stop_reason="ctrl_suffix",
        tokens_used=8,
        overflow_tokens=0,
        has_tail=False,
        trailer_offset=trailer_offset,
        input_tokens=5,
        max_new_tokens=10,
        body_budget=0,
        trailer_budget=10,
        tokens_reserved=18,
        body_tokens=0,
        trailer_tokens=8,
        tokens_body_overflow=0,
        tokens_trailer_overflow=0,
        suffix_triggered=True,
    )

    call_log = {"count": 0}

    def fake_generate_with_trailer(model, tokenizer, prompt, **kwargs):
        call_log["count"] += 1
        return replace(first) if call_log["count"] == 1 else replace(second)

    monkeypatch.setattr("src.agents_hf.generate_with_trailer", fake_generate_with_trailer)

    envelope, raw = agent.step("Solve", [])

    assert call_log["count"] >= 2
    assert raw.endswith(CTRL_SUFFIX)
    assert raw == body + completed

    control_meta = envelope["content"]["control"]
    telemetry = control_meta["telemetry"]

    assert telemetry["stopped_on"] == "ctrl_suffix"
    assert telemetry.get("trailer_only_retry") is True
    assert telemetry.get("closed_ctrl") is True
    assert telemetry.get("tokens_used_total") == first.tokens_used + second.tokens_used
    assert telemetry.get("tokens_reserved") == first.tokens_reserved + second.tokens_reserved
    assert "ERR_TRAILER_UNCLOSED" in control_meta.get("errors", [])
