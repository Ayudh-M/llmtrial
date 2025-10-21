import pytest
from typing import List

from src.agents_hf import HFChatAgent
from src.control_trailer import (
    envelope_from_payload,
    extract_control_trailer,
    normalise_canonical_text,
    validate_control_payload,
)
from src.model_loader import GenerationResult


class StubTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(ch) % 257 for ch in text]


class DummyStrategy:
    id = "dummy"
    name = "dummy"
    json_only = False
    decoding = None
    metadata = {}

    def decorate_prompts(self, text, context):
        return text


def test_extract_control_trailer_returns_last_block():
    text = (
        "analysis line one\n"
        "another line\n"
        "<<<CTRL{\"tag\":\"[PLAN]\",\"status\":\"PROPOSED\",\"intent\":\"PLAN\"}CTRL>>>"
    )

    extraction, failure = extract_control_trailer(text)
    assert failure is None
    assert extraction is not None
    assert "analysis line one" in extraction.body
    assert extraction.json_block == '{"tag":"[PLAN]","status":"PROPOSED","intent":"PLAN"}'
    assert extraction.trailer_start < extraction.trailer_end


def test_extract_control_trailer_requires_terminal_position():
    text = "message\n<<<CTRL{\"tag\":\"[PLAN]\",\"status\":\"PROPOSED\",\"intent\":\"PLAN\"}CTRL>>> extra"

    extraction, failure = extract_control_trailer(text)
    assert extraction is None
    assert failure is not None
    assert failure.reason == "not_at_end"


def test_extract_control_trailer_prefers_last_occurrence():
    text = (
        "alpha\n"
        "<<<CTRL{\"tag\":\"[PLAN]\",\"status\":\"PROPOSED\",\"intent\":\"PLAN\"}CTRL>>>\n"
        "omega\n"
        "<<<CTRL{\"tag\":\"[SOLVER]\",\"status\":\"READY_TO_SOLVE\",\"intent\":\"PROPOSE\"}CTRL>>>"
    )

    extraction, failure = extract_control_trailer(text)
    assert failure is None
    assert extraction is not None
    assert extraction.json_block.startswith('{"tag":"[SOLVER]"')


def test_extract_control_trailer_handles_body_json_payload():
    text = (
        '{"note": "keep {braces} in body"}\n'
        '<<<CTRL{"tag":"[PLAN]","status":"PROPOSED","intent":"PLAN"}CTRL>>>'
    )

    extraction, failure = extract_control_trailer(text)
    assert failure is None
    assert extraction is not None
    assert extraction.body.startswith('{"note"')


def test_extract_control_trailer_rejects_unbalanced_braces():
    text = "<<<CTRL{\"tag\":\"[PLAN\"CTRL>>>"

    extraction, failure = extract_control_trailer(text)
    assert extraction is None
    assert failure is not None
    assert failure.reason == "malformed_json"


def test_validate_control_payload_enforces_required_fields():
    valid = {
        "tag": "[SOLVER]",
        "status": "READY_TO_SOLVE",
        "intent": "PLAN",
        "final_solution": {"canonical_text": "ANSWER: 60"},
    }
    assert validate_control_payload(valid).ok

    invalid = {"tag": "", "status": "wrong", "intent": "nope"}
    result = validate_control_payload(invalid)
    assert not result.ok
    assert any("tag" in err.lower() for err in result.errors)
    assert any("status" in err.lower() for err in result.errors)
    assert any("intent" in err.lower() for err in result.errors)
    assert result.reason == "illegal_transition"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("ANSWER: 60", "ANSWER: 60"),
        ("ANSWER:   60.0  ", "ANSWER: 60"),
        ("ANSWER:  60 km/h", "ANSWER: 60 km/h"),
    ],
)
def test_normalise_canonical_text(raw, expected):
    assert normalise_canonical_text(raw) == expected


def test_envelope_from_payload_includes_intent_and_body():
    payload = {
        "tag": "[PLAN]",
        "status": "PROPOSED",
        "intent": "PLAN",
        "final_solution": {"canonical_text": "ANSWER: 42"},
    }
    env = envelope_from_payload(payload, body="details here", trailer='{"foo": "bar"}')
    assert env["tag"] == "[PLAN]"
    assert env["status"] == "PROPOSED"
    assert env["content"]["intent"] == "PLAN"
    assert env["content"]["body"] == "details here"
    assert env["final_solution"]["canonical_text"] == "ANSWER: 42"
    assert env["content"]["control"]["trailer"] == '{"foo": "bar"}'


def test_agent_step_retries_missing_trailer(monkeypatch):
    calls = []

    def fake_generate(tokenizer, model, messages, decoding=None, **kwargs):  # pragma: no cover - monkeypatched
        calls.append(messages)
        if len(calls) == 1:
            return GenerationResult(
                text="no trailer here",
                stop_reason="max_new_tokens",
                new_tokens=128,
                input_tokens=10,
                max_new_tokens=128,
                trailer_triggered=False,
                body_budget=96,
                reserved_tokens=32,
            )
        return GenerationResult(
            text=(
                "Body text\n"
                "<<<CTRL{\"tag\":\"[SOLVER]\",\"status\":\"READY_TO_SOLVE\",\"intent\":\"PLAN\",\"final_solution\":{\"canonical_text\":\"ANSWER: 1\"}}CTRL>>>"
            ),
            stop_reason="ctrl",
            new_tokens=64,
            input_tokens=10,
            max_new_tokens=128,
            trailer_triggered=True,
            body_budget=96,
            reserved_tokens=32,
        )

    monkeypatch.setattr("src.agents_hf.generate_json_only", fake_generate)

    agent = HFChatAgent(
        name="alpha",
        system_prompt="system",
        tokenizer=StubTokenizer(),
        model=None,
        strategy=DummyStrategy(),
    )

    envelope, raw = agent.step("do task", [], None)
    assert envelope["tag"] == "[SOLVER]"
    assert envelope["content"]["intent"] == "PLAN"
    assert len(calls) == 2
    retry_prompt = calls[1][1]["content"]
    assert "control trailer" in retry_prompt.lower()
    assert "<<<CTRL" in raw
    control_meta = envelope["content"]["control"]
    telemetry = control_meta["telemetry"]
    assert telemetry["retry_count"] == 1
    assert control_meta["first_error"] == "missing_trailer"
    assert control_meta["errors"] == ["missing_trailer"]
    assert telemetry["body_len"] == len("Body text")
    assert telemetry["trailer_len"] > 0
    assert telemetry["stopped_on_ctrl"] is True
    assert telemetry["stop_reason"] == "ctrl"
    assert control_meta["strategy_id"] == "dummy"
    assert telemetry["trailer_triggered"] is True
    assert telemetry["trailer_start"] >= 0
    assert telemetry["trailer_end"] > telemetry["trailer_start"]
    assert control_meta["raw_canonical"] == "ANSWER: 1"
    assert control_meta["normalised_canonical"] == "ANSWER: 1"
    assert telemetry["tokens_used_trailer"] > 0
    assert telemetry["tokens_reserved"] >= 0
    assert telemetry["tokens_used_body"] >= 0
    assert telemetry["tokens_overflow"] >= 0
    assert telemetry["has_tail"] is False


def test_agent_retries_when_trailer_not_at_end(monkeypatch):
    attempts: List[GenerationResult] = [
        GenerationResult(
            text=(
                "First body\n"
                "<<<CTRL{\"tag\":\"[PLAN]\",\"status\":\"PROPOSED\",\"intent\":\"PLAN\"}CTRL>>>."
            ),
            stop_reason="max_new_tokens",
            new_tokens=48,
            input_tokens=12,
            max_new_tokens=96,
            trailer_triggered=False,
            body_budget=64,
            reserved_tokens=32,
        ),
        GenerationResult(
            text=(
                "Clean body\n"
                "<<<CTRL{\"tag\":\"[SOLVER]\",\"status\":\"READY_TO_SOLVE\",\"intent\":\"PLAN\",\"final_solution\":{\"canonical_text\":\"ANSWER: 2\"}}CTRL>>>"
            ),
            stop_reason="ctrl",
            new_tokens=40,
            input_tokens=12,
            max_new_tokens=96,
            trailer_triggered=True,
            body_budget=64,
            reserved_tokens=32,
        ),
    ]

    def fake_generate(tokenizer, model, messages, decoding=None, **kwargs):  # pragma: no cover - monkeypatch
        return attempts.pop(0)

    monkeypatch.setattr("src.agents_hf.generate_json_only", fake_generate)

    strategy = DummyStrategy()
    strategy.metadata = {"body_style": "nl"}
    agent = HFChatAgent("beta", "system", StubTokenizer(), None, strategy)

    envelope, _ = agent.step("task", [], None)
    control_meta = envelope["content"]["control"]
    assert control_meta["first_error"] == "not_at_end"
    assert "not_at_end" in control_meta["errors"]
    telemetry = control_meta["telemetry"]
    assert telemetry["retry_count"] == 1
    assert telemetry["has_tail"] is False


@pytest.mark.parametrize("body_style", ["nl", "json", "pseudocode", "kqml", "dsl"])
def test_agent_telemetry_includes_core_fields(monkeypatch, body_style):
    payload = (
        "Body line\n"
        "<<<CTRL{\"tag\":\"[SOLVER]\",\"status\":\"READY_TO_SOLVE\",\"intent\":\"PLAN\",\"final_solution\":{\"canonical_text\":\"ANSWER: 5\"}}CTRL>>>"
    )

    def fake_generate(tokenizer, model, messages, decoding=None, **kwargs):  # pragma: no cover - monkeypatch
        return GenerationResult(
            text=payload,
            stop_reason="ctrl",
            new_tokens=60,
            input_tokens=14,
            max_new_tokens=120,
            trailer_triggered=True,
            body_budget=88,
            reserved_tokens=32,
        )

    monkeypatch.setattr("src.agents_hf.generate_json_only", fake_generate)

    strategy = DummyStrategy()
    strategy.metadata = {"body_style": body_style}
    agent = HFChatAgent("gamma", "system", StubTokenizer(), None, strategy)
    envelope, _ = agent.step("task", [], None)
    telemetry = envelope["content"]["control"]["telemetry"]
    for field in (
        "stopped_on",
        "tokens_reserved",
        "tokens_used_body",
        "tokens_used_trailer",
        "tokens_overflow",
        "trailer_start",
        "trailer_end",
        "has_tail",
    ):
        assert field in telemetry
    assert telemetry["tokens_used_trailer"] > 0
    assert telemetry["has_tail"] is False
