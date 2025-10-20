from src.agents_hf import HFChatAgent
from src.control_trailer import envelope_from_payload, extract_control_trailer, validate_control_payload


class DummyStrategy:
    json_only = False
    decoding = None
    metadata = None

    def decorate_prompts(self, text, context):
        return text


def test_extract_control_trailer_returns_last_block():
    text = (
        "analysis line one\n"
        "another line\n"
        "<<<CTRL{\"tag\":\"[PLAN]\",\"status\":\"PROPOSED\",\"intent\":\"PLAN\"}CTRL>>>"
    )

    body, trailer = extract_control_trailer(text)
    assert "analysis line one" in body
    assert trailer == '{"tag":"[PLAN]","status":"PROPOSED","intent":"PLAN"}'


def test_validate_control_payload_enforces_required_fields():
    valid = {
        "tag": "[SOLVER]",
        "status": "READY_TO_SOLVE",
        "intent": "PLAN",
        "final_solution": {"canonical_text": "ANSWER: 60"},
    }
    assert validate_control_payload(valid) == []

    invalid = {"tag": "", "status": "wrong", "intent": "nope"}
    errors = validate_control_payload(invalid)
    assert any("tag" in err.lower() for err in errors)
    assert any("status" in err.lower() for err in errors)
    assert any("intent" in err.lower() for err in errors)


def test_envelope_from_payload_includes_intent_and_body():
    payload = {
        "tag": "[PLAN]",
        "status": "PROPOSED",
        "intent": "PLAN",
        "final_solution": {"canonical_text": "ANSWER: 42"},
    }
    env = envelope_from_payload(payload, body="details here")
    assert env["tag"] == "[PLAN]"
    assert env["status"] == "PROPOSED"
    assert env["content"]["intent"] == "PLAN"
    assert env["content"]["body"] == "details here"
    assert env["final_solution"]["canonical_text"] == "ANSWER: 42"


def test_agent_step_retries_missing_trailer(monkeypatch):
    calls = []

    def fake_generate(tokenizer, model, messages, decoding=None):  # pragma: no cover - monkeypatched
        calls.append(messages)
        if len(calls) == 1:
            return "no trailer here"
        return (
            "Body text\n"
            "<<<CTRL{\"tag\":\"[SOLVER]\",\"status\":\"READY_TO_SOLVE\",\"intent\":\"PLAN\"}CTRL>>>"
        )

    monkeypatch.setattr("src.agents_hf.generate_json_only", fake_generate)

    agent = HFChatAgent(
        name="alpha",
        system_prompt="system",
        tokenizer=None,
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
