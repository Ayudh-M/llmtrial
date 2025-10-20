from src.agents_hf import (
    HFChatAgent,
    _extract_last_json,
    _retry_instructions,
    _validate_envelope_candidate,
)


class DummyStrategy:
    json_only = True
    decoding = None
    metadata = None

    def decorate_prompts(self, text, context):
        return text


def test_extract_last_json_returns_last_object():
    text = "noise {\"first\": 1} trailing {\"second\": {\"inner\": \"{brace}\"}} extra"
    assert _extract_last_json(text) == '{"second": {"inner": "{brace}"}}'


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


def test_step_retry_keeps_role_alternation(monkeypatch):
    calls = []

    def fake_generate(tokenizer, model, messages, decoding=None):  # pragma: no cover - monkeypatched
        calls.append(messages)
        if len(calls) == 1:
            return "no json here"
        return (
            '{"tag": "[CONTACT]", "status": "PROPOSED", '
            '"content": {"acl": "PROPOSE: ok => WAIT_FOR_PEER"}}'
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
    assert envelope["status"] == "PROPOSED"
    assert len(calls) == 2
    assert [msg["role"] for msg in calls[0]] == ["system", "user"]
    assert [msg["role"] for msg in calls[1]] == ["system", "user"]
    assert "Response did not contain a JSON object." in calls[1][1]["content"]
