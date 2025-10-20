from src.agents_hf import (
    _extract_last_json,
    _retry_instructions,
    _validate_envelope_candidate,
)


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
