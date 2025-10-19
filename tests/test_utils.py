import pytest

from src.utils import (
    ACLParseError,
    ALLOWED_DSL_INTENTS,
    ALLOWED_PERFORMATIVES,
    normalize_text,
    parse_acl_message,
    parse_dsl_message,
    sha256_hex,
)


def test_parse_acl_message_success():
    msg = parse_acl_message("PROPOSE: build plan => CONFIRM")
    assert msg.intent == "PROPOSE"
    assert msg.next_action == "CONFIRM"
    assert "build plan" in msg.content


@pytest.mark.parametrize("intent", [p for p in ALLOWED_PERFORMATIVES])
def test_parse_acl_message_allows_defined_intents(intent):
    result = parse_acl_message(f"{intent}: message")
    assert result.intent == intent


def test_parse_acl_message_rejects_unknown_intent():
    with pytest.raises(ACLParseError):
        parse_acl_message("UNKNOWN: hi")


def test_parse_dsl_message():
    parsed = parse_dsl_message("PLAN: compute output => EXECUTE")
    assert parsed["intent"] == "PLAN"
    assert parsed["next_action"] == "EXECUTE"


@pytest.mark.parametrize("intent", ALLOWED_DSL_INTENTS)
def test_parse_dsl_message_accepts_all_intents(intent):
    parsed = parse_dsl_message(f"{intent}: action")
    assert parsed["intent"] == intent


def test_normalize_text_removes_invisible_characters():
    text = "A\u200bB"
    assert normalize_text(text) == "AB"


def test_sha256_hex_consistent():
    assert sha256_hex("abc") == sha256_hex("abc")
