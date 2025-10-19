
import pytest

from src.utils import (
    ALLOWED_PERFORMATIVES,
    ACLParseError,
    ACLParseResult,
    normalize_text,
    parse_acl_message,
    sha256_hex,
)
def test_normalize_nfkc():
    assert normalize_text("Ａ B\u200B C") == "A B C"
def test_sha256():
    assert len(sha256_hex("abc")) == 64


def test_parse_acl_valid():
    msg = "PROPOSE: consider returning TRUE => WAIT_FOR_PEER"
    parsed = parse_acl_message(msg)
    assert isinstance(parsed, ACLParseResult)
    assert parsed.intent == "PROPOSE"
    assert parsed.content == "consider returning TRUE"
    assert parsed.next_action == "WAIT_FOR_PEER"


@pytest.mark.parametrize(
    "text,err",
    [
        ("", "cannot be empty"),
        ("hello", "must start"),
        ("ASK? what", "must start"),
        ("approve: ok", "Unknown intent"),
        ("PROPOSE:   ", "cannot be empty"),
        ("PROPOSE: idea =>   ", "cannot be empty"),
    ],
)
def test_parse_acl_invalid(text, err):
    with pytest.raises(ACLParseError) as exc:
        parse_acl_message(text)
    assert err in str(exc.value)


def test_allowed_performatives_contains_solved():
    assert "SOLVED" in ALLOWED_PERFORMATIVES
