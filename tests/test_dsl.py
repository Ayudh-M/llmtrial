import pytest

from src.utils import parse_dsl_message


def test_parse_dsl_message_without_next_action():
    parsed = parse_dsl_message("DEFINE: COMP(Pump, power=5kW)")
    assert parsed["intent"] == "DEFINE"
    assert parsed["next_action"] is None


def test_parse_dsl_message_rejects_empty_content():
    with pytest.raises(ValueError):
        parse_dsl_message("PLAN:   ")
