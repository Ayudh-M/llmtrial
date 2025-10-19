import pytest

from src.pseudocode import (
    validate_and_normalise_pseudocode,
    PseudocodeValidationError,
)


def test_validate_and_normalise_pseudocode_extracts_return():
    text = """
- STEP 1:  compute sum
- FOR item in items:
  - STEP loop: add item
- IF total > 0:
  - RETURN POSITIVE
- RETURN ZERO
"""

    normalized, final_return = validate_and_normalise_pseudocode(text)
    assert normalized.splitlines()[0] == "- STEP 1: compute sum"
    assert "  - RETURN POSITIVE" in normalized
    assert normalized.endswith("- RETURN ZERO")
    assert final_return == "ZERO"


def test_validator_rejects_bad_keyword():
    with pytest.raises(PseudocodeValidationError):
        validate_and_normalise_pseudocode("- START: nope")

