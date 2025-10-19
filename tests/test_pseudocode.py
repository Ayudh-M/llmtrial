import pytest

from src.pseudocode import (
    PSEUDOCODE_INSERT,
    PseudocodeValidationError,
    augment_system_prompt,
    validate_and_normalise_pseudocode,
)


def test_augment_system_prompt_appends_once():
    prompt = "You are helpful."
    augmented = augment_system_prompt(prompt)
    assert PSEUDOCODE_INSERT in augmented
    augmented_again = augment_system_prompt(augmented)
    assert augmented_again == augmented


def test_validate_and_normalise_success():
    text = """
- step 1: add numbers
- return total
"""
    normalised, return_value = validate_and_normalise_pseudocode(text)
    assert normalised == '- STEP 1: add numbers\n- RETURN total'
    assert return_value == 'total'


def test_validate_and_normalise_requires_return():
    with pytest.raises(PseudocodeValidationError):
        validate_and_normalise_pseudocode("- STEP 1: start")
