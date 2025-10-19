from __future__ import annotations

import textwrap
from typing import Tuple


class PseudocodeValidationError(ValueError):
    """Raised when pseudocode snippets fail validation."""


def _normalise_lines(source: str) -> str:
    dedented = textwrap.dedent(source).strip()
    lines = [line.rstrip() for line in dedented.splitlines()]
    return "\n".join(lines)


def _extract_return(lines: list[str]) -> str:
    for line in reversed(lines):
        stripped = line.strip()
        if stripped.startswith("return"):
            return stripped[len("return") :].strip()
    return ""


def validate_and_normalise_pseudocode(source: str) -> Tuple[str, str]:
    """Normalise pseudocode blocks and derive a return value.

    The normalisation is intentionally lightweight: we dedent the source,
    remove trailing whitespace, and capture the last explicit return value if
    present.  The helper mirrors the behaviour expected by the controller,
    which relies on consistent canonical text and an optional return payload.
    """

    if not isinstance(source, str):
        raise PseudocodeValidationError("Pseudocode must be provided as text.")

    normalised = _normalise_lines(source)
    if not normalised:
        raise PseudocodeValidationError("Pseudocode snippet cannot be empty.")

    lines = normalised.splitlines()
    return_value = _extract_return(lines)
    return normalised, return_value


__all__ = [
    "PseudocodeValidationError",
    "validate_and_normalise_pseudocode",
]
