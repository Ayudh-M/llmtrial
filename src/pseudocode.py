from __future__ import annotations

"""Helpers for enforcing structured pseudocode outputs."""

import re
from dataclasses import dataclass
from typing import Iterable, Tuple


PSEUDOCODE_INSERT = (
    "PSEUDOCODE FORMAT (strict):\n"
    "- Emit bullet steps beginning with '- STEP <n>: <action>'.\n"
    "- Use '- IF <condition>:' or '- FOR <iterator>:' for control flow.\n"
    "  Indent nested actions by two spaces per level and keep the leading '- '.\n"
    "- Finish the block with '- RETURN <final answer>'.\n"
    "Example:\n"
    "- STEP 1: Gather inputs\n"
    "- FOR each value in data:\n"
    "  - STEP loop: Accumulate sum\n"
    "- IF total > 0:\n"
    "  - RETURN POSITIVE\n"
    "- RETURN ZERO"
)


class PseudocodeValidationError(ValueError):
    """Raised when pseudocode fails format validation."""


@dataclass
class PseudocodeLine:
    level: int
    keyword: str
    text: str


ALLOWED_KEYWORDS = {"STEP", "IF", "FOR", "RETURN", "ELSE"}


def _normalise_spacing(text: str) -> str:
    return " ".join(text.split())


def augment_system_prompt(base: str) -> str:
    base = (base or "").strip()
    if PSEUDOCODE_INSERT in base:
        return base
    if base:
        return base + "\n\n" + PSEUDOCODE_INSERT
    return PSEUDOCODE_INSERT


def _parse_line(raw: str) -> PseudocodeLine:
    match = re.fullmatch(r"((?:  )*)-\s+(.*)", raw.rstrip())
    if not match:
        raise PseudocodeValidationError("Lines must start with '- ' and use two-space indents.")
    indent = match.group(1)
    if len(indent) % 2 != 0:
        raise PseudocodeValidationError("Indentation must be multiples of two spaces.")
    body = match.group(2).strip()
    if not body:
        raise PseudocodeValidationError("Bullet body cannot be empty.")
    keyword = body.split()[0]
    if keyword not in ALLOWED_KEYWORDS:
        raise PseudocodeValidationError(f"Keyword '{keyword}' is not allowed; use {sorted(ALLOWED_KEYWORDS)}.")
    return PseudocodeLine(level=len(indent) // 2, keyword=keyword, text=body)


def _normalise_body(line: PseudocodeLine) -> str:
    body = line.text
    if line.keyword == "STEP":
        match = re.match(r"STEP\s+([^:]+)\s*:\s*(.+)", body, flags=re.IGNORECASE)
        if not match:
            raise PseudocodeValidationError("STEP lines must look like 'STEP <label>: <action>'.")
        label = _normalise_spacing(match.group(1))
        action = _normalise_spacing(match.group(2))
        return f"STEP {label}: {action}"

    if line.keyword == "IF":
        match = re.match(r"IF\s+(.+?):\s*(.*)", body, flags=re.IGNORECASE)
        if not match:
            raise PseudocodeValidationError("IF lines must end with ':' and include a condition.")
        condition = _normalise_spacing(match.group(1))
        tail = _normalise_spacing(match.group(2))
        suffix = f" {tail}" if tail else ""
        return f"IF {condition}:{suffix}"

    if line.keyword == "FOR":
        match = re.match(r"FOR\s+(.+?):\s*(.*)", body, flags=re.IGNORECASE)
        if not match:
            raise PseudocodeValidationError("FOR lines must end with ':' and include a loop description.")
        iterator = _normalise_spacing(match.group(1))
        tail = _normalise_spacing(match.group(2))
        suffix = f" {tail}" if tail else ""
        return f"FOR {iterator}:{suffix}"

    if line.keyword == "ELSE":
        match = re.match(r"ELSE\s*:\s*(.*)", body, flags=re.IGNORECASE)
        if not match:
            raise PseudocodeValidationError("ELSE lines must end with ':'.")
        tail = _normalise_spacing(match.group(1))
        return "ELSE:" + (f" {tail}" if tail else "")

    if line.keyword == "RETURN":
        match = re.match(r"RETURN\s+(.+)", body, flags=re.IGNORECASE)
        if not match:
            raise PseudocodeValidationError("RETURN lines must include a value, e.g. 'RETURN RESULT'.")
        value = _normalise_spacing(match.group(1))
        return f"RETURN {value}"

    raise PseudocodeValidationError(f"Unsupported keyword '{line.keyword}'.")


def _validate_structure(lines: Iterable[PseudocodeLine]) -> None:
    prev_level = 0
    stack = [0]
    for idx, line in enumerate(lines):
        level = line.level
        if level > prev_level + 1:
            raise PseudocodeValidationError("Indentation cannot jump more than one level at a time.")
        if level > len(stack) - 1:
            stack.append(level)
        else:
            stack = stack[: level + 1]
        prev_level = level
        if idx == 0 and line.keyword != "STEP":
            raise PseudocodeValidationError("First line must be a STEP.")


def validate_and_normalise_pseudocode(text: str) -> Tuple[str, str]:
    """
    Validate pseudocode structure and return (normalised_text, final_return_value).
    Empty or whitespace-only text returns a pair of empty strings.
    """

    if not text or not text.strip():
        return "", ""

    raw_lines = [line for line in text.splitlines() if line.strip()]
    parsed = [_parse_line(line) for line in raw_lines]
    _validate_structure(parsed)

    normalized_lines = []
    last_return = ""
    for line in parsed:
        body = _normalise_body(line)
        normalized_lines.append(f"{'  ' * line.level}- {body}")
        if line.keyword == "RETURN":
            last_return = body.split(" ", 1)[1]

    if not last_return:
        raise PseudocodeValidationError("Pseudocode must contain at least one RETURN statement.")

    normalized_text = "\n".join(normalized_lines)
    return normalized_text, last_return

