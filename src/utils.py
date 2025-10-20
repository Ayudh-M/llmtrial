from __future__ import annotations

"""Utility helpers shared across the project."""

from dataclasses import dataclass
import hashlib
import json
import re
import unicodedata
from typing import Optional

__all__ = [
    "ALLOWED_PERFORMATIVES",
    "ALLOWED_DSL_INTENTS",
    "ACLParseResult",
    "ACLParseError",
    "parse_acl_message",
    "parse_dsl_message",
    "normalize_text",
    "sha256_hex",
    "to_json",
]


ALLOWED_PERFORMATIVES: tuple[str, ...] = (
    "INFORM",
    "REQUEST",
    "PROPOSE",
    "QUERY",
    "CRITIQUE",
    "PLAN",
    "CONFIRM",
    "SOLVED",
)

ALLOWED_DSL_INTENTS: tuple[str, ...] = (
    "DEFINE",
    "PLAN",
    "EXECUTE",
    "REVISE",
    "ASK",
    "CONFIRM",
    "SOLVED",
)


@dataclass(frozen=True)
class ACLParseResult:
    """Parsed representation of a symbolic ACL statement."""

    intent: str
    content: str
    next_action: Optional[str] = None


class ACLParseError(ValueError):
    """Raised when an ACL message is malformed."""


_ACL_PATTERN = re.compile(r"^(?P<intent>[A-Za-z_]+)\s*:\s*(?P<body>.*)$")


def parse_acl_message(text: str) -> ACLParseResult:
    """Parse a symbolic agent communication (ACL) message."""

    if not isinstance(text, str):
        raise ACLParseError("ACL message must be provided as a string.")
    stripped = text.strip()
    if not stripped:
        raise ACLParseError("ACL message cannot be empty.")

    match = _ACL_PATTERN.match(stripped)
    if not match:
        raise ACLParseError("ACL message must start with 'INTENT:'.")

    intent_raw = match.group("intent").strip().upper()
    if intent_raw not in ALLOWED_PERFORMATIVES:
        allowed = ", ".join(ALLOWED_PERFORMATIVES)
        raise ACLParseError(f"Unknown intent '{intent_raw}'. Allowed intents: {allowed}.")

    body = match.group("body").strip()
    if not body:
        raise ACLParseError("ACL message content cannot be empty.")

    next_action: Optional[str] = None
    if "=>" in body:
        before, after = body.split("=>", 1)
        body = before.strip()
        next_action = after.strip()
        if not next_action:
            raise ACLParseError("Next action after '=>' cannot be empty.")

    if not body:
        raise ACLParseError("ACL message content cannot be empty.")

    return ACLParseResult(intent=intent_raw, content=body, next_action=next_action)


_DSL_PATTERN = re.compile(
    r"^(?P<intent>[A-Za-z_]+)\s*:\s*(?P<content>.+?)(?:\s*=>\s*(?P<next>[A-Za-z_]+))?\s*$"
)


def parse_dsl_message(text: str) -> dict[str, Optional[str]]:
    """Validate and parse a DSL statement of the form ``INTENT: content => NEXT``."""

    if not isinstance(text, str):
        raise ValueError("DSL message must be provided as a string.")
    stripped = text.strip()
    if not stripped:
        raise ValueError("DSL message cannot be empty.")

    match = _DSL_PATTERN.match(stripped)
    if not match:
        raise ValueError("DSL message must match 'INTENT: content [=> NEXT]'.")

    intent = match.group("intent").strip().upper()
    if intent not in ALLOWED_DSL_INTENTS:
        allowed = ", ".join(ALLOWED_DSL_INTENTS)
        raise ValueError(f"Unknown DSL intent '{intent}'. Allowed intents: {allowed}.")

    content = match.group("content").strip()
    if not content:
        raise ValueError("DSL content cannot be empty.")

    next_action_raw = match.group("next")
    result = {"intent": intent, "content": content}
    if next_action_raw:
        next_action = next_action_raw.strip().upper()
        if next_action not in ALLOWED_DSL_INTENTS:
            allowed = ", ".join(ALLOWED_DSL_INTENTS)
            raise ValueError(
                f"Unknown DSL next action '{next_action}'. Allowed intents: {allowed}."
            )
        result["next_action"] = next_action
    else:
        result["next_action"] = None
    return result


def normalize_text(text: str) -> str:
    """Normalize unicode text using NFKC and strip invisible characters."""

    if not isinstance(text, str):
        text = str(text)
    normalized = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in normalized if unicodedata.category(ch) != "Cf").strip()


def sha256_hex(text: str) -> str:
    """Return the SHA-256 hex digest for *text*."""

    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_json(obj: object) -> str:
    """Serialize *obj* into pretty-printed JSON."""

    return json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True)
