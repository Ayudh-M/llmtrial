from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import dataclass
from typing import Optional


ALLOWED_PERFORMATIVES: tuple[str, ...] = (
    "PROPOSE",
    "CRITIQUE",
    "QUESTION",
    "PLAN",
    "SOLVED",
)


@dataclass(frozen=True)
class ACLParseResult:
    intent: str
    content: str
    next_action: Optional[str] = None


class ACLParseError(ValueError):
    """Raised when an ACL message cannot be parsed into intent/content."""


_ACL_PATTERN = re.compile(r"^(?P<intent>[A-Za-z_]+)\s*:\s*(?P<body>.*)$", re.DOTALL)


def parse_acl_message(text: str) -> ACLParseResult:
    """Parse a light-weight ACL style coordination message.

    The expected structure is ``INTENT: description [=> next_action]`` where ``INTENT`` must
    belong to :data:`ALLOWED_PERFORMATIVES`.
    """

    if not isinstance(text, str):
        raise ACLParseError("ACL message must be a string.")

    stripped = text.strip()
    if not stripped:
        raise ACLParseError("ACL message cannot be empty.")

    match = _ACL_PATTERN.match(stripped)
    if not match:
        raise ACLParseError(
            "ACL message must start with 'INTENT: ...' where INTENT is uppercase."
        )

    intent = match.group("intent").strip().upper()
    if intent not in ALLOWED_PERFORMATIVES:
        allowed = ", ".join(ALLOWED_PERFORMATIVES)
        raise ACLParseError(f"Unknown intent '{intent}'. Allowed intents: {allowed}.")

    body = match.group("body").strip()
    if not body:
        raise ACLParseError("ACL message body cannot be empty.")

    next_action: Optional[str] = None
    if "=>" in body:
        prefix, suffix = body.split("=>", 1)
        body = prefix.strip()
        next_action = suffix.strip()
        if not next_action:
            raise ACLParseError("Next action after '=>' cannot be empty.")

    if not body:
        raise ACLParseError("ACL message content cannot be empty.")

    return ACLParseResult(intent=intent, content=body, next_action=next_action)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def to_json(obj: object) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def normalize_text(text: str | None) -> str:
    if text is None:
        return ""
    normalised = unicodedata.normalize("NFKC", text)
    return "".join(ch for ch in normalised if unicodedata.category(ch) != "Cf")


__all__ = [
    "ACLParseError",
    "ACLParseResult",
    "ALLOWED_PERFORMATIVES",
    "normalize_text",
    "parse_acl_message",
    "sha256_hex",
    "to_json",
]
