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


_ACL_PATTERN = re.compile(r"^(?P<intent>[A-Za-z_]+)\s*:\s*(?P<body>.*)$")


def parse_acl_message(text: str) -> ACLParseResult:
    """Parse an ACL coordination message.

    Messages must follow ``INTENT: text [=> next_action]``. The intent must be one of
    :data:`ALLOWED_PERFORMATIVES`. When ``=>`` is present, the suffix is captured as the
    ``next_action`` directive.
    """

    if not isinstance(text, str):
        raise ACLParseError("ACL message must be a string.")
    stripped = text.strip()
    if not stripped:
        raise ACLParseError("ACL message cannot be empty.")

    m = _ACL_PATTERN.match(stripped)
    if not m:
        raise ACLParseError(
            "ACL message must start with 'INTENT: ...' where INTENT is uppercase."
        )

    intent = m.group("intent").strip().upper()
    if intent not in ALLOWED_PERFORMATIVES:
        allowed = ", ".join(ALLOWED_PERFORMATIVES)
        raise ACLParseError(f"Unknown intent '{intent}'. Allowed intents: {allowed}.")

    body = m.group("body").strip()
    if not body:
        raise ACLParseError("ACL message body cannot be empty.")

    next_action: Optional[str] = None
    if "=>" in body:
        before, after = body.split("=>", 1)
        body = before.strip()
        next_action = after.strip()
        if not next_action:
            raise ACLParseError("Next action after '=>' cannot be empty.")

    if not body:
        raise ACLParseError("ACL message content cannot be empty.")

    return ACLParseResult(intent=intent, content=body, next_action=next_action)


def normalize_text(text: str) -> str:
    """Normalize text using NFKC, drop zero-width spaces, and collapse whitespace."""

    if not isinstance(text, str):
        text = str(text)
    normalized = unicodedata.normalize("NFKC", text)
    normalized = normalized.replace("\u200b", "")
    collapsed = " ".join(normalized.split())
    return collapsed

import json, hashlib, unicodedata

def to_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2)


def sha256_hex(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def normalize_text(text: str) -> str:
    """Normalize text with NFKC and strip invisible format characters."""
    normed = unicodedata.normalize("NFKC", text or "")
    return "".join(ch for ch in normed if unicodedata.category(ch) != "Cf")
